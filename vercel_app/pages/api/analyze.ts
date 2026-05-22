import type { NextApiRequest, NextApiResponse } from 'next';
import sharp from 'sharp';

const TARGET_SIZE = 128;

// ─── Linear Algebra (Pure TypeScript) ────────────────────────────────────────

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function vecNorm(a: Float64Array): number {
  return Math.sqrt(dot(a, a));
}

function normalize(a: Float64Array): Float64Array {
  const n = vecNorm(a);
  if (n < 1e-12) return new Float64Array(a.length);
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] / n;
  return out;
}

/**
 * Power iteration: finds the top singular triplet (u, s, v) of matrix A.
 * A is represented as an array of Float64Array rows.
 */
function powerIter(
  A: Float64Array[],
  tol = 1e-8,
  maxIter = 300,
): { u: Float64Array; s: number; v: Float64Array } {
  const m = A.length;
  const n = A[0].length;

  // Initialise v randomly
  let v = new Float64Array(n);
  for (let j = 0; j < n; j++) v[j] = Math.random() - 0.5;
  v = normalize(v);

  let u = new Float64Array(m);
  let s = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    // u = A @ v
    u = new Float64Array(m);
    for (let i = 0; i < m; i++) u[i] = dot(A[i], v);
    s = vecNorm(u);
    if (s < 1e-12) break;
    u = normalize(u);

    // v_new = A^T @ u
    const vNew = new Float64Array(n);
    for (let j = 0; j < n; j++)
      for (let i = 0; i < m; i++) vNew[j] += A[i][j] * u[i];

    const vNewNorm = vecNorm(vNew);
    const vNewNormalized = normalize(vNew);

    // Check convergence
    let diff = 0;
    for (let j = 0; j < n; j++) diff += (vNewNormalized[j] - v[j]) ** 2;
    v = vNewNormalized;
    s = vNewNorm; // ||A^T u|| = s
    if (Math.sqrt(diff) < tol) break;
  }

  // Recompute u from v for accuracy
  u = new Float64Array(m);
  for (let i = 0; i < m; i++) u[i] = dot(A[i], v);
  s = vecNorm(u);
  u = normalize(u);

  return { u, s, v };
}

/** Deflate: A ← A - s·u·vᵀ */
function deflate(
  A: Float64Array[],
  u: Float64Array,
  s: number,
  v: Float64Array,
): Float64Array[] {
  return A.map((row, i) => {
    const out = new Float64Array(row.length);
    for (let j = 0; j < row.length; j++) out[j] = row[j] - s * u[i] * v[j];
    return out;
  });
}

/** Truncated SVD via deflated power iteration — returns top-k components. */
function truncatedSVD(
  AInput: Float64Array[],
  k: number,
): { U: Float64Array[]; S: number[]; Vt: Float64Array[] } {
  // Deep-copy so we don't mutate the caller's data
  let A = AInput.map(row => new Float64Array(row));
  const U: Float64Array[] = [];
  const S: number[] = [];
  const Vt: Float64Array[] = [];

  for (let i = 0; i < k; i++) {
    const { u, s, v } = powerIter(A);
    if (s < 1e-10) break;
    U.push(u);
    S.push(s);
    Vt.push(v);
    A = deflate(A, u, s, v);
  }
  return { U, S, Vt };
}

/** Reshape flat array into rows of Float64Array. */
function reshape(flat: Float64Array, rows: number, cols: number): Float64Array[] {
  const out: Float64Array[] = [];
  for (let i = 0; i < rows; i++) out.push(flat.slice(i * cols, (i + 1) * cols));
  return out;
}

// ─── Image Pre-processing ─────────────────────────────────────────────────────

function histEqualize(pixels: Float64Array): Float64Array {
  const N = pixels.length;
  const hist = new Int32Array(256);
  const q = new Uint8Array(N);

  for (let i = 0; i < N; i++) {
    const v = Math.min(255, Math.round(pixels[i] * 255));
    q[i] = v;
    hist[v]++;
  }

  const cdf = new Float64Array(256);
  cdf[0] = hist[0];
  for (let i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + hist[i];

  let cdfMin = 0;
  for (let i = 0; i < 256; i++) {
    if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
  }

  const out = new Float64Array(N);
  for (let i = 0; i < N; i++)
    out[i] = (cdf[q[i]] - cdfMin) / Math.max(1, N - cdfMin);
  return out;
}

async function preprocessImage(
  b64: string,
): Promise<{ pixels: Float64Array; detected: boolean }> {
  const base64Data = b64.includes(',') ? b64.split(',')[1] : b64;
  const buffer = Buffer.from(base64Data, 'base64');

  // Decode, resize, convert to grayscale → raw 8-bit pixel buffer
  const raw = await sharp(buffer)
    .resize(TARGET_SIZE, TARGET_SIZE)
    .grayscale()
    .raw()
    .toBuffer();

  const pixels = new Float64Array(TARGET_SIZE * TARGET_SIZE);
  for (let i = 0; i < raw.length; i++) pixels[i] = raw[i] / 255.0;

  return { pixels: histEqualize(pixels), detected: false };
}

// ─── Similarity Metrics ───────────────────────────────────────────────────────

function cosineSim(a: Float64Array, b: Float64Array): number {
  const na = vecNorm(a), nb = vecNorm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}

function euclideanDist(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

function ssimSimple(img1: Float64Array, img2: Float64Array): number {
  const C1 = 0.01 ** 2, C2 = 0.03 ** 2;
  const N = img1.length;
  let mu1 = 0, mu2 = 0;
  for (let i = 0; i < N; i++) { mu1 += img1[i]; mu2 += img2[i]; }
  mu1 /= N; mu2 /= N;

  let s1 = 0, s2 = 0, s12 = 0;
  for (let i = 0; i < N; i++) {
    s1  += (img1[i] - mu1) ** 2;
    s2  += (img2[i] - mu2) ** 2;
    s12 += (img1[i] - mu1) * (img2[i] - mu2);
  }
  s1 /= N; s2 /= N; s12 /= N;

  const num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2);
  const den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2);
  return Math.max(0, Math.min(1, den !== 0 ? num / den : 0));
}

function r4(v: number): number { return Math.round(v * 10000) / 10000; }

// ─── Core PCA/SVD Pipeline ────────────────────────────────────────────────────

function runAnalysis(face1: Float64Array, face2: Float64Array) {
  const N = face1.length;

  // Mean face & centred vectors
  const meanFace = new Float64Array(N);
  for (let i = 0; i < N; i++) meanFace[i] = (face1[i] + face2[i]) / 2;

  const c1 = new Float64Array(N), c2 = new Float64Array(N);
  for (let i = 0; i < N; i++) { c1[i] = face1[i] - meanFace[i]; c2[i] = face2[i] - meanFace[i]; }

  // Joint SVD of 2×N centred matrix → top-2 eigenfaces
  const { S: Sj, Vt: eigenfaces } = truncatedSVD([c1, c2], 2);

  // Project onto eigenspace
  const w1 = new Float64Array(eigenfaces.length);
  const w2 = new Float64Array(eigenfaces.length);
  for (let i = 0; i < eigenfaces.length; i++) {
    w1[i] = dot(eigenfaces[i], c1);
    w2[i] = dot(eigenfaces[i], c2);
  }

  // Individual face SVDs (128×128) — top-15 for visualisation
  const { S: S1 } = truncatedSVD(reshape(face1, TARGET_SIZE, TARGET_SIZE), 15);
  const { S: S2 } = truncatedSVD(reshape(face2, TARGET_SIZE, TARGET_SIZE), 15);

  const svInfo = (S: number[]) => {
    const total = S.reduce((a, v) => a + v * v, 0);
    return S.map((v, i) => ({
      rank: i + 1,
      value: v,
      variance_pct: total > 0 ? (v * v / total) * 100 : 0,
    }));
  };

  // Metrics
  const cosEigen  = cosineSim(w1, w2);
  const eucD      = euclideanDist(w1, w2);
  const eucSim    = 1.0 / (1.0 + eucD);
  const ssimVal   = ssimSimple(face1, face2);
  const cosPixel  = cosineSim(face1, face2);
  const composite =
    0.45 * Math.max(0, cosEigen) +
    0.25 * eucSim +
    0.20 * ssimVal +
    0.10 * Math.max(0, cosPixel);

  return {
    metrics: {
      cosine_similarity_eigenspace: r4(cosEigen),
      euclidean_distance_eigenspace: r4(eucD),
      euclidean_similarity_norm: r4(eucSim),
      ssim_pixel: r4(ssimVal),
      cosine_similarity_pixel: r4(cosPixel),
      composite_score: r4(composite),
    },
    eigenvalues:            Sj.map(s => s * s / 2),
    weights_face1:          Array.from(w1),
    weights_face2:          Array.from(w2),
    singular_values_face1:  svInfo(S1),
    singular_values_face2:  svInfo(S2),
    singular_values_joint:  Sj,
  };
}

// ─── Decision Maker ───────────────────────────────────────────────────────────

function makeDecision(composite: number, cosEigen: number, threshold = 0.70) {
  const isSame = composite >= threshold;
  let level: string, confidence: string, color: string;

  if      (cosEigen >= 0.95) { level = 'Identik';       confidence = 'Sangat Tinggi'; color = '#10b981'; }
  else if (cosEigen >= 0.85) { level = 'Sangat Mirip';  confidence = 'Tinggi';        color = '#22c55e'; }
  else if (cosEigen >= 0.70) { level = 'Mirip';         confidence = 'Sedang';        color = '#f59e0b'; }
  else if (cosEigen >= 0.55) { level = 'Kurang Mirip';  confidence = 'Rendah';        color = '#f97316'; }
  else                       { level = 'Tidak Mirip';   confidence = 'Sangat Rendah'; color = '#ef4444'; }

  return {
    is_same_person: isSame,
    verdict:        isSame ? 'Orang yang Sama' : 'Orang yang Berbeda',
    verdict_icon:   isSame ? '✅' : '❌',
    level, confidence, color,
    threshold_used: threshold,
  };
}

// ─── Next.js API Handler ──────────────────────────────────────────────────────

export const config = { api: { bodyParser: { sizeLimit: '10mb' } } };

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    return res.status(200).end();
  }

  if (req.method !== 'POST')
    return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { image1, image2, threshold = 0.70 } = req.body;
    if (!image1 || !image2)
      return res.status(400).json({ error: 'Kedua gambar diperlukan' });

    const [
      { pixels: face1, detected: det1 },
      { pixels: face2, detected: det2 },
    ] = await Promise.all([preprocessImage(image1), preprocessImage(image2)]);

    const result   = runAnalysis(face1, face2);
    const decision = makeDecision(
      result.metrics.composite_score,
      result.metrics.cosine_similarity_eigenspace,
      Number(threshold),
    );

    return res.status(200).json({
      success: true,
      decision,
      metrics:   result.metrics,
      math_data: {
        eigenvalues:           result.eigenvalues,
        weights_face1:         result.weights_face1,
        weights_face2:         result.weights_face2,
        singular_values_face1: result.singular_values_face1,
        singular_values_face2: result.singular_values_face2,
        singular_values_joint: result.singular_values_joint,
      },
      preprocessing: {
        face1_detected: det1,
        face2_detected: det2,
        image_size:     `${TARGET_SIZE}×${TARGET_SIZE}`,
      },
    });
  } catch (err: any) {
    return res.status(500).json({ error: String(err?.message ?? err), success: false });
  }
}
