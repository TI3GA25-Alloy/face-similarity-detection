"""
Entry point untuk Vercel Serverless Function.
Vercel membutuhkan file Python di folder api/ sebagai handler.
File ini cukup mengimpor `app` dari main.py.
"""
import sys
import os

# Tambahkan parent directory ke sys.path agar main.py bisa diimport
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: F401 — Vercel mendeteksi `app` sebagai ASGI handler
