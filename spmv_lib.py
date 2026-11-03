"""Host-side helpers for the SpMV study: matrix suite, CSR I/O, result parsing, cuTile kernel."""

from __future__ import annotations

import io
import pathlib
import subprocess
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

# Classic SpMV benchmark matrices (Williams et al., Bell & Garland) plus a few irregular ones.
SUITESPARSE = [
    ("Williams", "cant", "FEM cantilever"),
    ("Williams", "pdb1HYS", "protein, dense rows"),
    ("Bova", "rma10", "3D CFD"),
    ("Williams", "cop20k_A", "FEM accelerator"),
    ("Williams", "consph", "FEM spheres"),
    ("Williams", "mac_econ_fwd500", "economics, irregular"),
    ("Hamm", "scircuit", "circuit, irregular"),
    ("Williams", "webbase-1M", "web graph, power law"),
]
MIRRORS = ["https://suitesparse-collection-website.herokuapp.com/MM/{g}/{n}.tar.gz",
           "https://sparse.tamu.edu/MM/{g}/{n}.tar.gz"]


def clean(A) -> sp.csr_matrix:
    A = sp.csr_matrix(A, dtype=np.float32)
    A.sum_duplicates()
    A.sort_indices()
    A.indptr = A.indptr.astype(np.int32)
    A.indices = A.indices.astype(np.int32)
    return A


# ------------------------------------------------------------------ synthetic matrices
def laplace2d(k: int) -> sp.csr_matrix:
    """5-point finite-difference Laplacian on a k x k grid (structured, 5 nnz/row)."""
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k))
    I = sp.identity(k)
    return clean(sp.kron(I, T) + sp.kron(T, I))


def stencil27(k: int, seed: int = 0) -> sp.csr_matrix:
    """27-point stencil on a k^3 grid with random weights (structured, 27 nnz/row)."""
    t = sp.diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(k, k))
    A = sp.kron(sp.kron(t, t), t).tocsr()
    A.data = np.random.default_rng(seed).uniform(0.5, 1.5, A.nnz)
    return clean(A)


def random_uniform(n: int, per_row: int, seed: int = 1) -> sp.csr_matrix:
    """Erdos-Renyi style: per_row random columns in every row (unstructured, no locality)."""
    rng = np.random.default_rng(seed)
    cols = np.sort(rng.integers(0, n, (n, per_row)), axis=1).ravel()
    indptr = np.arange(0, n * per_row + 1, per_row)
    return clean(sp.csr_matrix((rng.uniform(-1, 1, n * per_row), cols, indptr), shape=(n, n)))


def powerlaw(n: int, alpha: float = 1.5, scale: float = 4.0, cap: int = 50_000, seed: int = 2):
    """Heavy-tailed row lengths (graph-like): most rows short, a few with tens of thousands of
    nonzeros, and 1% empty rows to exercise the empty-row corner case."""
    rng = np.random.default_rng(seed)
    lens = np.minimum(np.floor(rng.pareto(alpha, n) * scale).astype(np.int64) + 1, cap)
    lens[rng.random(n) < 0.01] = 0
    indptr = np.concatenate([[0], np.cumsum(lens)])
    cols = rng.integers(0, n, int(indptr[-1]))
    return clean(sp.csr_matrix((rng.uniform(-1, 1, cols.size), cols, indptr), shape=(n, n)))


def synthetic_suite(scale: float = 1.0):
    s = lambda v: max(8, int(v * scale))  # noqa: E731
    return {
        "laplace2d": (laplace2d(s(1024)), "2D Poisson, 5-point"),
        "stencil27_3d": (stencil27(s(80)), "3D 27-point stencil"),
        "random_uniform": (random_uniform(s(1_000_000), 16), "random, 16/row"),
        "powerlaw": (powerlaw(s(1_000_000)), "synthetic power law"),
    }


# ------------------------------------------------------------------ SuiteSparse download
def fetch_suitesparse(group: str, name: str, folder: pathlib.Path, timeout: int = 120):
    """Load a SuiteSparse matrix from folder/<name>.mtx or folder/<name>.tar.gz (for example downloaded
    in a browser from sparse.tamu.edu and uploaded). If neither is there, try one plain download."""
    folder.mkdir(parents=True, exist_ok=True)
    mtx, tgz = folder / f"{name}.mtx", folder / f"{name}.tar.gz"
    if not mtx.exists():
        if not (tgz.exists() and tgz.stat().st_size > 0):
            last = None
            for m in MIRRORS:
                try:
                    with urllib.request.urlopen(m.format(g=group, n=name), timeout=timeout) as r:
                        tgz.write_bytes(r.read())
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
            else:
                raise RuntimeError(f"not found locally and download failed ({last}); "
                                   f"download {name}.tar.gz in a browser and put it in {folder}/")
        with tarfile.open(tgz, mode="r:gz") as tf:
            member = next(m for m in tf.getmembers()
                          if m.name.endswith(f"/{name}.mtx") or m.name == f"{name}.mtx")
            mtx.write_bytes(tf.extractfile(member).read())
    return clean(scipy.io.mmread(str(mtx)))


# ------------------------------------------------------------------ CSR binary for the C++ bench
def write_bin(A: sp.csr_matrix, path: pathlib.Path):
    with open(path, "wb") as f:
        np.array([A.shape[0], A.shape[1], A.nnz], dtype=np.int64).tofile(f)
        A.indptr.astype(np.int32).tofile(f)
        A.indices.astype(np.int32).tofile(f)
        A.data.astype(np.float32).tofile(f)


def bench_x(m: int) -> np.ndarray:
    """Same deterministic x in [-1, 1) as spmv_bench.cu (64-bit LCG, top 53 bits)."""
    x = np.empty(m, dtype=np.float32)
    s = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for j in range(m):
        s = (s * 6364136223846793005 + 1442695040888963407) & mask
        x[j] = np.float32((s >> 11) / 9007199254740992.0 * 2.0 - 1.0)
    return x


def bench_x_fast(m: int) -> np.ndarray:
    """Vectorized bench_x via the LCG jump-ahead (identical values, fast for millions)."""
    a, c, mask = 6364136223846793005, 1442695040888963407, (1 << 64) - 1
    s0 = 0x9E3779B97F4A7C15
    out = np.empty(m, dtype=np.uint64)
    # states s_j = A_j * s0 + C_j with A_j = a^j, C_j = c (a^j - 1)/(a - 1), all mod 2^64
    A, C = 1, 0
    block = min(m, 4096)
    Ab, Cb = [], []
    for _ in range(block):
        A, C = (A * a) & mask, (C * a + c) & mask
        Ab.append(A); Cb.append(C)
    Ab = np.array(Ab, dtype=np.uint64); Cb = np.array(Cb, dtype=np.uint64)
    s = s0
    for start in range(0, m, block):
        k = min(block, m - start)
        with np.errstate(over="ignore"):
            out[start:start + k] = Ab[:k] * np.uint64(s) + Cb[:k]
        s = int(out[start + k - 1])
    return ((out >> np.uint64(11)).astype(np.float64) / 9007199254740992.0 * 2.0 - 1.0).astype(np.float32)


# ------------------------------------------------------------------ run + parse
RESULT_COLS = ["matrix", "kernel", "us", "us_min", "gflops", "gbs", "max_err_u", "mean_err_u",
               "launches", "bad_rows"]


def run_bench(binary: str, path: pathlib.Path, name: str, timeout: int = 1800):
    p = subprocess.run([binary, str(path), name], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{name}: spmv_bench failed\n{p.stderr[-2000:]}")
    return p.stdout


def parse(lines: str):
    res, err, mats, dev = [], [], [], None
    for ln in lines.splitlines():
        f = ln.strip().split(",")
        if f[0] == "RESULT":
            res.append(f[1:3] + [float(v) for v in f[3:9]] + [int(f[9]), int(f[10])])
        elif f[0] == "ERRBUCKET":
            err.append(dict(matrix=f[1], kernel=f[2], log2len=int(f[3]), rows=int(f[4]),
                            max_err_u=float(f[5])))
        elif f[0] == "MATRIX":
            mats.append(dict(matrix=f[1], rows=int(f[2]), cols=int(f[3]), nnz=int(f[4]),
                             mean_len=float(f[5]), max_len=int(f[6]), std_len=float(f[7]),
                             empty_rows=int(f[8])))
        elif f[0] == "DEVICE":
            dev = dict(name=f[1], sms=int(f[2]), l2=int(f[3]), cc=f[4], copy_gbs=float(f[5]))
    return pd.DataFrame(res, columns=RESULT_COLS), pd.DataFrame(err), pd.DataFrame(mats), dev


def cusp_tpr(mean_len: float) -> int:
    """Classic CSR-vector heuristic (CUSP): threads per row from the mean row length."""
    for t in (2, 4, 8, 16):
        if mean_len <= t:
            return t
    return 32


# ------------------------------------------------------------------ cuTile (Ampere or newer)
def cutile_groups(indptr: np.ndarray):
    """Rows grouped by row-length class. Each group gets a tile shape (R rows x W nonzeros)
    that cuTile compiles into its own specialized kernel."""
    lens = np.diff(indptr)
    p2 = np.maximum(4, 1 << np.ceil(np.log2(np.maximum(lens, 1))).astype(np.int64))
    p2 = np.minimum(p2, 4096)                 # everything longer shares one group
    groups = []
    for P in np.unique(p2):
        rows = np.nonzero(p2 == P)[0].astype(np.int32)
        W = int(min(P, 256))
        R = int(max(1, 1024 // W))
        groups.append(dict(rows=rows, W=W, R=R, max_len=int(lens[rows].max()), n=len(rows)))
    return groups


def emulate_cutile(indptr, indices, data, x, groups):
    """NumPy emulation of the cuTile kernel's tile logic (used to test the grouping on CPU)."""
    y = np.full(len(indptr) - 1, np.nan, dtype=np.float32)
    for g in groups:
        R, W = g["R"], g["W"]
        for blk in range(-(-g["n"] // R)):
            lane = blk * R + np.arange(R)
            rmask = lane < g["n"]
            rows = np.where(rmask, g["rows"][np.minimum(lane, g["n"] - 1)], 0)
            start, end = indptr[rows], indptr[rows + 1]
            acc = np.zeros(R, dtype=np.float32)
            off = np.arange(W)[None, :]
            for j in range(0, g["max_len"], W):
                k = start[:, None] + off + j
                m = (k < end[:, None]) & rmask[:, None]
                kk = np.where(m, k, 0)
                c = np.where(m, indices[kk], 0)
                v = np.where(m, data[kk], 0).astype(np.float32)
                xv = np.where(m, x[c], 0).astype(np.float32)
                acc = acc + (v * xv).sum(axis=1, dtype=np.float32)
            y[rows[rmask]] = acc[rmask]
    return y
