"""cuTile Python SpMV (CSR). Needs an Ampere-or-newer GPU and `pip install cuda-tile[tileiras]`."""
import cuda.tile as ct


@ct.kernel
def spmv_csr_tile(row_ids, row_ptr, col_idx, vals, x, y, n_rows, max_len,
                  R: ct.Constant[int], W: ct.Constant[int]):
    # One block owns R rows of a row-length group; each loop step gathers an R x W tile of nonzeros.
    pid = ct.bid(0)
    lane = ct.arange(R, dtype=ct.int32) + pid * R
    rmask = lane < n_rows
    rows = ct.gather(row_ids, lane, mask=rmask, padding_value=0)
    start = ct.expand_dims(ct.gather(row_ptr, rows), 1)          # (R, 1)
    end = ct.expand_dims(ct.gather(row_ptr, rows + 1), 1)
    rmask2 = ct.expand_dims(rmask, 1)
    off = ct.expand_dims(ct.arange(W, dtype=ct.int32), 0)         # (1, W)
    acc = ct.zeros((R,), dtype=ct.float32)
    for j in range(0, max_len, W):
        k = start + off + j                                       # (R, W) CSR positions
        m = (k < end) & rmask2
        c = ct.gather(col_idx, k, mask=m, padding_value=0)
        v = ct.gather(vals, k, mask=m, padding_value=0.0)
        xv = ct.gather(x, c, mask=m, padding_value=0.0)
        acc = acc + ct.sum(v * xv, axis=1)
    ct.scatter(y, rows, acc, mask=rmask)


def spmv(groups, row_ptr, col_idx, vals, x, y, stream):
    """y = A x: one launch per row-length group, each a specialization of the same kernel."""
    for g in groups:
        grid = ((g["n"] + g["R"] - 1) // g["R"], 1, 1)
        ct.launch(stream, grid, spmv_csr_tile,
                  (g["rows_dev"], row_ptr, col_idx, vals, x, y, g["n"], g["max_len"], g["R"], g["W"]))
