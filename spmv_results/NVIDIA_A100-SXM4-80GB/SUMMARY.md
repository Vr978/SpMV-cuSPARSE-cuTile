# SpMV summary: NVIDIA A100-SXM4-80GB (cc 8.0, copy BW 1721 GB/s)
* Matrices: 12 (8 SuiteSparse: cant, pdb1HYS, rma10, cop20k_A, consph, mac_econ_fwd500, scircuit, webbase-1M; 4 synthetic); largest 15,999,878 nnz; max row length 48,795.
* Correctness: all kernels within FP32 error bounds of the FP64 reference.
* Adaptive vs scalar: geomean 1.51x, max 18.51x (powerlaw).
* Adaptive vs cuSPARSE (best of 2 algs): geomean 0.76x of cuSPARSE throughput; within 10% or faster on 2/12 matrices; faster on 2/12.
* Adaptive bandwidth: median 28% of device copy bandwidth, max 45% (stencil27_3d).
* Irregular matrices (row-length CV > 1: webbase-1M, powerlaw): adaptive vs best single vector kernel geomean 1.50x.
* Mean-length heuristic vs autotuned threads-per-row: heuristic is 1.27x slower on geomean; it picked the best T on 1/12 matrices.
* Error, rows with >= 32,768 nonzeros: scalar 0.4u vs warp tree 0.1u (8.4x lower); max over all kernels/matrices 13.3u.
* cuTile (12 matrices): max error 3.36u, NaN rows 0; device time geomean 0.41x of cuSPARSE throughput and 0.54x of the CUDA C++ adaptive kernel; 5 specialized launches per SpMV (median).
