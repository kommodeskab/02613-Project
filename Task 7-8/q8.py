"""

Man skal bruge CUDA for at køre koden
Derfor, skriv:
    > voltash
    > module load python3/3.10
    > module load cuda/12.0
    > source .venv/bin/activate
I konsollen før du kører koden

Profile:
     nsys profile -o Task\ 9-10/cupy_profile python Task\ 9-10/q9.py 
Se analyse:
     nsys stats Task\ 9-10/cupy_profile.nsys-rep 

"""

from os.path import join
import sys
from numba import jit, cuda
import numpy as np
import time


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    i, j = cuda.grid(2)
    if 1 <= i < u.shape[0] - 1 and 1 <= j < u.shape[1] - 1:
        if interior_mask[i - 1, j - 1]:
            u_new[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])

def run_jacobi_gpu(u, interior_mask, max_iter, atol=1e-6):
    u_new = np.empty_like(u)
    d_u = cuda.to_device(u)
    d_u_new = cuda.to_device(u_new)
    d_interior_mask = cuda.to_device(interior_mask)

    threads_per_block = (16, 16)
    blocks_per_grid_x = int(np.ceil(u.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(u.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    for it in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](d_u, d_u_new, d_interior_mask)
        d_u.copy_to_host(u_new)

        delta = np.abs(u - u_new).max()
        u[:] = u_new

        if delta < atol:
            break
    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)

    start_time = time.time()
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = run_jacobi_gpu(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u
    end_time = time.time()
    print(f"Jacobi iterations took {end_time - start_time:.2f} seconds")

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))