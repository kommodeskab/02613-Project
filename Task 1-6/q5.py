"""
HOW TO RUN THIS CODE FROM THE COMMAND LINE:
python Task\ 1-6/q5.py
"""

from os.path import join
import os
import sys
import numpy as np
from tqdm import tqdm

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        # print(f" u : {u[1:-1, 1:-1][interior_mask]}")
        # print(f" shape and len of u : {u[1:-1, 1:-1][interior_mask].shape} {len(u[1:-1, 1:-1][interior_mask])}")
        # break
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

import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

def call_jacobi(args):
    return jacobi(*args)


def run_experiment(N, n_processes, scheduling="dynamic"):
    # Load data
    LOAD_DIR = r'/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
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

    # Prepare input arguments for parallel processing
    input_args = [(u0, mask, MAX_ITER, ABS_TOL) for u0, mask in zip(all_u0, all_interior_mask)]

    start_time = time.perf_counter()

    with Pool(n_processes) as pool:
        if scheduling == "static":
            results = list(tqdm(pool.starmap(jacobi, input_args),
                    total=len(input_args),
                    ncols=100,
                    smoothing=0.1,
                    desc=f"{scheduling.title()} {n_processes} procs"))

        elif scheduling == "dynamic":
            iterator = pool.imap(call_jacobi, input_args)
            results = list(tqdm(iterator, total=len(input_args)))
        else:
            raise ValueError("Unknown scheduling type!")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"{scheduling.title()} scheduling with {n_processes} processes took {elapsed_time:.2f} seconds.")

    return elapsed_time

if __name__ == '__main__':
    N = 64
    n_process_list = [1, 2, 4, 8, 16]
    times_static = []
    times_dynamic = []

    for n_proc in n_process_list:
        # times_static.append(run_experiment(N, n_proc, scheduling="static"))
        times_dynamic.append(run_experiment(N, n_proc, scheduling="dynamic"))

    # Plot
    plt.figure()
    # baseline_static = times_static[0]
    baseline_dynamic = times_dynamic[0]
    # plt.plot(n_process_list, [baseline_static / t for t in times_static], marker='o', label='Static scheduling')
    plt.plot(n_process_list, [baseline_dynamic / t for t in times_dynamic], marker='x', label='Dynamic scheduling')
    plt.xlabel('Number of processes')
    plt.ylabel('Speedup')
    plt.title('Parallel Dynamic Speedup Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    # savefig
    plt.savefig('Task 1-6/speedup_dynamic.png', dpi=300, bbox_inches='tight')

    # save times
    np.save('times_dynamic.npy', times_dynamic)