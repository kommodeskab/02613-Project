"""

Man skal bruge CUDA for at køre koden
Derfor, skriv:
    > voltash
    > module load python3/3.10
    > module load cuda/12.0
    > source .venv/bin/activate
I konsollen før du kører koden

Profile:
     nsys profile -o Task\ 9-10/<filename> python Task\ 9-10/q9.py 
Se analyse:
     nsys stats Task\ 9-10/<filename>.nsys-rep 

"""
from os.path import join
import sys
import cupy as cp
from tqdm import tqdm
import math

def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def cupy_jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = cp.copy(u)
    
    for _ in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[:, 1:-1, :-2] + u[:, 1:-1, 2:] + u[:, :-2, 1:-1] + u[:, 2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = cp.abs(u[:, 1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[:, 1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = cp.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = cp.sum(u_interior < 15) / u_interior.size * 100
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
        print(f"Found {len(building_ids)} building ids.")

    if len(sys.argv) < 2:
        N = 1
    elif sys.argv[1] == 'all':
        N = len(building_ids)
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]
    print(f"Using {N} building ids.")

    # Load floor plans
    all_u0 = cp.empty((N, 514, 514))
    all_interior_mask = cp.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    CHUNK_SIZE = 512
    
    # split all_u0 and all_interior_mask into chunks
    all_u0_splitted = cp.array_split(all_u0, math.ceil(N / CHUNK_SIZE))
    all_interior_mask_splitted = cp.array_split(all_interior_mask, math.ceil(N / CHUNK_SIZE))
    
    print(f"Splitting into {len(all_u0_splitted)} chunks of maximum size {CHUNK_SIZE} each.")
    
    all_u = []
    for u0_chunk, interior_mask_chunk in tqdm(zip(all_u0_splitted, all_interior_mask_splitted), total=len(all_u0_splitted)):
        all_u_chunk = cupy_jacobi(u0_chunk, interior_mask_chunk, MAX_ITER, ABS_TOL)
        all_u.append(all_u_chunk)
        
    all_u = cp.concatenate(all_u, axis=0)
        
    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))