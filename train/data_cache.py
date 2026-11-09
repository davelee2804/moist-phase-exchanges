import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


DEFAULT_INPUT_FILES = (
    'Dq_v_loss.txt',
    'Dq_l_loss.txt',
    'Dq_i_loss.txt',
    'mu_v_loss.txt',
    'mu_l_loss.txt',
    'mu_i_loss.txt',
    'temp_loss.txt',
    'rho_loss.txt',
    'qv_loss.txt',
    'ql_loss.txt',
    'qi_loss.txt',
    'eta_loss.txt',
)


def get_cache_path(full_path: str, cache_root: str) -> str:
    source_name = os.path.basename(full_path)
    digest = hashlib.md5(full_path.encode('utf-8')).hexdigest()[:12]
    return os.path.join(cache_root, f'{source_name}.{digest}.npy')


def ensure_cache_task(task):
    full_path, cache_path = task

    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(full_path):
        return (full_path, cache_path, 'hit', 0.0)

    t0 = time.perf_counter()
    arr = np.loadtxt(full_path, dtype=np.float64)
    np.save(cache_path, arr, allow_pickle=False)
    elapsed = time.perf_counter() - t0
    return (full_path, cache_path, 'built', elapsed)


def build_caches_parallel(file_paths, cache_root: str, max_workers: int, logger=None) -> None:
    tasks = [(full_path, get_cache_path(full_path, cache_root)) for full_path in file_paths]
    if logger is not None:
        logger.info(f'Cache prebuild with {max_workers} worker processes for {len(tasks)} files')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for full_path, cache_path, status, elapsed in executor.map(ensure_cache_task, tasks):
            if logger is None:
                continue
            source_name = os.path.basename(full_path)
            if status == 'hit':
                logger.info(f'Cache up-to-date: {source_name}')
            else:
                logger.info(f'Built cache: {cache_path} in {elapsed:.2f} s')


def load_array_cached(full_path: str, cache_root: str, logger=None):
    cache_path = get_cache_path(full_path, cache_root)
    source_name = os.path.basename(full_path)
    t0 = time.perf_counter()

    use_cache = (
        os.path.exists(cache_path) and
        os.path.getmtime(cache_path) >= os.path.getmtime(full_path)
    )

    if use_cache:
        if logger is not None:
            logger.info(f'Loading cache: {cache_path}')
        arr = np.load(cache_path, allow_pickle=False)
        if logger is not None:
            logger.info(f'Loaded {source_name} from cache in {time.perf_counter() - t0:.2f} s')
        return arr

    if logger is not None:
        logger.info(f'Parsing text file: {full_path}')
    arr = np.loadtxt(full_path, dtype=np.float64)
    try:
        np.save(cache_path, arr, allow_pickle=False)
        if logger is not None:
            logger.info(f'Cached binary copy: {cache_path}')
    except OSError as exc:
        if logger is not None:
            logger.warning(f'Could not write cache for {source_name}: {exc}')

    if logger is not None:
        logger.info(f'Loaded {source_name} from text in {time.perf_counter() - t0:.2f} s')
    return arr


def load_filtered_array(base_path: str, filename: str, use_mask, cache_root: str, logger=None):
    full_path = os.path.join(base_path, filename)
    arr = load_array_cached(full_path, cache_root, logger=logger)
    filtered = arr[use_mask]
    if logger is not None:
        logger.info(f'Filtered {filename}: selected {filtered.shape[0]} rows')
    return filtered


def prebuild_default_caches(base_path: str, cache_root: str, max_workers: int, logger=None) -> None:
    file_paths = [os.path.join(base_path, name) for name in DEFAULT_INPUT_FILES]
    build_caches_parallel(file_paths, cache_root, max_workers, logger=logger)
