"""Testbed for testing multiprocessing
"""

import multiprocessing as mp
import numpy as np
import time
from multiprocessing import Pool
import itertools
from functools import partial

# np.random.RandomState(100)
# np.random.seed(52)
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

rs = RandomState(MT19937(SeedSequence(100)))


# ? First we will see a solution without parallelization
def howmany_within_range(row, minimum, maximum):
    """Returns how many numbers lie within 'maximum'
       and 'minimum' in a given array.

    Args:
        row (LIST): The row which contains the data 
        minimum (int): Minimum number
        maximum (int): Max number
    """
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count


# ! Parallelize -> Take a function, run it multiple times on different processors
# ! We initialize a pool first with number of processes
# ! pass the function into pools
# ! Both apply and map take the function to be parallelized as the main argument.

# ! Using Simple Pool.app
def pool_app(row_data):
    # ! Create new function to run in the main if guard as described below.
    # ? Parallelizing using Pool.apply()
    # ? Step 1: Init multiprocessing.Pool()
    print('Using pool.apply()')
    pool = mp.Pool(mp.cpu_count())

    # ? Step 2: 'pool.apply' the 'howmany_within_range()'
    results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in row_data]

    # ? Step 3: Close the pool
    pool.close()

    print(results[:10])
    
    return results


# ! Now using Pool.map
def pool_map_app(row_data):
    print('Using Pool.map')
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(partial(howmany_within_range, minimum=4, maximum=8), [row for row in row_data])

    pool.close()
    print(results[:10])
    return results

# ! Starmap
def pool_star_app(row_data):
    print('Using Pool.starmap!')
    pool = mp.Pool(mp.cpu_count())
    # results = [pool.starmap(howmany_within_range(row, minimum=4, maximum=8)) for row in row_data]
    results = pool.starmap(howmany_within_range, [(row, 4, 8) for row in row_data])
    pool.close()
    print(results[:10])
    return results


def time_it(start, end, funct):
    ex_time = end - start
    print(f'Time taken by {funct} function is {ex_time:0.4} secs\n')

# ! NEED to have a if __name__ guard on the main module to avoid creating subprocesses recursively
if __name__ == '__main__':

    # ? Prepare data
    arr = np.random.randint(0, 10, size=[200000, 5])
    data = arr.tolist()
    data[:5]

    results = []
    start = time.time()
    # ? Adding timing information
    for row in data:
        results.append(howmany_within_range(row, minimum=4, maximum=8))
    end = time.time()
    print(results[:10])
    time_it(start, end, 'Normal')

    start = time.time()
    results = pool_app(data)
    end = time.time()
    time_it(start, end, 'Pool.app()')

    start = time.time()
    results_2 = pool_map_app(data)
    end = time.time()
    time_it(start, end, 'Pool.map()')

    start = time.time()
    results_3 = pool_star_app(data)
    end = time.time()
    time_it(start, end, 'Pool.starmap()')

# TODO Asynchronous Parallel Processing Alternates for above methods.
# TODO The asynchronous equivalents -> apply_async(), map_async() and starmap_async()
