# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:46:32 2022

@author: ncbcn
"""


import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

#There are 2 main objects in multiprocessing to implement parallel execution of a function:
#The Pool Class and the Process Class.

import numpy as np
from time import time

# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[200000, 5])
data = arr.tolist()
data[:5]
# Solution Without Paralleization

def howmany_within_range(row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

results = []
for row in data:
    results.append(howmany_within_range(row, minimum=4, maximum=8))
print(results[:10])


#The general way to parallelize any operation is to take a particular 
#function that should be run multiple times and make it run parallelly 
#in different processors.
#To do this, you initialize a Pool with n number of processors and pass the function 
#you want to parallelize to one of Pools parallization methods.

#Both apply and map take the function to be parallelized as the main argument.


# Parallelizing using Pool.apply()

import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]

# Step 3: Don't forget to close
pool.close()    

print(results[:10])

pool = mp.Pool(mp.cpu_count())
results = pool.map(howmany_within_range_rowonly, [row for row in data])

results = pool.starmap(howmany_within_range, [(row, 4, 8) for row in data])


#Asynchronous Parallel ync()Processing
#The asynchronous equivalents apply_async(), map_async() and starmap_as
