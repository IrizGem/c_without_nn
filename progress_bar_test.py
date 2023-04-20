import numpy as np
# from numba import njit
# from numba_progress import ProgressBar
# import time
# num_iterations = 10000
#
#
# def numba_function(num_iterations, progress_proxy):
#     for i in range(10000):
#         inside_loop(i, progress_proxy)
#         time.sleep(0.001)
#
#
# @njit(nogil=True)
# def inside_loop(i, progress_proxy):
#     print(i)
#     progress_proxy.update(1)
#
#
# with ProgressBar(total=num_iterations) as progress:
#     numba_function(num_iterations, progress)

matrix = np.array([[1, 2, 3], [6, 5, 4]])
print(matrix)
matrix = np.square(matrix)
print(matrix)
matrix = np.sum(matrix)
print(matrix)
