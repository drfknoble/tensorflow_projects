'''
'''
#%%
#pylint: disable=C0103
#pylint: disable=E1101
import numpy as np
import tensorflow as tf

x = 1
x1 = [1]
x2 = [[1]]

print(x)
print(x1)
print(x2)

y = [[1], [2]]
print('\ny:', y)
print(np.shape(y))

n = np.array(y)
print('\nn: ', n)
print(np.shape(n))
