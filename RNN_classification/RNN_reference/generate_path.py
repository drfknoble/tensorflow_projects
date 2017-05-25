'''
'''
#%%
#pylint: disable = C0103
#pylint: disable = E1101
import numpy as np
import tensorflow as tf

def heuristic(x, y, x_t, y_t):

    h = abs(x_t - x) + abs(y_t - y)

    return h

n = 4

x_o = 0 #np.random.choice(range(n))
y_o = 0 #np.random.choice(range(n))

x_t = 3 #np.random.choice(range(n))
y_t = 3 #np.random.choice(range(n))

print('Origin: ', '(', x_o, ',', y_o, ')')
print('Target: ', '(', x_t, ',', y_t, ')\n')

path_array = []

for i in range(2):

    x = x_o
    y = y_o

    h = heuristic(x, y, x_t, y_t)

    local_map = np.zeros([n, n], 'int')

    local_map[x, y] = 1

    path_history = [local_map]

    while True:

        viable = False

        delta = np.random.choice([-1, 1])

        direction = np.random.choice([0, 1])
        if direction == 0:
            if abs(x_t - (x + delta)) < abs(x_t - x):
                x = x + delta
                viable = True
        else:
            if abs(y_t - (y + delta)) < abs(y_t - y):
                y = y + delta
                viable = True

        if viable:

            local_map = np.zeros([n, n], 'int')
            local_map[x, y] = 1
            path_history.append(local_map)

            h = heuristic(x, y, x_t, y_t)

        if h == 0:
            break

    path_array.append(path_history)

    print(path_array[i])
print(np.shape(path_array))
