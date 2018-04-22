# In this file, we verify that our push-forward method returns the same result
# as that of Dr. Scott

import numpy as np
from Assignment_6_Utils import getStiff, getEulerStiff

def getScottStiff(F, n, cCons=0):
    if cCons != 0:
        D = getStiff(n, cCons)
    else:
        D = getStiff(n)
    
    index_map = [[0, 1, 0, 1, 0, 2, 1, 2, 2],
                 [0, 1, 1, 2, 2, 2, 0, 1, 0]]
    loop_map = [0, 1, 2, 3, 4, 5, 2, 3, 4]
    d = [[0.0 for i in range(6)] for j in range(6)]  # pushed forward values
    jac = np.linalg.det(F)
    
    for j in range(6):
        y = index_map[0][j]
        z = index_map[1][j]
        for i in range(j, 6):
            w = index_map[0][i]
            x = index_map[1][i]

            update_value = 0;
            for J in range(9):
                Y = index_map[0][J]
                Z = index_map[1][J]
                for I in range(9):
                    W = index_map[0][I]
                    X = index_map[1][I]

                    cons = D[loop_map[I]][loop_map[J]]
                    update_value += F[w][W]*F[x][X]*F[y][Y]*F[z][Z]*cons/jac
            d[i][j] = update_value
            if (i == 2) and (j == 2):
                print(w, x, y, z)

    for j in range(1, 6):
        for i in range(j):
            d[i][j] = d[j][i]

    return d

a = 0.1
F = [[a + 1, 0, 0], [0, 1, 0], [0, 0, 1]]

#D0 = getStiff(3, [2, 0.25], 1)  # Basic 3D stiffness
D1 = getEulerStiff(F, 3, [2, 0.25])  # My prediction
D2 = getScottStiff(F, 3, [2, 0.25])  # Dr Scott's value

print(D1)
print(D2)
print(np.array(D1) - np.array(D2))
