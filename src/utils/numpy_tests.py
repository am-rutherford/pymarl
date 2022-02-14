import numpy as np


print(np.array([]))
x = [1, 2, 3]
y = np.append(x, [[4, 5, 6], [7, 8, 9]])
print(x)
print(y)

y.resize((3,3))
print(y)

arr = np.array([1,2,3,4])
print(arr[-1])