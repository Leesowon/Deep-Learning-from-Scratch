import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))
print(np.dot(B, A))

A = np.array([[1,2,3], [4,5,6]]) #(2,3)
B = np.array([[1,2], [3,4], [5,6]]) #(3,2)
print(np.dot(A,B)) #(2,2)

X = np.array([1,2]) #(1,2)
W = np.array([[1,3,5], [2,4,6]]) #(2,3)
Y = np.dot(X,W)
print(Y) #(1,3)


