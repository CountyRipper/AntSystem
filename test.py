import numpy as np
print(np.zeros((1,3)))
path_matrix = np.zeros((3,5))
path_matrix[:,0]=np.random.permutation(range(0,5))[:3]
print("pathmatrix:",path_matrix)
for i in range(3):
    for j in range(1,5):
        path_matrix[i,j] = i+j
print("pathmatrix:",path_matrix)
current_city_list = path_matrix[:,0]
print(current_city_list)
unvisit_list = np.empty([3],dtype=set)
for m in range(3):
    unvisit_list[m] = set(range(5))
    unvisit_list[m].remove(current_city_list[m])
print(unvisit_list)
# matrix = np.empty([5,1],dtype=set)
# for i in range(5):
#     matrix[i] = set(range(5))
#     #print(i)
# print(matrix)
unv = np.zeros(len(unvisit_list[0]))
for i in range(len(unvisit_list[0])):
    unv[i] = i
print(unv)
l =  np.zeros(3)
for i in range(3):
    l[i] = i
print("max=",np.argmax(l))

