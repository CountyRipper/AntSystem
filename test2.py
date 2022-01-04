import numpy as np
list = np.random.rand(5)
print(list)
list = list.cumsum()
print(list)
list -= np.random.rand()
print(list)
print(np.where(list > 0))
print(np.power(16,1/5))