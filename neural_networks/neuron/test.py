import numpy as np
a = np.ones(shape=3)
a[1:] = np.random.rand(2)
print(a)


d={(444,1): 444, (2,2):2, (3,3):3, (1,1):1}
a = set(d.keys())

import time
def f(x):
    print(x)
    time.sleep(0.4)
    return True

a = list(d.keys())
np.random.shuffle(a)
while all(f(x) for x in a):
    np.random.shuffle(a)
