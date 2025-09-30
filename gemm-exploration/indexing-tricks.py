import numpy as np
import numpy.linalg as la

n = m = 128
bn = bm = 8

a = np.random.randn(n, m)
a_flat = a.flatten()

for i in range(n):
    for j in range(m):
        assert (a[i,j] - a_flat[i*n + j]) < 1e-14

tmp = np.zeros_like(a_flat) 
for iouter in range(n // bn):
    for j in range(m):
        for iinner in range(bn):
            tmp[(iouter*bn + iinner)*n + j] = a_flat[(iouter*bn + iinner)*n + j]
assert la.norm(tmp - a_flat) <= 1e-14

tmp = np.zeros_like(a_flat)
for iouter in range(n // bn):
    for jouter in range(m // bm):
        for iinner in range(bn):
            for jinner in range(bm):
                tmp[(iouter*bn + iinner)*n + (jouter*bm + jinner)] = \
                    a_flat[(iouter*bn + iinner)*n + (jouter*bm + jinner)]
assert la.norm(tmp - a_flat) <= 1e-14
