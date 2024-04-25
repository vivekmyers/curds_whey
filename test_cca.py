import numpy as jnp

def cca(X, Y):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    comp = min(p, q)
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    Qx, Rx = jnp.linalg.qr(X)
    Qy, Ry = jnp.linalg.qr(Y)

    U, S, V = jnp.linalg.svd(Qx.T @ Qy)
    V = V.T
    c = S[:comp]

    Tx = jnp.linalg.inv(Rx) @ U[:, :comp]
    Ty = jnp.linalg.inv(Ry) @ V[:, :comp]

    return Tx, Ty, c

def cca_full(X, Y):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    Qx, Rx = jnp.linalg.qr(X)
    Qy, Ry = jnp.linalg.qr(Y)

    comp = min(p, q)
    U, S, V = jnp.linalg.svd(Qx.T @ Qy)
    V = V.T

    # print("U: {} | S: {} | V: {}".format(U.shape, S.shape, V.shape))

    # NOTE: extend c to length q
    c = jnp.zeros((q, ))
    # c = c.at[:comp].set(S)
    c[:comp] = S

    # print("Rx: {} | U: {} | Ry: {} | V: {}".format(Rx.shape, U.shape, Ry.shape, V.shape))

    # NOTE: now Ty is an invertible qxq matrix
    Tx = jnp.linalg.inv(Rx) @ U
    Ty = jnp.linalg.inv(Ry) @ V

    return Tx, Ty, c

def eig_Q(X, Y):
    Q = jnp.linalg.inv(Y.T @ Y) @ Y.T @ X @ jnp.linalg.inv(X.T @ X) @ X.T @ Y
    cc, T = jnp.linalg.eig(Q)
    return T, cc

n = 5
p = 2
q = 3
X = jnp.random.normal(size=(n, p))
b = jnp.random.normal(size=(p, q))
Y = X @ b + jnp.random.normal(size=(n, q))

_, T1, c1 = cca(X, Y)
_, T2, c2 = cca_full(X, Y)
T, cc = eig_Q(X, Y)


# NOTE:
# Conclusion is that CCA implementation is correct!

print(c1**2)
print(cc)

print(T1)
print(T2)
# T is normalized to one
print(T)

print(T2.T @ Y.T @ Y @ T2)

# NOTE:
# experiment to show that we should use full cca
# in beta calculation, we have T @ D @ T^{-1}
# here we compare the two ways of calculation

# res1 = res2 when q <= p
# res1 != res2 when q > p

res1 = T1 @ jnp.diag(c1) @ jnp.linalg.pinv(T1)
res2 = T2 @ jnp.diag(c2) @ jnp.linalg.inv(T2)

print("res1: {}".format(res1))
print("res2: {}".format(res2))
