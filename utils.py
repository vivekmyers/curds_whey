import jax
import jax.numpy as jnp

def normalize(X):
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    forward = lambda x: (x - mu) / sigma
    reverse = lambda x: x * sigma + mu
    return forward(X), forward, reverse

# http://numerical.recipes/whp/notes/CanonCorrBySVD.pdf
@jax.jit
def cca(X, Y):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    comp = min(p, q)
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    Qx, Rx = jnp.linalg.qr(X)
    Qy, Ry = jnp.linalg.qr(Y)

    U, S, V = jnp.linalg.svd(Qx.T @ Qy)
    c = S[:comp]

    Tx = jnp.linalg.pinv(Rx) @ U[:, :comp]
    Ty = jnp.linalg.pinv(Ry) @ V[:, :comp]

    return Tx, Ty, c
