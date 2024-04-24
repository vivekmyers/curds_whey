import jax
import jax.numpy as jnp
import utils

@jax.jit
def ols(X, Y, Xt):
    beta = jnp.linalg.inv(X.T @ X) @ X.T @ Y
    return Xt @ beta


@jax.jit
def ridge(X, Y, Xt, lam):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    X, fX, rX = utils.normalize(X)
    Y, fY, rY = utils.normalize(Y)
    beta_scaled = jnp.linalg.inv(X.T @ X + lam * jnp.eye(p)) @ X.T @ Y
    return rY(fX(Xt) @ beta_scaled)


@jax.jit
def curds_nocv(X, Y, Xt):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    X, fX, rX = utils.normalize(X)
    Y, fY, rY = utils.normalize(Y)

    ncomp = min(p, q)
    Tx, Ty, c = utils.cca(X, Y)
    c2 = c ** 2

    r = p / n
    dscale = c2 / (c2 + r * (1 - c2))
    D = jnp.diag(dscale)

    c_beta = jnp.linalg.pinv(X.T @ X) @ X.T @ Y @ Ty
    beta_scaled = c_beta @ D @ Ty.T

    return rY(fX(Xt) @ beta_scaled)


@jax.jit
def curds_gcv(X, Y, Xt):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    X, fX, rX = utils.normalize(X)
    Y, fY, rY = utils.normalize(Y)
    ncomp = min(p, q)
    Tx, Ty, c = utils.cca(X, Y)
    c2 = c ** 2

    Xc, Yc = X @ Tx, Y @ Ty
    r = p / n
    dscale = (1 - r) * (c2 - r) / ((1 - r) ** 2 * c2 + r**2 * (1 - c2))
    dscale = jnp.maximum(dscale, 0)
    D = jnp.diag(dscale)
    c_beta = jnp.linalg.pinv(X.T @ X) @ X.T @ Yc
    beta_scaled = c_beta @ D @ Ty.T

    return rY(fX(Xt) @ beta_scaled)

@jax.jit
def curds_nocv_pinv(X, Y, Xt):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    X, fX, rX = utils.normalize(X)
    Y, fY, rY = utils.normalize(Y)
    
    Tx, Ty, c = utils.cca(X, Y)
    c2 = c ** 2

    r = p / n
    dscale = c2 / (c2 + r * (1 - c2))
    D = jnp.diag(dscale)

    c_beta = jnp.linalg.pinv(X.T @ X) @ X.T @ Y @ Ty
    beta_scaled = c_beta @ D @ jnp.linalg.pinv(Ty)

    return rY(fX(Xt) @ beta_scaled)


@jax.jit
def curds_nocv_cca_full(X, Y, Xt):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    X, fX, rX = utils.normalize(X)
    Y, fY, rY = utils.normalize(Y)

    ncomp = min(p, q)
    Tx, Ty, c = utils.cca_full(X, Y)
    c2 = c ** 2

    r = p / n
    dscale = c2 / (c2 + r * (1 - c2))
    D = jnp.diag(dscale)

    c_beta = jnp.linalg.pinv(X.T @ X) @ X.T @ Y @ Ty
    beta_scaled = c_beta @ D @ jnp.linalg.inv(Ty)

    return rY(fX(Xt) @ beta_scaled)
