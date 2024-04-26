import jax
import jax.numpy as jnp
import utils

@jax.jit
def ols(X, Y, Xt):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    if n >= p:
        beta = jnp.linalg.inv(X.T @ X) @ X.T @ Y
    else:
        beta = X.T @ jnp.linalg.inv(X @ X.T) @ Y
    return Xt @ beta


@jax.jit
def ridge_norm(X, Y, Xt, lam):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    X, fX, rX = utils.normalize(X)
    Y, fY, rY = utils.normalize(Y)
    beta_scaled = jnp.linalg.inv(X.T @ X + lam * jnp.eye(p)) @ X.T @ Y
    return rY(fX(Xt) @ beta_scaled)


@jax.jit
def ridge(X, Y, Xt, lam):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    beta_scaled = jnp.linalg.inv(X.T @ X + lam * jnp.eye(p)) @ X.T @ Y
    return Xt @ beta_scaled


@jax.jit
def curds_gcv_cca_full(X, Y, Xt):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    X, fX, rX = utils.normalize(X)
    Y, fY, rY = utils.normalize(Y)

    Tx, Ty, c = utils.cca_full(X, Y)
    c2 = c ** 2

    r = p / n
    dscale = (1 - r) * (c2 - r) / ((1 - r) ** 2 * c2 + r**2 * (1 - c2))
    dscale = jnp.maximum(dscale, 0)
    D = jnp.diag(dscale)

    c_beta = jnp.linalg.pinv(X.T @ X) @ X.T @ Y @ Ty
    beta_scaled = c_beta @ D @ jnp.linalg.inv(Ty)

    return rY(fX(Xt) @ beta_scaled)


@jax.jit
def curds_nocv_cca_full(X, Y, Xt):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    X, fX, rX = utils.normalize(X)
    Y, fY, rY = utils.normalize(Y)

    Tx, Ty, c = utils.cca_full(X, Y)
    c2 = c ** 2

    r = p / n
    dscale = c2 / (c2 + r * (1 - c2))
    D = jnp.diag(dscale)

    c_beta = jnp.linalg.pinv(X.T @ X) @ X.T @ Y @ Ty
    beta_scaled = c_beta @ D @ jnp.linalg.inv(Ty)

    return rY(fX(Xt) @ beta_scaled)

@jax.jit
def curds_gcv_cca_full_nonorm(X, Y, Xt):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    Tx, Ty, c = utils.cca_full(X, Y)
    c2 = c ** 2

    r = p / n
    dscale = (1 - r) * (c2 - r) / ((1 - r) ** 2 * c2 + r**2 * (1 - c2))
    dscale = jnp.maximum(dscale, 0)
    D = jnp.diag(dscale)

    c_beta = jnp.linalg.pinv(X.T @ X) @ X.T @ Y @ Ty
    
    beta_scaled = c_beta @ D @ jnp.linalg.pinv(Ty)

    return Xt @ beta_scaled


@jax.jit
def curds_nocv_cca_full_nonorm(X, Y, Xt):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    Tx, Ty, c = utils.cca_full(X, Y)
    c2 = c ** 2

    r = p / n
    dscale = c2 / (c2 + r * (1 - c2))
    D = jnp.diag(dscale)

    c_beta = jnp.linalg.pinv(X.T @ X) @ X.T @ Y @ Ty
    beta_scaled = c_beta @ D @ jnp.linalg.pinv(Ty)

    return Xt @ beta_scaled

@jax.jit
def curds_nocv_cca_eig(X, Y, Xt):
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape == (n, p)
    assert Y.shape == (n, q)

    Ty, c2 = utils.cca_eig(X, Y)

    r = p / n
    dscale = c2 / (c2 + r * (1 - c2))
    D = jnp.diag(dscale)

    c_beta = jnp.linalg.pinv(X.T @ X) @ X.T @ Y @ Ty
    beta_scaled = c_beta @ D @ jnp.linalg.pinv(Ty)

    return Xt @ beta_scaled
