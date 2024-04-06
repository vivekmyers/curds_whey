import jax
import jax.numpy as jnp
import models
import tqdm
import functools
import matplotlib.pyplot as plt
import collections


@functools.partial(jax.jit, static_argnames=["n", "p", "q"])
def gen_data(key, n, p, q, rho, eps):
    rng1, rng2, rng3, rng4, rng5 = jax.random.split(key, 5)
    X = jax.random.normal(rng1, (n, p))
    beta = (1 - rho) * jax.random.normal(rng2, (p, q)) + rho * (
        jax.random.normal(rng3, (1, 1)) + jax.random.normal(rng4, (p, 1))
    ) / 2
    noise = jax.random.normal(rng5, (n, q))
    Y = X @ beta + eps * noise
    return X, Y


@functools.partial(jax.jit, static_argnames=["model", "n", "p", "q"])
def eval_trial(key, model, n, p, q, rho, eps, **kwargs):
    rng1, rng2 = jax.random.split(key)
    X, Y = gen_data(rng1, n, p, q, rho, eps)
    Xt, Yt = gen_data(rng2, n, p, q, rho, eps)
    Yhat = model(X, Y, Xt, **kwargs)
    return jnp.mean((Yt - Yhat) ** 2)


@functools.partial(jax.jit, static_argnames=["model", "n", "p", "q", "trials"])
def evaluate(key, model, n=100, p=10, q=15, trials=1000, rho=0.1, eps=0.1, **kwargs):
    results = jax.vmap(lambda x: eval_trial(x, model, n, p, q, rho, eps, **kwargs))(
        jax.random.split(key, trials)
    )
    stderr = jnp.std(results) / jnp.sqrt(trials)
    return jnp.mean(results), stderr


def ablate_param(key, name, vals, title=None):
    plt.figure(figsize=(7, 4))
    plt.style.use("ggplot")
    plt.rc("text", usetex=True)
    results = collections.defaultdict(list)

    for val in tqdm.tqdm(vals):
        results["Curds"].append(evaluate(key, models.curds, **{name: val}))
        results["OLS"].append(evaluate(key, models.ols, **{name: val}))
        results["Ridge 0.1"].append(evaluate(key, models.ridge, lam=0.1, **{name: val}))
        results["Ridge 1"].append(evaluate(key, models.ridge, lam=1.0, **{name: val}))
        results["Ridge 10"].append(evaluate(key, models.ridge, lam=10, **{name: val}))

    for k, data in results.items():
        mean, stderr = zip(*data)
        p = plt.plot(vals, mean, label=k)
        plt.errorbar(vals, mean, yerr=stderr, capsize=5, fmt="o", color=p[0].get_color())

    plt.xlabel(name)
    plt.ylabel("MSE")
    plt.legend()
    plt.title(title or f"Ablate {name}")
    plt.savefig(f"ablation_{name}.png", dpi=300)


key = jax.random.key(0)

ablate_param(key, "n", [25, 50, 100, 150, 200], title="Ablation of dataset size")
ablate_param(key, "p", [5, 10, 20, 40, 80], title="Ablation of input dimension")
ablate_param(key, "q", [5, 10, 20, 40, 80], title="Ablation of output dimension")
ablate_param(key, "rho", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9], title="Ablation of parameter correlations")
ablate_param(key, "eps", [0.01, 0.1, 0.3, 0.6, 1.0], title="Ablation of noise level")

