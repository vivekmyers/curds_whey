import jax
import jax.numpy as jnp
import models
import tqdm
import functools
import matplotlib.pyplot as plt
import collections
import argparse


@functools.partial(jax.jit, static_argnames=["p", "q"])
def sample_beta(key, p, q, rho):
    if args.beta == "gaussian":
        rng1, rng2, rng3 = jax.random.split(key, 3)
        beta = (1 - rho) * jax.random.normal(rng1, (p, q)) + rho * (
            jax.random.normal(rng2, (1, 1)) + jax.random.normal(rng3, (p, 1))
        ) / 2
        return beta
    elif args.beta == "uniform":
        rng1, rng2, rng3 = jax.random.split(key, 3)
        beta = (1 - rho) * jax.random.uniform(rng1, (p, q)) + rho * (
            jax.random.uniform(rng2, (1, 1)) + jax.random.uniform(rng3, (p, 1))
        ) / 2
        return beta
    elif args.beta == "constant":
        return jnp.ones((p, q))


@functools.partial(jax.jit, static_argnames=["n", "p", "q"])
def gen_data(key, n, p, q, beta, eps):
    rng1, rng2 = jax.random.split(key, 2)
    X = jax.random.normal(rng1, (n, p))
    noise = jax.random.normal(rng2, (n, q))
    Y = X @ beta + eps * noise
    return X, Y


@functools.partial(jax.jit, static_argnames=["model", "n", "p", "q"])
def eval_trial(key, model, n, p, q, rho, eps, **kwargs):
    rng1, rng2, rng3 = jax.random.split(key, 3)
    beta = sample_beta(rng1, p, q, rho)
    X, Y = gen_data(rng2, n, p, q, beta, eps)
    Xt, Yt = gen_data(rng3, n, p, q, beta, eps)
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
    plt.rc("font", family="serif")
    results = collections.defaultdict(list)

    for val in tqdm.tqdm(vals):
        # results["Curds"].append(evaluate(key, models.curds_nocv, **{name: val}))
        # results["Curds GCV"].append(evaluate(key, models.curds_gcv, **{name: val}))
        results["Ridge 0.001"].append(evaluate(key, models.ridge, lam=0.001, **{name: val}))
        results["Ridge 0.01"].append(evaluate(key, models.ridge, lam=0.01, **{name: val}))
        results["Ridge 0.1"].append(evaluate(key, models.ridge, lam=0.1, **{name: val}))
        # results["Ridge 10"].append(evaluate(key, models.ridge, lam=10, **{name: val}))
        results["OLS"].append(evaluate(key, models.ols, **{name: val}))
        results["Curds cca_full"].append(evaluate(key, models.curds_nocv_cca_full, **{name: val}))
        # results["Curds cca_eig"].append(evaluate(key, models.curds_nocv_cca_eig, **{name: val}))
        results["Curds gcv"].append(evaluate(key, models.curds_gcv_cca_full, **{name: val}))
        # results["Curds cca"].append(evaluate(key, models.curds_nocv_pinv, **{name: val}))

    for k, data in results.items():
        mean, stderr = zip(*data)
        p = plt.plot(vals, mean, label=k)
        plt.errorbar(vals, mean, yerr=stderr, capsize=5, fmt="o", color=p[0].get_color())

    plt.xlabel(name)
    plt.ylabel("MSE")
    plt.legend()
    title = title or f"Ablate {name}"
    title += f" ({args.beta} model)"
    plt.title(title)
    plt.savefig(f"ablation_{name}_{args.beta}.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, default="n")
    parser.add_argument("--beta", type=str, choices=["gaussian", "uniform", "constant"], default="gaussian")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    key = jax.random.key(args.seed)

    if args.param == "n":
        ablate_param(key, "n", [25, 50, 100, 150, 200], title="Ablation of dataset size")
    if args.param == "p":
        ablate_param(key, "p", [5, 10, 20, 40, 80], title="Ablation of input dimension")
    if args.param == "q":
        ablate_param(key, "q", [5, 10, 20, 40, 80], title="Ablation of output dimension")
    if args.param == "rho":
        ablate_param(key, "rho", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9], title="Ablation of parameter correlations")
    if args.param == "eps":
        ablate_param(key, "eps", [0.01, 0.1, 0.3, 0.6, 1.0], title="Ablation of noise level")

