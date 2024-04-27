import jax
import jax.numpy as jnp
import models
import tqdm
import functools
import matplotlib.pyplot as plt
import collections
import argparse
import re
import pickle


@functools.partial(jax.jit, static_argnames=["p", "q"])
def sample_beta(key, p, q, rho):
    rng1, rng2, rng3, rng4 = jax.random.split(key, 4)
    beta = (
        args.beta_noise * jax.random.normal(rng1, (p, q))
        + jax.random.normal(rng2, (1, q))
        + jax.random.normal(rng3, (1, 1))
        + jax.random.normal(rng4, (p, 1))
    )
    return beta


@functools.partial(jax.jit, static_argnames=["n", "p", "q"])
def gen_data(key, n, p, q, rho, beta, eps):
    rng1, rng2 = jax.random.split(key, 2)
    # X = jax.random.normal(rng1, (n, p))
    cov = jnp.eye(p) * (1 - rho) + rho
    # X = X @ jnp.linalg.cholesky(cov)
    X = jax.random.multivariate_normal(rng1, jnp.zeros(p), cov, (n,))
    noise = jax.random.normal(rng2, (n, q))
    Y = X @ beta + eps * noise
    return X, Y


@functools.partial(jax.jit, static_argnames=["n", "p", "q"])
def gen_data_fixed(key, n, p, q, eps, X_fixed, beta_fixed):
    noise = jax.random.normal(key, (n, q))
    Y = X_fixed @ beta_fixed + eps * noise
    return Y


@functools.partial(jax.jit, static_argnames=["model", "n", "p", "q", "fixed"])
def eval_trial(key, model, n, p, q, rho, eps, fixed, **kwargs):
    if fixed:
        beta_fixed = kwargs.pop("beta_fixed")
        X_fixed = kwargs.pop("X_fixed")
        rng1, rng2 = jax.random.split(key, 2)
        X = X_fixed
        Xt = X_fixed
        Y = gen_data_fixed(rng1, n, p, q, eps, X_fixed, beta_fixed)
        Yt = gen_data_fixed(rng2, n, p, q, eps, X_fixed, beta_fixed)
    else:
        rng1, rng2, rng3 = jax.random.split(key, 3)
        beta = sample_beta(rng1, p, q, rho)
        X, Y = gen_data(rng2, n, p, q, rho, beta, eps)
        Xt, Yt = gen_data(rng3, n, p, q, rho, beta, eps)
    Yhat = model(X, Y, Xt, **kwargs)
    return jnp.mean((Yt - Yhat) ** 2)


@functools.partial(jax.jit, static_argnames=["model", "n", "p", "q", "trials", "fixed"])
def evaluate(key, model, n, p, q, trials, rho, eps, fixed, **kwargs):

    key, rng1 = jax.random.split(key)
    if fixed:
        beta_fixed = sample_beta(rng1, p, q, rho)
        kwargs["beta_fixed"] = beta_fixed
        key, rng1, rng2 = jax.random.split(key, 3)
        X_fixed, _ = gen_data(rng2, n, p, q, rho, beta_fixed, eps)
        kwargs["X_fixed"] = X_fixed

    results = jax.vmap(
        lambda x: eval_trial(x, model, n, p, q, rho, eps, fixed, **kwargs)
    )(jax.random.split(key, trials))
    stderr = jnp.std(results) / jnp.sqrt(trials)
    return jnp.mean(results), stderr

def config_desc(config, sweep):
    desc = []
    for k, v in config.items():
        if k == sweep:
            continue
        if sweep == "pq" and k in ["p", "q"]:
            continue
        if k == "eps":
            desc.append(f"$\\epsilon={v}$")
        elif k == "rho":
            desc.append(f"$\\rho={v}$")
        elif k in ["n", "p", "q"]:
            desc.append(f"${k}={v}$")
    desc.append(f"$\\beta_0={args.beta_noise}$")
    parsed = f'({", ".join(desc)})'
    return parsed

def format_key(key):
    if key == "pq":
        return "$p = q$"
    if key in ["n", "p", "q"]:
        return f"${key}$"
    if key == "rho":
        return f"$\\rho$"
    if key == "eps":
        return f"$\\epsilon$"
    return key

def ablate_param(key, name, vals, title=None):
    plt.figure(figsize=(7, 4))
    plt.style.use("ggplot")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    results = collections.defaultdict(list)
    config = {
        "n": args.n,
        "p": args.p,
        "q": args.q,
        "trials": args.trials,
        "rho": args.rho,
        "eps": args.eps,
        "fixed": args.fixed,
    }

    target = f"ablation_{name}"
    if args.fixed:
        target += "_fixed"
    if args.suffix:
        target += f"_{args.suffix}"

    if not args.nocompute:
        for val in tqdm.tqdm(vals):
            kwargs = config.copy()
            if name == "pq":
                kwargs["p"] = val
                kwargs["q"] = val
            else:
                kwargs[name] = val

            if args.curds_only:
                kwargs.pop("eps")

                def run_eps(eps):
                    results[f"Curds GCV ($\\epsilon={eps}$)"].append(evaluate(key, models.curds_gcv_cca_full_nonorm, eps=eps, **kwargs))

                for eps in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
                    run_eps(eps)


            else:
                results["Ridge 0.01"].append(evaluate(key, models.ridge, lam=0.01, **kwargs))
                results["Ridge 0.1"].append(evaluate(key, models.ridge, lam=0.1, **kwargs))
                results["OLS"].append(evaluate(key, models.ols, **kwargs))
                results["Curds"].append(evaluate(key, models.curds_nocv_cca_full_nonorm, **kwargs))
                results["Curds GCV"].append(evaluate(key, models.curds_gcv_cca_full_nonorm, **kwargs))
        pickle.dump(results, open(f"{target}.pkl", "wb"))

    if not args.noplot:
        results = pickle.load(open(f"{target}.pkl", "rb"))
        for k, data in results.items():
            mean, stderr = zip(*data)
            mean = jnp.array(mean)
            stderr = jnp.array(stderr)
            nan_mask = jnp.isnan(mean)
            xval = jnp.array(vals)
            # mean = mean[~nan_mask]
            # stderr = stderr[~nan_mask]
            # xval = xval[~nan_mask]
            p = plt.plot(xval, mean, alpha=0.7, label=k)
            plt.errorbar(xval, mean, yerr=stderr, capsize=3, fmt="o", color=p[0].get_color(), markersize=3)

        plt.yscale("log")
        plt.xlabel(format_key(name))
        plt.ylabel("MSE")
        plt.legend()
        title = title or f"Ablate {name}"
        title += f" {config_desc(config, name)}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"{target}.png", dpi=300)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default="n")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--q", type=int, default=20)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--eps", type=float, default=5.0)
    # parser.add_argument('--shift', type=float, default=5.0)
    parser.add_argument("--beta_noise", type=float, default=0.2)
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--curds_only', action='store_true')
    parser.add_argument('--nocompute', action='store_true')
    parser.add_argument('--noplot', action='store_true')
    args = parser.parse_args()
    key = jax.random.key(args.seed)

    # ablate_param(key, "lam", [0.] + list(jnp.exp(jnp.linspace(-5, 0, 100))), title="Ablation of lambda")
    if args.sweep == "pq":
        ablate_param(
            key,
            "pq",
            [15, 30, 40, 50, 60, 70, 75, 80, 85, 90, 92, 94, 95, 98, 100, 102, 105, 110, 120, 130, 150],
            title="Input/output dimension",
        )
    if args.sweep == "n":
        ablate_param(
            key,
            "n",
            [5, 10, 15, 20, 25, 50, 100, 150, 200],
            title="Dataset size",
        )
    if args.sweep == "p":
        ablate_param(
            key,
            "p",
            [15, 30, 40, 50, 60, 70, 75, 80, 85, 90, 92, 94, 95, 98, 100, 102, 105, 110, 120, 130, 150],
            title="Input dimension",
        )
    if args.sweep == "q":
        ablate_param(
            key, "q",
            [15, 30, 40, 50, 60, 70, 75, 80, 85, 90, 92, 94, 95, 98, 100, 102, 105, 110, 120, 130, 150],
            title="Output dimension"
        )
    if args.sweep == "rho":
        ablate_param(
            key,
            "rho",
            [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            title="Predictor covariance",
        )
    if args.sweep == "eps":
        ablate_param(
            key,
            "eps",
            [0.1, 0.2, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
            title="Noise level",
        )
