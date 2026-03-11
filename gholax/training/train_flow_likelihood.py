import jax.numpy as jnp
import equinox as eqx
from flowjax.flows import masked_autoregressive_flow
from flowjax.distributions import Normal
from flowjax.train import fit_to_data
from flowjax.bijections import RationalQuadraticSpline
from datetime import datetime
import yaml
import numpy as np
import sys
import jax
from scipy.stats import ks_2samp

def train_posterior_flow(theta, weights, train_split=0.8, flow_settings={}):
    
    key = jax.random.key(int(datetime.now().strftime("%Y%m%d%s")))

    N = theta.shape[0]
    Ntrain = int(N * train_split)

    idx = jax.random.choice(key, N, shape=(Ntrain,), replace=True, p=weights)
    theta_training = theta[idx]

    mean = jnp.mean(theta_training, axis=0)
    std = jnp.std(theta_training, axis=0)

    u_training = (theta_training - mean) / std    
    
    key, subkey = jax.random.split(key)
    if flow_settings.get("type", "maf") == "maf": 
        flow = masked_autoregressive_flow(
            subkey,
            base_dist=Normal(jnp.zeros(u_training.shape[1])),
            transformer=RationalQuadraticSpline(knots=flow_settings.get("knots", 16),
                                                interval=flow_settings.get("interval", 4)),
        )    
    else:
        raise NotImplementedError("Currently only MAF flows are supported.")
    
    key, subkey = jax.random.split(key)
    flow, losses = fit_to_data(subkey, flow, u_training,\
                            learning_rate=flow_settings.get("learning_rate", 1e-3))
    
    return flow, mean, std

def load_getdist_samples(file, param_names, ignore_rows=0.3):
    import getdist
    samples = getdist.mcsamples.MCSamples(root=file, settings={'ignore_rows': ignore_rows})
    
    def extract_theta_from_getdist(gd, params):
        chain_names = [p.name for p in gd.paramNames.names]
        idx = [chain_names.index(p) for p in params]
        return jnp.asarray(gd.samples)[:, idx]
    
    theta = extract_theta_from_getdist(samples, param_names)
    w_np = getattr(samples, "weights", None)
    return theta, w_np

def save_flow(path, flow, config, mean, std, bounds, params):
    eqx.tree_serialise_leaves(path + "_flow.eqx", flow)

    with open(path + "_config.yaml", "w") as f:
        yaml.dump(config, f)

def sample_from_flow(flow, mean, std, n_samples, key):
    """Draw samples from the trained flow and transform back to parameter space."""
    u = np.array(flow.sample(key, (n_samples,)))
    return u * np.array(std) + np.array(mean)


def compute_mmd(x, y, n_max=2000, seed=0):
    """Maximum Mean Discrepancy with an RBF kernel (median bandwidth heuristic).

    Subsamples to n_max per set to keep the O(n^2) kernel computation tractable.
    A larger value indicates greater dissimilarity between the two distributions.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) > n_max:
        x = x[rng.choice(len(x), n_max, replace=False)]
    if len(y) > n_max:
        y = y[rng.choice(len(y), n_max, replace=False)]

    all_pts = np.vstack([x, y])
    dists_sq = np.sum((all_pts[:, None] - all_pts[None, :]) ** 2, axis=-1)
    positive = dists_sq[dists_sq > 0]
    h = np.sqrt(np.median(positive) / 2) if len(positive) > 0 else 1.0

    def rbf(a, b):
        d2 = np.sum((a[:, None] - b[None, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2 * h ** 2))

    n, m = len(x), len(y)
    kxx = rbf(x, x)
    kyy = rbf(y, y)
    kxy = rbf(x, y)
    mmd2 = (
        (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        - 2 * kxy.mean()
    )
    return float(np.sqrt(max(mmd2, 0.0)))


def compute_sliced_wasserstein(x, y, n_proj=200, seed=0):
    """Sliced Wasserstein distance: average 1D Wasserstein over random projections.

    Subsamples to the smaller of the two sets so projections are comparable.
    Robust to high dimensions; captures differences in the joint distribution.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    k = min(len(x), len(y))
    x = x[rng.choice(len(x), k, replace=False)]
    y = y[rng.choice(len(y), k, replace=False)]

    d = x.shape[1]
    directions = rng.normal(size=(n_proj, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    sw = 0.0
    for v in directions:
        sw += np.mean(np.abs(np.sort(x @ v) - np.sort(y @ v)))
    return sw / n_proj


def per_param_ks_test(theta_true, theta_flow, param_names):
    """Two-sample KS test on each parameter's 1D marginal.

    Returns a dict {param: {'statistic': float, 'p_value': float}}.
    A small p-value (< 0.05) flags that the marginal is poorly reproduced.
    """
    results = {}
    for i, p in enumerate(param_names):
        stat, pval = ks_2samp(theta_true[:, i], theta_flow[:, i])
        results[p] = {'statistic': float(stat), 'p_value': float(pval)}
    return results


def compare_posteriors_getdist(theta_true, weights_true, theta_flow, param_names, save_path):
    """Save a GetDist triangle plot comparing the true and flow posteriors."""
    import getdist
    import getdist.plots as gdp

    samples_true = getdist.MCSamples(
        samples=np.asarray(theta_true),
        weights=weights_true,
        names=param_names,
        label='MCMC',
    )
    samples_flow = getdist.MCSamples(
        samples=np.asarray(theta_flow),
        names=param_names,
        label='Flow',
    )

    g = gdp.get_subplot_plotter()
    g.triangle_plot([samples_true, samples_flow], filled=True)
    plot_path = save_path + '_flow_comparison.pdf'
    g.export(plot_path)
    print(f"Saved triangle plot to {plot_path}")


def run_flow_diagnostics(flow, mean, std, theta_true, weights_true, param_names,
                         save_path, n_flow_samples=10000, seed=0):
    """Run goodness-of-fit diagnostics and print a summary.

    Tests:
      - Per-parameter two-sample KS test on 1D marginals
      - Maximum Mean Discrepancy (joint, kernel-based)
      - Sliced Wasserstein distance (joint, projection-based)
      - Per-parameter mean and std comparison
      - GetDist triangle plot saved to {save_path}_flow_comparison.pdf
    """
    key = jax.random.key(seed)
    theta_flow = sample_from_flow(flow, mean, std, n_flow_samples, key)

    # Resample true samples according to weights for unweighted comparison
    rng = np.random.default_rng(seed)
    theta_true = np.asarray(theta_true)
    w = np.asarray(weights_true, dtype=float)
    w = w / w.sum()
    idx = rng.choice(len(theta_true), size=n_flow_samples, replace=True, p=w)
    theta_true_resampled = theta_true[idx]

    print("\n=== Flow Posterior Diagnostics ===")

    # 1D marginal KS tests
    ks_results = per_param_ks_test(theta_true_resampled, theta_flow, param_names)
    print(f"\n{'Per-parameter KS test (1D marginals):'}")
    print(f"  {'Parameter':<24} {'KS statistic':>14} {'p-value':>12}")
    for p, r in ks_results.items():
        flag = '  *' if r['p_value'] < 0.05 else ''
        print(f"  {p:<24} {r['statistic']:>14.4f} {r['p_value']:>12.4f}{flag}")

    # Mean and std comparison
    print(f"\n{'Per-parameter mean and std comparison:'}")
    print(f"  {'Parameter':<24} {'mean_true':>12} {'mean_flow':>12} {'std_true':>10} {'std_flow':>10}")
    for i, p in enumerate(param_names):
        print(f"  {p:<24} {theta_true_resampled[:, i].mean():>12.4f} "
              f"{theta_flow[:, i].mean():>12.4f} "
              f"{theta_true_resampled[:, i].std():>10.4f} "
              f"{theta_flow[:, i].std():>10.4f}")

    # Joint distribution tests
    mmd_val = compute_mmd(theta_true_resampled, theta_flow)
    sw_val = compute_sliced_wasserstein(theta_true_resampled, theta_flow)
    print(f"\nMaximum Mean Discrepancy (joint):   {mmd_val:.6f}")
    print(f"Sliced Wasserstein distance (joint): {sw_val:.6f}")

    # Triangle plot
    compare_posteriors_getdist(
        theta_true, weights_true, theta_flow, param_names, save_path
    )


if __name__ == "__main__":
    config_name = sys.argv[1]
    with open(config_name, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    chain_root = config["chain_root"]
    priors = config['params']
    param_names = list(priors.keys())
    
    theta, weights = load_getdist_samples(chain_root, param_names)
    
    flow, mean, std = train_posterior_flow(theta, weights, flow_settings=config)
    config_out = {**config, "mean": mean.tolist(), "std": std.tolist()}

    save_flow(config["save_path"], flow, config_out, mean, std, param_names)

    run_flow_diagnostics(
        flow, mean, std, theta, weights, param_names,
        save_path=config["save_path"],
        n_flow_samples=config.get("n_diagnostic_samples", 10000),
    )