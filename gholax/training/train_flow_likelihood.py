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

if __name__ == "__main__":
    config_name = sys.argv[1]
    with open(config_name, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    chain_root = config["chain_root"]
    priors = config['priors']
    param_names = list(priors.keys())
    
    theta, weights = load_getdist_samples(chain_root, param_names)
    
    flow, mean, std = train_posterior_flow(theta, weights, flow_settings=config)
    config_out = {**config, "mean": mean.tolist(), "std": std.tolist()} 
    
    save_flow(config["save_path"], flow, config_out, mean, std, param_names)