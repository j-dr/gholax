#!/usr/bin/python3
import yaml
import sys
from . import sampler
from .util import Model
import jax
import numpy as np
import os
jax.config.update("jax_default_matmul_precision", "float32")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".25"
#jax.config.update("jax_log_compiles", True)

def main():
    """CLI entry point for running gholax inference.

    Reads a YAML config from sys.argv[1], optionally restarts from
    checkpoint (sys.argv[2]), instantiates the model and sampler,
    runs the sampler, and saves the resulting chain to text files.
    """
    from .util.distributed import maybe_init_distributed

    with open(sys.argv[1], 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    # Must run before the first device use: joins multi-process (multi-node)
    # runs into one global JAX mesh. No-op for single-process runs.
    distributed = maybe_init_distributed(cfg)
    if distributed:
        print(
            f"jax.distributed initialized: process {jax.process_index()} of "
            f"{jax.process_count()}, {jax.local_device_count()} local / "
            f"{jax.device_count()} global devices",
            flush=True,
        )

    if len(sys.argv) > 2:
        restart = bool(int(sys.argv[2]))
        if restart:
            print('Trying to restart from checkpoint.', flush=True)
    else:
        restart = False
        
    model = Model(cfg)

    scfg = cfg['sampler']
    sampler_type = list(scfg.keys())[0]
    scfg[sampler_type]['restart'] = restart
    s = getattr(sampler, sampler_type)(cfg)
    
    samples, param_names = s.run(model, cfg['output_file'])

    if distributed:
        # One global run: every process holds the full gathered chains;
        # process 0 writes them all under rank tag 0.
        if jax.process_index() == 0:
            for i in range(samples.shape[0]):
                np.savetxt(f"{cfg['output_file']}.samples.0.{i}.txt", samples[i,...], header=' '.join(param_names))
        return

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        for i in range(samples.shape[0]):
            np.savetxt(f"{cfg['output_file']}.samples.{rank}.{i}.txt", samples[i,...], header=' '.join(param_names))
    except ImportError as e:
        for i in range(samples.shape[0]):
            np.savetxt(f"{cfg['output_file']}.samples.0.{i}.txt", samples[i,...], header=' '.join(param_names))

