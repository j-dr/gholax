#!/usr/bin/python3
import yaml
import sys
import os
import jax
import numpy as np
jax.config.update("jax_default_matmul_precision", "float32")
#jax.config.update("jax_log_compiles", True)

def _setup_compilation_cache(cfg):
    """Point jax at a persistent compilation cache so repeated jobs and
    restarts skip recompilation. Configure with the top-level YAML key
    `compilation_cache_dir` (null disables); multi-process safe. Default is
    $SCRATCH/gholax/jax_cache when SCRATCH is set (NERSC), else ~/.cache."""
    default = (
        os.path.join(os.environ['SCRATCH'], 'gholax', 'jax_cache')
        if os.environ.get('SCRATCH')
        else '~/.cache/gholax/jax_cache'
    )
    cache_dir = cfg.get('compilation_cache_dir', default)
    if cache_dir:
        jax.config.update("jax_compilation_cache_dir", os.path.expanduser(cache_dir))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

def main():
    """CLI entry point for running gholax inference.

    Reads a YAML config from sys.argv[1], optionally restarts from
    checkpoint (sys.argv[2]), instantiates the model and sampler,
    runs the sampler, and saves the resulting chain to text files.
    """
    from .util.distributed import maybe_init_distributed

    with open(sys.argv[1], 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    # Config-only; must run before the first compile, and is safe before
    # maybe_init_distributed (no device use).
    _setup_compilation_cache(cfg)

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

    # Imported only after maybe_init_distributed: this import chain reaches
    # interpax, which materializes a jnp array at module scope and so brings
    # up the XLA backend. Importing it earlier makes
    # jax.distributed.initialize() fail ("must be called before any JAX calls
    # that might initialise the XLA backend") and leaves every task on a node
    # holding all of that node's GPUs, which then OOMs.
    from . import sampler
    from .util import Model

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

