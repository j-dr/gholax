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

    with open(sys.argv[1], 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)
        
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
    
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        for i in range(samples.shape[0]):
            np.savetxt(f"{cfg['output_file']}.samples.{rank}.{i}.txt", samples[i,...], header=' '.join(param_names))        
    except ImportError as e:
        for i in range(samples.shape[0]):
            np.savetxt(f"{cfg['output_file']}.samples.0.{i}.txt", samples[i,...], header=' '.join(param_names))     
        
