# gholax
Differentiable likelihood through surrogate models (gholas).

## Installation
### Dependencies
- `h5py` for reading emulator data
- `numpy`, `scipy`
- `mpi4py` for running multiple chains simultaneously
- `jax`, `jaxlib`
- `blackjax` for sampling algorithms
- `optax` for minimization
- `interpax` for theory calculations

### NERSC Installation
At NERSC you can run `sh setup_nersc_env.sh` and this should create a functional conda environment,
that you can activate as follows:

```bash
module load python
mamba activate gholax
```
This is equipped with a jupyter kernel named `gholax` that you can use with NERSC's jupyterlab.


