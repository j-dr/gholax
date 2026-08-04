"""Multi-device and multi-process (multi-node) infrastructure.

Builds the 2-D device mesh ('chains', 'model') used to distribute sampling:
chains are data-parallel across the 'chains' axis, and each chain's posterior
evaluation/gradient may additionally be sharded across the 'model' axis
(bin-pair sharding of the projection stage in Nx2PT likelihoods).

Multi-node runs use jax.distributed: one process per GPU launched by slurm
(e.g. `srun --ntasks-per-node=4 --gpus-per-task=1 run-gholax cfg.yaml`), all
processes forming a single global mesh and a single logical run.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp

MODEL_AXIS = "model"
CHAIN_AXIS = "chains"


class ModelShardingContext:
    """Pair-axis sharding of a likelihood pipeline across the 'model' mesh axis.

    Valid only inside jax.shard_map over a mesh containing `axis_name`
    (lax.axis_index is used to select each device's block). Modules slice
    their bin-pair axis to a local block with `slice_local`; the full axis is
    reassembled (replicated across the axis) with `gather_full` at the
    model-vector boundary via zero-padding + psum.

    `pair_counts` maps spectrum type -> full (unpadded) pair-axis length. It
    is populated by Limber at trace time (Limber normalizes every per-pair
    array to the canonical pair axis first, so it is the source of truth) and
    read by downstream modules whose inputs are already local blocks.
    """

    def __init__(self, axis_name, n_shards):
        self.axis_name = axis_name
        self.n_shards = n_shards
        self.pair_counts = {}

    def n_padded(self, n_full):
        """Pair-axis length padded up to a multiple of n_shards."""
        return -(-n_full // self.n_shards) * self.n_shards

    def slice_local(self, a):
        """Zero-pad axis 0 to a multiple of n_shards, return this shard's block.

        Padded rows are zero; every per-pair quantity in the sharded region is
        multiplicative in at least one sliced array, so padded rows carry
        zeros through to the gather, where they are dropped.
        """
        n_full = a.shape[0]
        pad = self.n_padded(n_full) - n_full
        if pad:
            a = jnp.concatenate(
                [a, jnp.zeros((pad,) + a.shape[1:], a.dtype)], axis=0
            )
        block = a.shape[0] // self.n_shards
        idx = jax.lax.axis_index(self.axis_name)
        return jax.lax.dynamic_slice_in_dim(a, idx * block, block, axis=0)

    def gather_full(self, a_local, n_full):
        """Reassemble the full pair axis from local blocks, replicated.

        Implemented as zero-pad + psum rather than all_gather so the result
        is provably replicated across the model axis (keeps every device of a
        chain group in lockstep, and satisfies shard_map's replication rules).
        """
        block = a_local.shape[0]
        idx = jax.lax.axis_index(self.axis_name)
        padded = jax.lax.dynamic_update_slice_in_dim(
            jnp.zeros((block * self.n_shards,) + a_local.shape[1:], a_local.dtype),
            a_local,
            idx * block,
            axis=0,
        )
        return jax.lax.psum(padded, self.axis_name)[:n_full]


def maybe_init_distributed(cfg=None):
    """Initialize jax.distributed for multi-process runs.

    Called before any other JAX device use. Initializes when the config sets
    `distributed: true`, or automatically when running as a multi-task slurm
    step (SLURM_NTASKS > 1) unless the config sets `distributed: false`.
    No-op otherwise, so single-process behavior is unchanged.

    Args:
        cfg: Top-level YAML config dict (or None).

    Returns:
        True if jax.distributed was initialized.
    """
    explicit = None if cfg is None else cfg.get("distributed", None)
    if explicit is False:
        return False

    slurm_multi = int(os.environ.get("SLURM_NTASKS", "1")) > 1
    if not (explicit or slurm_multi):
        return False

    # Coordinator address/process ids are auto-detected from the slurm
    # environment (jax.distributed SlurmCluster support).
    jax.distributed.initialize()
    return True


def build_mesh(sampler_cfg):
    """Build the ('chains', 'model') device mesh from sampler config keys.

    Config keys (both optional):
        n_chains: number of chains (mesh 'chains' axis size). Default:
            total device count // model_shards.
        model_shards: devices sharing each chain's model evaluation
            (mesh 'model' axis size). Default 1.

    Returns None when neither key is present and this is a single-process run,
    in which case callers keep the legacy pmap path (one chain per local
    device). Multi-process runs always get a mesh: the pmap path cannot span
    processes.

    Args:
        sampler_cfg: The config dict for the selected sampler (e.g.
            cfg['sampler']['NUTS']).

    Returns:
        jax.sharding.Mesh with axes ('chains', 'model'), or None.
    """
    has_keys = "n_chains" in sampler_cfg or "model_shards" in sampler_cfg
    if not has_keys and jax.process_count() == 1:
        return None

    model_shards = int(sampler_cfg.get("model_shards", 1))
    n_devices = len(jax.devices())
    if n_devices % model_shards != 0:
        raise ValueError(
            f"model_shards={model_shards} does not divide device count {n_devices}"
        )
    n_chains = int(sampler_cfg.get("n_chains", n_devices // model_shards))
    if n_chains * model_shards != n_devices:
        raise ValueError(
            f"n_chains * model_shards = {n_chains}*{model_shards} must equal "
            f"the total device count {n_devices}"
        )

    devices = np.array(jax.devices()).reshape(n_chains, model_shards)
    return jax.sharding.Mesh(devices, (CHAIN_AXIS, MODEL_AXIS))


def is_io_process():
    """True on the process responsible for checkpoint/output IO."""
    return jax.process_index() == 0


def gather_to_host(x):
    """Return x as a host numpy array with the full global value.

    For single-process runs this is a plain device-to-host transfer. For
    multi-process runs, shards held by other processes are all-gathered so
    every host sees the complete array (needed for R-hat over all chains and
    for checkpoint writes).
    """
    if jax.process_count() == 1:
        return np.asarray(x)
    from jax.experimental import multihost_utils

    return np.asarray(multihost_utils.process_allgather(x, tiled=True))
