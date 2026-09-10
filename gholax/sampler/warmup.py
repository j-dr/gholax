"""Sampler-agnostic warmup: metric and step-size adaptation.

Owns the machinery any gradient sampler needs before it can sample - an
inverse mass matrix and a step size - independently of which sampler will
consume them.  Tree-depth tuning is optional and NUTS-specific.
"""

import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from ..util.distributed import is_io_process


def _stale_warmup_parameters(output_file, prefix="nuts"):
    """True when a finished-warmup file coexists with a newer intermediate
    checkpoint, i.e. a later warmup was started for the same prefix."""
    if not output_file:
        return False
    params = f"{output_file}.{prefix}_warmup_parameters.json"
    inter = f"{output_file}.{prefix}_warmup_intermediate.json"
    return (
        os.path.exists(params)
        and os.path.exists(inter)
        and os.path.getmtime(inter) > os.path.getmtime(params)
    )

def _spd_sqrt(A, power=0.5):
    lam, V = jnp.linalg.eigh(A)
    lam = jnp.clip(lam, 1e-30)
    return (V * lam**power) @ V.T


def _spd_guard(S):
    """Symmetrize and floor eigenvalues so the result is SPD."""
    S = 0.5 * (S + S.T)
    lam, V = jnp.linalg.eigh(S)
    lam = jnp.maximum(lam, 1e-6 * jnp.max(lam))
    return (V * lam) @ V.T


def _fisher_metric(C, G, rank, cutoff, reg, n=None):
    """Inverse mass matrix minimizing the sample Fisher divergence to a
    standard normal (Seyboldt, Carlson & Carpenter 2026): C = cov(draws),
    G = cov(scores).  Per-coordinate scales sigma^2 = sqrt(diag C / diag G)
    are read off first; the correlation structure comes from the geometric
    mean of the sigma-whitened covariances, Tikhonov-regularized toward the
    identity there (reg, raised to at least dim/n so directions n < dim
    samples cannot identify fall back to the diagonal scale).  rank None ->
    full whitened geometric mean; "auto"/int -> keep only eigen-directions
    with eigenvalue outside [1/cutoff, cutoff] (int: that many farthest
    from 1)."""
    dim = C.shape[0]
    if n is not None:
        reg = max(reg, dim / max(n, 1))
    cx = jnp.clip(jnp.diag(C), 1e-30)
    cg = jnp.clip(jnp.diag(G), 1e-30)
    sig = jnp.sqrt(jnp.sqrt(cx / cg))
    Cw = C / jnp.outer(sig, sig) + reg * jnp.eye(dim)
    Gw = G * jnp.outer(sig, sig) + reg * jnp.eye(dim)
    # Sw = Gw^-1/2 (Gw^1/2 Cw Gw^1/2)^1/2 Gw^-1/2 solves Sw Gw Sw = Cw
    Gh, Gmh = _spd_sqrt(Gw, 0.5), _spd_sqrt(Gw, -0.5)
    Sw = Gmh @ _spd_sqrt(Gh @ Cw @ Gh, 0.5) @ Gmh
    if rank is not None:
        lam, U = jnp.linalg.eigh(0.5 * (Sw + Sw.T))
        if rank == "auto":
            keep = (lam >= cutoff) | (lam <= 1.0 / cutoff)
        else:
            order = jnp.argsort(-jnp.abs(jnp.log(jnp.clip(lam, 1e-30))))
            keep = jnp.zeros(dim, bool).at[order[: int(rank)]].set(True)
        lam_k = jnp.where(keep, lam - 1.0, 0.0)
        Sw = jnp.eye(dim) + (U * lam_k) @ U.T
        print(f"  fisher low-rank metric: kept {int(jnp.sum(keep))} directions "
              f"outside [1/{cutoff:g}, {cutoff:g}]", flush=True)
    return _spd_guard(Sw * jnp.outer(sig, sig))


def stuck_chains(pos):
    """Chains that never moved in a window, and the ones that did.

    A chain whose every proposal diverges never moves.  Left alone it
    feeds frozen values into the pooled tail covariance and the mixing
    R-hat for the rest of warmup, and is still frozen at sampling.
    Positions are exchangeable before convergence, so re-seeding from a
    chain that did move is initialization, not a chain edit.

    Args:
        pos: (window_steps, n_chains, dim) window positions.

    Returns:
        (stuck_idx, live_idx).  Both empty when nothing is stuck or when
        no chain moved at all (a global failure re-seeding cannot fix).
    """
    empty = jnp.zeros((0,), dtype=int)
    if pos.shape[0] < 2 or pos.shape[1] < 2:
        return empty, empty
    moved = jnp.abs(jnp.diff(pos, axis=0)).sum(axis=(0, 2)) > 0
    stuck, live = jnp.where(~moved)[0], jnp.where(moved)[0]
    if stuck.size == 0 or live.size == 0:
        return empty, empty
    return stuck, live


ALGORITHMS = ("window", "adaptive_window", "pooled_window", "mclmc")
NUTS_ALGORITHMS = ("window", "adaptive_window", "pooled_window")
MCLMC_ALGORITHMS = ("mclmc", "pooled_window")

# Tree depth is tuned by NUTS only; rejected in another sampler's block.
TREE_DEPTH_FIELDS = (
    "max_doublings",
    "sampling_max_num_doublings",
    "sampling_depth_auto",
    "depth_cap_quantile",
    "depth_cap_margin",
    "depth_cap_margin_saturated",
)

# Derived from key *presence*, never written by the user.
_DERIVED_FIELDS = ("step_size_search", "sampling_depth_auto")

# Flat sampler-level key -> WarmupConfig field.  Accepted with a
# DeprecationWarning; the nested `warmup:` block is the new spelling.
LEGACY_KEYS = {
    "warmup_algorithm": "algorithm",
    "n_steps_warmup": "n_steps",
    "warmup_init_file": "init_file",
    "diagonal_mass_matrix": "diagonal_mass_matrix",
    "diagonal_preconditioning": "diagonal_mass_matrix",
    "step_size_init": "step_size_init",
    "target_acceptance_rate": "target_acceptance_rate",
    "pooled_window_steps": "window_steps",
    "pooled_window_max_steps": "max_steps",
    "pooled_window_max_window": "max_window",
    "pooled_window_terminal_steps": "terminal_steps",
    "pooled_window_consecutive_windows": "consecutive_windows",
    "pooled_window_rtol_mass": "rtol_mass",
    "pooled_window_rtol_step": "rtol_step",
    "pooled_window_mass_stat": "mass_stat",
    "pooled_window_mixing_rhat": "mixing_rhat",
    "pooled_window_mixing_quantile": "mixing_quantile",
    "pooled_window_min_tail_steps": "min_tail_steps",
    "pooled_window_allow_unconverged": "allow_unconverged",
    "pooled_window_metric_estimator": "metric_estimator",
    "pooled_window_fisher_cutoff": "fisher_cutoff",
    "pooled_window_fisher_reg": "fisher_reg",
    "pooled_window_dense_rank": "dense_rank",
    "pooled_window_dense_update": "dense_update",
    "pooled_window_max_doublings": "max_doublings",
    "max_num_doublings": "sampling_max_num_doublings",
    "depth_cap_quantile": "depth_cap_quantile",
    "depth_cap_margin": "depth_cap_margin",
    "depth_cap_margin_saturated": "depth_cap_margin_saturated",
    "adaptive_warmup_stage_steps": "stage_steps",
    "adaptive_warmup_max_steps": "adaptive_max_steps",
    "adaptive_warmup_min_steps": "adaptive_min_steps",
    "adaptive_warmup_rtol_mass": "adaptive_rtol_mass",
    "adaptive_warmup_rtol_step": "adaptive_rtol_step",
}

# Per-sampler overrides of the shared defaults.
SAMPLER_WARMUP_DEFAULTS = {
    "NUTS": {},
    "MCLMC": {
        "algorithm": "mclmc",
        "n_steps": 5000,
        "step_size_init": 0.01,
        "target_acceptance_rate": 0.65,
    },
}

# Deprecated sampler attribute -> WarmupConfig field, for samplers that
# still expose the flat names (see NUTS.__getattr__).
LEGACY_ATTRS = {
    key: field
    for key, field in LEGACY_KEYS.items()
    if key not in ("diagonal_preconditioning", "max_num_doublings")
}
LEGACY_ATTRS["max_num_doublings_auto"] = "sampling_depth_auto"
LEGACY_ATTRS["step_size_search"] = "step_size_search"

_WARNED = set()


def _warn_legacy(sampler, key, field):
    if (sampler, key) in _WARNED:
        return
    _WARNED.add((sampler, key))
    warnings.warn(
        f"sampler.{sampler}.{key} is deprecated; use "
        f"sampler.{sampler}.warmup.{field}",
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class WarmupConfig:
    """Metric/step-size adaptation settings, independent of the sampler.

    Constructed from a sampler config by `from_sampler_config`, which is
    also where key *presence* is turned into the derived `step_size_search`
    and `sampling_depth_auto` flags: those two options change behaviour by
    being absent, not by their value.
    """

    # selection and budget
    algorithm: str = "pooled_window"
    n_steps: int = 500
    init_file: str = None
    restart: bool = False

    # shared metric / step size
    diagonal_mass_matrix: bool = True
    step_size_init: float = 0.05
    step_size_search: bool = True
    target_acceptance_rate: float = 0.8

    # pooled window
    window_steps: int = 25
    max_steps: int = 200
    max_window: int = 40
    terminal_steps: int = None
    consecutive_windows: int = 2
    rtol_mass: float = None
    rtol_step: float = None
    mass_stat: str = "rms_diag"
    mixing_rhat: float = 1.2
    mixing_quantile: float = 1.0
    min_tail_steps: int = 8
    allow_unconverged: bool = True
    metric_estimator: str = "covariance"
    fisher_cutoff: float = 1.5
    fisher_reg: float = 1e-4
    dense_rank: object = None
    dense_update: bool = True

    # tree depth (NUTS only)
    max_doublings: int = 8
    sampling_max_num_doublings: int = 10
    sampling_depth_auto: bool = True
    depth_cap_quantile: float = 0.9
    depth_cap_margin: int = 1
    depth_cap_margin_saturated: int = 2

    # adaptive window
    stage_steps: int = 100
    adaptive_max_steps: int = 1000
    adaptive_min_steps: int = 200
    adaptive_rtol_mass: float = 0.05
    adaptive_rtol_step: float = 0.05

    def __post_init__(self):
        set_ = lambda k, v: object.__setattr__(self, k, v)
        if self.algorithm not in ALGORITHMS:
            raise ValueError(
                f"warmup_algorithm must be one of {ALGORITHMS}, "
                f"got '{self.algorithm}'"
            )
        set_("mixing_rhat", float(self.mixing_rhat))
        set_("mixing_quantile", float(self.mixing_quantile))
        if not 0.0 < self.mixing_quantile <= 1.0:
            raise ValueError("pooled_window_mixing_quantile must be in (0, 1]")
        if self.mixing_rhat < 1.0:
            raise ValueError("pooled_window_mixing_rhat must be >= 1")
        set_("min_tail_steps", int(self.min_tail_steps))
        if self.min_tail_steps < 2:
            raise ValueError("pooled_window_min_tail_steps must be >= 2")
        set_("consecutive_windows", int(self.consecutive_windows))
        if self.consecutive_windows < 1:
            raise ValueError("pooled_window_consecutive_windows must be >= 1")
        if self.mass_stat not in ("rms_diag", "max_diag"):
            raise ValueError("pooled_window_mass_stat must be 'rms_diag' or 'max_diag'")
        if self.metric_estimator not in ("covariance", "fisher"):
            raise ValueError(
                "pooled_window_metric_estimator must be 'covariance' or 'fisher'"
            )
        set_("fisher_cutoff", float(self.fisher_cutoff))
        set_("fisher_reg", float(self.fisher_reg))
        if self.dense_rank not in (None, "auto") and not (
            isinstance(self.dense_rank, int) and self.dense_rank >= 0
        ):
            raise ValueError(
                "pooled_window_dense_rank must be None, 'auto' or a non-negative int"
            )
        set_("dense_update", bool(self.dense_update))
        # None sentinels resolve here so the engine only sees numbers.
        if self.terminal_steps is None:
            set_("terminal_steps", self.window_steps)
        set_("terminal_steps", int(self.terminal_steps))
        if self.terminal_steps < 1:
            raise ValueError("pooled_window_terminal_steps must be >= 1")
        for key in ("rtol_mass", "rtol_step"):
            if getattr(self, key) is None:
                legacy = getattr(self, "adaptive_" + key)
                set_(key, legacy if self.mass_stat == "max_diag" else 0.1)
            set_(key, float(getattr(self, key)))
        set_("depth_cap_margin", int(self.depth_cap_margin))
        set_("depth_cap_margin_saturated", int(self.depth_cap_margin_saturated))

    @classmethod
    def from_sampler_config(cls, c, *, sampler, tree_depth=True):
        """Build from a `sampler.<name>` config dict.

        Reads the nested `warmup:` block, folds in the deprecated flat keys
        (nested wins), rejects unknown keys, and resolves the presence-derived
        flags.
        """
        names = {f.name for f in fields(cls)}
        block = c.get("warmup", {})
        if not isinstance(block, Mapping):
            raise ValueError(f"sampler.{sampler}.warmup must be a mapping")
        allowed = names - set(_DERIVED_FIELDS)
        if not tree_depth:
            allowed -= set(TREE_DEPTH_FIELDS)
        unknown = set(block) - allowed
        if unknown:
            depth = unknown & set(TREE_DEPTH_FIELDS)
            if depth and not tree_depth:
                raise ValueError(
                    f"tree-depth options apply to NUTS only: {sorted(depth)}"
                )
            raise ValueError(
                f"unknown key(s) in sampler.{sampler}.warmup: {sorted(unknown)}; "
                f"valid keys: {sorted(allowed)}"
            )

        merged = dict(SAMPLER_WARMUP_DEFAULTS.get(sampler, {}))
        defaulted = set(merged)
        for key, field_ in LEGACY_KEYS.items():
            if key not in c:
                continue
            _warn_legacy(sampler, key, field_)
            if field_ in block:
                warnings.warn(
                    f"sampler.{sampler}.{key} ignored: "
                    f"sampler.{sampler}.warmup.{field_} is also set",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            merged[field_] = c[key]
        merged.update(block)
        # Explicitly given by the user, as opposed to a per-sampler default.
        explicit = (set(merged) - defaulted) | (set(merged) & set(block))

        merged["step_size_search"] = "step_size_init" not in explicit
        merged["sampling_depth_auto"] = "sampling_max_num_doublings" not in explicit
        # rtol precedence: explicit rtol_* > (explicit adaptive_rtol_* or
        # mass_stat max_diag) ? adaptive_rtol_* : 0.1
        mass_stat = merged.get("mass_stat", "rms_diag")
        for key in ("rtol_mass", "rtol_step"):
            legacy = "adaptive_" + key
            if key in explicit:
                continue
            if legacy in explicit or mass_stat == "max_diag":
                merged[key] = merged.get(legacy, getattr(cls, legacy, None))
            else:
                merged[key] = 0.1

        cfg = cls(**merged)
        algorithms = NUTS_ALGORITHMS if tree_depth else MCLMC_ALGORITHMS
        if cfg.algorithm not in algorithms:
            raise ValueError(
                f"warmup_algorithm must be one of {algorithms}, got '{cfg.algorithm}'"
            )
        return cfg


@dataclass(frozen=True)
class WarmupHost:
    """What the warmup engine needs from the sampler that owns it.

    Built by BaseSampler._warmup_host so the engine never holds the sampler
    itself: the coupling is these six values plus the config.
    """

    chains_per_device: int = 1
    param_names: object = None
    sample_transform: bool = False
    mesh: object = None
    prefix: str = "nuts"

    def is_io_process(self):
        return is_io_process()


class WarmupCheckpoint:
    """Read/write the warmup checkpoint files for one sampler prefix.

    Paths and JSON keys are a public API (gholax.util.postprocess_chain and
    the samplers' restart paths read them); the prefix keeps them
    sampler-specific.
    """

    def __init__(self, output_file, host):
        self.host = host
        prefix = host.prefix
        self.intermediate = (
            f"{output_file}.{prefix}_warmup_intermediate.json" if output_file else None
        )
        self.parameters = (
            f"{output_file}.{prefix}_warmup_parameters.json" if output_file else None
        )

    def read_intermediate(self, check_transform=True):
        """The checkpoint dict, or None when there is nothing to resume."""
        if not self.intermediate or not os.path.exists(self.intermediate):
            return None
        with open(self.intermediate, "r") as fp:
            ck = json.load(fp)
        if check_transform and bool(ck.get("sample_transform", False)) != bool(
            self.host.sample_transform
        ):
            raise RuntimeError(
                "Restart refused: warmup checkpoint uses a different sampling "
                "coordinate convention (sample_transform mismatch)."
            )
        return ck

    def write_intermediate(self, payload):
        if not self.intermediate or not self.host.is_io_process():
            return
        with open(self.intermediate, "w") as fp:
            json.dump(payload, fp)

    def is_stale(self, output_file):
        return _stale_warmup_parameters(output_file, self.host.prefix)


@dataclass
class WarmupRequest:
    """Per-call inputs to a warmup run."""

    logdensity_fn: object
    initial_positions: object
    rng_key: object
    initial_inverse_mass_matrix: object = None
    initial_step_size: object = None
    output_file: str = None


@dataclass
class WarmupResult:
    """Adapted sampling parameters and where the chains ended up."""

    inverse_mass_matrix: object
    step_size: object
    positions: object = None
    state: object = None
    converged: bool = False
    calibrated: bool = False
    max_num_doublings: int = None
    diagnostics: dict = field(default_factory=dict)


class Warmup:
    """Metric and step-size adaptation, decoupled from the sampler.

    Consumes a WarmupConfig and a WarmupHost; returns a WarmupResult. Tree
    depth is tuned only when tune_tree_depth is set (NUTS).
    """

    def __init__(self, config, host, *, tune_tree_depth=True):
        self.config = config
        self.host = host
        self.tune_tree_depth = tune_tree_depth

    def run(self, req):
        if self.config.algorithm == "adaptive_window":
            return self.run_adaptive_window(req)
        if self.config.algorithm == "pooled_window":
            return self.run_pooled_window(req)
        raise NotImplementedError(
            f"warmup algorithm '{self.config.algorithm}' is not run by this engine"
        )

    def run_adaptive_window(self, req):
        """Window adaptation in stages, stopping once mass matrix and step
        size converge.

        Every stage is checkpointed; with restart an existing checkpoint
        resumes from that stage.  A warm start (initial_step_size) seeds the
        convergence check so adaptation can stop after a single stage.
        """
        cfg = self.config
        ckpt = WarmupCheckpoint(req.output_file, self.host)

        prev_mass = None
        prev_step = None
        position = req.initial_positions
        total_steps = 0
        rng_key = req.rng_key
        step_size_init = cfg.step_size_init

        # The provided mass matrix seeds the first stage only; later stages
        # warm-start from the previous stage's adapted one.
        current_imm = req.initial_inverse_mass_matrix

        if req.initial_step_size is not None:
            step_size_init = float(req.initial_step_size)
            if req.initial_inverse_mass_matrix is not None:
                prev_mass = jnp.asarray(req.initial_inverse_mass_matrix)
                prev_step = jnp.asarray(step_size_init)

        ck = ckpt.read_intermediate(check_transform=False) if cfg.restart else None
        if ck is not None:
            position = jnp.array(ck["position"])
            prev_mass = jnp.array(ck["inverse_mass_matrix"])
            prev_step = jnp.asarray(ck["step_size"])
            total_steps = ck["total_steps"]
            current_imm = None
            print(
                f"Resuming adaptive warmup from checkpointed step {total_steps}",
                flush=True,
            )

        while total_steps < cfg.adaptive_max_steps:
            rng_key, sub_key = jax.random.split(rng_key)

            warmup_kwargs = {
                "is_mass_matrix_diagonal": cfg.diagonal_mass_matrix,
                "progress_bar": False,
                "initial_step_size": step_size_init,
                "target_acceptance_rate": cfg.target_acceptance_rate,
            }
            if current_imm is not None:
                warmup_kwargs["initial_inverse_mass_matrix"] = current_imm

            warmup = blackjax.window_adaptation(
                blackjax.nuts, req.logdensity_fn, **warmup_kwargs
            )
            (state, parameters), _ = warmup.run(sub_key, position, cfg.stage_steps)
            current_imm = None

            mass = parameters["inverse_mass_matrix"]
            step = parameters["step_size"]
            total_steps += cfg.stage_steps

            ckpt.write_intermediate(
                {
                    "inverse_mass_matrix": np.asarray(mass).tolist(),
                    "step_size": float(step),
                    "position": np.asarray(state.position).tolist(),
                    "total_steps": total_steps,
                }
            )

            if prev_mass is not None and total_steps >= cfg.adaptive_min_steps:
                mass_change = float(
                    jnp.max(jnp.abs(mass - prev_mass) / (jnp.abs(prev_mass) + 1e-10))
                )
                step_change = float(
                    jnp.abs(step - prev_step) / (jnp.abs(prev_step) + 1e-10)
                )
                print(
                    f"Adaptive warmup step {total_steps}: "
                    f"max_rel_mass_change={mass_change:.4f}, "
                    f"rel_step_change={step_change:.4f}",
                    flush=True,
                )
                if (
                    mass_change < cfg.adaptive_rtol_mass
                    and step_change < cfg.adaptive_rtol_step
                ):
                    print(f"Warmup converged after {total_steps} steps", flush=True)
                    return WarmupResult(
                        inverse_mass_matrix=mass,
                        step_size=step,
                        state=state,
                        converged=True,
                        diagnostics={"total_steps": total_steps},
                    )

            prev_mass = mass
            prev_step = step
            position = state.position

        print(
            f"Warmup reached max {cfg.adaptive_max_steps} steps without convergence",
            flush=True,
        )
        return WarmupResult(
            inverse_mass_matrix=mass,
            step_size=step,
            state=state,
            converged=False,
            diagnostics={"total_steps": total_steps},
        )

    def run_pooled_window(self, req):
        """Cross-chain windowed warmup that keeps NUTS end to end.

        All chains step together through windows that double in length each
        boundary (Stan-style, from warmup.window_steps up to
        warmup.max_window) with a fixed diagonal
        inverse mass matrix, while the step size is dual-averaged every step
        from the cross-chain mean acceptance (Stan-style within-window
        feedback). At each window boundary the mass matrix is re-estimated
        from the pooled within-chain position variance, Stan-regularized
        toward the current inverse mass matrix, and
        dual averaging is restarted at its averaged step size (as blackjax
        window_adaptation does at slow-window ends). Pooling yields
        n_chains samples per sequential step, so far fewer sequential steps
        are needed than in single-chain window adaptation. Boundaries are
        checkpointed to {output_file}.<sampler>_warmup_intermediate.json and
        resumable with restart=True. Stops when mass matrix and step size
        stabilize (warmup.rtol_mass/rtol_step; warm starts seed the
        check) or at warmup.max_steps.

        Returns:
            A WarmupResult carrying the final per-chain positions.
        """
        from blackjax.adaptation.step_size import (
            dual_averaging_adaptation,
            find_reasonable_step_size,
        )
        from blackjax.mcmc import nuts as nuts_mcmc

        cfg = self.config
        if self.host.mesh is not None:
            raise ValueError(
                "warmup_algorithm 'pooled_window' is not supported in mesh "
                "mode (n_chains/model_shards)"
            )
        jlp = req.logdensity_fn
        rng_key = req.rng_key
        initial_positions = req.initial_positions
        initial_inverse_mass_matrix = req.initial_inverse_mass_matrix
        initial_step_size = req.initial_step_size
        ckpt = WarmupCheckpoint(req.output_file, self.host)

        n_chains, dim = initial_positions.shape
        n_window = cfg.window_steps
        param_names = self.host.param_names
        kernel = nuts_mcmc.build_kernel()

        da_init, da_update, da_final = dual_averaging_adaptation(
            target=cfg.target_acceptance_rate
        )

        # Adaptation runs pmap(vmap): K chains vmapped per device, devices
        # synchronized through an axis-name pmean so the per-step
        # dual-averaging update sees the acceptance averaged across ALL
        # chains — every device advances an identical da_state. This uses
        # all GPUs during warmup and spreads the memory footprint.
        # Depth cap bounds per-step cost while the metric is still poor: an
        # untuned mass matrix drives NUTS to max depth, and vmap lockstep
        # makes every chain pay the deepest tree.
        nd = jax.local_device_count()
        assert n_chains % nd == 0
        K = n_chains // nd
        max_doublings = cfg.max_doublings
        vkernel = jax.vmap(
            lambda k, s, step, imm: kernel(
                k, s, jlp, step, imm, max_num_doublings=max_doublings
            ),
            in_axes=(0, 0, None, None),
        )
        vinit = jax.vmap(lambda p: nuts_mcmc.init(p, jlp))

        #reshape for pmap
        def _dev_split(tree):
            return jax.tree.map(
                lambda x: x.reshape((nd, K) + x.shape[1:]), tree
            )

        #inv of _dev_split
        def _flat_positions(states):
            return np.asarray(states.position).reshape(n_chains, -1)

        # imm / da_state are traced arguments so every window reuses one
        # compilation. Step size feeds back per step (Stan-style), which is
        # what keeps dual averaging stable; updating it only at boundaries
        # gives no within-window feedback and ping-pongs.
        def _run_window(dev_key, states, da_state, imm, n_steps):
            def one_step(carry, k):
                states, da_state = carry
                step = jnp.exp(da_state.log_step_size)
                states, infos = vkernel(
                    jax.random.split(k, K), states, step, imm
                )
                acc = jax.lax.pmean(jnp.mean(infos.acceptance_rate), "d")
                da_state = da_update(da_state, acc)
                return (states, da_state), (
                    states.position,
                    states.logdensity_grad,
                    acc,
                    infos.num_integration_steps,
                )

            (states, da_state), (pos, grad, acc, n_leap) = jax.lax.scan(
                one_step, (states, da_state), jax.random.split(dev_key, n_steps)
            )
            return states, da_state, pos, grad, acc, n_leap

        # ``n_steps`` is static so a final partial window performs exactly the
        # number of transitions left in the strict pooled-step budget.  A
        # dynamic lax.scan length would either fail to compile or tempt the
        # caller to run a full window and merely discard the excess samples.
        # JAX caches the ordinary window and (usually) one short remainder
        # specialization.
        prun_window = jax.pmap(
            _run_window,
            axis_name="d",
            in_axes=(0, 0, None, None),
            static_broadcasted_argnums=(4,),
        )

        def run_window(key, states, da_state, imm, n_steps):
            states, da_dev, pos, grad, acc, n_leap = prun_window(
                jax.random.split(key, nd), states, da_state, imm, n_steps
            )
            # da_state is device-invariant (pmean-synchronized); host
            # consumers get flat (n_steps, n_chains, ...) arrays.
            da_state = jax.tree.map(lambda x: x[0], da_dev)
            pos = jnp.moveaxis(pos, 0, 1).reshape(n_steps, n_chains, -1)
            grad = jnp.moveaxis(grad, 0, 1).reshape(n_steps, n_chains, -1)
            n_leap = jnp.moveaxis(n_leap, 0, 1).reshape(n_steps, n_chains)
            return states, da_state, pos, grad, acc[0], n_leap

        positions = initial_positions
        imm = (
            jnp.asarray(initial_inverse_mass_matrix)
            if initial_inverse_mass_matrix is not None
            else jnp.ones(dim)
        )
        step_size = jnp.asarray(
            initial_step_size
            if initial_step_size is not None
            else cfg.step_size_init
        )
        prev_mass = None
        prev_step = None
        # ``next_window_chunks`` is the size of the next growing boundary,
        # rather than the size of the boundary just completed.  Keeping this
        # distinction in the checkpoint makes a restart schedule-identical.
        next_window_chunks = 1
        stable_boundaries = 0
        resumed_dense_updated = False
        if initial_inverse_mass_matrix is not None and initial_step_size is not None:
            # Warm start: seed the convergence check so a single window can
            # suffice.
            prev_mass = imm
            prev_step = step_size
        total_steps = 0
        resumed_converged = False

        ck = ckpt.read_intermediate() if cfg.restart else None
        if ck is not None:
            if "positions" in ck:  # written by this warmup mode
                positions = jnp.array(ck["positions"])
                imm = jnp.array(ck["inverse_mass_matrix"])
                step_size = jnp.asarray(ck["step_size"])
                prev_mass = imm
                prev_step = step_size
                total_steps = ck["total_steps"]
                stable_boundaries = int(ck.get("stable_boundaries", 0))
                resumed_dense_updated = bool(ck.get("dense_updated", False))
                resumed_converged = bool(ck.get("warmup_converged", False)) and not bool(
                    ck.get("calibrated", False)
                )
                bw_ratio = ck.get("between_within_ratio", float("nan"))
                tail_rhat = ck.get("tail_rhat", float("nan"))
                completed_window_chunks = int(ck.get("window_chunks", 1))
                if "next_window_chunks" in ck:
                    next_window_chunks = int(ck["next_window_chunks"])
                else:
                    # Checkpoints written before this field was introduced
                    # stored the completed boundary.  Advance it once when
                    # reading those checkpoints.
                    completed_chunks = int(ck.get("window_chunks", 1))
                    next_window_chunks = min(
                        2 * completed_chunks,
                        max(1, cfg.max_window // n_window),
                    )
                print(
                    f"Resuming pooled window warmup from checkpointed step "
                    f"{total_steps}",
                    flush=True,
                )

        if cfg.step_size_search and initial_step_size is None and total_steps == 0:
            # Halve/double from a large guess until acceptance crosses target:
            # a few gradient evaluations replace whole windows spent crawling
            # from a mismatched hand-set initial step size.
            rng_key, srch_key = jax.random.split(rng_key)
            step_size = jnp.asarray(
                find_reasonable_step_size(
                    srch_key,
                    lambda eps: lambda k, s: kernel(
                        k, s, jlp, eps, imm,
                        max_num_doublings=max_doublings,
                    ),
                    nuts_mcmc.init(positions[0], jlp),
                    0.5,
                    target_accept=cfg.target_acceptance_rate,
                )
            )
            print(
                f"Initial step size search: {float(step_size):.3g}", flush=True
            )

        print(
            f"Running pooled window warmup ({n_chains} chains, "
            f"{n_window}-step windows, max {cfg.max_steps} "
            f"steps)",
            flush=True,
        )
        da_state = da_init(float(step_size))
        states = _dev_split(vinit(positions))

        n_leap = None
        def dense_from_tail(tail, step, states, rng_key, reseed=True,
                            grad_tail=None):
            """Dense imm from the pooled within-chain tail (covariance of
            draws, or the Fisher-divergence geometric mean of draw and
            score covariances when scores are given and the estimator is
            "fisher"), eigenvalue-guarded, with eps re-seeded."""
            xc = tail - tail.mean(axis=0, keepdims=True)
            n_cov = xc.shape[0] * xc.shape[1]
            C = jnp.einsum("tcd,tce->de", xc, xc) / max(n_cov - 1, 1)
            rank = cfg.dense_rank
            if cfg.metric_estimator == "fisher" and grad_tail is not None:
                gc = grad_tail - grad_tail.mean(axis=0, keepdims=True)
                G = jnp.einsum("tcd,tce->de", gc, gc) / max(n_cov - 1, 1)
                C = _fisher_metric(
                    C, G, rank, cfg.fisher_cutoff,
                    cfg.fisher_reg, n=n_cov,
                )
            elif rank is None:
                alpha = dim / (dim + n_cov)
                C = (1 - alpha) * C + alpha * jnp.diag(jnp.diag(C))
            else:
                # diagonal + low rank: keep correlation eigen-directions above noise
                sd = jnp.sqrt(jnp.clip(jnp.diag(C), 1e-30))
                R = C / jnp.outer(sd, sd)
                lr, Ur = jnp.linalg.eigh(R)
                if rank == "auto":
                    # keep directions above noise on either side
                    q = np.sqrt(dim / max(n_cov, 1))
                    keep = (lr > (1 + q) ** 2) | (lr < (1 - q) ** 2)
                else:
                    order = jnp.argsort(-jnp.abs(lr - 1.0))
                    keep = jnp.zeros(dim, bool).at[order[: int(rank)]].set(True)
                n_keep = int(jnp.sum(keep))
                lr_k = jnp.where(keep, lr - 1.0, 0.0)
                R_lr = jnp.eye(dim) + (Ur * lr_k) @ Ur.T
                C = R_lr * jnp.outer(sd, sd)
                print(f"  low-rank metric: kept {n_keep} correlation "
                      f"directions outside the noise bulk", flush=True)
            lam, V = jnp.linalg.eigh(C)
            ok = jnp.isfinite(lam) & (lam > 0)
            med = jnp.nanmedian(jnp.where(ok, lam, jnp.nan))
            med = jnp.where(jnp.isfinite(med) & (med > 0), med, jnp.asarray(1.0))
            lam = jnp.where(ok, lam, med)
            lam = jnp.clip(jnp.maximum(lam, 1e-4 * med), 1e-6, 1e6)
            imm = (V * lam) @ V.T
            ev = jnp.linalg.eigvalsh(imm)
            print(
                f"Dense metric updated from pooled tail covariance "
                f"({n_cov} samples): eigenvalue range "
                f"[{float(ev.min()):.3e}, {float(ev.max()):.3e}]",
                flush=True,
            )
            if not reseed:
                return imm, step, rng_key
            rng_key, srch_key = jax.random.split(rng_key)
            step = float(
                find_reasonable_step_size(
                    srch_key,
                    lambda eps: lambda k, s: kernel(
                        k, s, jlp, eps, imm, max_num_doublings=max_doublings,
                    ),
                    nuts_mcmc.init(jnp.asarray(_flat_positions(states))[0], jlp),
                    float(step),
                    target_accept=cfg.target_acceptance_rate,
                )
            )
            print(f"Step size re-seeded for updated metric: {step:.3g}", flush=True)
            return imm, step, rng_key

        calibration_leap = None
        converged = False
        calibrated = False
        dense_updated = resumed_dense_updated
        last_diag = {}

        def calibrate(states, step, imm, rng_key, total_steps):
            """Fixed-metric epsilon calibration after convergence; returns
            (states, step, calibration_leap, total_steps, calibrated, rng_key)
            and writes the calibrated checkpoint."""
            terminal_steps = min(
                max(0, int(cfg.terminal_steps)),
                cfg.max_steps - total_steps,
            )
            if not terminal_steps:
                return states, step, None, total_steps, False, rng_key
            rng_key, terminal_key = jax.random.split(rng_key)
            terminal_da = da_init(float(step))
            states, terminal_da, _, _, _, terminal_leap = run_window(
                terminal_key, states, terminal_da, imm, terminal_steps
            )
            step = da_final(terminal_da)
            total_steps += terminal_steps
            print(
                f"Pooled fixed-metric calibration: {terminal_steps} steps, "
                f"step_size={float(step):.5g}",
                flush=True,
            )
            ckpt.write_intermediate(
                {
                    "inverse_mass_matrix": np.asarray(imm).tolist(),
                    "step_size": float(step),
                    "positions": _flat_positions(states).tolist(),
                    "sample_transform": bool(self.host.sample_transform),
                    "total_steps": total_steps,
                    "between_within_ratio": bw_ratio,
                    "tail_rhat": tail_rhat,
                    "stable_boundaries": stable_boundaries,
                    "window_chunks": completed_window_chunks,
                    "next_window_chunks": k,
                    "warmup_converged": True,
                    "calibrated": True,
                    **last_diag,
                }
            )
            return states, step, terminal_leap, total_steps, True, rng_key
        # Stan-style growing windows built from k base-length scans (one
        # compiled executable): short windows early for fast metric feedback,
        # doubling each boundary so the tail statistics and mixing check are
        # judged on enough samples to be useful.
        k_max = max(1, cfg.max_window // n_window)
        k = min(next_window_chunks, k_max)
        # Even the longest window's pooled tail must overdetermine the
        # dim x dim covariance, or dense updates can never happen (the
        # runtime gate below would skip every boundary).
        max_win = k_max * n_window
        max_tail = max_win - max_win // 5
        if (
            cfg.dense_update
            and imm.ndim == 2
            and n_chains * max_tail <= dim
        ):
            raise ValueError(
                f"pooled_window_dense_update can never run: the longest "
                f"window's pooled tail has {n_chains} * {max_tail} <= {dim} "
                f"(n_params) samples; raise chains_per_device, "
                f"pooled_window_steps, or pooled_window_max_window."
            )
        while total_steps < cfg.max_steps:
            if resumed_converged:
                print(
                    "Resuming at a converged boundary; running calibration",
                    flush=True,
                )
                converged = True
                states, step, calibration_leap, total_steps, calibrated, rng_key = (
                    calibrate(states, float(step_size), imm, rng_key, total_steps)
                )
                step_size = jnp.asarray(step)
                print(f"Pooled warmup converged after {total_steps} steps", flush=True)
                break
            pos_c, grad_c, acc_c, leap_c = [], [], [], []

            remaining = cfg.max_steps - total_steps
            boundary_steps = min(k * n_window, remaining)
            if boundary_steps <= 0:
                break
            n_subwindows = (boundary_steps + n_window - 1) // n_window
            for i in range(n_subwindows):
                sub_steps = min(n_window, boundary_steps - i * n_window)
                rng_key, sub_key = jax.random.split(rng_key)
                states, da_state, p_i, g_i, a_i, l_i = run_window(
                    sub_key, states, da_state, imm, sub_steps
                )
                pos_c.append(p_i)
                grad_c.append(g_i)
                acc_c.append(a_i)
                leap_c.append(l_i)
                total_steps += sub_steps
            pos = jnp.concatenate(pos_c, axis=0)
            grad = jnp.concatenate(grad_c, axis=0)
            acc = jnp.concatenate(acc_c, axis=0)
            n_leap = jnp.concatenate(leap_c, axis=0)
            w_len = boundary_steps

            # re-seed stuck chains at window edges
            stuck, live = stuck_chains(pos)
            if stuck.size:
                rng_key, seed_key = jax.random.split(rng_key)
                donors = jax.random.choice(seed_key, live, (stuck.size,))
                flat = jnp.asarray(_flat_positions(states))
                states = _dev_split(vinit(flat.at[stuck].set(flat[donors])))
                pos = pos.at[:, stuck, :].set(pos[:, donors, :])
                grad = grad.at[:, stuck, :].set(grad[:, donors, :])
                print(
                    f"Re-seeded {stuck.size} stuck chain(s) "
                    f"{stuck.tolist()} from live chains",
                    flush=True,
                )

            # Per-chain diagnostics: fraction of window steps each chain
            # moved and its current log density (checkpointed for post-hoc
            # questions like "which chains never left the seed").
            chain_moved = np.asarray(
                (jnp.abs(jnp.diff(pos, axis=0)).sum(-1) > 0).mean(0)
            )
            chain_logp = np.asarray(jnp.asarray(states.logdensity)).reshape(-1)
            n_slow = int((chain_moved < 0.5).sum())
            last_diag.update(chain_logp=chain_logp.tolist(),
                             chain_moved_frac=chain_moved.tolist())
            print(
                f"  chains: logp median {np.median(chain_logp):.1f} "
                f"[{chain_logp.min():.1f}, {chain_logp.max():.1f}], "
                f"{n_slow} moved <50% of steps",
                flush=True,
            )

            # pos: (w_len, n_chains, dim)
            tail_start = min(max(w_len // 5, 0), max(w_len - 1, 0))
            tail = pos[tail_start:]
            grad_tail = grad[tail_start:]
            tail_ddof = 1 if tail.shape[0] > 1 else 0
            within = jnp.mean(jnp.var(tail, axis=0, ddof=tail_ddof), axis=0)

            between_ddof = 1 if n_chains > 1 else 0
            between = jnp.var(
                jnp.mean(tail, axis=0), axis=0, ddof=between_ddof
            )
            bw_ratio = float(jnp.max(between / (within + 1e-30)))
            tail_rhat = float("inf")
            mixing_ready = tail.shape[0] >= cfg.min_tail_steps

            # coordinates no chain has moved in have zero within-variance
            # and an undefined (inf) R-hat; judge mixing on the rest and
            # report the frozen count separately
            moving = within > 0
            n_frozen = int(jnp.sum(~moving))
            if mixing_ready and n_chains > 1 and bool(jnp.any(moving)):
                from blackjax.diagnostics import potential_scale_reduction

                rhat_vec = potential_scale_reduction(
                    tail[:, :, moving], chain_axis=1, sample_axis=0
                )
                q = cfg.mixing_quantile
                tail_rhat = float(
                    jnp.max(rhat_vec) if q >= 1.0 else jnp.quantile(rhat_vec, q)
                )
                if param_names is not None:
                    idx_moving = np.flatnonzero(np.asarray(moving))
                    top = np.argsort(-np.asarray(rhat_vec))[:5]
                    print("  worst tail R-hat: " + ", ".join(
                        f"{param_names[idx_moving[t]]}={float(rhat_vec[t]):.2f}"
                        for t in top)
                        + (f" (gate uses q{q:g}={tail_rhat:.2f})" if q < 1.0 else ""),
                        flush=True)
            mixing_ok = (
                mixing_ready
                and tail_rhat <= cfg.mixing_rhat
            )
            
            # Keep the boundary update conservative when chains disagree, but
            # do not interpret this shrinkage factor as an ESS estimate.
            
            disagreement = max(tail_rhat - 1.0, 0.0) if mixing_ready else 1.0
            tail_weight_steps = tail.shape[0] / (1.0 + disagreement)
            w = tail_weight_steps / (tail_weight_steps + 5)
            step = da_final(da_state)
            seed_step = float(step)
            if imm.ndim == 2:
                # Dense (hessian_dense) metric: frozen through warmup unless
                # pooled_window_dense_update re-estimates it from the tail.
                # eps is re-searched only on the first update (the
                # Hessian->covariance jump; find_reasonable_step_size has
                # factor-2 granularity); after that dual averaging carries it.
                mass = imm
                # The pooled tail must overdetermine the dim x dim
                # covariance or the estimate is singular.
                tail_ok = tail.shape[0] * n_chains > dim
                if (
                    cfg.dense_update
                    and mixing_ready
                    and not tail_ok
                ):
                    print(
                        f"Skipping dense metric update: pooled tail has "
                        f"{tail.shape[0] * n_chains} samples <= {dim} "
                        f"params; raise chains_per_device or "
                        f"pooled_window_steps for per-window updates.",
                        flush=True,
                    )
                if cfg.dense_update and mixing_ready and tail_ok:
                    C, seed_step, rng_key = dense_from_tail(
                        tail, seed_step, states, rng_key,
                        reseed=not dense_updated, grad_tail=grad_tail,
                    )
                    
                    # Evidence-weighted EMA toward the previous metric at
                    # every boundary (w -> 0 when chains disagree), so the
                    # seed metric is relaxed as soon as the pooled tail
                    # carries information instead of waiting for mixing.
                    mass = w * C + (1 - w) * imm
                    dense_updated = True
            else:
                mass = w * within + (1 - w) * imm
            
            # Boundary: take the averaged step size and restart dual
            # averaging around it with the new mass matrix (blackjax
            # window_adaptation's slow_final).
            da_state = da_init(seed_step)
            mean_acc = float(jnp.mean(acc))

            converged = False
            if prev_mass is not None:
                # Dense: per-parameter variances (element-wise ratios blow
                # up on near-zero off-diagonals; Frobenius of the full
                # matrix is dominated by estimator noise ~ sqrt(dim/n_eff)).
                d_new = jnp.diag(mass) if mass.ndim == 2 else mass
                d_old = jnp.diag(prev_mass) if mass.ndim == 2 else prev_mass
                rel = jnp.abs(d_new - d_old) / (jnp.abs(d_old) + 1e-10)
                if cfg.mass_stat == "rms_diag":
                    mass_change = float(jnp.sqrt(jnp.mean(rel**2)))
                else:
                    mass_change = float(jnp.max(rel))
                step_change = float(
                    jnp.abs(step - prev_step) / (jnp.abs(prev_step) + 1e-10)
                )
                # The max-over-dims change statistic has a noise floor set by
                # the tail-variance estimator: ~sqrt(2/n_eff) per dim,
                # inflated ~sqrt(2 ln d) by the max. A configured rtol below
                # that floor means nothing, so gate against
                # max(rtol, floor); eps co-moves with the metric at roughly
                # half the relative rate.
                mass_floor = 0.0
                if n_chains > 1 and tail.shape[0] > 1:
                    from blackjax.diagnostics import effective_sample_size

                    n_eff = float(jnp.nanmin(effective_sample_size(
                        tail, chain_axis=1, sample_axis=0
                    )))
                    if np.isfinite(n_eff) and n_eff > 0:
                        # per-variance relative error ~sqrt(2/n_eff); the
                        # max over d of them is inflated by ~sqrt(2 ln d),
                        # the RMS is not
                        infl = (
                            np.sqrt(2 * np.log(max(dim, 2)))
                            if cfg.mass_stat == "max_diag" else 1.0
                        )
                        mass_floor = 1.5 * infl * np.sqrt(2 / n_eff)
                rtol_mass = max(cfg.rtol_mass, mass_floor)
                rtol_step = max(cfg.rtol_step, 0.5 * mass_floor)
                stat_name = (
                    "rms_rel_mass_change" if cfg.mass_stat == "rms_diag"
                    else "max_rel_mass_change"
                )
                print(
                    f"Pooled warmup step {total_steps} (window {w_len}): "
                    f"{stat_name}={mass_change:.4f}, "
                    f"rel_step_change={step_change:.4f}, "
                    f"rtol_mass={rtol_mass:.3f}, rtol_step={rtol_step:.3f} "
                    f"(noise floor {mass_floor:.3f}), "
                    f"mean_acceptance={mean_acc:.3f}, "
                    f"max_between_within_ratio={bw_ratio:.2f}, "
                    f"max_tail_rhat={tail_rhat:.3f}, "
                    f"tail_steps={tail.shape[0]}"
                    + (f", frozen_dims={n_frozen}" if n_frozen else "")
                    + (
                        " (mixing gate not met)"
                        if not mixing_ok
                        else ""
                    ),
                    flush=True,
                )
                stable = (
                    mass_change < rtol_mass
                    and step_change < rtol_step
                    and mixing_ok
                )
                stable_boundaries = stable_boundaries + 1 if stable else 0
                converged = (
                    stable_boundaries
                    >= cfg.consecutive_windows
                )

            prev_mass = mass
            prev_step = step
            imm = mass
            step_size = jnp.asarray(step)
            completed_window_chunks = k
            next_window_chunks = min(2 * k, k_max)
            k = next_window_chunks

            if converged and (
                cfg.max_steps - total_steps
                < cfg.terminal_steps
            ):
                converged = False

            ckpt.write_intermediate(
                {
                    "inverse_mass_matrix": np.asarray(mass).tolist(),
                    "step_size": float(step),
                    "positions": _flat_positions(states).tolist(),
                    "sample_transform": bool(self.host.sample_transform),
                    "total_steps": total_steps,
                    "between_within_ratio": bw_ratio,
                    "tail_rhat": tail_rhat,
                    "stable_boundaries": stable_boundaries,
                    # Keep the old field for readers of existing
                    # checkpoints, but make the restart contract
                    # explicit with the next boundary size.
                    "window_chunks": completed_window_chunks,
                    "next_window_chunks": next_window_chunks,
                    "warmup_converged": bool(converged),
                    "calibrated": False,
                    "dense_updated": bool(dense_updated),
                    **last_diag,
                }
            )

            if converged:
                print(f"Pooled warmup convergence detected at {total_steps} steps", flush=True)
                if imm.ndim == 2 and not cfg.dense_update:
                    imm, step, rng_key = dense_from_tail(
                        tail, step, states, rng_key
                    )
                states, step, calibration_leap, total_steps, calibrated, rng_key = (
                    calibrate(states, float(step), imm, rng_key, total_steps)
                )
                step_size = jnp.asarray(step)
                print(
                    f"Pooled warmup converged after {total_steps} steps",
                    flush=True,
                )
                break
        else:
            print(
                f"Pooled warmup reached max {cfg.max_steps} "
                f"steps without convergence",
                flush=True,
            )

        if not converged:
            message = (
                "Pooled warmup failed to converge within "
                f"{cfg.max_steps} steps; refusing to start "
                "sampling. Set pooled_window_allow_unconverged=true only "
                "for an explicitly exploratory run."
            )
            if not cfg.allow_unconverged:
                raise RuntimeError(message)
            print("WARNING: " + message, flush=True)

        result = WarmupResult(
            inverse_mass_matrix=imm,
            step_size=step_size,
            positions=jnp.asarray(_flat_positions(states)),
            converged=bool(converged),
            calibrated=bool(calibrated),
            diagnostics=dict(total_steps=total_steps, **last_diag),
        )
        if (
            (self.tune_tree_depth and cfg.sampling_depth_auto)
            and calibration_leap is not None
            and converged
            and calibrated
        ):
            # Sampling depth cap from the last (converged) window: under vmap
            # lockstep the deepest chain sets the per-step cost, so cap just
            # above the bulk of the depth distribution instead of blackjax's
            # default 10.
            depth = jnp.ceil(
                jnp.log2(calibration_leap.astype(jnp.float32) + 1.0)
            )
            qs = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            dq = [int(jnp.quantile(depth, q)) for q in qs]
            print(
                "Warmup tree depth percentiles: "
                + ", ".join(f"q{q:g}={d}" for q, d in zip(qs, dq)),
                flush=True,
            )
            q_depth = int(jnp.quantile(depth, cfg.depth_cap_quantile))
            saturated = q_depth >= cfg.max_doublings
            margin = (
                cfg.depth_cap_margin_saturated if saturated
                else cfg.depth_cap_margin
            )
            result.max_num_doublings = min(
                q_depth + margin, cfg.sampling_max_num_doublings
            )
            print(
                f"Sampling depth cap: {result.max_num_doublings} "
                f"(q{cfg.depth_cap_quantile:g} warmup depth {q_depth}"
                + (f", at the warmup cap {cfg.max_doublings}: "
                   f"margin {margin}" if saturated else "")
                + ")",
                flush=True,
            )
        return result
