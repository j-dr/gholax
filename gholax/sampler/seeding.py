"""Sampler-agnostic seeding: MAP search, Pathfinder, and the initial metric.

Owns everything a sampler needs before adaptation starts - where the chains
begin and what curvature they begin with - independently of which sampler
will consume them.  Feeds gholax.sampler.warmup, which refines the metric.
"""

import json
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

from ..util.distributed import gather_to_host, is_io_process

# Every sampler reaches the same ladder (Seeder.initial_metric).  NUTS used
# to reject "mclmc" and MCLMC to reject "hessian_dense"/"pathfinder"/
# "fisher_seeds" only because each had its own open-coded dispatch;
# "pathfinder" and "fisher_seeds" still require pathfinder_init.
METRIC_CHOICES = (
    "ones", "hessian", "hessian_dense", "pathfinder", "fisher_seeds", "mclmc",
)

# Flat sampler-level key -> SeedingConfig field.  Accepted with a
# DeprecationWarning; the nested `seeding:` block is the new spelling.
LEGACY_KEYS = {
    "minimize_and_sample": "minimize_and_sample",
    "minimize_n_starts": "n_starts",
    "minimize_start_scale": "start_scale",
    "pathfinder_init": "pathfinder_init",
    "pathfinder_resample": "pathfinder_resample",
    "pathfinder_n_paths": "pathfinder_n_paths",
    "pathfinder_start_scale": "pathfinder_start_scale",
    "pathfinder_elbo_samples": "pathfinder_elbo_samples",
    "pathfinder_maxiter": "pathfinder_maxiter",
    "pathfinder_maxcor": "pathfinder_maxcor",
    "mass_matrix_init": "mass_matrix_init",
}

# Deprecated sampler attribute -> SeedingConfig field (see the samplers'
# __getattr__).  Every legacy key is also an attribute here.
LEGACY_ATTRS = dict(LEGACY_KEYS)

# Per-sampler overrides of the shared defaults.  `metric_fallback` is set
# here, never by the user: NUTS leaves an unrecognized mass_matrix_init as
# None (blackjax then uses its own identity default) while MCLMC builds an
# explicit ones vector.
SAMPLER_SEEDING_DEFAULTS = {
    "NUTS": {"pathfinder_init": True},
    "MCLMC": {
        "minimize_and_sample": True,
        "mass_matrix_init": "ones",
        "metric_fallback": "ones",
    },
    "Minimize": {"minimize_and_sample": True},
    "MetropolisHastings": {},
}

# Fields the user may not set: derived from the sampler, not the config.
_INTERNAL_FIELDS = ("metric_fallback",)

_WARNED = set()


def _warn_legacy(sampler, key, field):
    if (sampler, key) in _WARNED:
        return
    _WARNED.add((sampler, key))
    warnings.warn(
        f"sampler.{sampler}.{key} is deprecated; use "
        f"sampler.{sampler}.seeding.{field}",
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class SeedingConfig:
    """MAP / Pathfinder / initial-metric settings, independent of the sampler.

    Unlike WarmupConfig there are no presence-derived flags: no seeding
    option changes behaviour by being absent.  The two cross-key defaults
    (pathfinder_n_paths, pathfinder_start_scale) are None sentinels resolved
    in __post_init__, so they hold regardless of key order and of whether
    the sibling field arrived nested or flat.
    """

    # MAP search
    minimize_and_sample: bool = False
    n_starts: int = 4
    start_scale: float = 0.5
    n_adam: int = 300
    n_lbfgs_restarts: int = 3

    # Pathfinder.  These keep their prefix: pathfinder_start_scale and
    # start_scale are different knobs and must not collide.
    pathfinder_init: bool = False
    pathfinder_resample: bool = False
    pathfinder_n_paths: int = None
    pathfinder_start_scale: float = None
    pathfinder_elbo_samples: int = 20
    pathfinder_maxiter: int = 100
    pathfinder_maxcor: int = 10

    # initial metric
    mass_matrix_init: str = None
    metric_fallback: str = None
    mclmc_mm_n_tune_steps: int = 600
    mclmc_mm_n_steps: int = 1200
    mclmc_mm_n_chains: int = 8

    def __post_init__(self):
        set_ = lambda k, v: object.__setattr__(self, k, v)
        set_("minimize_and_sample", bool(self.minimize_and_sample))
        set_("pathfinder_init", bool(self.pathfinder_init))
        set_("pathfinder_resample", bool(self.pathfinder_resample))
        set_("n_starts", int(self.n_starts))
        if self.n_starts < 1:
            raise ValueError("minimize_n_starts must be >= 1")
        set_("start_scale", float(self.start_scale))
        set_("n_adam", int(self.n_adam))
        set_("n_lbfgs_restarts", int(self.n_lbfgs_restarts))
        # cross-key sentinels
        if self.pathfinder_n_paths is None:
            set_("pathfinder_n_paths", self.n_starts)
        set_("pathfinder_n_paths", int(self.pathfinder_n_paths))
        if self.pathfinder_n_paths < 1:
            raise ValueError("pathfinder_n_paths must be >= 1")
        if self.pathfinder_start_scale is None:
            set_("pathfinder_start_scale", self.start_scale)
        set_("pathfinder_start_scale", float(self.pathfinder_start_scale))
        for key in ("pathfinder_elbo_samples", "pathfinder_maxiter",
                    "pathfinder_maxcor", "mclmc_mm_n_tune_steps",
                    "mclmc_mm_n_steps", "mclmc_mm_n_chains"):
            set_(key, int(getattr(self, key)))
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be >= 1")
        if self.metric_fallback not in (None, "ones"):
            raise ValueError("metric_fallback must be None or 'ones'")
        # Messages kept verbatim: tests match on them.
        if self.mass_matrix_init == "fisher_seeds" and not self.pathfinder_init:
            raise ValueError(
                "mass_matrix_init: fisher_seeds requires pathfinder_init: true"
            )
        if self.mass_matrix_init == "pathfinder" and not self.pathfinder_init:
            raise ValueError(
                "mass_matrix_init: pathfinder requires pathfinder_init: true"
            )

    @classmethod
    def from_sampler_config(cls, c, *, sampler):
        """Build from a `sampler.<name>` config dict.

        Reads the nested `seeding:` block, folds in the deprecated flat keys
        (nested wins), rejects unknown keys, and derives mass_matrix_init
        when the config leaves it unset.
        """
        names = {f.name for f in fields(cls)}
        block = c.get("seeding", {})
        if not isinstance(block, Mapping):
            raise ValueError(f"sampler.{sampler}.seeding must be a mapping")
        allowed = names - set(_INTERNAL_FIELDS)
        unknown = set(block) - allowed
        if unknown:
            raise ValueError(
                f"unknown key(s) in sampler.{sampler}.seeding: {sorted(unknown)}; "
                f"valid keys: {sorted(allowed)}"
            )

        merged = dict(SAMPLER_SEEDING_DEFAULTS.get(sampler, {}))
        for key, field_ in LEGACY_KEYS.items():
            if key not in c:
                continue
            _warn_legacy(sampler, key, field_)
            if field_ in block:
                warnings.warn(
                    f"sampler.{sampler}.{key} ignored: "
                    f"sampler.{sampler}.seeding.{field_} is also set",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            merged[field_] = c[key]
        merged.update(block)

        # mass_matrix_init's default depends on the sampler, so it cannot be
        # resolved in __post_init__.
        if merged.get("mass_matrix_init") is None:
            merged["mass_matrix_init"] = {
                "NUTS": "pathfinder" if merged.get("pathfinder_init") else "hessian_dense",
            }.get(sampler, "ones")
        return cls(**merged)


@dataclass(frozen=True)
class FisherParams:
    """The three numbers the fisher_seeds metric needs from the warmup config.

    Passed in rather than importing WarmupConfig, so seeding stays free of
    any warmup type.
    """

    rank: object = "auto"
    cutoff: float = 1.5
    reg: float = 1e-4


@dataclass(frozen=True)
class SeedingHost:
    """What the seeding engine needs from the sampler that owns it.

    Unlike WarmupHost this carries behaviour (map_chains, constrain) as well
    as values, because minimization runs under the sampler's pmap/shard_map
    and the results file is written in the prior's physical space.
    """

    map_chains: object = None
    mesh: object = None
    chains_per_device: int = 1
    param_names: object = None
    constrain: object = None
    devices: object = None

    def is_io_process(self):
        return is_io_process()

    def __post_init__(self):
        if self.map_chains is None:
            object.__setattr__(self, "map_chains", lambda fn, **kw: jax.vmap(fn))
        if self.devices is None:
            object.__setattr__(self, "devices", tuple(jax.local_devices()))


class SeedingOutputs:
    """Writes the seeding result files for one output prefix.

    Paths and JSON keys are a public API: gholax.util.postprocess_chain and
    the analysis notebooks read .minimization_results.json.  Do not change
    paths or keys.
    """

    def __init__(self, output_file, host):
        self.host = host
        self.minimization = (
            f"{output_file}.minimization_results.json" if output_file else None
        )
        self.pathfinder = (
            f"{output_file}.pathfinder_init.json" if output_file else None
        )

    def write_minimization(self, x_opt, values):
        if not self.minimization or not self.host.is_io_process():
            return
        payload = {"x_opt": np.asarray(x_opt).tolist()}
        if self.host.constrain is not None:
            # physical-space copy: x_opt is sampling-space, whose meaning
            # depends on the prior transform
            payload["x_opt_physical"] = np.asarray(
                self.host.constrain(jnp.asarray(x_opt))
            ).tolist()
        payload["value"] = np.asarray(values).tolist()
        with open(self.minimization, "w") as fp:
            json.dump(payload, fp)

    def write_pathfinder(self, payload):
        if not self.pathfinder or not self.host.is_io_process():
            return
        with open(self.pathfinder, "w") as fp:
            json.dump(payload, fp)


def proposal_covariance_from_metric(imm, max_sigma=1.0):
    """Metropolis-Hastings proposal covariance from an inverse mass matrix.

    The cap is MetropolisHastings' own, lifted verbatim from its bespoke
    init_covariance: hessian branch: keep the curvature's correlation
    structure but never propose wider than the prior (1 in normalized
    space).  It applies to any curvature source, so every metric the seeding
    ladder can produce reaches the proposal through here.
    """
    imm = jnp.asarray(imm)
    hx = jnp.linalg.inv(imm) if imm.ndim == 2 else jnp.diag(1.0 / imm)
    hx_sigmas = jnp.sqrt(jnp.diag(hx))
    hxc = jnp.einsum("ij, i, j -> ij", hx, 1 / hx_sigmas, 1 / hx_sigmas)
    proposal_corr = jnp.linalg.inv(hxc)
    proposal_sigmas = jnp.minimum(max_sigma, 1 / hx_sigmas)
    return jnp.einsum(
        "ij, i, j -> ij", proposal_corr, proposal_sigmas, proposal_sigmas
    )


@dataclass
class SeedingRequest:
    """Per-call inputs to a seeding run."""

    log_posterior: object
    logdensity_fn: object
    initial_positions: object
    n_chains: int
    rng_key: object = None
    pathfinder_key: object = None
    output_file: str = None
    chi2_threshold: float = 1
    skip_minimize: bool = False
    skip_metric: bool = False


@dataclass
class SeedingResult:
    """Where the chains start and what curvature they start with."""

    positions: object
    inverse_mass_matrix: object = None
    map_position: object = None
    values: object = None
    pathfinder_imm: object = None
    metric_info: dict = field(default_factory=dict)


class Seeder:
    """MAP search, Pathfinder seeding, and the initial metric.

    Decoupled from the sampler: consumes a SeedingConfig and a SeedingHost,
    returns a SeedingResult.
    """

    def __init__(self, config, host, *, fisher_params=None):
        self.config = config
        self.host = host
        self.fisher_params = fisher_params
        self.pathfinder_imm = None
        self.pathfinder_positions = None
        self.map_values = None

    def run(self, req):
        """MAP -> Pathfinder -> initial metric, in that order.

        The pathfinder key is supplied by the caller and never split here:
        the split site fixes every downstream key, and therefore the whole
        starting cloud of a production run.
        """
        cfg = self.config
        positions = req.initial_positions
        pf_key = None if req.skip_minimize else req.pathfinder_key
        map_position = None
        if cfg.minimize_and_sample and not req.skip_minimize:
            positions = self.minimize(
                req.log_posterior, positions, req.n_chains, req.output_file,
                chi2_threshold=req.chi2_threshold, pathfinder_key=pf_key,
            )
        x_h = positions[0]
        map_position = x_h
        if pf_key is not None:
            positions = (
                self.pathfinder_positions
                if self.pathfinder_positions is not None
                else self.pathfinder_init(
                    req.logdensity_fn, x_h, req.n_chains, pf_key, req.output_file
                )
            )
        imm, info = None, {}
        if not req.skip_metric:
            imm, info = self.initial_metric(
                req.logdensity_fn, x_h, positions, req.rng_key
            )
        return SeedingResult(
            positions=positions,
            inverse_mass_matrix=imm,
            map_position=map_position,
            values=self.map_values,
            pathfinder_imm=self.pathfinder_imm,
            metric_info=info,
        )

    def initial_metric(self, jlp, x_map, positions, rng_key):
        """The mass_matrix_init ladder, shared by every sampler.

        Returns (inverse_mass_matrix, info).  An unrecognized value falls
        back silently, as both open-coded ladders did: to None (blackjax
        then uses its own identity default) unless the sampler asked for an
        explicit ones vector via metric_fallback.
        """
        cfg = self.config
        kind = cfg.mass_matrix_init
        x_h = x_map

        if kind == "hessian":
            print("Estimating initial mass matrix from Hessian diagonal...", flush=True)
            if not cfg.minimize_and_sample:
                x_h = self.best_fit_position(jlp, x_h)
            init_imm = self.hessian_mass_matrix(jlp, x_h)
            print(f"  imm range: [{float(init_imm.min()):.4f}, {float(init_imm.max()):.4f}]", flush=True)
        elif kind == "hessian_dense":
            print("Estimating dense initial mass matrix from full Hessian...", flush=True)
            if not cfg.minimize_and_sample:
                x_h = self.best_fit_position(jlp, x_h)
            init_imm = self.hessian_mass_matrix_dense(jlp, x_h)
            ev = jnp.linalg.eigvalsh(init_imm)
            print(
                f"  dense imm eigenvalue range: "
                f"[{float(ev.min()):.3e}, {float(ev.max()):.3e}]",
                flush=True,
            )
        elif kind == "fisher_seeds":
            # Fisher-divergence metric from the Pathfinder seed cloud and
            # its scores: n_chains draws with gradients, no extra warmup
            fp = self.fisher_params
            if fp is None:
                raise ValueError(
                    "mass_matrix_init: fisher_seeds needs fisher_params "
                    "(rank, cutoff, reg)"
                )
            from .warmup import _fisher_metric

            seeds = jnp.asarray(positions)
            gs = jax.vmap(jax.grad(jlp))(seeds)
            xc = seeds - seeds.mean(0); gc = gs - gs.mean(0)
            C = xc.T @ xc / max(seeds.shape[0] - 1, 1)
            G = gc.T @ gc / max(seeds.shape[0] - 1, 1)
            init_imm = _fisher_metric(
                C, G, fp.rank, fp.cutoff, fp.reg, n=int(seeds.shape[0]),
            )
            ev = jnp.linalg.eigvalsh(init_imm)
            print(
                f"Dense initial mass matrix from Fisher divergence of the "
                f"seed cloud: eigenvalue range [{float(ev.min()):.3e}, "
                f"{float(ev.max()):.3e}]",
                flush=True,
            )
        elif kind == "pathfinder":
            init_imm = self.pathfinder_imm
            ev = jnp.linalg.eigvalsh(init_imm)
            print(
                f"Dense initial mass matrix from Pathfinder covariance: "
                f"eigenvalue range [{float(ev.min()):.3e}, "
                f"{float(ev.max()):.3e}]",
                flush=True,
            )
        elif kind == "mclmc":
            print("Estimating initial mass matrix from short MCLMC runs...", flush=True)
            init_imm, info = self.mclmc_mass_matrix(jlp, positions, rng_key)
            print(
                f"  rung={info['rung']}, "
                f"n_survivors={info['n_survivors']}, "
                f"imm range: [{float(init_imm.min()):.4f}, "
                f"{float(init_imm.max()):.4f}]",
                flush=True,
            )
            return init_imm, dict(info, kind=kind)
        elif cfg.metric_fallback == "ones":
            init_imm = jnp.ones((jnp.asarray(positions).shape[-1],))
            return init_imm, {"kind": "ones"}
        else:
            init_imm = None
        return init_imm, {"kind": kind}


    def best_fit_position(self, jlp, position, n_adam=None,
                          n_lbfgs_restarts=None, n_starts=None,
                          start_scale=None, rng_key=None):
        """Multi-start Adam warm-start then restarted L-BFGS to the MAP.

        A single descent can settle in a data-likelihood ridge far from the
        joint optimum (observed with FlowLikelihood configs: 3x2pt curvature
        traps the minimizer ~59 logp above the CMB+BAO-compatible basin), so
        The basin search is a vmapped Adam-only pass over all starts (a
        simple scan graph that compiles quickly; memory is fine with
        gradient checkpointing on every likelihood block). L-BFGS — whose
        zoom line search vmaps into an hours-long XLA compile — polishes
        only the argmin winner, reusing the cached single-point compile.
        """
        best = self.adam_multistart(jlp, position, n_adam, n_starts,
                                    start_scale, rng_key)
        if best is None:
            return position
        best_p, best_v = self.lbfgs_polish(jlp, *best, n_lbfgs_restarts)
        print(f"  best fit logp: {float(-best_v):.2f}", flush=True)
        return best_p

    def adam_multistart(self, jlp, position, n_adam=None, n_starts=None,
                        start_scale=None, rng_key=None):
        """Vmapped multi-start Adam; returns the (position, -logp) of the
        best start, or None when every start failed."""
        cfg = self.config
        if n_adam is None:
            n_adam = cfg.n_adam
        if n_starts is None:
            n_starts = cfg.n_starts
        if start_scale is None:
            start_scale = cfg.start_scale
        if rng_key is None:
            rng_key = jax.random.key(0)
        starts = self.jittered_starts(
            jlp, position, n_starts, start_scale, rng_key
        )
        ps, vs = jax.vmap(
            lambda x0: self.adam_descent(jlp, x0, n_adam)
        )(starts)
        vs = jnp.where(vs < 1e30, vs, jnp.inf)
        i = int(jnp.argmin(vs))
        if not bool(jnp.isfinite(vs[i])):
            print("  minimization failed; using initial position", flush=True)
            return None
        spread = (float(vs.max() - vs.min())
                  if bool(jnp.isfinite(vs).all()) else float("nan"))
        print(f"  adam multi-start: best -logp {float(vs[i]):.2f} "
              f"({n_starts} starts, spread {spread:.2f})", flush=True)
        print("  per-start -logp (start 0 = reference): "
              + ", ".join(f"{float(v):.1f}" for v in vs), flush=True)
        return ps[i], vs[i]

    def jittered_starts(self, jlp, position, n_starts, start_scale, rng_key):
        """position plus n_starts-1 Hessian-guided jittered copies.

        Per-coordinate scale ~ sqrt(imm) at position concentrates the
        displacement in soft/degenerate directions (where distinct basins
        live) and barely moves stiff ones; the scale is backed off until the
        candidate is in-box.
        """
        jn = jax.jit(lambda p: -jlp(p))
        dim = position.shape[0]
        w = jnp.sqrt(self.hessian_mass_matrix(jlp, position))
        w = w / jnp.max(w)
        starts = [position]
        for i in range(n_starts - 1):
            ki = jax.random.fold_in(rng_key, i)
            for scale in (start_scale, start_scale / 5, 0.0):
                cand = position + scale * w * jax.random.normal(ki, (dim,))
                if bool(jn(cand) < 1e30):
                    break
            starts.append(cand)
        return jnp.stack(starts)

    def pathfinder_init(self, jlp, x_map, n_chains, rng_key, output_file):
        """Multi-path Pathfinder chain seeding (Zhang et al. 2022).

        L-BFGS paths start from x_map and Hessian-jittered copies.  Along
        each path the Gaussian N(theta_l - Sigma_l grad_l, Sigma_l), with
        Sigma_l the L-BFGS inverse-Hessian estimate, is scored by a Monte
        Carlo ELBO and the best iterate kept (done here rather than with
        blackjax's own scan, whose mean has the opposite sign).  Draws are
        pooled and either importance-resampled with mixture-proposal
        weights (pathfinder_resample) or taken as ELBO-weighted mixture
        draws.
        x_map itself is untouched: it stays the MAP for model comparison
        and the Hessian metric.
        """
        from blackjax.vi import pathfinder

        cfg = self.config
        n_paths = cfg.pathfinder_n_paths
        n_elbo = cfg.pathfinder_elbo_samples
        resample = cfg.pathfinder_resample
        dim = x_map.shape[0]
        k_start, k_pf, k_draw, k_pick = jax.random.split(rng_key, 4)
        starts = self.jittered_starts(
            jlp, x_map, n_paths,
            cfg.pathfinder_start_scale, k_start,
        )
        print(f"Running Pathfinder ({n_paths} paths, resample={resample})",
              flush=True)
        vlp = jax.jit(jax.vmap(jlp))

        def gaussian(pos, grad, alpha, beta, gamma):
            sig = jnp.diag(alpha) + beta @ gamma @ beta.T
            return pos - sig @ grad, jnp.linalg.cholesky(sig)

        def logq(mu, L, x):
            z = jax.scipy.linalg.solve_triangular(L, (x - mu).T, lower=True)
            return (-0.5 * jnp.sum(z**2, axis=0)
                    - jnp.sum(jnp.log(jnp.diag(L)))
                    - 0.5 * dim * jnp.log(2 * jnp.pi))

        @jax.jit
        def elbo_path(key, path):
            def one(k, pos, grad, alpha, beta, gamma):
                mu, L = gaussian(pos, grad, alpha, beta, gamma)
                x = mu + (L @ jax.random.normal(k, (dim, n_elbo))).T
                lp = vlp(x)
                e = jnp.mean(lp - logq(mu, L, x))
                ok = jnp.all(jnp.isfinite(L)) & jnp.all(lp > -1e30)
                return jnp.where(ok, e, -jnp.inf)
            keys = jax.random.split(key, path.position.shape[0])
            # sequential to avoid OOM
            
            return jax.lax.map(
                lambda a: one(*a),
                (keys, path.position, path.grad_position,
                 path.alpha, path.beta, path.gamma),
            )

        approx = jax.jit(lambda k, x0: pathfinder.approximate(
            k, jlp, x0, num_samples=2,
            maxiter=cfg.pathfinder_maxiter,
            maxcor=cfg.pathfinder_maxcor,
        )[1].path)
        gauss, elbos, n_iter = [], [], []
        per_path = 2 * n_chains // n_paths + 1
        for k, x0 in zip(jax.random.split(k_pf, n_paths), starts):
            k1, k2 = jax.random.split(k)
            path = approx(k1, x0)
            e = elbo_path(k2, path)
            l = int(jnp.argmax(e))
            elbos.append(float(e[l])); n_iter.append(l)
            gauss.append(gaussian(path.position[l], path.grad_position[l],
                                  path.alpha[l], path.beta[l], path.gamma[l]))
        print("  per-path best ELBO: " + ", ".join(
            f"{e:.1f} (iter {l})" for e, l in zip(elbos, n_iter)), flush=True)
        phi = jnp.concatenate([
            mu + (L @ jax.random.normal(k, (dim, per_path))).T
            for (mu, L), k in zip(gauss, jax.random.split(k_draw, n_paths))
        ])
        logp = vlp(phi)
        ok = jnp.isfinite(logp) & (logp > -1e30)
        print(f"  finite draws {int(ok.sum())}/{phi.shape[0]}", flush=True)
        phi, logp = phi[ok], logp[ok]
        info = {"elbo": elbos, "best_iter": n_iter, "n_finite": int(ok.sum()),
                "resample": resample}
        # Plain-mixture mode weights paths by exp(ELBO)
        # importance-sampling mode instead uses the uniform mixture the
        # draws actually came from and lets the weights do the work.
        log_pm = jnp.asarray(elbos) - jax.scipy.special.logsumexp(
            jnp.asarray(elbos))
        info["path_weights"] = np.exp(np.asarray(log_pm)).tolist()
        path_of = jnp.repeat(jnp.arange(n_paths), per_path)[ok]
        w = jnp.exp(log_pm)[path_of]
        w = w / w.sum()
        if resample:
            lq = jax.scipy.special.logsumexp(
                jnp.stack([logq(mu, L, phi) for mu, L in gauss]), axis=0
            ) - jnp.log(n_paths)
            logw = logp - lq
            w = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
            n_eff = float(1.0 / jnp.sum(w**2))
            info["weight_ess"] = n_eff
            print(f"  importance-weight ESS {n_eff:.1f} of {phi.shape[0]} "
                  f"draws", flush=True)
            if n_eff < 2:
                print("  weights degenerate; falling back to mixture draws",
                      flush=True)
                w = jnp.exp(log_pm)[path_of]
                w = w / w.sum()
        idx = jax.random.choice(
            k_pick, phi.shape[0], (n_chains,),
            replace=phi.shape[0] < n_chains, p=w,
        )
        positions = phi[idx]
        info["n_unique"] = int(jnp.unique(idx).shape[0])
        # ELBO-best path's L-BFGS covariance: bulk-scale dense metric
        # candidate (mass_matrix_init: pathfinder)
        Lb = gauss[int(jnp.argmax(jnp.asarray(elbos)))][1]
        self.pathfinder_imm = Lb @ Lb.T
        print(f"  seeded {n_chains} chains from {info['n_unique']} distinct "
              f"draws; logp range [{float(logp[idx].min()):.1f}, "
              f"{float(logp[idx].max()):.1f}]", flush=True)
        if self.host.is_io_process():
            with open(f"{output_file}.pathfinder_init.json", "w") as fp:
                json.dump({**info, "positions": positions.tolist(),
                           "inverse_mass_matrix":
                               np.asarray(self.pathfinder_imm).tolist()}, fp)
        return positions

    def polish_and_pathfinder(self, log_posterior, position, n_chains,
                               pathfinder_key, output_file):
        """Adam multi-start, then L-BFGS polish (thread, spare device) in
        parallel with Pathfinder seeding from the unpolished winner.
        Returns the polished MAP; seeds go to self.pathfinder_positions."""
        jlp = jax.jit(log_posterior)
        best = self.adam_multistart(jlp, position)
        if best is None:
            self.pathfinder_positions = self.pathfinder_init(
                jlp, position, n_chains, pathfinder_key, output_file
            )
            return position
        p0, v0 = best
        devs = self.host.devices
        pol_dev = devs[1] if len(devs) > 1 else devs[0]
        out = {}

        def polish():
            with jax.default_device(pol_dev):
                pp = jax.device_put(p0, pol_dev)
                out["x"], out["v"] = self.lbfgs_polish(jlp, pp, v0)

        print(f"  L-BFGS polish on {pol_dev} in parallel with Pathfinder",
              flush=True)
        t = threading.Thread(target=polish, daemon=True)
        t.start()
        self.pathfinder_positions = self.pathfinder_init(
            jlp, p0, n_chains, pathfinder_key, output_file
        )
        t.join()
        x_opt = jax.device_put(out["x"], devs[0])
        print(f"  best fit logp: {float(-out['v']):.2f}", flush=True)
        return x_opt

    def adam_descent(self, jlp, position, n_adam=None):
        """Adam descent tracking the best valid point.

        Adam has no line search, so the -FLT_MAX out-of-box plateau and fp32
        noise that stall jaxopt's zoom line search don't affect it; updates
        that land out of the box (logp <= -1e30) are rejected. Pure JAX
        (vmap-compatible); returns (best position, -logp), -logp >= 1e30 if
        no valid point was seen.
        """
        import optax

        if n_adam is None:
            n_adam = self.config.n_adam
        jnlp = jax.jit(lambda p: -jlp(p))
        vgrad = jax.value_and_grad(jnlp)

        opt = optax.adam(optax.cosine_decay_schedule(1e-2, n_adam, 1e-1))

        def adam_step(carry, _):
            p, s, best_p, best_v = carry
            v, g = vgrad(p)
            updates, s = opt.update(g, s, p)
            p_candidate = optax.apply_updates(p, updates)
            # Evaluate the candidate at the position being recorded.  The
            # previous implementation compared ``v = f(p)`` but stored
            # ``p_new``; on a non-monotone Adam step this paired the old
            # objective with a different position and could select a point
            # that was not actually the best one seen.
            v_candidate = jnlp(p_candidate)
            valid = (
                jnp.isfinite(v_candidate)
                & (v_candidate < 1e30)
                & (v_candidate > -1e30)
            )
            p = jnp.where(valid, p_candidate, p)
            v = jnp.where(valid, v_candidate, v)
            better = valid & (v < best_v)
            best_p = jnp.where(better, p, best_p)
            best_v = jnp.where(better, v, best_v)
            return (p, s, best_p, best_v), None

        v0 = jnlp(position)
        v0_valid = (
            jnp.isfinite(v0) & (v0 < 1e30) & (v0 > -1e30)
        )
        v0 = jnp.where(v0_valid, v0, jnp.inf)
        (p, _, best_p, best_v), _ = jax.lax.scan(
            adam_step, (position, opt.init(position), position, v0),
            None, length=n_adam,
        )
        vp = jnlp(p)
        take = (
            jnp.isfinite(vp)
            & (vp < 1e30)
            & (vp > -1e30)
            & (vp <= best_v)
        )
        best_p = jnp.where(take, p, best_p)
        best_v = jnp.where(take, vp, best_v)
        return best_p, best_v

    def lbfgs_polish(self, jlp, position, value, n_lbfgs_restarts=None):
        """Restarted single-point L-BFGS polish (fresh curvature memory each
        restart), keeping improvements. Returns (position, -logp)."""
        if n_lbfgs_restarts is None:
            n_lbfgs_restarts = self.config.n_lbfgs_restarts
        jnlp = jax.jit(lambda p: -jlp(p))
        vgrad = jax.value_and_grad(jnlp)
        solver = jaxopt.LBFGS(fun=vgrad, value_and_grad=True)
        best_p, best_v = position, value
        for _ in range(n_lbfgs_restarts):
            res = solver.run(best_p)
            val = jnlp(res.params)
            if bool(val < 1e30) and bool(val < best_v - 0.1):
                best_p, best_v = res.params, val
            else:
                break
        return best_p, best_v

    def hessian_mass_matrix(self, jlp, position):
        """Estimate diagonal inverse mass matrix from the Hessian of the log posterior.

        Uses exact forward-over-reverse Hessian-vector products (the RK4
        spectral-equivalence rewrite made the model forward-mode
        differentiable; finite differences produced clip-saturated entries
        in soft directions). Requires dim HVP evaluations.

        Elements are clamped to [1e-6, 1e6] to guard against degenerate
        curvature far from the MAP.

        Args:
            jlp: JIT-compiled log posterior function (scalar output).
            position: 1D JAX array of parameter values (normalized space).

        Returns:
            1D JAX array of shape (dim,) representing the diagonal
            inverse mass matrix.
        """
        grad_fn = jax.grad(lambda p: -jlp(p))
        dim = len(position)

        def hvp_diag(i):
            v = jnp.zeros(dim).at[i].set(1.0)
            return jax.jvp(grad_fn, (position,), (v,))[1][i]

        diag_H = jax.lax.map(hvp_diag, jnp.arange(dim))
        # Non-positive curvature (unconverged MAP / flat directions) would
        # clip to the floor and give that direction a ~1e6 inverse mass,
        # which collapses dual averaging (observed eps -> 4e-9). Use the
        # median positive curvature as a neutral scale instead.
        positive = jnp.isfinite(diag_H) & (diag_H > 0)
        pos_med = jnp.nanmedian(jnp.where(positive, diag_H, jnp.nan))
        # ``nanmedian`` is NaN when the whole diagonal is non-positive or
        # non-finite.  Such a Hessian is possible before the MAP polish (and
        # for genuinely flat targets); use the identity metric in that case
        # so warmup receives finite parameters instead of poisoning dual
        # averaging with NaNs.
        pos_med = jnp.where(
            jnp.isfinite(pos_med) & (pos_med > 0), pos_med, jnp.asarray(1.0)
        )
        diag_H = jnp.where(positive, diag_H, pos_med)
        # Relative floor: near-zero positive curvature is as damaging as
        # negative (caps imm at 1e4 x the median imm).
        diag_H = jnp.maximum(diag_H, 1e-4 * pos_med)
        return 1.0 / jnp.clip(diag_H, 1e-6, 1e6)

    def hessian_mass_matrix_dense(self, jlp, position):
        """Dense inverse mass matrix from the full Hessian at `position`.

        Same HVP count as the diagonal estimate (one per coordinate), but
        keeps the full matrix: the inverse Hessian approximates the local
        posterior covariance, so a dense metric absorbs the correlations a
        diagonal metric cannot. The eigenvalue analog of the diagonal median
        guard handles indefiniteness: non-positive eigenvalues are replaced
        by the median positive one, with a relative floor of 1e-4 x median.

        Returns:
            (dim, dim) symmetric positive-definite inverse mass matrix.
        """
        grad_fn = jax.grad(lambda p: -jlp(p))
        dim = len(position)

        def hvp_row(i):
            v = jnp.zeros(dim).at[i].set(1.0)
            return jax.jvp(grad_fn, (position,), (v,))[1]

        H = jax.lax.map(hvp_row, jnp.arange(dim))
        H = 0.5 * (H + H.T)
        lam, V = jnp.linalg.eigh(H)
        positive = jnp.isfinite(lam) & (lam > 0)
        pos_med = jnp.nanmedian(jnp.where(positive, lam, jnp.nan))
        pos_med = jnp.where(
            jnp.isfinite(pos_med) & (pos_med > 0), pos_med, jnp.asarray(1.0)
        )
        lam = jnp.where(positive, lam, pos_med)
        lam = jnp.clip(jnp.maximum(lam, 1e-4 * pos_med), 1e-6, 1e6)
        return (V * (1.0 / lam)) @ V.T

    def mclmc_mass_matrix(self, log_posterior, initial_positions, rng_key,
                          n_tune_steps=None, n_steps=None, n_chains=None):
        """Estimate a diagonal inverse mass matrix from short MCLMC runs.

        A short single-chain blackjax MCLMC tune finds (L, eps); n_chains
        unadjusted MCLMC chains are then run at a conservative eps/2 from
        jittered starts. Chains with any non-finite position in the second
        half are dropped; the estimate is the mean over surviving chains of
        the within-chain variance over the second half of steps. Fallback
        ladder: retry once at eps/4, then _hessian_mass_matrix, then ones.
        Result is clamped to [1e-8, 1e8].

        Args:
            log_posterior: Log posterior function (scalar output).
            initial_positions: (n, dim) array of starting positions.
            rng_key: JAX random key.
            n_tune_steps: Steps for the single-chain L/eps tune.
            n_steps: Steps per estimation chain.
            n_chains: Number of estimation chains.

        Returns:
            Tuple of (imm, info_dict) where imm is a (dim,) diagonal inverse
            mass matrix and info_dict records n_survivors, eps, and rung.
        """
        import blackjax
        import blackjax.mcmc.integrators
        import blackjax.mcmc.mclmc

        cfg = self.config
        if n_tune_steps is None:
            n_tune_steps = cfg.mclmc_mm_n_tune_steps
        if n_steps is None:
            n_steps = cfg.mclmc_mm_n_steps
        if n_chains is None:
            n_chains = cfg.mclmc_mm_n_chains
        jlp = jax.jit(log_posterior)
        dim = initial_positions.shape[1]
        rng_key, tune_key, init_key, jit_key = jax.random.split(rng_key, 4)

        # Rung 0: short single-chain tune for (L, eps).
        L, eps, tuned_imm = jnp.sqrt(dim), 0.01, jnp.ones(dim)
        try:
            tune_state = blackjax.mcmc.mclmc.init(
                position=initial_positions[0], logdensity_fn=jlp,
                rng_key=init_key,
            )
            kernel = lambda imm: blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=jlp,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                inverse_mass_matrix=imm,
            )
            _, tp, _ = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=kernel,
                num_steps=n_tune_steps,
                state=tune_state,
                rng_key=tune_key,
                diagonal_preconditioning=True,
            )
            if (
                jnp.isfinite(tp.L)
                and jnp.isfinite(tp.step_size)
                and bool(jnp.all(jnp.isfinite(tp.inverse_mass_matrix)))
            ):
                L, eps, tuned_imm = tp.L, tp.step_size, jnp.maximum(
                    tp.inverse_mass_matrix,
                    1e-8 * jnp.max(tp.inverse_mass_matrix),
                )
            else:
                print(
                    "MCLMC mass-matrix tune produced non-finite params; "
                    "using conservative defaults.",
                    flush=True,
                )
        except Exception as e:
            print(
                f"MCLMC mass-matrix tune failed ({e}); using conservative "
                "defaults.",
                flush=True,
            )

        step_kernel = blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=jlp,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            inverse_mass_matrix=tuned_imm,
        )

        idx = jnp.arange(n_chains) % initial_positions.shape[0]
        starts = initial_positions[idx]
        starts = starts + 0.01 * jax.random.normal(jit_key, starts.shape)

        def run_chain(pos, key, step_size):
            init_k, run_k = jax.random.split(key)
            state = blackjax.mcmc.mclmc.init(
                position=pos, logdensity_fn=jlp, rng_key=init_k,
            )

            def step(state, k):
                state, _ = step_kernel(k, state, L=L, step_size=step_size)
                return state, state.position

            _, positions = jax.lax.scan(
                step, state, jax.random.split(run_k, n_steps)
            )
            return positions

        def try_rung(step_size, key):
            keys = jax.random.split(key, n_chains)
            positions = jax.vmap(run_chain, in_axes=(0, 0, None))(
                starts, keys, step_size
            )  # (n_chains, n_steps, dim)
            second = positions[:, n_steps // 2:, :]
            finite = jnp.all(jnp.isfinite(second), axis=(1, 2))
            n_surv = int(jnp.sum(finite))
            if n_surv < max(n_chains // 2, 1):
                return None, n_surv
            var = jnp.var(second, axis=1)  # within-chain variance
            imm = jnp.mean(var[finite], axis=0)
            if not bool(jnp.all(jnp.isfinite(imm))):
                return None, n_surv
            # fp32 variance can go slightly negative for tight params
            imm = jnp.maximum(imm, 1e-8 * jnp.max(imm))
            return imm, n_surv

        for rung, step_size in (("eps/2", eps / 2), ("eps/4", eps / 4)):
            rng_key, run_key = jax.random.split(rng_key)
            imm, n_surv = try_rung(step_size, run_key)
            if imm is not None:
                info = {
                    "n_survivors": n_surv,
                    "eps": float(step_size),
                    "rung": rung,
                }
                return jnp.clip(imm, 1e-8, 1e8), info
            print(
                f"MCLMC mass-matrix estimation at {rung} "
                f"(eps={float(step_size):.3e}) failed: only {n_surv}/"
                f"{n_chains} chains finite.",
                flush=True,
            )

        print(
            "MCLMC mass-matrix estimation failed at all step sizes; "
            "falling back to Hessian mass matrix.",
            flush=True,
        )
        try:
            imm = self.hessian_mass_matrix(
                jlp, self.best_fit_position(jlp, initial_positions[0])
            )
            if bool(jnp.all(jnp.isfinite(imm))):
                return (
                    jnp.clip(imm, 1e-8, 1e8),
                    {"n_survivors": 0, "eps": float(eps / 4),
                     "rung": "hessian"},
                )
        except Exception as e:
            print(f"Hessian fallback failed ({e}).", flush=True)
        print(
            "Hessian mass-matrix fallback failed; using ones.", flush=True
        )
        return (
            jnp.ones(dim),
            {"n_survivors": 0, "eps": float(eps / 4), "rung": "ones"},
        )

    def minimize(
        self, log_posterior, initial_positions, n_chains, output_file,
        chi2_threshold=1, pathfinder_key=None, mode="auto",
    ):
        """Run L-BFGS from every chain's start, save the results, and move
        chains whose minimum is worse than chi2_threshold x the best one to
        the best position.

        With pathfinder_key (identical starts only) the L-BFGS polish of the
        Adam winner runs in a thread on a spare device while Pathfinder seeds
        the chains from the unpolished winner; the seeds are left in
        self.pathfinder_positions and the polished MAP is still saved."""
        # minimize negative log posterior
        jnlp = jax.jit(lambda p: -log_posterior(p))
        vgrad = jax.value_and_grad(jnlp)
        solver = jaxopt.LBFGS(fun=vgrad, value_and_grad=True)

        print("Running minimization before sampling", flush=True)
        self.pathfinder_positions = None
        # A *value* test, not the random_start flag: a restart or an upstream
        # stage can also hand over identical positions, and those runs must
        # keep taking this branch.
        identical = bool(jnp.all(initial_positions == initial_positions[0]))
        if mode == "auto" and identical:
            # identical starts (random_start: false): minimize once and
            # broadcast instead of K concurrent L-BFGS grads (OOM at K=32).
            if pathfinder_key is not None:
                x_opt = self.polish_and_pathfinder(
                    log_posterior, initial_positions[0], n_chains,
                    pathfinder_key, output_file,
                )
            else:
                x_opt = self.best_fit_position(
                    log_posterior, initial_positions[0]
                )
            initial_positions = np.tile(
                np.asarray(x_opt), (n_chains, 1)
            )
            values = np.full(n_chains, float(jnlp(x_opt)))
        else:
            minimize_map = self.host.map_chains(solver.run)
            res = minimize_map(initial_positions)
            initial_positions = gather_to_host(res.params)
            values = gather_to_host(res.state.value)
        SeedingOutputs(output_file, self.host).write_minimization(
            initial_positions, values
        )
        self.map_values = values
        if mode != "auto":
            # per_chain: no chi2 re-tiling (Minimize wants every chain's own
            # optimum, not the best one broadcast).
            return initial_positions

        chi2_ratio = values / np.min(values)
        initial_positions_min = jnp.tile(
            initial_positions[np.argmin(values)], n_chains
        ).reshape(n_chains, -1)
        initial_positions = jnp.where(
            chi2_ratio[:, None] > chi2_threshold,
            initial_positions_min,
            initial_positions,
        )
        return initial_positions
