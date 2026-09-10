"""blackjax version-compatibility tests for MCLMC adaptation.

blackjax 1.2.5 has no `params` kwarg on mclmc_find_L_and_step_size; gholax
must fall back to calling without it (warning that the initial guess is
ignored) and must not claim warm-start convergence on the first round.
"""

import blackjax
import jax
import jax.numpy as jnp
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

import gholax.sampler.mclmc as mclmc_mod
from gholax.sampler import MCLMC


def _mclmc(cfg=None):
    return MCLMC({"sampler": {"MCLMC": cfg or {}}})


def _toy_jlp():
    return jax.jit(lambda x: -0.5 * jnp.sum(x**2))


def _initial(jlp, dim=2):
    state = blackjax.mcmc.mclmc.init(
        position=jnp.zeros(dim), logdensity_fn=jlp, rng_key=jax.random.key(0)
    )
    params = MCLMCAdaptationState(
        L=jnp.sqrt(dim), step_size=0.01, inverse_mass_matrix=jnp.ones(dim)
    )
    return state, params


def test_adapt_unadjusted_runs_on_installed_blackjax():
    """The real adaptation call must work whether or not `params` exists."""
    jlp = _toy_jlp()
    state, params0 = _initial(jlp)
    sampler = _mclmc({"n_steps_warmup": 100})
    _, params = sampler._adapt_unadjusted(jlp, state, jax.random.key(1), params0)
    assert float(params.step_size) > 0
    assert float(params.L) > 0


def _identity_stub(self, jlp, state, rng_key, params, diagonal_preconditioning=None):
    return state, params


def test_no_round1_convergence_when_params_not_injectable(monkeypatch, capsys):
    """With params ignored, warm_start must not allow round-1 convergence."""
    monkeypatch.setattr(mclmc_mod, "_UNADJ_ADAPT_HAS_PARAMS", False)
    monkeypatch.setattr(MCLMC, "_adapt_unadjusted", _identity_stub)

    jlp = _toy_jlp()
    state, params0 = _initial(jlp)
    sampler = _mclmc({"warmup_tolerance": 0.2, "max_warmup_rounds": 5})
    sampler._adapt_with_convergence(
        jlp, state, jax.random.key(1), params0, warm_start=True
    )
    out = capsys.readouterr().out
    assert "is ignored" in out
    assert "converged after 1 rounds" not in out
    assert "converged after 2 rounds" in out


def test_round1_convergence_allowed_when_params_injectable(monkeypatch, capsys):
    monkeypatch.setattr(mclmc_mod, "_UNADJ_ADAPT_HAS_PARAMS", True)
    monkeypatch.setattr(MCLMC, "_adapt_unadjusted", _identity_stub)

    jlp = _toy_jlp()
    state, params0 = _initial(jlp)
    sampler = _mclmc({"warmup_tolerance": 0.2, "max_warmup_rounds": 5})
    sampler._adapt_with_convergence(
        jlp, state, jax.random.key(1), params0, warm_start=True
    )
    out = capsys.readouterr().out
    assert "is ignored" not in out
    assert "converged after 1 rounds" in out


def test_shared_warmup_defaults_match_legacy():
    """MCLMC's own defaults must survive the shared WarmupConfig."""
    s = _mclmc()
    assert s.warmup_config.algorithm == "mclmc"
    assert (s.n_steps_warmup, s.step_size_init, s.target_acceptance_rate) == (
        5000, 0.01, 0.65
    )
    assert s.diagonal_preconditioning is True
    assert s.warmup_init_file is None


def test_pooled_pre_adaptation_is_opt_in(monkeypatch, tmp_path):
    """Default MCLMC never runs the pooled engine; opt-in seeds L/eps/imm."""
    from gholax.sampler.warmup import Warmup, WarmupResult

    calls = []

    def fake(self, req):
        calls.append(req)
        return WarmupResult(
            inverse_mass_matrix=jnp.full(req.initial_positions.shape[1], 3.0),
            step_size=jnp.asarray(0.25),
            positions=req.initial_positions,
        )

    monkeypatch.setattr(Warmup, "run_pooled_window", fake)
    seen = {}

    def fake_adapt(self, jlp, state, key, params, output_file=None, warm_start=False):
        seen["params"] = params
        return state, params

    monkeypatch.setattr(mclmc_mod.MCLMC, "_adapt_with_convergence", fake_adapt)
    monkeypatch.setattr(
        mclmc_mod.MCLMC, "_run_convergence_loop",
        lambda self, *a, **k: (jnp.zeros((1, 1, 2)), jnp.zeros((1, 1))),
    )

    from tests.test_sampler_consolidation import ToyModel

    cfg = {"minimize_and_sample": False, "warmup": {"algorithm": "pooled_window"}}
    _mclmc(cfg).run(ToyModel(), str(tmp_path / "a"))
    assert len(calls) == 1
    assert float(seen["params"].step_size) == 0.25
    assert float(seen["params"].inverse_mass_matrix[0]) == 3.0

    calls.clear()
    _mclmc({"minimize_and_sample": False}).run(ToyModel(), str(tmp_path / "b"))
    assert not calls
