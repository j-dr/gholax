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
