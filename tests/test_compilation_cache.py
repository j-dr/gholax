"""The CLI must wire the YAML compilation_cache_dir key into jax config.

Runs in a subprocess so the global jax config of the test session is
untouched and the checks see a pristine (unset) starting state.
"""

import subprocess
import sys

PROBE = """
import os
import jax
from gholax.cli import _setup_compilation_cache

# explicit null disables: config stays unset
_setup_compilation_cache({'compilation_cache_dir': None})
assert jax.config.jax_compilation_cache_dir is None

# explicit dir is honored, ~ expanded, and min compile time lowered to 1s
_setup_compilation_cache({'compilation_cache_dir': '~/my_cache'})
assert jax.config.jax_compilation_cache_dir == os.path.expanduser('~/my_cache')
assert jax.config.jax_persistent_cache_min_compile_time_secs == 1

# absent key falls back to $SCRATCH when set (NERSC), else the user cache
_setup_compilation_cache({})
if os.environ.get('SCRATCH'):
    expected = os.path.join(os.environ['SCRATCH'], 'gholax', 'jax_cache')
else:
    expected = os.path.expanduser('~/.cache/gholax/jax_cache')
assert jax.config.jax_compilation_cache_dir == expected
print("OK")
"""


def test_compilation_cache_config_wiring():
    res = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout
