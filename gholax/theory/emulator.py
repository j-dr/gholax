import os

import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np
import yaml


def activation(x, alpha, beta):
    """
    Swish-like activation function with learnable parameters.

    Args:
        x: Input array
        alpha: Scaling parameter for sigmoid
        beta: Beta parameter controlling the activation shape

    Returns:
        Activated output array
    """
    return jnp.multiply(
        jnp.add(
            beta,
            jnp.multiply(jax.nn.sigmoid(jnp.multiply(alpha, x)), jnp.subtract(1, beta)),
        ),
        x,
    )


def _resolve_emu_config(config, abspath=False, data_dir=None,
                        empty_data_dir_on_abspath=False):
    """Resolve an emulator YAML config name/dict to (cfg dict, data_dir).

    Relative config names are resolved against data_dir (default:
    emu_weights/ next to this file). With abspath=True the config is opened
    as given (or used directly if already a dict);
    empty_data_dir_on_abspath reproduces MultiSpectrumEmulator's convention
    of blanking data_dir so weight paths come verbatim from the config.
    """
    if not abspath:
        if data_dir is None:
            data_dir = "/".join(
                [
                    os.path.dirname(os.path.realpath(__file__)),
                    "emu_weights",
                ]
            )

            if type(config) is not dict:
                config_abspath = "/".join(
                    [
                        data_dir,
                        config,
                    ]
                )
        else:
            if type(config) is not dict:
                config_abspath = "/".join(
                    [
                        data_dir,
                        config,
                    ]
                )

        with open(config_abspath, "r") as fp:
            cfg = yaml.load(fp, Loader=yaml.SafeLoader)
    else:
        if type(config) is not dict:
            config_abspath = config
            with open(config_abspath, "r") as fp:
                cfg = yaml.load(fp, Loader=yaml.SafeLoader)
        else:
            cfg = config

        if empty_data_dir_on_abspath:
            data_dir = ""

    return cfg, data_dir


def _load_h5_weights(filebase, scale_As=False, As_param_order=None,
                     first_layer_param_order=None, param_idx=None):
    """Load MLP weights from filebase.h5 into a dict of jnp arrays.

    Args:
        filebase: Path to the weight file (without .h5 extension).
        scale_As: Multiply the As entry of param_mean/param_sigmas by 1e9.
        As_param_order: Parameter order used to locate As (index 0 if None).
        first_layer_param_order: When given, the first W layer's input width
            is asserted against its length and the layer is reordered by
            param_idx; param_mean/param_sigmas are reordered as well.
        param_idx: Index list mapping the weights' parameter order to the
            caller's input parameter order.
    """
    out = {}
    with h5.File("{}.h5".format(filebase), "r") as weights:
        for k in weights:
            if k in ["W", "b", "alphas", "betas"]:
                w = []
                for i, wi in enumerate(weights[k]):
                    w.append(weights[k][wi][:].astype(np.float32))
                    if (first_layer_param_order is not None) & (i == 0) & (k == "W"):
                        assert w[i].shape[0] == len(first_layer_param_order)
                        w[i] = w[i][param_idx, :]
                    w[i] = jnp.array(w[i]).astype(jnp.float32)

            elif k in ["param_mean", "param_sigmas"]:
                w = np.array(weights[k][f'{k}_0']).astype(np.float32)
                if scale_As:
                    if As_param_order is not None:
                        w[As_param_order.index("As")] *= 1e9
                    else:
                        w[0] *= 1e9
                if first_layer_param_order is not None:
                    w = w[param_idx]

                w = jnp.array(w).astype(jnp.float32)

            else:
                w = jnp.array(weights[k][f'{k}_0']).astype(jnp.float32)

            out[k] = w
    return out


def _build_scan_stacks(W, b, alphas, betas):
    """Pre-stack hidden layer weights for the scan-based forward pass.

    Returns (use_scan, W_hidden, b_hidden, alphas_hidden, betas_hidden);
    the stacks are None unless all middle hidden layers share one shape.
    """
    mid_W = W[1:-1]
    use_scan = (
        len(W) > 2
        and len(mid_W) > 0
        and all(w.shape == mid_W[0].shape for w in mid_W)
    )
    if not use_scan:
        return False, None, None, None, None
    return (
        True,
        jnp.stack(mid_W),
        jnp.stack(b[1:-1]),
        jnp.stack(alphas[1:]),
        jnp.stack(betas[1:]),
    )


def _mlp_step(x, wandb):
    """Single hidden layer step for use with jax.lax.scan."""
    W_i, b_i, alpha_i, beta_i = wandb
    x = x @ W_i + b_i
    x = activation(x, alpha_i, beta_i)
    return x, None


def _hidden_forward(x, W, b, alphas, betas, use_scan, hidden_stacks):
    """Run the input + hidden layers of the MLP (everything before the
    linear output layer), via scan when the hidden stacks are available."""
    if use_scan:
        # First hidden layer
        x = x @ W[0] + b[0]
        x = activation(x, alphas[0], betas[0])

        # Middle hidden layers via scan
        x, _ = jax.lax.scan(_mlp_step, x, hidden_stacks)
    else:
        for i in range(len(W) - 1):
            x = x @ W[i] + b[i]
            x = activation(x, alphas[i], betas[i])
    return x


_DEFAULT_PARAM_ORDER = ["As", "ns", "H0", "w", "ombh2", "omch2", "logmnu", "z"]


def _build_range_arrays(param_ranges, param_order):
    """(lo, hi) arrays aligned with param_order (+-inf where a parameter has
    no range), or (None, None) when param_ranges is empty.  Names not in
    param_order raise, as does lo > hi."""
    if not param_ranges:
        return None, None
    if param_order is None:
        raise ValueError("param_ranges given but the input parameter order is unknown")
    lo = np.full(len(param_order), -np.inf, dtype=np.float32)
    hi = np.full(len(param_order), np.inf, dtype=np.float32)
    for name, (a, b) in param_ranges.items():
        if name not in param_order:
            raise ValueError(
                f"param_ranges key {name!r} not in emulator inputs {param_order}"
            )
        if a > b:
            raise ValueError(f"param_ranges for {name!r}: lo {a} > hi {b}")
        lo[param_order.index(name)] = a
        hi[param_order.index(name)] = b
    return jnp.asarray(lo), jnp.asarray(hi)


def _clip_params(parameters, lo, hi):
    """Clip emulator inputs to the training box when ranges are set."""
    if lo is None:
        return parameters
    return jnp.clip(parameters, lo, hi)


def _range_dict(param_ranges):
    return {k: (float(v[0]), float(v[1])) for k, v in (param_ranges or {}).items()}


class Emulator(object):
    """MLP emulator for single-spectrum power spectrum predictions.

    Uses PCA-compressed output with sinh denormalization. Weights are
    loaded from HDF5 files.
    """

    def __init__(self, filebase, kmin=1e-3, kmax=0.5, scale_As=True,
                 param_ranges=None, param_order=None):
        """Initialize the emulator and load weights.

        Args:
            filebase: Path to the HDF5 weight file (without .h5 extension).
            kmin: Minimum wavenumber for the k grid.
            kmax: Maximum wavenumber for the k grid.
            scale_As: If True, scale As by 1e9 in normalization parameters.
            param_ranges: Optional {name: [lo, hi]} training box; inputs are
                clipped to it in predict.
            param_order: Input parameter names (defaults to the 8-parameter
                gholax order when the network has 8 inputs).
        """
        super(Emulator, self).__init__()
        self.scale_As = scale_As
        self.load(filebase)

        self.n_parameters = self.W[0].shape[0]
        if param_order is None and self.n_parameters == len(_DEFAULT_PARAM_ORDER):
            param_order = _DEFAULT_PARAM_ORDER
        self.param_ranges = _range_dict(param_ranges)
        self.param_lo, self.param_hi = _build_range_arrays(
            self.param_ranges, param_order
        )
        self.n_components = self.W[-1].shape[-1]
        self.n_layers = len(self.W)
        self.nk = self.sigmas.shape[0]
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), self.nk)

    def load(self, filebase):
        """Load neural network weights from an HDF5 file.

        Args:
            filebase: Path to the weight file (without .h5 extension).
        """
        with h5.File("{}.h5".format(filebase), "r") as weights:
            for k in weights:
                if k in ["W", "b", "alphas", "betas"]:
                    w = []
                    for i, wi in enumerate(weights[k]):
                        w.append(jnp.array(weights[k][wi]))
                elif k in ["param_mean", "param_sigmas"]:
                    w = jnp.array(weights[k][f'{k}_0']).astype(jnp.float32)
                    if self.scale_As:
                        w = w.at[0].set(w[0] * 1e9)
                else:
                    w = jnp.array(weights[k][f'{k}_0']).astype(jnp.float32)

                setattr(self, k, w)

    def predict(self, parameters):
        """Run the MLP forward pass to predict the power spectrum.

        Args:
            parameters: Array of shape (n_samples, n_parameters) with cosmological
                parameters.

        Returns:
            Array of predicted power spectrum values.
        """
        parameters = _clip_params(parameters, self.param_lo, self.param_hi)
        x = (parameters - self.param_mean) / self.param_sigmas

        for i in range(self.n_layers - 1):
            # linear network operation
            x = x @ self.W[i] + self.b[i]

            # non-linear activation function
            x = activation(x, self.alphas[i], self.betas[i])

        # linear output layer
        x = ((x @ self.W[-1]) + self.b[-1]) * self.pc_sigmas[
            : self.n_components
        ] + self.pc_mean[: self.n_components]
        x = (
            jnp.sinh((x @ self.v[:, : self.n_components].T) * self.sigmas + self.mean)
            * self.fstd
        )

        return x


class MultiSpectrumEmulator(object):
    """MLP emulator for multiple power spectrum components simultaneously.

    Predicts n_spec spectra at once, optionally using sigma8(z) as a
    transformed variable and scaling by sigma8(z)^2.
    """

    def __init__(
        self,
        config,
        input_param_order=None,
        abspath=False,
        data_dir=None,
        s8_tvar=None,
        scale_As_spec=None,
        scale_As_d=None,
        scale_by_s8zsq=None,
    ):
        """Initialize the multi-spectrum emulator.

        Args:
            config: Path to a YAML config file or a config dict.
            input_param_order: Ordering of input parameters (default: from config).
            abspath: If True, treat config/weight paths as absolute.
            data_dir: Directory containing emulator weight files.
            s8_tvar: If True, replace the last parameter with sigma8(z).
            scale_As_spec: If True, scale As by 1e9 for spectrum normalization.
            scale_As_d: If True, scale As by 1e9 for sigma8(z) emulator.
            scale_by_s8zsq: If True, multiply output by sigma8(z)^2.
        """
        super(MultiSpectrumEmulator, self).__init__()

        cfg, data_dir = _resolve_emu_config(
            config, abspath=abspath, data_dir=data_dir,
            empty_data_dir_on_abspath=True,
        )

        spec_base = cfg["spec_base"]
        s8z_base = cfg["s8z_base"]
        kmin = float(cfg["kmin"])
        kmax = float(cfg["kmax"])
        self.n_spec = cfg["nspec"]

        self.param_order_spec = cfg["param_order_spec"]
        self.param_order_d = cfg["param_order_d"]
        if input_param_order is None:
            input_param_order = self.param_order_spec

        self.param_idx_spec = [
            self.param_order_spec.index(p) for p in input_param_order
        ]

        self.input_param_order = input_param_order
        self.param_ranges = _range_dict(cfg.get("param_ranges", None))

        if s8_tvar is None:
            self.s8_tvar = bool(cfg.get("s8_tvar", True))
        else:
            self.s8_tvar = s8_tvar

        if scale_As_d is None:
            self.scale_As_d = bool(cfg.get("scale_As_d", True))
        else:
            self.scale_As_d = scale_As_d

        if scale_As_spec is None:
            self.scale_As_spec = bool(cfg.get("scale_As_spec", False))
        else:
            self.scale_As_spec = scale_As_spec

        if scale_by_s8zsq is None:
            self.scale_by_s8zsq = bool(cfg.get("scale_by_s8zsq", True))
        else:
            self.scale_by_s8zsq = scale_by_s8zsq

        if not abspath:
            self.sigma8z_emu = ScalarEmulator(
                s8z_base,
                scale_As=self.scale_As_d,
                data_dir=data_dir,
                input_param_order=input_param_order,
                weight_param_order=self.param_order_d,
                param_ranges=self.param_ranges,
            )
        else:
            self.sigma8z_emu = ScalarEmulator(
                s8z_base,
                scale_As=self.scale_As_d,
                input_param_order=input_param_order,
                weight_param_order=self.param_order_d,
                param_ranges=self.param_ranges,
            )

        self.load_spec(f"{data_dir}/{spec_base}")

        self.n_parameters = self.W[0].shape[0]
        self.param_lo, self.param_hi = _build_range_arrays(
            self.param_ranges, self.input_param_order
        )
        self.n_components = self.W[-1].shape[-1]
        self.n_layers = len(self.W)
        self.nk = self.sigmas.shape[0] // self.n_spec
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), self.nk)

        (
            self._use_scan,
            self._W_hidden,
            self._b_hidden,
            self._alphas_hidden,
            self._betas_hidden,
        ) = _build_scan_stacks(self.W, self.b, self.alphas, self.betas)

    def load_spec(self, filebase):
        """Load spectrum emulator weights from an HDF5 file.

        Args:
            filebase: Path to the weight file (without .h5 extension).
        """
        loaded = _load_h5_weights(
            filebase,
            scale_As=self.scale_As_spec,
            As_param_order=self.param_order_spec,
            first_layer_param_order=self.param_order_spec,
            param_idx=self.param_idx_spec,
        )
        for k, w in loaded.items():
            setattr(self, k, w)

    def predict(self, parameters):
        """Run the MLP forward pass to predict multiple spectra.

        Args:
            parameters: Array of shape (n_samples, n_parameters).

        Returns:
            Array of shape (n_samples, n_spec, nk) with predicted spectra.
        """
        parameters = _clip_params(parameters, self.param_lo, self.param_hi)
        if self.s8_tvar:
            s8z = self.sigma8z_emu.predict(parameters)[:, 0]
            parameters = parameters.at[:, -1].set(s8z)

        x = (parameters - self.param_mean) / self.param_sigmas
        x = _hidden_forward(
            x, self.W, self.b, self.alphas, self.betas, self._use_scan,
            (self._W_hidden, self._b_hidden, self._alphas_hidden, self._betas_hidden),
        )

        # Linear output layer
        x = ((x @ self.W[-1]) + self.b[-1]) * self.pc_sigmas[
            : self.n_components
        ] + self.pc_mean[: self.n_components]
        x = (
            jnp.sinh((x @ self.v[:, : self.n_components].T * self.sigmas + self.mean))
            * self.fstd
        )

        if self.scale_by_s8zsq:
            if not self.s8_tvar:
                s8z = self.sigma8z_emu.predict(parameters)[:, 0]
            x = x * (s8z[:, None] ** 2)
        return x.reshape(-1, self.n_spec, self.nk)


class ScalarEmulator(object):
    """MLP emulator for scalar quantities (e.g., sigma8(z), chi(z), E(z))."""

    def __init__(
        self,
        filebase_or_config,
        scale_As=None,
        data_dir=None,
        input_param_order=None,
        weight_param_order=None,
        param_ranges=None,
    ):
        """Initialize the scalar emulator.

        Args:
            filebase_or_config: Name of the weight file (resolved relative to
                data_dir), a YAML config filename (ending in .yaml), or a dict
                with keys 'filebase', 'scale_As', and 'param_order'.
            scale_As: If True, scale As by 1e9 in normalization parameters.
                Overrides config value when explicitly provided.
            data_dir: Directory containing weight files (default: emu_weights/).
            input_param_order: Ordering of input parameters at call time.
                Overrides config value when explicitly provided.
            weight_param_order: Ordering of parameters used during training.
                Overrides config value when explicitly provided.
            param_ranges: {name: [lo, hi]} training box; inputs are clipped
                to it in predict. Overrides config value when provided.
        """
        super(ScalarEmulator, self).__init__()

        # Resolve config
        if isinstance(filebase_or_config, dict):
            cfg = filebase_or_config
            filebase = cfg["filebase"]
            cfg_param_order = cfg.get("param_order", None)
            cfg_scale_As = cfg.get("scale_As", True)
            cfg_param_ranges = cfg.get("param_ranges", None)
        elif isinstance(filebase_or_config, str) and filebase_or_config.endswith(".yaml"):
            cfg_path = filebase_or_config
            if data_dir is None:
                cfg_dir = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "emu_weights"
                )
            else:
                cfg_dir = data_dir
            with open(os.path.join(cfg_dir, cfg_path), "r") as f:
                cfg = yaml.safe_load(f)
            filebase = cfg["filebase"]
            cfg_param_order = cfg.get("param_order", None)
            cfg_scale_As = cfg.get("scale_As", True)
            cfg_param_ranges = cfg.get("param_ranges", None)
        else:
            filebase = filebase_or_config
            cfg_param_order = None
            cfg_scale_As = True
            cfg_param_ranges = None

        # Apply config defaults, kwargs override
        self.scale_As = cfg_scale_As if scale_As is None else scale_As

        if weight_param_order is None and cfg_param_order is not None:
            weight_param_order = cfg_param_order
        if input_param_order is None and cfg_param_order is not None:
            input_param_order = cfg_param_order

        self.input_param_order = input_param_order
        self.weight_param_order = weight_param_order
        if self.input_param_order is not None:
            self.param_idx = [
                self.weight_param_order.index(p) for p in self.input_param_order
            ]

        self.load(filebase, data_dir=data_dir)

        self.n_parameters = self.W[0].shape[0]
        self.n_components = self.W[-1].shape[-1]
        self.n_layers = len(self.W)
        order = self.input_param_order or self.weight_param_order
        if order is None and self.n_parameters == len(_DEFAULT_PARAM_ORDER):
            order = _DEFAULT_PARAM_ORDER
        self.param_ranges = _range_dict(
            cfg_param_ranges if param_ranges is None else param_ranges
        )
        self.param_lo, self.param_hi = _build_range_arrays(
            self.param_ranges, order
        )

        (
            self._use_scan,
            self._W_hidden,
            self._b_hidden,
            self._alphas_hidden,
            self._betas_hidden,
        ) = _build_scan_stacks(self.W, self.b, self.alphas, self.betas)

    def load(self, filebase, data_dir=None):
        """Load neural network weights from an HDF5 file.

        Args:
            filebase: Name of the weight file (without .h5 extension).
            data_dir: Directory containing weight files.
        """
        if data_dir is None:
            emu_abspath = "/".join(
                [
                    os.path.dirname(os.path.realpath(__file__)),
                    "emu_weights",
                    filebase,
                ]
            )
        else:
            emu_abspath = "/".join(
                [
                    data_dir,
                    filebase,
                ]
            )

        loaded = _load_h5_weights(
            emu_abspath,
            scale_As=self.scale_As,
            As_param_order=self.weight_param_order,
            first_layer_param_order=self.input_param_order,
            param_idx=getattr(self, "param_idx", None),
        )
        for k, w in loaded.items():
            setattr(self, k, w)

    def predict(self, parameters):
        """Run the MLP forward pass to predict a scalar quantity.

        Args:
            parameters: Array of shape (n_samples, n_parameters).

        Returns:
            Array of predicted scalar values.
        """
        parameters = _clip_params(parameters, self.param_lo, self.param_hi)
        x = (parameters - self.param_mean) / self.param_sigmas
        x = _hidden_forward(
            x, self.W, self.b, self.alphas, self.betas, self._use_scan,
            (self._W_hidden, self._b_hidden, self._alphas_hidden, self._betas_hidden),
        )

        # linear output layer
        x = ((x @ self.W[-1]) + self.b[-1]) * self.pc_sigmas + self.pc_mean

        return x


class PijEmulator(object):
    """Emulator for P_ij basis spectra, composed of individual Emulator instances.

    Wraps multiple single-spectrum Emulators, one per P_ij component,
    with a shared sigma8(z) emulator.
    """

    def __init__(
        self, config, abspath=False, data_dir=None, scale_As=True, s8_tvar=True
    ):
        """Initialize the P_ij emulator.

        Args:
            config: Path to a YAML config file or a config dict.
            abspath: If True, treat paths as absolute.
            data_dir: Directory containing weight files.
            scale_As: If True, scale As by 1e9 for normalization.
            s8_tvar: If True, use sigma8(z) as a transformed variable.
        """
        cfg, data_dir = _resolve_emu_config(
            config, abspath=abspath, data_dir=data_dir
        )

        pij_emu_bases = cfg["pij_bases"]
        s8z_base = cfg["s8z_base"]
        kmin = float(cfg["kmin"])
        kmax = float(cfg["kmax"])
        self.n_spec = len(pij_emu_bases)
        self.s8_tvar = s8_tvar

        self.input_param_order = cfg.get("param_order_spec", None)
        self.param_order_d = cfg.get("param_order_d", None)
        self.param_ranges = _range_dict(cfg.get("param_ranges", None))
        # input order used for range alignment only; input_param_order stays
        # None for configs without param_order_spec (callers branch on it)
        self.effective_param_order = (
            self.input_param_order or self.param_order_d or _DEFAULT_PARAM_ORDER
        )
        self.param_lo, self.param_hi = _build_range_arrays(
            self.param_ranges, self.effective_param_order
        )

        self.pij_emus = []

        for i in range(self.n_spec):
            if not abspath:
                self.pij_emus.append(
                    Emulator(
                        f"{data_dir}/{pij_emu_bases[i]}",
                        kmin=kmin,
                        kmax=kmax,
                        scale_As=scale_As,
                        param_ranges=self.param_ranges,
                        param_order=self.effective_param_order,
                    )
                )
            else:
                self.pij_emus.append(
                    Emulator(pij_emu_bases[i], kmin=kmin, kmax=kmax, scale_As=scale_As,
                             param_ranges=self.param_ranges,
                             param_order=self.effective_param_order)
                )

        if not abspath:
            self.sigma8z_emu = ScalarEmulator(
                s8z_base, scale_As=scale_As, data_dir=data_dir,
                param_ranges=self.param_ranges,
            )
        else:
            self.sigma8z_emu = ScalarEmulator(
                s8z_base, scale_As=scale_As, param_ranges=self.param_ranges
            )

        # Pre-stack weights across all n_spec emulators for scan-based predict.
        self._stacked_weights = self._stack_emulator_weights()

    def _stack_emulator_weights(self):
        """Stack weights from all individual emulators into arrays for scan."""
        emus = self.pij_emus
        nc = emus[0].n_components
        return (
            jnp.stack([e.param_mean for e in emus]),
            jnp.stack([e.param_sigmas for e in emus]),
            jnp.stack([e.W[0] for e in emus]),
            jnp.stack([e.W[-1] for e in emus]),
            jnp.stack([jnp.stack(e.W[1:-1]) for e in emus]),
            jnp.stack([e.b[-1] for e in emus]),
            jnp.stack([jnp.stack(e.b[:-1]) for e in emus]),
            jnp.stack([jnp.stack(e.alphas) for e in emus]),
            jnp.stack([jnp.stack(e.betas) for e in emus]),
            jnp.stack([e.pc_sigmas[:nc] for e in emus]),
            jnp.stack([e.pc_mean[:nc] for e in emus]),
            jnp.stack([e.v[:, :nc] for e in emus]),
            jnp.stack([e.sigmas for e in emus]),
            jnp.stack([e.mean for e in emus]),
            jnp.stack([e.fstd for e in emus]),
        )

    def predict(self, parameters):
        """Predict all P_ij components for the given parameters.

        Args:
            parameters: Array of shape (n_samples, n_parameters).

        Returns:
            Array of shape (n_samples, n_spec, nk) with predicted P_ij spectra.
        """
        parameters = _clip_params(parameters, self.param_lo, self.param_hi)
        if self.s8_tvar:
            s8z = self.sigma8z_emu.predict(parameters)[:, 0]
            parameters = parameters.at[:, -1].set(s8z)

        def scan_fn(carry, xs_i):
            _, pred = predict_scan(parameters, xs_i)
            return carry, pred

        _, pij = jax.lax.scan(scan_fn, None, self._stacked_weights)
        # pij shape: (n_spec, n_samples, nk)
        return pij.transpose(1, 0, 2)


def predict_scan(parameters, xs):
    """
    Neural network prediction function for scanning over parameters.

    Args:
        parameters: Input parameter array
        xs: Tuple containing all neural network weights, biases, and normalization constants

    Returns:
        Tuple of (parameters, predictions) where predictions are the NN outputs
    """
    (
        param_mean,
        param_sigmas,
        W_0,
        W_m1,
        W,
        b_m1,
        b,
        alphas,
        betas,
        pc_sigmas,
        pc_mean,
        v,
        sigmas,
        mean,
        fstd,
    ) = xs

    x = (parameters - param_mean) / param_sigmas
    x = x @ W_0 + b[0]
    x = activation(x, alphas[0], betas[0])

    x, _ = jax.lax.scan(_mlp_step, x, (W, b[1:], alphas[1:], betas[1:]))

    # linear output layer
    x = ((x @ W_m1) + b_m1) * pc_sigmas + pc_mean
    x = jnp.sinh((x @ v.T) * sigmas + mean) * fstd

    return parameters, x
