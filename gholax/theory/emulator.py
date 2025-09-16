import os

import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np
import yaml


class NNPowerSpectrumInterpolator(object):
    def __init__(self, emu, sigma8z_emu, cosmo_params, nz_max=200, nonu=False):
        self.emu = emu
        self.sigma8z_emu = sigma8z_emu
        self.cosmo_params = jnp.copy(cosmo_params)
        # set zero neutrino mass to 1e-2 eV (this is the boundary of our training data)
        if not nonu:
            if self.cosmo_params[-1] != 0:
                self.cosmo_params = self.cosmo_params.at[-1].set(
                    jnp.log10(self.cosmo_params[-1])
                )
            else:
                self.cosmo_params = self.cosmo_params.at[-1].set(-2)

        self.cparam_grid = jnp.zeros((nz_max, len(self.cosmo_params) + 1))
        self.cparam_grid = self.cparam_grid.at[:, :-1].set(self.cosmo_params)

    def P(self, z, k):
        self.cparam_grid = self.cparam_grid.at[: len(z), -1].set(z)
        sigma8z = self.sigma8z_emu.predict(self.cparam_grid[: len(z)])[:, 0]
        self.cparam_grid = self.cparam_grid.at[: len(z), -1].set(sigma8z)

        k_emu, p = self.emu.predict(self.cparam_grid[: len(z)])
        p_result = jnp.zeros((len(k), len(z)))
        for i in range(len(z)):
            p_result = p_result.at[:, i].set(jnp.interp(k, k_emu, p[:, i]))

        return p_result


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


class Emulator(object):
    def __init__(self, filebase, kmin=1e-3, kmax=0.5, scale_As=True):
        super(Emulator, self).__init__()
        self.scale_As = scale_As
        self.load(filebase)

        self.n_parameters = self.W[0].shape[0]
        self.n_components = self.W[-1].shape[-1]
        self.n_layers = len(self.W)
        self.nk = self.sigmas.shape[0]
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), self.nk)

    def load(self, filebase):
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
    def __init__(
        self,
        config,
        input_param_order=None,
        abspath=False,
        data_dir=None,
        s8_tvar=True,
        scale_As_spec=False,
        scale_As_d=True,
        scale_by_s8zsq=True,
    ):
        super(MultiSpectrumEmulator, self).__init__()

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
                            os.path.dirname(os.path.realpath(__file__)),
                            "emu_weights",
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

            data_dir = ""

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
            )
        else:
            self.sigma8z_emu = ScalarEmulator(
                s8z_base,
                scale_As=self.scale_As_d,
                input_param_order=input_param_order,
                weight_param_order=self.param_order_d,
            )

        self.load_spec(f"{data_dir}/{spec_base}")

        self.n_parameters = self.W[0].shape[0]
        self.n_components = self.W[-1].shape[-1]
        self.n_layers = len(self.W)
        self.nk = self.sigmas.shape[0] // self.n_spec
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), self.nk)

    def load_spec(self, filebase):
        with h5.File("{}.h5".format(filebase), "r") as weights:
            for k in weights:
                if k in ["W", "b", "alphas", "betas"]:
                    w = []
                    for i, wi in enumerate(weights[k]):
                        w.append(weights[k][wi][:].astype(np.float32))
                        if (self.param_order_spec is not None) & (i == 0) & (k == "W"):
                            assert w[i].shape[0] == len(self.param_order_spec)
                            w[i] = w[i][self.param_idx_spec, :]
                        w[i] = jnp.array(w[i]).astype(jnp.float32)

                elif k in ["param_mean", "param_sigmas"]:
                    w = np.array(weights[k][f'{k}_0']).astype(np.float32)
                    if self.scale_As_spec:
                        if self.param_order_spec is not None:
                            w[self.param_order_spec.index("As")] *= 1e9
                        else:
                            w[0] *= 1e9
                    if self.input_param_order is not None:
                        w = w[self.param_idx_spec]

                    w = jnp.array(w).astype(jnp.float32)

                else:
                    w = jnp.array(weights[k][f'{k}_0']).astype(jnp.float32)

                setattr(self, k, w)

    def predict(self, parameters):
        if self.s8_tvar:
            s8z = self.sigma8z_emu.predict(parameters)[:, 0]
            parameters = parameters.at[:, -1].set(s8z)

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
            jnp.sinh((x @ self.v[:, : self.n_components].T * self.sigmas + self.mean))
            * self.fstd
        )

        if self.scale_by_s8zsq:
            if not self.s8_tvar:
                s8z = self.sigma8z_emu.predict(parameters)[:, 0]
            x = x * (s8z[:, None] ** 2)
        return x.reshape(-1, self.n_spec, self.nk)


class ScalarEmulator(object):
    def __init__(
        self,
        filebase,
        scale_As=True,
        data_dir=None,
        input_param_order=None,
        weight_param_order=None,
    ):
        super(ScalarEmulator, self).__init__()
        self.scale_As = scale_As

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

    def load(self, filebase, data_dir=None):
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

        with h5.File("{}.h5".format(emu_abspath), "r") as weights:
            for k in weights:
                if k in ["W", "b", "alphas", "betas"]:
                    w = []
                    for i, wi in enumerate(weights[k]):
                        w.append(weights[k][wi][:].astype(np.float32))
                        if (self.input_param_order is not None) & (i == 0) & (k == "W"):
                            assert w[i].shape[0] == len(self.input_param_order)
                            w[i] = w[i][self.param_idx, :]
                        w[i] = jnp.array(w[i]).astype(jnp.float32)

                elif k in ["param_mean", "param_sigmas"]:
                    w = np.array(weights[k][f'{k}_0']).astype(np.float32)
                    if self.scale_As:
                        if self.weight_param_order is not None:
                            w[self.weight_param_order.index("As")] *= 1e9
                        else:
                            w[0] *= 1e9
                    if self.input_param_order is not None:
                        w = w[self.param_idx]

                    w = jnp.array(w).astype(jnp.float32)

                else:
                    w = jnp.array(weights[k][f'{k}_0']).astype(jnp.float32)

                setattr(self, k, w)

    def predict(self, parameters):
        x = (parameters - self.param_mean) / self.param_sigmas

        for i in range(self.n_layers - 1):
            # linear network operation
            x = x @ self.W[i] + self.b[i]

            # non-linear activation function
            x = activation(x, self.alphas[i], self.betas[i])

        # linear output layer
        x = ((x @ self.W[-1]) + self.b[-1]) * self.pc_sigmas + self.pc_mean

        return x


class PijEmulator(object):
    def __init__(
        self, config, abspath=False, data_dir=None, scale_As=True, s8_tvar=True
    ):
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
                            os.path.dirname(os.path.realpath(__file__)),
                            "emu_weights",
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

        pij_emu_bases = cfg["pij_bases"]
        s8z_base = cfg["s8z_base"]
        kmin = float(cfg["kmin"])
        kmax = float(cfg["kmax"])
        self.n_spec = len(pij_emu_bases)
        self.s8_tvar = s8_tvar

        self.pij_emus = []

        for i in range(self.n_spec):
            if not abspath:
                self.pij_emus.append(
                    Emulator(
                        f"{data_dir}/{pij_emu_bases[i]}",
                        kmin=kmin,
                        kmax=kmax,
                        scale_As=scale_As,
                    )
                )
            else:
                self.pij_emus.append(
                    Emulator(pij_emu_bases[i], kmin=kmin, kmax=kmax, scale_As=scale_As)
                )

        if not abspath:
            self.sigma8z_emu = ScalarEmulator(
                s8z_base, scale_As=scale_As, data_dir=data_dir
            )
        else:
            self.sigma8z_emu = ScalarEmulator(s8z_base, scale_As=scale_As)

    def predict(self, parameters):
        if self.s8_tvar:
            s8z = self.sigma8z_emu.predict(parameters)[:, 0]
            parameters = parameters.at[:, -1].set(s8z)

        npred = len(parameters)

        pij = jnp.zeros((npred, self.n_spec, len(self.pij_emus[0].k)))

        for i in range(self.n_spec):
            pij_temp = self.pij_emus[i].predict(parameters)
            pij = pij.at[:, i, :].set(pij_temp)

        return pij


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

    def MLP(x, wandb):
        W_i, b_i, alpha_i, beta_i = wandb
        x = x @ W_i + b_i
        x = activation(x, alpha_i, beta_i)
        return x, None

    x, _ = jax.lax.scan(MLP, x, (W, b[1:], alphas[1:], betas[1:]))

    # linear output layer
    x = ((x @ W_m1) + b_m1) * pc_sigmas + pc_mean
    x = jnp.sinh((x @ v.T) * sigmas + mean) * fstd

    return parameters, x
