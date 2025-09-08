from ...util.likelihood_module import LikelihoodModule
from ...data_vector.two_point_spectrum import field_types
from jax.scipy.integrate import trapezoid
from jax.scipy.interpolate import RegularGridInterpolator
from interpax import interp1d
import jax.numpy as jnp
import numpy as np
import jax


class ProjectionKernels(LikelihoodModule):
    def __init__(
        self,
        observed_data_vector,
        zmin=0.0,
        zmax=2.0,
        nz=125,
        shifted_nz=[],
        **config,
    ):
        self.observed_data_vector = observed_data_vector
        self.shifted_nz = shifted_nz
        self.z = jnp.linspace(zmin, zmax, nz)

        if self.observed_data_vector.zeff_weighting:
            self.required_projection_kernels = {
                "c_dd": ["w_d", "w_mag"],
                "c_dk": ["w_d_dk", "w_k", "w_mag_dk", "w_ia"],
                "c_kk": ["w_k", "w_ia"],
                "c_bb": ["w_ia"],
                "c_dcmbk": ["w_d_dcmbk", "w_cmbk", "w_mag_dcmbk"],
                "c_cmbkcmbk": ["w_cmbk"],
            }
        else:
            self.required_projection_kernels = {
                "c_dd": ["w_d", "w_mag"],
                "c_dk": ["w_d", "w_k", "w_mag", "w_ia"],
                "c_kk": ["w_k", "w_ia"],
                "c_bb": ["w_ia"],
                "c_dcmbk": ["w_d", "w_cmbk", "w_mag"],
                "c_cmbkcmbk": ["w_cmbk"],
            }

        self.kernel_nz = {
            "w_d": "nz_d",
            "w_k": "nz_s",
            "w_mag": "nz_d",
            "w_ia": "nz_s",
            "w_d_dk": "nz_d_dk",
            "w_mag_dk": "nz_d_dk",
            "w_d_dcmbk": "nz_d_cmbk",
            "w_mag_dcmbk": "nz_d_cmbk",
        }

        if ("c_dcmbk" in self.observed_data_vector.spectrum_info) | (
            "c_cmbkcmbk" in self.observed_data_vector.spectrum_info
        ):
            self.required_inputs.append("chi_star")

        all_kernels = []
        self.mag_spec = []

        for spec_type in self.observed_data_vector.spectrum_types:
            if spec_type in self.required_projection_kernels.keys():
                for i in range(2):
                    if "d" in field_types[spec_type][i]:
                        self.n_dbins = self.observed_data_vector.nz_d.shape[0]
                    elif "gamma" in field_types[spec_type][i]:
                        self.n_sbins = self.observed_data_vector.nz_s.shape[0]

                kernels = self.required_projection_kernels[spec_type]
                #                k_temp = []
                for k in kernels:
                    if "mag" in k:
                        self.mag_spec.append(spec_type)
                    all_kernels.append(k)

        self.all_kernels = list(np.unique(all_kernels))
        self.indexed_params = {}
        self.output_requirements = {}
        for k in self.all_kernels:
            self.output_requirements[k] = ["e_z", "chi_z", "omegam"]
            self.indexed_params[k] = []
            if self.kernel_nz[k] in self.shifted_nz:
                self.output_requirements[k].append(f"{self.kernel_nz[k]}_shifted")
            if "mag" in k:
                self.output_requirements[k].extend(
                    [
                        f"smag_{i}"
                        for i in self.observed_data_vector.spectrum_info[
                            self.mag_spec[0]
                        ]["bins0"]
                    ]
                )
                for i in range(self.observed_data_vector.nz_d.shape[0]):
                    if k == "w_mag_dk":
                        for j in range(self.observed_data_vector.nz_s.shape[0]):
                            if (
                                i
                                in self.observed_data_vector.spectrum_info[
                                    self.mag_spec[0]
                                ]["bins0"]
                            ):
                                self.indexed_params[k].append(f"smag_{i}")
                            else:
                                self.indexed_params[k].append("NA")  # returns zero
                    else:
                        if (
                            i
                            in self.observed_data_vector.spectrum_info[
                                self.mag_spec[0]
                            ]["bins0"]
                        ):
                            self.indexed_params[k].append(f"smag_{i}")
                        else:
                            self.indexed_params[k].append("NA")  # returns zero
            self.indexed_params[k] = np.array(self.indexed_params[k])[:, None]

    def compute_w_d(self, e_z, chi_z, omega_m, nz, smags, z, chi_star):
        w_d = jnp.zeros_like(nz)

        f = lambda carry, nz_i: (carry, nz_i * e_z)
        _, w_d = jax.lax.scan(f, None, nz)

        w_d = w_d / trapezoid(w_d, x=chi_z, axis=-1)[:, None]
        zeff_d = trapezoid(z[None, :] * w_d**2 / chi_z[None, :] ** 2, x=chi_z)
        zeff_d = zeff_d / trapezoid(w_d**2 / chi_z[None, :] ** 2, x=chi_z)

        return w_d, zeff_d, jnp.zeros_like(zeff_d)

    def compute_w_d_dk(self, e_z, chi_z, omega_m, nz, smags, z, chi_star):
        return self.compute_w_d(e_z, chi_z, omega_m, nz, smags, z, chi_star)

    def compute_w_d_dcmbk(self, e_z, chi_z, omega_m, nz, smags, z, chi_star):
        return self.compute_w_d(e_z, chi_z, omega_m, nz, smags, z, chi_star)

    def compute_w_ia(self, e_z, chi_z, omega_m, nz, smags, z, chi_star):
        rho_c = 2.7754e11
        c1_bar = 5e-14
        om_fid = 0.31
        w_ia, zeff_ia, ichi_inv = self.compute_w_d(
            e_z, chi_z, omega_m, nz, smags, z, chi_star
        )

        return -om_fid * rho_c * c1_bar * w_ia, zeff_ia, ichi_inv

    def compute_w_k(self, e_z, chi_z, omega_m, nz, smags, z, chi_star):
        w_k = jnp.zeros_like(nz)
        cmax = jnp.max(chi_z) * 1.1  # what is the point of the * 1.1

        def zupper(carry, x):
            return carry, jnp.linspace(x, cmax, chi_z.shape[0])

        _, chivalp = jax.lax.scan(zupper, None, chi_z)
        chivalp = chivalp.T
        zvalp = interp1d(
            chivalp.reshape(-1), chi_z, z, extrap=True, method="linear"
        ).reshape(chivalp.shape)

        def f(carry, nz_i):
            dndz_n = jnp.interp(zvalp, z, nz_i, left=0, right=0)
            Ez = interp1d(
                zvalp.reshape(-1), z, e_z, extrap=True, method="linear"
            ).reshape(zvalp.shape)
            g = (chivalp - chi_z[None, :]) / chivalp
            g = g * dndz_n * Ez / 2997.925
            g = chi_z * trapezoid(g, x=chivalp, axis=0)
            w_k = 1.5 * omega_m / 2997.925**2 * (1 + z) * g
            return carry, w_k

        _, w_k = jax.lax.scan(f, None, nz)

        ichi_eff = trapezoid(nz / chi_z[None, :], x=z, axis=1)

        return w_k, jnp.ones_like(ichi_eff), ichi_eff

    def compute_c_cmbk(self, e_z, chi_z, omega_m, nz, smags, z, chi_star):
        w_cmbk = 1.5 * omega_m * (1.0 / 2997.925) ** 2 * (1 + z)
        w_cmbk *= chi_z * (chi_star - chi_z) / chi_star
        ichi_eff = 1 / chi_star

        return w_cmbk, jnp.ones_like(ichi_eff), ichi_eff

    def compute_w_mag(self, e_z, chi_z, omega_m, nz, smags, z, chi_star):
        w_mag, zeff, ichi_eff = self.compute_w_k(
            e_z, chi_z, omega_m, nz, smags, z, chi_star
        )

        def f(ii, x):
            w_mag_i, smag = x
            w_mag_i = w_mag_i * (5 * smag - 2)
            ii += 1
            return ii, w_mag_i

        _, w_mag = jax.lax.scan(f, 0, [w_mag, smags])
        return w_mag, zeff, ichi_eff

    def compute_w_mag_dk(self, e_z, chi_z, omega_m, nz, smags, z, chi_star):
        return self.compute_w_mag(e_z, chi_z, omega_m, nz, smags, z, chi_star)

    def compute_w_mag_dcmbk(self, e_z, chi_z, omega_m, nz, smags, z, chi_star):
        return self.compute_w_mag(e_z, chi_z, omega_m, nz, smags, z, chi_star)

    def compute(self, state, params_values):
        param_vec = jnp.array(list(params_values.values()))
        e_z = state["e_z_limber"]
        chi_z_proj = state["chi_z_limber"]
        z_limber = state["z_limber"]

        interpolated_nzs = {}

        for k in self.all_kernels:
            nz_name = self.kernel_nz[k]
            if nz_name in interpolated_nzs:
                continue
            if nz_name in self.shifted_nz:
                nz = state[f"{nz_name}_shifted"]
            else:
                nz = getattr(self.observed_data_vector, nz_name)

            def f(carry, nz_i):
                nz_i = interp1d(z_limber, self.z, nz_i, extrap=True)
                nz_i = jnp.where(nz_i > 0, nz_i, 0)
                nz_i = nz_i / trapezoid(nz_i, x=z_limber)

                return carry, nz_i

            _, nz = jax.lax.scan(f, None, nz)
            interpolated_nzs[nz_name] = nz

        for k in self.all_kernels:
            nz_name = self.kernel_nz[k]
            nz = interpolated_nzs[nz_name]

            # get chistar if using cmb lensing
            if k == "w_cmbk":
                chi_star = state["chi_star"]
            else:
                chi_star = None

            kernel = getattr(self, f"compute_{k}")

            # setup magnification coefficient array
            if "mag" in k:
                smag = param_vec[self.param_indices[k]]
            else:
                smag = jnp.zeros(0)

            w, z_eff, ichi_eff = kernel(
                e_z, chi_z_proj, state["omegam"], nz, smag, z_limber, chi_star
            )
            if k in ["w_d_dk", "w_mag_dk"]:
                w = w.reshape((self.n_dbins, self.n_sbins, -1))

            state[k] = w

            state[f"zeff_{k}"] = z_eff
            state[f"chi_inv_eff_{k}"] = ichi_eff

        return state
