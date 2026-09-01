from ...util.likelihood_module import LikelihoodModule
from ...data_vector.two_point_spectrum import field_types
from jax.lax import scan
from copy import copy
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)


class AngularPowerSpectrum_to_CorrelationFunction(LikelihoodModule):
    def __init__(
        self,
        observed_data_vector,
        spectrum_types,
        spectrum_info,
        n_ell=200,
        l_max=3001,
        **config,
    ):
        self.observed_data_vector = observed_data_vector
        self.spectrum_types = spectrum_types
        self.spectrum_info = spectrum_info
        self.n_ell = n_ell
        self.l_max = l_max
        self.ell = jnp.logspace(jnp.log10(2), jnp.log10(self.l_max), self.n_ell)
        self.cl_tag = config.get("cl_tag", "_mbias")
        self.all_spectra = {}
        self.output_requirements = {}
        for t in self.observed_data_vector.spectrum_types:
            self.all_spectra[t] = []
            for ii, i in enumerate(spectrum_info[t]["bins0"]):
                if spectrum_info[t]["use_cross"]:
                    if field_types[t][0] == field_types[t][1]:
                        bins1 = spectrum_info[t]["bins1"][ii:]
                    else:
                        bins1 = spectrum_info[t]["bins1"][:]
                    for j in bins1:
                        self.all_spectra[t].append((i, j))
                        # Add output requirements:
                        if t == "w_theta":
                            if i == j:
                                self.output_requirements[f"w_{i}_{j}_obs"] = [f"c_dd_{i}_{j}{self.cl_tag}"]
                        elif t == "gamma_t":
                            self.output_requirements[f"gamma_{i}_{j}_obs"] = [f"c_dk_{i}_{j}{self.cl_tag}"]
                        elif t == "xi_plus":
                            self.output_requirements[f"xi_pos_{i}_{j}_obs"] = [f"c_kk_{i}_{j}{self.cl_tag}"]
                            self.output_requirements[f"xi_neg_{i}_{j}_obs"] = [f"c_kk_{i}_{j}{self.cl_tag}"]
                else:
                    self.all_spectra[t].append((i, i))
                    if t == "w_theta":
                        self.output_requirements[f"w_{i}_{i}_obs"] = [f"c_dd_{i}_{i}{self.cl_tag}"]

    def compute(self, state, params_values):
        # Build theta edges from bin centers stored in the data vector
        # Use first spectrum type's separations (assume same grid for all)
        t0 = self.spectrum_types[0]
        centers = jnp.array(self.observed_data_vector.spectrum_info[t0]["separation"])
        half_gaps = jnp.diff(centers) / 2
        thetas = jnp.concatenate([
            jnp.array([centers[0] - half_gaps[0]]),
            centers[:-1] + half_gaps,
            jnp.array([centers[-1] + half_gaps[-1]])
        ])
        mu = jnp.cos(thetas)
        mu = jnp.clip(mu, -1 + 1e-6, 1 - 1e-6)
        P_ell_theta = legendre_poly(mu, self.l_max+1)
        dP_ell_theta = derive_legendre_poly(P_ell_theta, mu)
        w_kernel = build_w_kernel(P_ell_theta, mu, self.l_max)
        ell_w = jnp.arange(2, self.l_max+1)
        gamma_kernel = build_gamma_kernel(P_ell_theta, mu, self.l_max)
        ell_gamma = jnp.arange(2, self.l_max+1)
        xi_pos_kernel, xi_neg_kernel = build_xi_kernels(P_ell_theta, dP_ell_theta, mu, self.l_max)
        ell_xi = jnp.arange(2, self.l_max+1)

        

        for (i,j) in self.all_spectra.get("w_theta", []):
            if i == j:
                C_dd_ii = state[f"c_dd{self.cl_tag}"][i, :]
                C_dd_interp = jnp.interp(ell_w, self.ell, C_dd_ii)
                w_theta_ii = jnp.sum(w_kernel * C_dd_interp[:, jnp.newaxis], axis=0)
                state[f"w_{i}_{i}_obs"] = w_theta_ii
        for (i,j) in self.all_spectra.get("gamma_t", []):
            n_bins = self.observed_data_vector.spectrum_info["gamma_t"]["n_bins1_tot"]
            flat_idx = i * n_bins + j
            C_dk_ij = state[f"c_dk{self.cl_tag}"][flat_idx, :]
            C_dk_interp = jnp.interp(ell_gamma, self.ell, C_dk_ij)
            gamma_theta_ij = jnp.sum(gamma_kernel * C_dk_interp[:, jnp.newaxis], axis=0)
            state[f"gamma_{i}_{j}_obs"] = gamma_theta_ij
        for (i,j) in self.all_spectra.get("xi_plus", []):
            n_bins = self.observed_data_vector.spectrum_info["xi_plus"]["n_bins1_tot"]
            flat_idx = i * n_bins + j
            C_EE_ij = state[f"c_kk{self.cl_tag}"][flat_idx, :]
            C_BB_ij = state.get(f"c_bb{self.cl_tag}", jnp.zeros_like(state[f"c_kk{self.cl_tag}"]))[flat_idx, :]
            C_EE_interp = jnp.interp(ell_xi, self.ell, C_EE_ij)
            C_BB_interp = jnp.interp(ell_xi, self.ell, C_BB_ij)

            xi_pos_ij = jnp.sum(xi_pos_kernel * (C_EE_interp + C_BB_interp)[:, jnp.newaxis], axis=0)
            xi_neg_ij = jnp.sum(xi_neg_kernel * (C_EE_interp - C_BB_interp)[:, jnp.newaxis], axis=0)
            state[f"xi_pos_{i}_{j}_obs"] = xi_pos_ij
            state[f"xi_neg_{i}_{j}_obs"] = xi_neg_ij
        return state

    def get_model_from_state(self, state):
        cf_keys = {
            "w_theta": "w",
            "gamma_t": "gamma",
            "xi_plus":  "xi_pos",
            "xi_minus": "xi_neg",
        }
        model = []
        for t in self.observed_data_vector.spectrum_types:
            cf_key = cf_keys[t]
            for (i, j) in self.all_spectra[t]:
                if t == "w_theta" and i != j:
                    continue
                model.append(state[f"{cf_key}_{i}_{j}_obs"])
        return jnp.hstack(model)

def build_w_kernel(P_ell_theta, mu, l_max):
        ell = jnp.arange(2, l_max+1)
        prefacing_term = (2*ell+1) / (4*jnp.pi)
        P_bar = legendre_poly_bar(P_ell_theta, mu, ell)
        return (prefacing_term[:, jnp.newaxis] * P_bar)

def build_gamma_kernel(P_ell_theta, mu, l_max):
        ell = jnp.arange(2, l_max+1)
        prefacing_term = (2*ell+1)/(4*jnp.pi*ell*(ell+1))
        P_bar = legendre_order_2_bar(mu, P_ell_theta, l_max)
        return (prefacing_term[:, jnp.newaxis] * P_bar)

def build_xi_kernels(P_ell_theta, dP_ell_theta, mu, l_max):
        ell = jnp.arange(2, l_max+1)
        G_pos_theta, G_neg_theta = G_posneg_bar(mu, P_ell_theta, dP_ell_theta, l_max)
        prefacing_term = (2*ell+1) / (2*jnp.pi*(ell*(ell+1))**2)
        xi_pos = prefacing_term[:, jnp.newaxis] * G_pos_theta
        xi_neg = prefacing_term[:, jnp.newaxis] * G_neg_theta
        return xi_pos, xi_neg



def legendre_poly(abscissa: jnp.array, l_max: int):
    P0 = jnp.ones_like(abscissa)
    P1 = abscissa

    def step(carry, l):
        P_l_prev, P_l = carry
        P_l_next = ((2*l+1) * abscissa*P_l - l*P_l_prev) / (l+1)
        carry = P_l, P_l_next
        return carry, P_l_next

    _, P_ls = scan(step, (P0, P1), jnp.arange(1, l_max))
    return jnp.vstack([P0, P1, P_ls])

def derive_legendre_poly(P_l: jnp.array, mu: jnp.array):
    l_max = P_l.shape[0] - 1
    dP0 = jnp.zeros_like(mu)
    def step(dP_prev, l):
        new_dP = (l+1) * P_l[l] + mu * dP_prev # computes dP_(l+1)
        return new_dP, new_dP
    _, dP_rest = scan(step, dP0, jnp.arange(0, l_max)) 
    return jnp.vstack([dP0, dP_rest]) # aligns indexing with l

def legendre_poly_bar(P_l: jnp.array, mu: jnp.array, ell: jnp.array):
    # ell = jnp.arange(2, P_l.shape[0]-1)
    delta_P = P_l[3:, :] - P_l[1:-2, :] # skip first element as it corresponds to l=0 and only use l>=1
    num = delta_P[:, 1:] - delta_P[:, :-1] 
    denom = (2*ell[:, jnp.newaxis] + 1) * (mu[1:] - mu[:-1])
    return num / denom
    
def legendre_order_2_bar(mu: jnp.array, P_l: jnp.array, l_max: int):
    mu_mins = mu[:-1]
    mu_maxs = mu[1:]
    mu_mins_slice = slice(0, -1)
    mu_maxs_slice = slice(1, None)
    l_sub1_slice = slice(1,-2) # skip first element as it corresponds to l=0 and only use l>=1
    l_slice = slice(2, -1)
    l_add1_slice = slice(3, None)
    l = jnp.arange(2, l_max+1)[:, jnp.newaxis]

    inv_diff = 1 / (mu_mins-mu_maxs)[jnp.newaxis, :]
    term_1 = (l + 2/(2*l+1)) * (P_l[l_sub1_slice, mu_mins_slice] - P_l[l_sub1_slice, mu_maxs_slice])
    term_2 = (2-l) * (mu_mins*P_l[l_slice, mu_mins_slice] - mu_maxs*P_l[l_slice, mu_maxs_slice])
    term_3 = -(2/(2*l+1)) * (P_l[l_add1_slice, mu_mins_slice] - P_l[l_add1_slice, mu_maxs_slice])
    return inv_diff * (term_1 + term_2 + term_3)

def G_posneg_bar(mu: jnp.array, P_l: jnp.array, dP_l: jnp.array, l_max: int):
    l = jnp.arange(2,l_max+1)[:, jnp.newaxis]
    theta_mins_slice = slice(0, -1)
    theta_maxs_slice = slice(1, None)
    l_sub1_slice = slice(1,-2) # skip first element as it corresponds to l=0 and only use l>=1
    l_slice = slice(2, -1)
    l_add1_slice = slice(3, None)
    assert not jnp.any(jnp.abs(mu) >= 1.0)

    P_diff = lambda P, l_slice: P[l_slice, theta_mins_slice] - P[l_slice, theta_maxs_slice]
    P_mu_diff = lambda P, l_slice: (mu[theta_mins_slice] * P[l_slice, theta_mins_slice]
                                  - mu[theta_maxs_slice] * P[l_slice, theta_maxs_slice])

    term_1 = -1*(l*(l-1)/2)*(l+2/(2*l+1))*P_diff(P_l, l_sub1_slice)
    term_2 = -1*(l*(l-1)*(2-l)/2) * P_mu_diff(P_l, l_slice)
    term_3 =    (l*(l-1)/(2*l+1)) * P_diff(P_l, l_add1_slice)
    term_4 =    (4-l)*P_diff(dP_l, l_slice)
    term_5 =    (l+2)*((P_mu_diff(dP_l, l_sub1_slice))-P_diff(P_l, l_sub1_slice))
    term_6 =  2*(l-1)*(P_mu_diff(dP_l, l_slice) - P_diff(P_l, l_slice))
    term_7 =  2*(l+2)*(P_diff(dP_l, l_sub1_slice))
    inv_diff = 1/(mu[theta_mins_slice]-mu[theta_maxs_slice])
    
    G_pos = inv_diff * (term_1 + term_2 + term_3 + term_4 + term_5 + term_6 - term_7)
    G_neg = inv_diff * (term_1 + term_2 + term_3 + term_4 + term_5 - term_6 + term_7)
    return G_pos, G_neg # potentially look at aligning with l indexing