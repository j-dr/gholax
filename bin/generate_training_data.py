import sys

import h5py
import numpy as np
import yaml
from mpi4py import MPI
from yaml import Loader

from gholax.util.model import Model
from gholax.training.design import Design


comm = MPI.COMM_WORLD 
rank = comm.Get_rank() 
nproc = comm.Get_size()


def generate_models(
    params,
    param_names,
    model,
    emu_info,
    nstart=0,
    nend=10,
    params_fast=None,
    param_names_fast=None,
    nfast_per_slow=1,
):
    """
    Generate training data for emulator by running model predictions in parallel.
    
    Args:
        params: Parameter sampling design object
        param_names: List of parameter names
        model: Physics model object for generating predictions
        emu_info: Dictionary containing emulation configuration
        nstart: Starting index for parameter sampling (default: 0)
        nend: Ending index for parameter sampling (default: 10) 
        params_fast: Optional fast parameter sampling design
        param_names_fast: Optional list of fast parameter names
        nfast_per_slow: Number of fast samples per slow sample (default: 1)
    """
    # Determine if we're using fast-slow parameter sampling
    if params_fast is None:
        allslow = True  # Only slow parameters
        if nfast_per_slow != 1:
            print("Setting nfast_per_slow=1")
            nfast_per_slow = 1
    else:
        allslow = False  # Using both slow and fast parameters
        assert nfast_per_slow > 1  # Must have multiple fast samples per slow sample

    # Calculate total number of parameter combinations to evaluate
    npars = nend - nstart

    # Combine slow and fast parameter names if using fast-slow sampling
    if param_names_fast is not None:
        param_names_all = param_names + param_names_fast
    else:
        param_names_all = param_names

    npars_slow = (npars + nfast_per_slow - 1) // nfast_per_slow  
    npars_this = ((npars_slow + nproc - 1) // nproc) * nfast_per_slow 

    # Initialize output dictionary to store results
    out = {"params": np.zeros((npars_this, len(param_names_all)))}
    count = 0  
    
    # Main loop: distribute slow parameter sets across MPI ranks
    for n in range(npars_slow):
    
        if n % nproc == rank:
    
            if rank == 0:
                print(n, flush=True)

            # Sample the slow parameters for this iteration
            pslow = params.sample(n)
            
            # For each slow parameter set, generate multiple fast parameter combinations
            for m in range(nfast_per_slow):
                if not allslow:
                    # fast-slow sampling: combine slow and fast parameters
                    fvec = params_fast.sample(n * nfast_per_slow + m)
                    pvec = np.concatenate([pslow, fvec])
                else:
                    # Single-speed sampling: only slow parameters
                    pvec = pslow

                # Convert parameter vector to dictionary and store
                pars = dict(zip(param_names_all, pvec))
                out["params"][count] = pvec

                
                if "likelihood" in emu_info:
                    # Loop over all likelihood modules specified in configuration
                    for like in emu_info["likelihood"]:
                        try:
                            # Run the model to generate predictions for this parameter set
                            state = model.generate_training_data(like, pars)

                            # Check if multiple observables are requested for this likelihood
                            if hasattr(emu_info["likelihood"][like], "__iter__"):
                                # Multiple observables: extract each one
                                for l in emu_info["likelihood"][like]:
                                    try:
                                        pred = state[l]
                                    except Exception as e:
                                        print(e, flush=True)
                                        print(
                                            f"params for failed eval: {pars}",
                                            flush=True,
                                        )
                                        continue

                                    # Create unique name and initialize storage if needed
                                    name = "{}.{}".format(like, l)
                                    if name not in out:
                                        tsize = [npars_this]
                                        [tsize.append(d) for d in pred.shape]
                                        out[name] = np.zeros(tsize)
                                    out[name][count] = pred

                            else:
                                # Single observable: extract it directly
                                attr = emu_info["likelihood"][like]
                                pred = state[attr]
                                name = "{}.{}".format(like, attr)
                                if name not in out:
                                    tsize = [npars_this]
                                    [tsize.append(d) for d in pred.shape]
                                    out[name] = np.zeros(tsize)
                                out[name][count] = pred
                        except Exception:
                            # Skip this parameter set if model evaluation fails
                            continue

                count += 1  

    # Save this rank's results to a separate HDF5 file
    for k in out:
        with h5py.File("{}.{}".format(emu_info["output_filename"], rank), "w") as fp:
            for k in out:
                shape = out[k].shape
                fp.create_dataset(k, shape)
                fp[k][:] = out[k]

    # Wait for all ranks to finish writing their individual files
    comm.Barrier()
    
    # Rank 0 creates the master file with external links to all rank files
    if rank == 0:
        with h5py.File(emu_info["output_filename"], "w") as fp:
            for k in out:
                # Create external links to each rank's data
                for n in range(nproc):
                    fp["{}_{}".format(k, n)] = h5py.ExternalLink(
                        "{}.{}".format(emu_info["output_filename"], n), k
                    )


if __name__ == "__main__":

    info_txt = sys.argv[1]
    with open(info_txt, "rb") as fp:
        info = yaml.load(fp, Loader=Loader)

    # Extract emulation configuration and initialize model
    emu_info = info["emulate"]
    model = Model(info_txt)
    
    # Set up parameter configuration for training
    param_names = model.setup_training_config(emu_info["likelihood"])
    bounds = np.array(model.prior.get_minimizer_bounds())
    
    # Extract sampling configuration with defaults
    nstart = emu_info.pop("nstart", 0)  # Starting parameter index
    nend = emu_info.pop("nend", 100)    # Ending parameter index
    param_names_fast = emu_info.pop("param_names_fast", None)  # Fast parameter names
    nfast = emu_info.pop("nfast_per_slow", 1)  # Fast samples per slow sample
    design_scheme = emu_info.pop("design_scheme", "qrs")  # Sampling scheme (qrs, sobol, lhs)
    seed = emu_info.pop("seed", 0)  # Random seed for reproducibility

    # Set random seed for reproducible parameter sampling
    np.random.seed(seed)

    # Set up parameter sampling designs based on fast-slow configuration
    if param_names_fast is not None:
        # fast-slow sampling: separate slow and fast parameters
        param_names = [p for p in param_names if p not in param_names_fast]
        fast_idx = [model.prior.params.index(f) for f in param_names_fast]
        slow_idx = [model.prior.params.index(f) for f in param_names]

        # Extract parameter bounds for slow and fast parameters
        bounds_fast = bounds[fast_idx]
        bounds_slow = bounds[slow_idx]

        # Create separate sampling designs for slow and fast parameters
        params = Design(
            bounds_slow[:, 0],    # Lower bounds for slow parameters
            bounds_slow[:, 1],    # Upper bounds for slow parameters
            scheme=design_scheme,
            ntot=(nend - nstart),
        )
        params_fast = Design(
            bounds_fast[:, 0],    # Lower bounds for fast parameters
            bounds_fast[:, 1],    # Upper bounds for fast parameters
            scheme=design_scheme,
            ntot=(nend - nstart),
        )
    else:
        # Single-speed sampling: all parameters treated equally
        bounds = bounds[[model.prior.params.index(p) for p in param_names]]
        
        params = Design(
            bounds[:, 0], bounds[:, 1], scheme=design_scheme, ntot=(nend - nstart)
        )
        params_fast = None

    # Generate the training data using MPI parallelization
    generate_models(
        params,
        param_names,
        model,
        emu_info,
        nstart=nstart,
        nend=nend,
        params_fast=params_fast,
        param_names_fast=param_names_fast,
        nfast_per_slow=nfast,
    )
