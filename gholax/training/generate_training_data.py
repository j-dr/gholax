import sys
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import h5py
import numpy as np
import yaml
from mpi4py import MPI
from yaml import Loader
from time import time
from ..util.model import Model
from .design import Design

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
    n_checkpoint=2,
    sample_offset=0,
    rank_offset=0,
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
    end = time()    
    start = time()
    for n in range(npars_slow):
        
        if n % nproc == rank:
            if rank == 0:
                print(n, flush=True)
                if n > 0:
                    print("Time for last {} samples: {:.2f} seconds".format(nfast_per_slow, end - start), flush=True)            
            
            start = time()
        

            # Sample the slow parameters for this iteration
            pslow = params.sample(n + sample_offset)
            
            # For each slow parameter set, generate multiple fast parameter combinations
            for m in range(nfast_per_slow):
                if not allslow:
                    # fast-slow sampling: combine slow and fast parameters
                    fvec = params_fast.sample((n + sample_offset) * nfast_per_slow + m)
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
                        except Exception as e:
                            print(e, flush=True)
                            print(
                                f"params for failed eval: {pars}",
                                flush=True,
                            )
                            continue
            
                count += 1  
            end = time()                

            if (n + 1) % n_checkpoint == 0:
                with h5py.File(f"{emu_info['output_filename']}.{rank + rank_offset}", "w") as fp:
                    for k in out:
                        fp.create_dataset(k, data=out[k][:count])
                    fp.attrs['nend'] = nend
                    fp.attrs['nfast_per_slow'] = nfast_per_slow
                    fp.attrs['sample_offset'] = sample_offset
                if rank == 0:
                    print(f"Checkpoint at slow step {n+1}", flush=True)

    # Save this rank's results to a separate HDF5 file
    rank_filename = "{}.{}".format(emu_info["output_filename"], rank + rank_offset)
    with h5py.File(rank_filename, "w") as fp:
        for k in out:
            fp.create_dataset(k, data=out[k][:count])
        fp.attrs['nend'] = nend
        fp.attrs['nfast_per_slow'] = nfast_per_slow
        fp.attrs['sample_offset'] = sample_offset

    # Wait for all ranks to finish writing their individual files
    comm.Barrier()

    # Rank 0 creates the master file with external links to all rank files
    if rank == 0:
        from .reformat import _link_rank_files
        _link_rank_files(emu_info["output_filename"])


def generate_training_data():
    """CLI entry point for MPI-parallel emulator training data generation.

    Reads a YAML config file from sys.argv[1], sets up the model and
    parameter sampling design, then calls generate_models to evaluate
    the model at each design point in parallel across MPI ranks.

    An optional second argument specifies additional samples to generate,
    appending to existing training data:
        generate-training-data config.yaml 5000
    """

    info_txt = sys.argv[1]
    n_additional = int(sys.argv[2]) if len(sys.argv) > 2 else None

    with open(info_txt, "rb") as fp:
        info = yaml.load(fp, Loader=Loader)

    # Extract emulation configuration and initialize model
    emu_info = info["emulate"]
    model = Model(info_txt)

    # Set up parameter configuration for training
    param_names = model.setup_training_config(emu_info["likelihood"])
    if rank == 0:
        print("Sampling over parameters:", param_names, flush=True)

    bounds = np.array(model.prior.get_minimizer_bounds())

    # Extract sampling configuration with defaults
    nstart = emu_info.get("nstart", 0)
    original_nend = emu_info.get("nend", 100)
    param_names_fast = emu_info.get("param_names_fast", None)
    nfast = emu_info.get("nfast_per_slow", 1)
    n_checkpoint = emu_info.get("n_checkpoint", 2)
    design_scheme = emu_info.get("design_scheme", "qrs")
    seed = emu_info.get("seed", 0)

    # Set random seed for reproducible parameter sampling
    np.random.seed(seed)

    # Compute restart offsets
    sample_offset = 0
    rank_offset = 0
    if n_additional is not None:
        import glob as globmod
        output_filename = emu_info["output_filename"]
        existing_rank_files = sorted(
            [f for f in globmod.glob(f"{output_filename}.*")
             if f.rsplit('.', 1)[-1].isdigit()],
            key=lambda f: int(f.rsplit('.', 1)[-1])
        )
        rank_offset = len(existing_rank_files)

        # Compute sample_offset from metadata in latest rank file
        original_npars = original_nend - nstart
        original_npars_slow = (original_npars + nfast - 1) // nfast
        sample_offset = original_npars_slow  # fallback from config

        if existing_rank_files and rank == 0:
            latest_rank_file = existing_rank_files[-1]
            try:
                with h5py.File(latest_rank_file, 'r') as fp:
                    if 'sample_offset' in fp.attrs:
                        prev_offset = int(fp.attrs['sample_offset'])
                        prev_nend = int(fp.attrs['nend'])
                        prev_nfast = int(fp.attrs.get('nfast_per_slow', 1))
                        prev_npars_slow = (prev_nend + prev_nfast - 1) // prev_nfast
                        sample_offset = prev_offset + prev_npars_slow
            except Exception as e:
                if rank == 0:
                    print(f"Warning: could not read metadata from {latest_rank_file}: {e}",
                          flush=True)

        sample_offset = comm.bcast(sample_offset, root=0)

        nend = n_additional
        ntot = sample_offset * nfast + n_additional

        if rank == 0:
            print(f"Restart mode: generating {n_additional} additional samples", flush=True)
            print(f"  sample_offset={sample_offset}, rank_offset={rank_offset}", flush=True)
    else:
        nend = original_nend
        ntot = nend - nstart

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
            bounds_slow[:, 0],
            bounds_slow[:, 1],
            scheme=design_scheme,
            ntot=ntot,
        )
        params_fast_design = Design(
            bounds_fast[:, 0],
            bounds_fast[:, 1],
            scheme=design_scheme,
            ntot=ntot,
        )
    else:
        # Single-speed sampling: all parameters treated equally
        bounds = bounds[[model.prior.params.index(p) for p in param_names]]

        params = Design(
            bounds[:, 0], bounds[:, 1], scheme=design_scheme, ntot=ntot
        )
        params_fast_design = None

    # Generate the training data using MPI parallelization
    generate_models(
        params,
        param_names,
        model,
        emu_info,
        nstart=nstart,
        nend=nend,
        params_fast=params_fast_design,
        param_names_fast=param_names_fast,
        nfast_per_slow=nfast,
        n_checkpoint=n_checkpoint,
        sample_offset=sample_offset,
        rank_offset=rank_offset,
    )
