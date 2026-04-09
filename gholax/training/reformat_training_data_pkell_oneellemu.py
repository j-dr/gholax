import sys, os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import numpy as np
from yaml import Loader
from gholax.util.model import Model
from mpi4py import MPI
import numpy as np
import h5py
import yaml
import sys

def load_training_data(key, dataset, downsample=None, spec_idx=None):
    
    keys = list(dataset.keys())
    keys = [k for k in keys if key in k]
    if downsample is not None:
        keys = keys[::downsample]
    nproc = len(keys)
    print(key)
    print(key in keys)
    for i, k in enumerate(keys):
        rank = k.split('_')[-1]
        d = dataset[k][:]
        p = dataset['params_{}'.format(rank)][:]
        if i==0:
            if spec_idx is not None:
                size_i = [d.shape[0], d.shape[2], d.shape[3]]
            else:
                size_i = d.shape
                
            size = [size_i[0]*nproc]
            psize = [size_i[0]*nproc, p.shape[1]]
            [size.append(size_i[i]) for i in range(1, len(size_i))]
            
            Y = np.zeros(size)
            X = np.zeros(psize)

        X[i * size_i[0]:(i + 1) * size_i[0]] = p

        if spec_idx is not None:
            Y[i * size_i[0]:(i + 1) * size_i[0]] = d[:,spec_idx,...]
        else:
            Y[i * size_i[0]:(i + 1) * size_i[0]] = d
            
    return X, Y


if __name__ == '__main__':

    info_txt = sys.argv[1]
    with open(info_txt, 'r') as fp:
        info = yaml.load(fp, Loader=Loader)
    model = Model(info_txt)

    emu_info = info['emulate']
    param_names_fast = emu_info.pop('param_names_fast', None)
    param_names = model.prior.params
    
    if param_names_fast is not None:
        param_names = [p for p in param_names if p not in param_names_fast]
        param_names = param_names + param_names_fast
        
    emu_info = info['emulate']
    training_data_filename = emu_info['output_filename']
    training_data = h5py.File(training_data_filename, 'r+')

    print(training_data_filename, flush=True)
    training_data = h5py.File(training_data_filename, 'r+')
    z = model.likelihoods['RSDPK'].likelihood_pipeline[5].z
    k =  model.likelihoods['RSDPK'].likelihood_pipeline[5].k
    nspec = 13
    nell = 4
    Ptrain_all, Ftrain_all_spec = load_training_data('RSDPK.p_ij_ell_no_ap_redshift_space_bias_grid', training_data)
    
    rescale_by_s8sq = True
    
    for l in range(nell):
        print(l, flush=True)
        Ftrain_all = Ftrain_all_spec[:,:,l,:,:] #islkz
        if (l==0):
            _, sigma8_z_all = load_training_data('RSDPK.sigma8_z', training_data)
            idx = np.any(Ptrain_all>0, axis=1)
            Ptrain = Ptrain_all[idx]
            sigma8_z = sigma8_z_all[idx]
            del Ptrain_all, sigma8_z_all

            Ptrain_z = np.zeros((len(Ptrain)*len(z),Ptrain.shape[1]+1))
            Ptrain_z[:,-1] = sigma8_z.flatten()    

            for j in range(len(z)):
                Ptrain_z[j::len(z), :-1] = Ptrain

            Ptrain = Ptrain_z

            try:
                training_data.create_dataset('params', Ptrain.shape)
            except:
                del training_data['params']
                training_data.create_dataset('params', Ptrain.shape)

            training_data['params'][:] = Ptrain
            del Ptrain

        Ftrain = Ftrain_all[idx]
        Ftrain = np.einsum('iskz->izsk', Ftrain).reshape(-1,len(k)*nspec) 
        
        if rescale_by_s8sq:
            Ftrain /= Ptrain_z[:,-1,None]**2

        try:
            training_data.create_dataset(f'pkell{l}', Ftrain.shape)
        except:
            del training_data[f'pkell{l}']
            training_data.create_dataset(f'pkell{l}', Ftrain.shape)

        training_data[f'pkell{l}'][:] = Ftrain
        del Ftrain

    del Ftrain_all_spec
        
    training_data.close()    
