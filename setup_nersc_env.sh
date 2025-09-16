#!/bin/bash
module load python
mamba create gholax --clone base
source activate gholax
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install --upgrade "jax[cuda12]==0.4.38"
mamba install numpy scipy pyyaml setuptools ipython jupyter cython gsl matplotlib h5py
pip install interpax optax
pip install git+https://github.com/blackjax-devs/blackjax.git
pip install git+https://github.com/sfschen/spinosaurus.git 
pip install git+https://github.com/sfschen/velocileptors.git
pip install git+https://github.com/AemulusProject/aemulus_heft.git
python -m pip cache purge
pip install classy
pip install getdist --upgrade-strategy only-if-needed
cp -r /pscratch/sd/j/jderose/emu_weights gholax/theory/emu_weights
pip install .

python3 -m ipykernel install --user --name gholax --display-name gholax
