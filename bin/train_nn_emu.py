import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml
import h5py
import json
from yaml import Loader

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class Emulator(tf.keras.Model):
    """
    Neural network emulator for cosmological power spectra using TensorFlow/Keras.
    
    This class implements a multi-layer perceptron with custom activation functions
    for emulating cosmological power spectra. It uses principal component analysis
    for dimensionality reduction and includes normalization layers.
    
    Attributes:
        n_parameters (int): Number of input parameters
        n_hidden (list): List of hidden layer sizes  
        n_components (int): Number of principal components to use
        nks (int): Number of k-values in output spectra
        architecture (list): Complete network architecture [input, hidden..., output]
        n_layers (int): Total number of layers
        W (list): Weight matrices for each layer
        b (list): Bias vectors for each layer
        alphas (list): Alpha parameters for custom activation function
        betas (list): Beta parameters for custom activation function
        pc_sigmas (array): Principal component standard deviations
        pc_mean (array): Principal component means
        param_mean (array): Input parameter means for normalization
        param_sigmas (array): Input parameter standard deviations for normalization
        v (array): Principal component basis vectors
        mean (array): Output data means
        sigmas (array): Output data standard deviations
        fstd (array): Function standard deviations for asinh scaling
    """

    def __init__(self, n_params, nks, pc_sigmas, pc_mean, v,
                 n_hidden=[100, 100, 100], n_components=10,
                 mean=None, sigmas=None, fstd=None,
                 param_mean=None, param_sigmas=None):
        """
        Initialize the neural network emulator.
        
        Args:
            n_params (int): Number of input parameters
            nks (int): Number of k-values in output spectra
            pc_sigmas (array): Principal component standard deviations
            pc_mean (array): Principal component means
            v (array): Principal component basis vectors
            n_hidden (list, optional): Hidden layer sizes. Defaults to [100, 100, 100].
            n_components (int, optional): Number of principal components. Defaults to 10.
            mean (array, optional): Output data means for denormalization
            sigmas (array, optional): Output data standard deviations for denormalization
            fstd (array, optional): Function standard deviations for asinh scaling
            param_mean (array, optional): Input parameter means for normalization
            param_sigmas (array, optional): Input parameter standard deviations for normalization
        """
        super(Emulator, self).__init__()

        trainable = True

        self.n_parameters = n_params
        self.n_hidden = n_hidden
        self.n_components = n_components
        self.nks = nks

        self.architecture = [self.n_parameters] + \
            self.n_hidden + [self.n_components]
        self.n_layers = len(self.architecture) - 1

        self.W = []
        self.b = []
        self.alphas = []
        self.betas = []
        self.pc_sigmas = pc_sigmas
        self.pc_mean = pc_mean
        self.param_mean = param_mean
        self.param_sigmas = param_sigmas
        self.v = v
        self.mean = mean
        self.sigmas = sigmas
        self.fstd = fstd

        for i in range(self.n_layers):
            self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., np.sqrt(
                2./self.n_parameters)), name="W_" + str(i), trainable=trainable))
            self.b.append(tf.Variable(
                tf.zeros([self.architecture[i+1]]), name="b_" + str(i), trainable=trainable))
        for i in range(self.n_layers-1):
            self.alphas.append(tf.Variable(tf.random.normal(
                [self.architecture[i+1]]), name="alphas_" + str(i), trainable=trainable))
            self.betas.append(tf.Variable(tf.random.normal(
                [self.architecture[i+1]]), name="betas_" + str(i), trainable=trainable))

    def activation(self, x, alpha, beta):
        """
        Custom swish-like activation function with learnable parameters.
        
        Args:
            x (tensor): Input tensor
            alpha (tensor): Scaling parameter for sigmoid component
            beta (tensor): Offset parameter for activation shape
            
        Returns:
            tensor: Activated output tensor
        """
        return tf.multiply(tf.add(beta, tf.multiply(tf.sigmoid(tf.multiply(alpha, x)), tf.subtract(1.0, beta))), x)

    @tf.function
    def call(self, parameters):
        """
        Forward pass of the neural network emulator.
        
        Args:
            parameters (tensor): Input parameter tensor of shape (batch_size, n_params)
            
        Returns:
            tensor: Predicted power spectra of shape (batch_size, nks)
        """
        outputs = []
        x = parameters

        for i in range(self.n_layers - 1):

            # linear network operation
            x = tf.add(tf.matmul(x, self.W[i]), self.b[i])

            # non-linear activation function
            x = self.activation(x, self.alphas[i], self.betas[i])

        # linear output layer
        x = tf.add(tf.multiply(tf.add(tf.matmul(
            x, self.W[-1]), self.b[-1]), self.pc_sigmas[:self.n_components]), self.pc_mean[:self.n_components])
        x = tf.matmul(x, self.v[:, :self.n_components].T)

        return x
    
    def save(self, filebase):
        """
        Save the trained emulator model to JSON file.
        
        Args:
            filebase (str): Base filename for saving the model (without extension)
        """
        W = [self.W.weights[i].numpy().tolist() for i in range(len(self.W.weights))]
        b = [self.b.weights[i].numpy().tolist() for i in range(len(self.b.weights))]
        alpha = [self.alphas.weights[i].numpy().tolist() for i in range(len(self.alphas.weights))]
        beta = [self.betas.weights[i].numpy().tolist() for i in range(len(self.betas.weights))]
        pc_sigmas = self.pc_sigmas[:self.n_components].tolist()
        pc_mean = self.pc_mean[:self.n_components].tolist()
        v = self.v[:,:self.n_components].tolist()
        sigmas = self.sigmas.tolist()
        mean = self.mean.tolist()
        param_sigmas = self.param_sigmas.tolist()
        param_mean = self.param_mean.tolist()
        fstd = self.fstd.tolist()

        out = {'W':W, 'b':b,
               'alphas':alpha,
               'betas':beta,
               'pc_sigmas':pc_sigmas, 
               'pc_mean':pc_mean,
               'v':v,
               'sigmas':sigmas,
               'mean':mean,
               'param_sigmas':param_sigmas,
               'param_mean':param_mean,
               'fstd':fstd}

        with open('{}.json'.format(filebase), 'w') as fp:
            json.dump(out, fp)


    def load(self, filebase):
        """
        Load a trained emulator model from JSON file.
        
        Args:
            filebase (str): Base filename for loading the model (without extension)
        """
        with open('{}.json'.format(filebase), 'r') as fp:
            weights = json.load(fp)
            
            for k in weights:
                if k in ['W', 'b', 'alphas', 'betas']:
                    for i, wi in enumerate(weights[k]):
                        weights[k][i] = np.array(wi).astype(np.float32)
#                         tf.Variable(np.array(wi).astype(np.float32), name="W_" + str(i), trainable=True))
                else:
                    weights[k] = np.array(weights[k]).astype(np.float32)

                setattr(self,k, weights[k])
                
def train_emu(Ptrain, Ftrain, validation_frac=0.2,
              n_hidden=[100, 100, 100], n_pcs=20,
              n_epochs=1000, fstd=None, pmean=None,
              pstd=None, outfile=None, lrs=None,
              nbatchs=None, restart_file=None):
    """
    Train a neural network emulator for cosmological spectra.
    
    Args:
        Ptrain: Training parameter array
        Ftrain: Training target function array
        validation_frac: Fraction of data to use for validation (default: 0.2)
        n_hidden: List of hidden layer sizes (default: [100, 100, 100])
        n_pcs: Number of principal components to use (default: 20)
        n_epochs: Number of training epochs (default: 1000)
        fstd: Optional target function standard deviation
        pmean: Optional parameter mean
        pstd: Optional parameter standard deviation
        outfile: Optional output file path for saving model
        lrs: Optional learning rate schedule
        nbatchs: Optional number of batches
        restart_file: Optional restart file path
    """

    iis = np.random.rand(len(Ptrain)) > validation_frac

    Pval = Ptrain[~iis, :]
    Fval = Ftrain[~iis, :]

    Ptrain = Ptrain[iis, :]
    Ftrain = Ftrain[iis, :]

    # Construct Principle Components
    mean = np.mean(Ftrain, axis=0)
    mean = np.array(mean, dtype='float32')
    sigmas = np.std(Ftrain, axis=0)
    sigmas = np.array(sigmas, dtype='float32')
    Ftrain = (Ftrain - mean) / sigmas
    Fval = (Fval - mean) / sigmas

    cov_matrix = np.cov(Ftrain.T)
    w, v = np.linalg.eigh(cov_matrix)
    # flip to rank in ascending eigenvalue
    w = np.flip(w)
    v = np.flip(v, axis=1)
    v = np.array(v, dtype='float32')
    pc_train = np.dot(Ftrain, v)

    pc_mean = np.mean(pc_train, axis=0)
    pc_sigmas = np.std(pc_train, axis=0)

    # Now start the emulator and run it
    emulator = Emulator(n_params=Ptrain.shape[-1], nks=Ftrain.shape[-1],
                        pc_sigmas=pc_sigmas, pc_mean=pc_mean, v=v,
                        sigmas=sigmas, mean=mean, fstd=fstd,
                        param_mean=pmean, param_sigmas=pstd,
                        n_components=n_pcs, n_hidden=n_hidden)
    if restart_file is not None:
        emulator.load(restart_file)

    emulator.compile(optimizer='adam', loss='mse', metrics=['mse'])

    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

    for lr, nbatch in zip(lrs, nbatchs):
        print("Using learning rate, batch size:  %.2e, %d." % (lr, nbatch))

        emulator.optimizer.lr = lr
        emulator.fit(Ptrain, Ftrain, epochs=n_epochs, batch_size=nbatch,
                     validation_data=(Pval, Fval), callbacks=[es], verbose=2)

        if outfile is not None:
            emulator.save(outfile)

    return emulator


if __name__ == '__main__':

    info_txt = sys.argv[1]
    with open(info_txt, 'rb') as fp:
        emu_info = yaml.load(fp, Loader=Loader)

    training_data_filename = emu_info['training_filename']
    output_path = emu_info['output_path']
    restart = emu_info.pop('restart', False)
    learning_rate = emu_info.pop('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    learning_rate = [float(l) for l in learning_rate]
    nbatchs = emu_info.pop('nbatchs', [320, 640, 1280, 2560, 5120])
    nbatchs = [int(n) for n in nbatchs]

    emu_target = emu_info['target']
    emu_params = emu_info['param_dataset']
    training_data = h5py.File(training_data_filename, 'r')

    n_hidden = emu_info['n_hidden']
    n_pcs = emu_info['n_pcs']
    n_epochs = emu_info['n_epochs']
    use_asinh = emu_info['use_asinh']
    scale_by_std = emu_info['scale_by_std']
    training_set_offset = emu_info.pop('training_set_offset', 0)
    downsample = int(emu_info.pop('downsample', 1))

    # data should already be cleaned

    Ptrain = training_data[emu_params][:]
    Ftrain = training_data[emu_target][:]
    idx = np.isfinite(Ftrain).all(axis=1)
    Ftrain = Ftrain[idx]
    Ptrain = Ptrain[idx]
    del idx

#    idx = (np.abs(Ftrain)<1e7).all(axis=1)
    Ptrain = Ptrain[training_set_offset::downsample]
    Ftrain = Ftrain[training_set_offset::downsample]

    if use_asinh:
        if scale_by_std:
            print('use asinh, scaled')
            sys.stdout.flush()            
            Fstd = np.std(Ftrain, axis=0)
            Ftrain = np.arcsinh(Ftrain/Fstd)
        else:
            Ftrain = np.arcsinh(Ftrain)
            Fstd = np.ones(Ftrain.shape[-1])

    Pmean = np.mean(Ptrain,axis=0)
    Psigmas = np.std(Ptrain,axis=0)
    Ptrain = (Ptrain - Pmean)/Psigmas

    if restart:
        restart_file = output_path
        output_path = output_path + '_restarted'
    else:
        restart_file = None
    
    emu = train_emu(Ptrain, Ftrain, validation_frac=0.2,
                    n_hidden=n_hidden, n_pcs=n_pcs,
                    n_epochs=n_epochs, fstd=Fstd,
                    pmean=Pmean, pstd=Psigmas,
                    outfile=output_path,
                    lrs=learning_rate, nbatchs=nbatchs,
                    restart_file=restart_file)

    emu.save(output_path)
