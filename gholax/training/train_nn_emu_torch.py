import sys
import numpy as np
import yaml
import h5py
import json
from yaml import Loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0
        
    def __call__(self, val_loss, model, epoch=0):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
                print(f"Restored best weights from epoch {self.best_epoch}")
            return True
        return False
        
    def save_checkpoint(self, model):
        if self.restore_best_weights:
            # Deep copy to avoid reference issues
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

class TunableActivation(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, learnable=True):
        super(TunableActivation, self).__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))
            self.register_buffer('beta', torch.tensor(beta))
    
    def forward(self, x):
        # More stable formulation
        alpha_clamped = torch.clamp(self.alpha, -5, 5)
        beta_clamped = torch.clamp(self.beta, 0.1, 0.9)
        
        # Swish-like activation with tunable parameters
        sigmoid_term = torch.sigmoid(alpha_clamped * x)
        return x * (beta_clamped + sigmoid_term * (1.0 - beta_clamped))

class Emulator(nn.Module):
    def __init__(self, n_params, pc_sigmas, pc_mean, v, 
                 n_hidden=[100, 100, 100], n_components=10,
                 mean=None, sigmas=None, fstd=None,
                 param_mean=None, param_sigmas=None, device=None):
        super(Emulator, self).__init__()
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}", flush=True)
        
        # Convert and register buffers for non-trainable tensors
        self.register_buffer('pc_sigmas', self._to_tensor(pc_sigmas))
        self.register_buffer('pc_mean', self._to_tensor(pc_mean))
        self.register_buffer('v', self._to_tensor(v))
        self.n_components = n_components
        
        # Store normalization parameters if provided
        if param_mean is not None:
            self.register_buffer('param_mean', self._to_tensor(param_mean))
        if param_sigmas is not None:
            self.register_buffer('param_sigmas', self._to_tensor(param_sigmas))
        if mean is not None:
            self.register_buffer('mean', self._to_tensor(mean))
        if sigmas is not None:
            self.register_buffer('sigmas', self._to_tensor(sigmas))            
        if fstd is not None:
            self.register_buffer('fstd', self._to_tensor(fstd))            
        
        layers = []
        
        # Input projection
        layers.append(nn.Linear(n_params, n_hidden[0]))
        layers.append(TunableActivation())
        
        # Hidden layers
        for i in range(len(n_hidden) - 1):
            layers.append(nn.Linear(n_hidden[i], n_hidden[i+1]))
            layers.append(TunableActivation())
        
        # Output projection
        layers.append(nn.Linear(n_hidden[-1], n_components))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
    
    def _to_tensor(self, data):
        """Convert data to tensor and move to device"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def _initialize_weights(self):
        """Improved initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for better gradient flow
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    # Small random bias to break symmetry
                    nn.init.normal_(module.bias, mean=0, std=0.01)
                    
            elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
            elif isinstance(module, TunableActivation):
                # Initialize with slight randomness
                if hasattr(module.alpha, 'data'):
                    nn.init.normal_(module.alpha, mean=1.0, std=0.05)
                if hasattr(module.beta, 'data'):
                    nn.init.uniform_(module.beta, 0.4, 0.6)
    
    def forward(self, x):
        # Forward through network
        y = self.network(x)

        y = y * self.pc_sigmas[:self.n_components] + self.pc_mean[:self.n_components]
        y = torch.matmul(y, self.v[:, :self.n_components].T)
        
        return y
    
    def train_with_adaptive_batching(self, X, y, Xtest, ytest, patience=400, min_delta=1e-5,
                                    phases=None, use_gradient_accumulation=True,
                                    accumulation_steps=4):
        
        if phases is None:
            phases = [
                {'lr': 5e-3, 'epochs': 500, 'initial_bs': 320, 'warmup': 50},
                {'lr': 1e-3, 'epochs': 500, 'initial_bs': 640, 'warmup': 0},
                {'lr': 5e-4, 'epochs': 500, 'initial_bs': 1280, 'warmup': 0},
                {'lr': 1e-4, 'epochs': 500, 'initial_bs': 2560, 'warmup': 0}
            ]
        
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        Xtest = Xtest.to(self.device)
        ytest = ytest.to(self.device)
        
        # Validate inputs
        if torch.isnan(X).any() or torch.isnan(y).any():
            raise ValueError("Input data contains NaN values")
        
        print(f"Training on device: {self.device}", flush=True)
        print(f"Data shapes - X: {X.shape}, y: {y.shape}", flush=True)
        
        n_samples = X.shape[0]
        best_val_loss = float('inf')
        
        # Use Huber loss for robustness
        loss_fn = nn.HuberLoss(delta=1.0)
        
        for phase_idx, phase in enumerate(phases):
            print(f"\nPhase {phase_idx + 1}/{len(phases)}", flush=True)
            print(f"LR: {phase['lr']}, Initial Batch Size: {phase['initial_bs']}", flush=True)
            
            # AdamW with weight decay for better generalization
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=phase['lr'],
                weight_decay=1e-5,
                betas=(0.9, 0.999)
            )
            
            # Cosine annealing with warm restarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=50,  # Restart every 50 epochs
                T_mult=2,  # Double the period after each restart
                eta_min=phase['lr'] * 0.01
            )
            
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
            
            for epoch in range(phase['epochs']):
                self.train()
                
                # Learning rate warmup
                if epoch < phase.get('warmup', 0):
                    warmup_lr = phase['lr'] * (epoch + 1) / phase['warmup']
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                
                current_lr = optimizer.param_groups[0]['lr']
                
                # Dynamic batch size
                lr_reduction_factor = max(1, phase['lr'] / current_lr)
                current_bs = min(
                    int(phase['initial_bs'] * np.sqrt(lr_reduction_factor)),
                    n_samples // 2
                )
                
                # Shuffle data
                perm = torch.randperm(n_samples)
                X_shuffled = X[perm]
                y_shuffled = y[perm]
                
                # Training loop with gradient accumulation
                epoch_loss = 0
                n_batches = (n_samples + current_bs - 1) // current_bs
                
                optimizer.zero_grad()
                for i in range(n_batches):
                    start_idx = i * current_bs
                    end_idx = min((i + 1) * current_bs, n_samples)
                    
                    output = self.forward(X_shuffled[start_idx:end_idx])
                    loss = loss_fn(output, y_shuffled[start_idx:end_idx])
                    
                    # Scale loss for gradient accumulation
                    if use_gradient_accumulation:
                        loss = loss / accumulation_steps
                    
                    loss.backward()
                    
                    # Update weights after accumulation steps
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == n_batches:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item() * (accumulation_steps if use_gradient_accumulation else 1)
                
                # Validation
                self.eval()
                with torch.no_grad():
                    val_pred = self.forward(Xtest)
                    val_loss = F.mse_loss(val_pred, ytest).item()
                    val_mae = F.l1_loss(val_pred, ytest).item()
                
                # Update learning rate
                scheduler.step()
                
                # Logging
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}, BS: {current_bs}, "
                          f"LR: {current_lr:.2e}, Train Loss: {epoch_loss/n_batches:.6f}, "
                          f"Val MSE: {val_loss:.6f}, Val MAE: {val_mae:.6f}", flush=True)
                
                # Early stopping
                if early_stopping(val_loss, self, epoch):
                    print(f"Early stopping triggered at epoch {epoch+1} in phase {phase_idx+1}", flush=True)
                    break
                
                # Track best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
        
        print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}", flush=True)
        return self
    
    def save(self, filepath):
        """
        Save model weights and configuration to a JSON file.
        
        Args:
            filepath (str): Path to save the JSON file
        """
        # Create a dictionary to store all model information
        model_dict = {
            'W': [],
            'b': [],
            'alphas': [],
            'betas': [],
        }
        
        # Save PC parameters
        model_dict['pc_sigmas'] = self.pc_sigmas.cpu().numpy().tolist()
        model_dict['pc_mean'] = self.pc_mean.cpu().numpy().tolist()
        model_dict['v'] = self.v.cpu().numpy().tolist()
        model_dict['param_sigmas'] = self.param_sigmas.cpu().numpy().tolist() if hasattr(self, 'param_sigmas') else None
        model_dict['param_mean'] = self.param_mean.cpu().numpy().tolist() if hasattr(self, 'param_mean') else None
        model_dict['sigmas'] = self.sigmas.cpu().numpy().tolist() if hasattr(self, 'sigmas') else None
        model_dict['mean'] = self.mean.cpu().numpy().tolist() if hasattr(self, 'mean') else None
        model_dict['fstd'] = self.fstd.cpu().numpy().tolist() if hasattr(self, 'fstd') else None
        
        # Save all model weights
        state_dict = self.state_dict()
        for name, param in state_dict.items():
            if 'weight' in name:
                model_dict['W'].append(param.cpu().numpy().T.tolist())
            elif 'bias' in name:
                model_dict['b'].append(param.cpu().numpy().tolist())
            elif 'alpha' in name:
                param = param.cpu().numpy().tolist()
                if param < -5:
                    param = -5
                elif param > 5:
                    param = 5
                model_dict['alphas'].append(param)
            elif 'beta' in name:
                param = param.cpu().numpy().tolist()
                if param < 0.1:
                    param = 0.1
                elif param > 0.9:
                    param = 0.9 
                model_dict['betas'].append(param)
        
        # Save to JSON file
        with open(filepath+'.json', 'w') as f:
            json.dump(model_dict, f)
        
        print(f"Model saved to {filepath}")
        
        # Print summary
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters saved: {total_params:,}")
        print(f"File size: {os.path.getsize(filepath+'.json') / 1024 / 1024:.2f} MB")    

if __name__ == '__main__':

    info_txt = sys.argv[1]
    with open(info_txt, 'rb') as fp:
        emu_info = yaml.load(fp, Loader=Loader)

    training_data_filename = emu_info['training_filename']
    output_path = emu_info['output_path']
    learning_rate = emu_info.pop('learning_rate', [5e-3, 1e-3, 5e-4, 1e-4])
    batch_size = emu_info.pop('batch_size', [320, 640, 1280, 2560])
    n_epochs = emu_info['n_epochs']
    
    phases = [
        {'lr': learning_rate[0], 'epochs':n_epochs, 'initial_bs': int(batch_size[0]), 'warmup': 50},
        {'lr': learning_rate[1], 'epochs':n_epochs, 'initial_bs': int(batch_size[1]), 'warmup': 0},
        {'lr': learning_rate[2], 'epochs':n_epochs, 'initial_bs': int(batch_size[2]), 'warmup': 0},
        {'lr': learning_rate[3], 'epochs':n_epochs, 'initial_bs': int(batch_size[3]), 'warmup': 0}
    ]    

    emu_target = emu_info['target']
    emu_params = emu_info['param_dataset']
    training_data = h5py.File(training_data_filename, 'r')

    n_hidden = emu_info['n_hidden']
    n_pcs = emu_info['n_pcs']
    use_asinh = emu_info['use_asinh']
    scale_by_std = emu_info['scale_by_std']
    downsample = int(emu_info.pop('downsample', 1))

    # data should already be cleaned
    Ptrain = training_data[emu_params][::downsample]
    Ftrain = training_data[emu_target][::downsample]
    idx = np.isfinite(Ftrain).all(axis=1)
    Ftrain = Ftrain[idx]
    Ptrain = Ptrain[idx]

    if use_asinh:
        if scale_by_std:
            print('use asinh, scaled', flush=True)
            Fstd = np.std(Ftrain, axis=0)
            Ftrain = np.arcsinh(Ftrain/Fstd)
        else:
            Ftrain = np.arcsinh(Ftrain)
            Fstd = np.ones(Ftrain.shape[-1])

    Pmean = np.mean(Ptrain,axis=0)
    Psigmas = np.std(Ptrain,axis=0)
    Ptrain = (Ptrain - Pmean)/Psigmas

    validation_frac = 0.2
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

    emu = Emulator(Ptrain.shape[-1], torch.Tensor(pc_sigmas), torch.Tensor(pc_mean), torch.Tensor(v), n_components=n_pcs,
                   n_hidden=n_hidden, mean=torch.Tensor(mean), sigmas=torch.Tensor(sigmas),
                   param_mean=torch.Tensor(Pmean), param_sigmas=torch.Tensor(Psigmas), fstd=torch.Tensor(Fstd))
    emu.train_with_adaptive_batching(torch.Tensor(Ptrain), torch.Tensor(Ftrain), torch.Tensor(Pval), torch.Tensor(Fval), phases=phases)    
    emu.save(output_path)
