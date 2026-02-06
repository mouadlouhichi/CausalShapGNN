"""
Early Stopping utilities for CausalShapGNN
"""

import numpy as np
import torch
import os
from typing import Optional, Callable


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.
    """
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'max',
                 checkpoint_path: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize EarlyStopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy, 'min' for loss
            checkpoint_path: Path to save best model checkpoint
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
            self.best_score = -np.inf
        else:
            self.is_better = lambda new, best: new < best - min_delta
            self.best_score = np.inf
    
    def __call__(self, score: float, model: torch.nn.Module, 
                 epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to save if improved
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
            if self.checkpoint_path:
                self.save_checkpoint(model)
            
            if self.verbose:
                print(f"  [EarlyStopping] New best score: {score:.4f}")
            
            return False
        else:
            self.counter += 1
            
            if self.verbose:
                print(f"  [EarlyStopping] No improvement. "
                      f"Counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  [EarlyStopping] Stopping training. "
                          f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
            
            return False
    
    def save_checkpoint(self, model: torch.nn.Module):
        """Save model checkpoint"""
        if self.checkpoint_path:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            if self.verbose:
                print(f"  [EarlyStopping] Saved checkpoint to {self.checkpoint_path}")
    
    def load_best_model(self, model: torch.nn.Module):
        """Load best model from checkpoint"""
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            model.load_state_dict(torch.load(self.checkpoint_path))
            if self.verbose:
                print(f"  [EarlyStopping] Loaded best model from {self.checkpoint_path}")
        return model
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.early_stop = False
        if self.mode == 'max':
            self.best_score = -np.inf
        else:
            self.best_score = np.inf


class ModelCheckpoint:
    """
    Save model checkpoints based on monitored metric.
    """
    
    def __init__(self,
                 checkpoint_dir: str = './checkpoints',
                 filename: str = 'model_{epoch:03d}_{score:.4f}.pt',
                 monitor: str = 'val_recall@20',
                 mode: str = 'max',
                 save_best_only: bool = True,
                 save_last: bool = True,
                 max_checkpoints: int = 5):
        """
        Initialize ModelCheckpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            filename: Filename template for checkpoints
            monitor: Metric to monitor
            mode: 'max' or 'min'
            save_best_only: Only save when metric improves
            save_last: Always save the last checkpoint
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.max_checkpoints = max_checkpoints
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_score = -np.inf if mode == 'max' else np.inf
        self.checkpoints = []
    
    def __call__(self, model: torch.nn.Module, metrics: dict, epoch: int):
        """
        Save checkpoint if criteria met.
        
        Args:
            model: Model to save
            metrics: Dictionary of metrics
            epoch: Current epoch
        """
        score = metrics.get(self.monitor, 0)
        
        should_save = False
        is_best = False
        
        if self.mode == 'max' and score > self.best_score:
            self.best_score = score
            should_save = True
            is_best = True
        elif self.mode == 'min' and score < self.best_score:
            self.best_score = score
            should_save = True
            is_best = True
        
        if not self.save_best_only:
            should_save = True
        
        if should_save:
            filepath = os.path.join(
                self.checkpoint_dir,
                self.filename.format(epoch=epoch, score=score)
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'is_best': is_best
            }, filepath)
            
            self.checkpoints.append(filepath)
            
            # Remove old checkpoints
            while len(self.checkpoints) > self.max_checkpoints:
                old_ckpt = self.checkpoints.pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_path)
        
        # Save last model
        if self.save_last:
            last_path = os.path.join(self.checkpoint_dir, 'last_model.pt')
            torch.save(model.state_dict(), last_path)


class LRSchedulerWithWarmup:
    """
    Learning rate scheduler with warmup.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int = 5,
                 total_epochs: int = 100,
                 min_lr: float = 1e-6,
                 warmup_start_lr: float = 1e-7,
                 scheduler_type: str = 'cosine'):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer instance
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            min_lr: Minimum learning rate
            warmup_start_lr: Starting LR for warmup
            scheduler_type: Type of scheduler ('cosine', 'linear', 'step')
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.scheduler_type = scheduler_type
        
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """
        Update learning rate.
        
        Args:
            epoch: Current epoch (optional, uses internal counter if not provided)
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # Post-warmup phase
            progress = (self.current_epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            
            if self.scheduler_type == 'cosine':
                lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                     (1 + np.cos(np.pi * progress))
            elif self.scheduler_type == 'linear':
                lr = self.base_lr - (self.base_lr - self.min_lr) * progress
            else:
                lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']