"""
Logging utilities for CausalShapGNN
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json


class Logger:
    """
    Logger class for CausalShapGNN training and evaluation.
    """
    
    def __init__(self, 
                 name: str = 'causalshapgnn',
                 log_dir: str = './logs',
                 level: int = logging.INFO,
                 console: bool = True,
                 file: bool = True):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            console: Whether to log to console
            file: Whether to log to file
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.log_file = log_file
        else:
            self.log_file = None
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration dictionary"""
        self.info("Configuration:")
        self.info("-" * 40)
        for key, value in config.items():
            if isinstance(value, dict):
                self.info(f"  {key}:")
                for k, v in value.items():
                    self.info(f"    {k}: {v}")
            else:
                self.info(f"  {key}: {value}")
        self.info("-" * 40)
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics dictionary"""
        if prefix:
            self.info(f"{prefix}:")
        for key, value in metrics.items():
            self.info(f"  {key}: {value:.4f}")
    
    def log_epoch(self, epoch: int, train_loss: Dict[str, float],
                  val_metrics: Optional[Dict[str, float]] = None):
        """Log epoch summary"""
        self.info(f"Epoch {epoch}")
        self.info(f"  Train Loss: {train_loss.get('total', 0):.4f}")
        
        for key, value in train_loss.items():
            if key != 'total':
                self.info(f"    {key}: {value:.4f}")
        
        if val_metrics:
            self.info("  Validation Metrics:")
            for key, value in val_metrics.items():
                self.info(f"    {key}: {value:.4f}")


class TensorBoardLogger:
    """
    TensorBoard logger for training visualization.
    """
    
    def __init__(self, log_dir: str = './runs'):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = log_dir
        self.writer = None
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(os.path.join(log_dir, timestamp))
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars"""
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram"""
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_embedding(self, tag: str, embeddings, metadata=None, step: int = 0):
        """Log embeddings for visualization"""
        if self.enabled and self.writer:
            self.writer.add_embedding(embeddings, metadata=metadata, 
                                      global_step=step, tag=tag)
    
    def log_figure(self, tag: str, figure, step: int):
        """Log matplotlib figure"""
        if self.enabled and self.writer:
            self.writer.add_figure(tag, figure, step)
    
    def close(self):
        """Close writer"""
        if self.enabled and self.writer:
            self.writer.close()


class ExperimentLogger:
    """
    Complete experiment logger combining file and TensorBoard logging.
    """
    
    def __init__(self, 
                 experiment_name: str,
                 log_dir: str = './experiments',
                 use_tensorboard: bool = True):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Base directory for logs
            use_tensorboard: Whether to use TensorBoard
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.logger = Logger(
            name=experiment_name,
            log_dir=self.experiment_dir,
            console=True,
            file=True
        )
        
        if use_tensorboard:
            self.tb_logger = TensorBoardLogger(
                os.path.join(self.experiment_dir, 'tensorboard')
            )
        else:
            self.tb_logger = None
        
        self.metrics_history = []
    
    def log_config(self, config: Dict[str, Any]):
        """Log and save configuration"""
        self.logger.log_config(config)
        
        # Save config to JSON
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_epoch(self, epoch: int, train_loss: Dict[str, float],
                  val_metrics: Optional[Dict[str, float]] = None):
        """Log epoch with both file and TensorBoard"""
        self.logger.log_epoch(epoch, train_loss, val_metrics)
        
        if self.tb_logger:
            for key, value in train_loss.items():
                self.tb_logger.log_scalar(f'train/{key}', value, epoch)
            
            if val_metrics:
                for key, value in val_metrics.items():
                    self.tb_logger.log_scalar(f'val/{key}', value, epoch)
        
        # Store metrics
        entry = {'epoch': epoch, **train_loss}
        if val_metrics:
            entry.update({f'val_{k}': v for k, v in val_metrics.items()})
        self.metrics_history.append(entry)
    
    def log_test_results(self, metrics: Dict[str, float]):
        """Log final test results"""
        self.logger.info("=" * 50)
        self.logger.info("FINAL TEST RESULTS")
        self.logger.info("=" * 50)
        self.logger.log_metrics(metrics)
        
        # Save to file
        results_path = os.path.join(self.experiment_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_metrics_history(self):
        """Save metrics history to CSV"""
        import csv
        
        if not self.metrics_history:
            return
        
        csv_path = os.path.join(self.experiment_dir, 'metrics_history.csv')
        
        keys = self.metrics_history[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.metrics_history)
    
    def close(self):
        """Close all loggers"""
        self.save_metrics_history()
        if self.tb_logger:
            self.tb_logger.close()


def setup_logger(log_dir: str = './logs', name: str = 'causalshapgnn') -> Logger:
    """
    Setup and return a logger instance.
    
    Args:
        log_dir: Directory for log files
        name: Logger name
        
    Returns:
        Logger instance
    """
    return Logger(name=name, log_dir=log_dir)