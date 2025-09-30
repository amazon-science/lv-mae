import os
import json
import torch.distributed as dist
from typing import Any, Dict, List
from pprint import pprint

class Logger:
    """Simple logger with tqdm support."""
    
    def __init__(self, log_dir=None):
        """Initialize the logger.
        
        Args:
            log_dir: Directory where logs will be saved
        """
        self.log_dir = log_dir
        self.rank = 0 if not dist.is_available() or not dist.is_initialized() else dist.get_rank()
        
        # Try to import tqdm
        try:
            from tqdm import tqdm
            self.tqdm = tqdm
        except ImportError:
            # Fall back to print if tqdm is not available
            self.tqdm = type('FakeTqdm', (), {'write': print})
        
        # Create log directory if it doesn't exist
        if self.rank == 0 and log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.metrics_file = os.path.join(log_dir, "metrics.jsonl")
            self.hparams_file = os.path.join(log_dir, "hparams.json")
    
    def stop(self):
        """Stop the logger."""
        pass
    
    def log(self, name: str, data: Any, step=None):
        """Log a metric value at a specific step."""
        if self.rank == 0:
            self.tqdm.write(f'{name}: {data}' if step is None else f'step {step}, {name}: {data:.4e}')
            
            # Log to file if log_dir is provided
            if hasattr(self, 'metrics_file'):
                with open(self.metrics_file, "a") as f:
                    f.write(json.dumps({
                        "name": name,
                        "value": float(data) if isinstance(data, (int, float)) else str(data),
                        "step": step
                    }) + "\n")
    
    def log_hparams(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.rank == 0:
            self.tqdm.write('hyperparameters:')
            pprint(params)
            
            # Log to file if log_dir is provided
            if hasattr(self, 'hparams_file'):
                with open(self.hparams_file, "w") as f:
                    json.dump(params, f, indent=2)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if self.rank == 0:
            self.tqdm.write('params:')
            pprint(params)
    
    def add_tags(self, tags: List[str]):
        """Add tags."""
        if self.rank == 0:
            self.tqdm.write('tags:')
            pprint(tags)
    
    def log_name_params(self, name: str, params: Any):
        """Log named parameters."""
        if self.rank == 0:
            self.tqdm.write(f'{name}:')
            pprint(params)


def get_logger(log_dir=None):
    """Get a logger instance.
    
    Args:
        log_dir: Directory where logs will be saved
        
    Returns:
        Logger instance
    """
    return Logger(log_dir)