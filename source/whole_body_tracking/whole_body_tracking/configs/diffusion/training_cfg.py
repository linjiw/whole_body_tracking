"""Training configuration for BeyondMimic Stage 2 diffusion model.

This module defines the configuration for training hyperparameters,
optimization settings, and logging.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any


@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings."""
    
    type: Literal["adam", "adamw", "sgd"] = "adamw"
    """Optimizer type."""
    
    learning_rate: float = 1e-4
    """Base learning rate."""
    
    weight_decay: float = 0.01
    """Weight decay for regularization."""
    
    betas: tuple = (0.9, 0.999)
    """Adam beta parameters."""
    
    eps: float = 1e-8
    """Adam epsilon for numerical stability."""
    
    gradient_clip: float = 1.0
    """Maximum gradient norm for clipping."""
    
    gradient_accumulation_steps: int = 1
    """Number of steps to accumulate gradients."""


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    
    type: Literal["cosine", "linear", "constant", "warmup_cosine"] = "warmup_cosine"
    """Scheduler type."""
    
    warmup_steps: int = 1000
    """Number of warmup steps."""
    
    warmup_start_lr: float = 1e-7
    """Starting learning rate for warmup."""
    
    min_lr: float = 1e-6
    """Minimum learning rate for cosine schedule."""
    
    num_cycles: float = 0.5
    """Number of cosine cycles."""
    
    power: float = 1.0
    """Power for polynomial decay."""


@dataclass
class DataConfig:
    """Configuration for data loading and augmentation."""
    
    batch_size: int = 256
    """Training batch size."""
    
    val_batch_size: int = 256
    """Validation batch size."""
    
    num_workers: int = 4
    """Number of data loading workers."""
    
    pin_memory: bool = True
    """Whether to pin memory for faster GPU transfer."""
    
    shuffle: bool = True
    """Whether to shuffle training data."""
    
    drop_last: bool = True
    """Whether to drop last incomplete batch."""
    
    # Data augmentation
    augment_noise: bool = True
    """Whether to add noise augmentation during training."""
    
    noise_scale: float = 0.01
    """Scale of noise augmentation."""
    
    action_delay_prob: float = 0.5
    """Probability of applying action delay."""
    
    action_delay_range: tuple = (0, 100)
    """Range of action delay in milliseconds."""


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    use_wandb: bool = True
    """Whether to use Weights & Biases for logging."""
    
    wandb_project: str = "beyondmimic-stage2"
    """WandB project name."""
    
    wandb_entity: Optional[str] = None
    """WandB entity (username or team)."""
    
    wandb_tags: Optional[list] = None
    """Tags for WandB run."""
    
    log_interval: int = 100
    """Steps between logging metrics."""
    
    log_gradient_norm: bool = True
    """Whether to log gradient norms."""
    
    log_learning_rate: bool = True
    """Whether to log learning rate."""
    
    log_samples: bool = True
    """Whether to log sample trajectories."""
    
    sample_interval: int = 5000
    """Steps between sampling for visualization."""
    
    num_samples: int = 4
    """Number of samples to generate for logging."""


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing."""
    
    checkpoint_dir: str = "./checkpoints"
    """Directory for saving checkpoints."""
    
    save_interval: int = 5000
    """Steps between checkpoint saves."""
    
    save_best: bool = True
    """Whether to save best model based on validation loss."""
    
    save_last: bool = True
    """Whether to always save the last checkpoint."""
    
    keep_last_n: int = 5
    """Number of recent checkpoints to keep."""
    
    resume_from: Optional[str] = None
    """Path to checkpoint to resume from."""


@dataclass
class ValidationConfig:
    """Configuration for validation settings."""
    
    val_interval: int = 1000
    """Steps between validation runs."""
    
    val_split: float = 0.1
    """Fraction of data to use for validation."""
    
    early_stopping: bool = False
    """Whether to use early stopping."""
    
    early_stopping_patience: int = 10
    """Number of validation steps without improvement before stopping."""
    
    early_stopping_min_delta: float = 1e-4
    """Minimum change to consider as improvement."""


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Core components
    optimizer: OptimizerConfig = None
    """Optimizer configuration."""
    
    scheduler: SchedulerConfig = None
    """Learning rate scheduler configuration."""
    
    data: DataConfig = None
    """Data loading configuration."""
    
    logging: LoggingConfig = None
    """Logging configuration."""
    
    checkpoint: CheckpointConfig = None
    """Checkpoint configuration."""
    
    validation: ValidationConfig = None
    """Validation configuration."""
    
    # Training settings
    num_epochs: int = 1000
    """Number of training epochs."""
    
    max_steps: Optional[int] = None
    """Maximum number of training steps (overrides num_epochs if set)."""
    
    seed: int = 42
    """Random seed for reproducibility."""
    
    device: str = "cuda"
    """Device to use for training."""
    
    mixed_precision: bool = False
    """Whether to use mixed precision training."""
    
    compile_model: bool = False
    """Whether to compile model with torch.compile (PyTorch 2.0+)."""
    
    # Loss weights
    state_loss_weight: float = 1.0
    """Weight for state prediction loss."""
    
    action_loss_weight: float = 1.0
    """Weight for action prediction loss."""
    
    def __post_init__(self):
        """Initialize defaults if not provided."""
        if self.optimizer is None:
            self.optimizer = OptimizerConfig()
        if self.scheduler is None:
            self.scheduler = SchedulerConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.checkpoint is None:
            self.checkpoint = CheckpointConfig()
        if self.validation is None:
            self.validation = ValidationConfig()
    
    def get_total_steps(self, dataset_size: int) -> int:
        """Calculate total training steps.
        
        Args:
            dataset_size: Size of training dataset
            
        Returns:
            Total number of training steps
        """
        steps_per_epoch = dataset_size // self.data.batch_size
        if self.max_steps is not None:
            return self.max_steps
        return steps_per_epoch * self.num_epochs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging."""
        def _to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: _to_dict(v) for k, v in obj.__dict__.items()}
            return obj
        
        return _to_dict(self)


# Pre-defined configurations
def get_debug_config() -> TrainingConfig:
    """Get configuration for debugging (small, fast)."""
    config = TrainingConfig()
    config.data.batch_size = 32
    config.optimizer.learning_rate = 1e-3
    config.num_epochs = 10
    config.validation.val_interval = 100
    config.logging.log_interval = 10
    config.checkpoint.save_interval = 500
    return config


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_long_training_config() -> TrainingConfig:
    """Get configuration for long training runs."""
    config = TrainingConfig()
    config.num_epochs = 2000
    config.optimizer.learning_rate = 5e-5
    config.scheduler.warmup_steps = 5000
    config.validation.early_stopping = True
    config.validation.early_stopping_patience = 20
    return config