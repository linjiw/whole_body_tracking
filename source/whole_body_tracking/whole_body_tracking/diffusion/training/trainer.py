"""DDPM training loop for state-action diffusion model.

This module implements the training infrastructure for the diffusion model
including optimization, logging, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import os
import json
from tqdm import tqdm
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")

from ..models.diffusion_model import StateActionDiffusionModel
from ..data.trajectory_dataset import TrajectoryDataset


class DDPMTrainer:
    """Trainer for DDPM diffusion model.
    
    Implements the training loop with:
    - EMA (Exponential Moving Average) for stable sampling
    - Gradient clipping for stability
    - WandB logging
    - Checkpoint saving/loading
    """
    
    def __init__(
        self,
        model: StateActionDiffusionModel,
        train_dataset: TrajectoryDataset,
        val_dataset: Optional[TrajectoryDataset] = None,
        learning_rate: float = 1e-4,
        batch_size: int = 256,
        num_epochs: int = 1000,
        warmup_steps: int = 1000,
        gradient_clip: float = 1.0,
        ema_decay: float = 0.9999,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 100,
        val_interval: int = 1000,
        save_interval: int = 5000,
        use_wandb: bool = True,
        wandb_project: str = "beyondmimic-stage2",
        device: str = "cuda",
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4
    ):
        """Initialize trainer.
        
        Args:
            model: Diffusion model to train
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for LR scheduler
            gradient_clip: Gradient clipping value
            ema_decay: EMA decay rate
            checkpoint_dir: Directory for checkpoints
            log_interval: Steps between logging
            val_interval: Steps between validation
            save_interval: Steps between checkpoints
            use_wandb: Whether to use WandB logging
            wandb_project: WandB project name
            device: Device to train on
            early_stopping: Whether to enable early stopping
            early_stopping_patience: Number of validation steps without improvement
            early_stopping_min_delta: Minimum change to consider as improvement
        """
        self.model = model.to(device)
        self.device = device
        
        # Datasets and loaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        else:
            self.val_loader = None
        
        # Optimization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler(warmup_steps, num_epochs * len(self.train_loader))
        
        # EMA model for stable sampling
        self.ema_model = self._create_ema_model()
        self.ema_decay = ema_decay
        
        # Training params
        self.num_epochs = num_epochs
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.use_wandb = use_wandb
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(project=wandb_project, config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "model_layers": model.transformer.transformer.num_layers,
                "hidden_dim": model.hidden_dim,
                "ema_decay": ema_decay
            })
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metric tracking
        from .metrics import MetricTracker
        self.metric_tracker = MetricTracker(window_size=100)
        
        # Early stopping settings
        self.early_stopping_enabled = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
    
    def _create_scheduler(self, warmup_steps: int, total_steps: int):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_ema_model(self) -> StateActionDiffusionModel:
        """Create EMA copy of model."""
        ema_model = StateActionDiffusionModel(
            state_dim=self.model.state_dim,
            action_dim=self.model.action_dim,
            hidden_dim=self.model.hidden_dim,
            num_layers=self.model.transformer.transformer.num_layers,
            history_length=self.model.history_length,
            future_length_states=self.model.future_length_states,
            future_length_actions=self.model.future_length_actions,
            num_timesteps=self.model.num_timesteps
        ).to(self.device)
        
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        
        return ema_model
    
    @torch.no_grad()
    def _update_ema(self):
        """Update EMA model parameters."""
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_losses = []
        epoch_state_losses = []
        epoch_action_losses = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            loss, loss_components = self.model.training_step(batch, return_loss_components=True)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Optimization step
            self.optimizer.step()
            self.scheduler.step()
            
            # Update EMA
            self._update_ema()
            
            # Record losses
            epoch_losses.append(loss.item())
            epoch_state_losses.append(loss_components['state_loss'])
            epoch_action_losses.append(loss_components['action_loss'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
            
            # Logging
            if self.global_step % self.log_interval == 0:
                self._log_metrics({
                    'train/loss': loss.item(),
                    'train/state_loss': loss_components['state_loss'],
                    'train/action_loss': loss_components['action_loss'],
                    'train/lr': self.scheduler.get_last_lr()[0]
                })
            
            # Validation
            if self.val_loader and self.global_step % self.val_interval == 0:
                val_metrics = self.validate()
                self._log_metrics({f'val/{k}': v for k, v in val_metrics.items()})
                
                # Update metric tracker
                self.metric_tracker.update(val_metrics)
                
                # Check for early stopping (optional)
                if self.early_stopping_enabled:
                    if not self.metric_tracker.is_improving(
                        'loss', 
                        patience=self.early_stopping_patience,
                        min_delta=self.early_stopping_min_delta
                    ):
                        print("Early stopping triggered - validation loss not improving")
                        return {
                            'loss': sum(epoch_losses) / len(epoch_losses),
                            'state_loss': sum(epoch_state_losses) / len(epoch_state_losses),
                            'action_loss': sum(epoch_action_losses) / len(epoch_action_losses),
                            'early_stopped': True
                        }
                
                self.model.train()
            
            # Checkpointing
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint()
            
            self.global_step += 1
        
        return {
            'loss': sum(epoch_losses) / len(epoch_losses),
            'state_loss': sum(epoch_state_losses) / len(epoch_state_losses),
            'action_loss': sum(epoch_action_losses) / len(epoch_action_losses)
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation with comprehensive metrics.
        
        Returns:
            Dictionary of validation metrics
        """
        if not self.val_loader:
            return {}
        
        # Import metrics here to avoid circular dependency
        from .metrics import DiffusionMetrics, MetricTracker
        
        # Initialize metrics calculator
        metrics_calc = DiffusionMetrics(
            state_dim=self.model.state_dim,
            action_dim=self.model.action_dim
        )
        
        self.model.eval()
        val_losses = []
        val_state_losses = []
        val_action_losses = []
        all_metrics = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get loss
            loss, loss_components = self.model.training_step(batch, return_loss_components=True)
            
            val_losses.append(loss.item())
            val_state_losses.append(loss_components['state_loss'])
            val_action_losses.append(loss_components['action_loss'])
            
            # Sample predictions for metrics computation
            # Note: We need to generate predictions to compute quality metrics
            t = torch.randint(0, self.model.num_timesteps, (batch['future_states'].shape[0],), device=self.device)
            
            # Get noisy inputs
            state_noise = torch.randn_like(batch['future_states'])
            action_noise = torch.randn_like(batch['future_actions'])
            
            noisy_states = self.model.state_noise_schedule.q_sample(
                batch['future_states'], t, state_noise
            )
            noisy_actions = self.model.action_noise_schedule.q_sample(
                batch['future_actions'], t, action_noise
            )
            
            # Predict clean trajectory
            pred_states, pred_actions = self.model(
                noisy_states, noisy_actions,
                batch['history_states'], batch['history_actions'],
                t
            )
            
            # Compute comprehensive metrics
            batch_metrics = metrics_calc.compute_all_metrics(
                pred_states=pred_states,
                pred_actions=pred_actions,
                target_states=batch['future_states'],
                target_actions=batch['future_actions'],
                loss=loss.item(),
                state_loss=loss_components['state_loss'],
                action_loss=loss_components['action_loss']
            )
            
            all_metrics.append(batch_metrics)
        
        # Aggregate metrics
        val_loss = sum(val_losses) / len(val_losses)
        
        # Average all metrics across batches
        aggregated_metrics = {
            'loss': val_loss,
            'state_loss': sum(val_state_losses) / len(val_state_losses),
            'action_loss': sum(val_action_losses) / len(val_action_losses)
        }
        
        # Add detailed metrics
        if all_metrics:
            metric_keys = all_metrics[0].to_dict().keys()
            for key in metric_keys:
                if key not in aggregated_metrics:
                    values = [m.to_dict()[key] for m in all_metrics]
                    aggregated_metrics[key] = sum(values) / len(values)
        
        # Update best model based on primary metric
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(is_best=True)
        
        return aggregated_metrics
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Total training steps: {self.num_epochs * len(self.train_loader)}")
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
            self._log_metrics({f'epoch/train_{k}': v for k, v in train_metrics.items()})
            
            # End-of-epoch validation
            if self.val_loader:
                val_metrics = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}")
                self._log_metrics({f'epoch/val_{k}': v for k, v in val_metrics.items()})
        
        print("Training completed!")
        self.save_checkpoint(is_final=True)
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to WandB."""
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics, step=self.global_step)
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': {
                'state_dim': self.model.state_dim,
                'action_dim': self.model.action_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.transformer.transformer.num_layers,
                'history_length': self.model.history_length,
                'future_length_states': self.model.future_length_states,
                'future_length_actions': self.model.future_length_actions,
                'num_timesteps': self.model.num_timesteps
            }
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        elif is_final:
            path = self.checkpoint_dir / "final_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        # Also save config separately for easy loading
        config_path = self.checkpoint_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(checkpoint['config'], f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")
    
    @torch.no_grad()
    def sample_trajectories(
        self,
        num_samples: int = 1,
        use_ema: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Sample trajectories from the model.
        
        Args:
            num_samples: Number of trajectories to sample
            use_ema: Whether to use EMA model
            
        Returns:
            Dictionary with sampled states and actions
        """
        model = self.ema_model if use_ema else self.model
        model.eval()
        
        # Get a random batch from training data for history
        batch = next(iter(self.train_loader))
        batch = {k: v[:num_samples].to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Sample future trajectories
        sampled_states, sampled_actions = model.sample(
            batch['history_states'],
            batch['history_actions']
        )
        
        return {
            'history_states': batch['history_states'],
            'history_actions': batch['history_actions'],
            'sampled_states': sampled_states,
            'sampled_actions': sampled_actions,
            'gt_states': batch['future_states'],
            'gt_actions': batch['future_actions']
        }