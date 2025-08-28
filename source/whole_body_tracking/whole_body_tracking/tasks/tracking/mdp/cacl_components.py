"""Capability-Aware Curriculum Learning (CACL) components for motion tracking.

This module provides core components for CACL:
- CompetenceAssessor: Evaluates agent's current capability
- DifficultyEstimator: Pre-computes motion segment difficulties
- CapabilityMatcher: Matches tasks to agent competence
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List


class CompetenceAssessor(nn.Module):
    """Lightweight competence assessment network.
    
    Evaluates agent capability based on performance history.
    Uses a simple MLP for computational efficiency.
    """
    
    def __init__(self, 
                 history_length: int = 100,
                 hidden_dim: int = 128, 
                 num_envs: int = 4096,
                 device: str = "cuda"):
        """Initialize competence assessor.
        
        Args:
            history_length: Length of performance history to consider
            hidden_dim: Hidden layer dimension
            num_envs: Maximum number of environments
            device: Computation device
        """
        super().__init__()
        self.device = device
        self.history_length = history_length
        
        # Feature extraction dimension (3 features per history analysis)
        feature_dim = 3
        
        # Simple MLP for efficiency
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(), 
            nn.Linear(32, 1),  # Scalar competence
            nn.Sigmoid()  # Normalize to [0, 1]
        ).to(device)
        
        # Pre-allocate for efficiency
        self.features_buffer = torch.zeros((num_envs, feature_dim), device=device)
    
    @torch.no_grad()
    def assess(self, performance_history: torch.Tensor) -> torch.Tensor:
        """Fast competence assessment without gradient computation.
        
        Args:
            performance_history: [batch, history_len, 2] tensor with (success, difficulty)
            
        Returns:
            Scalar competence values [batch] in range [0, 1]
        """
        # Extract features from history
        features = self._extract_features(performance_history)
        
        # Single forward pass
        competence = self.network(features).squeeze(-1)
        
        return competence
    
    def _extract_features(self, history: torch.Tensor) -> torch.Tensor:
        """Extract relevant features from performance history.
        
        Args:
            history: [batch, history_len, 2] tensor
            
        Returns:
            Features tensor [batch, 3]
        """
        batch_size = history.shape[0]
        
        # Success rate over entire history
        success_rate = history[:, :, 0].mean(dim=1)
        
        # Improvement trend (recent vs early performance)
        recent_window = min(20, self.history_length // 5)
        recent_success = history[:, -recent_window:, 0].mean(dim=1)
        early_success = history[:, :recent_window, 0].mean(dim=1)
        improvement_trend = recent_success - early_success
        
        # Maximum difficulty successfully handled
        # Only consider successful attempts (success == 1)
        success_mask = history[:, :, 0] > 0.5
        difficulties_handled = history[:, :, 1].clone()
        difficulties_handled[~success_mask] = 0
        max_difficulty_handled = difficulties_handled.max(dim=1)[0]
        
        # Stack features
        features = torch.stack([
            success_rate,
            improvement_trend,
            max_difficulty_handled
        ], dim=-1)
        
        return features
    
    def train_step(self, 
                   history: torch.Tensor, 
                   future_performance: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> float:
        """Single training step for the competence network.
        
        Args:
            history: Past performance history [batch, history_len, 2]
            future_performance: Future performance outcomes [batch]
            optimizer: Optimizer for network parameters
            
        Returns:
            Training loss value
        """
        # Forward pass with gradients (don't use assess which has @torch.no_grad)
        features = self._extract_features(history)
        predicted_competence = self.network(features).squeeze(-1)
        
        # Loss: MSE between predicted competence and actual future performance
        loss = nn.functional.mse_loss(predicted_competence, future_performance)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class DifficultyEstimator:
    """Pre-computed difficulty estimation for efficiency.
    
    Computes difficulty scores for all motion frames during initialization
    to avoid runtime overhead.
    """
    
    def __init__(self, 
                 motion_loader,
                 feature_list: Optional[List[str]] = None,
                 device: str = "cpu"):
        """Initialize difficulty estimator.
        
        Args:
            motion_loader: MotionLoader instance with motion data
            feature_list: List of features to use for difficulty estimation
            device: Computation device
        """
        self.device = device
        self.motion_loader = motion_loader
        
        # Default features if not specified
        if feature_list is None:
            self.feature_list = [
                "velocity", 
                "acceleration",
                "angular_velocity"
            ]
        else:
            self.feature_list = feature_list
            
        # Pre-compute all difficulties
        self.difficulties = self._compute_all_difficulties()
        
        # Store statistics for normalization
        self.mean_difficulty = self.difficulties.mean().item()
        self.std_difficulty = self.difficulties.std().item()
    
    def _compute_all_difficulties(self) -> torch.Tensor:
        """Pre-compute difficulties for all motion frames.
        
        Returns:
            Normalized difficulty scores [n_frames] in range [0, 1]
        """
        n_frames = self.motion_loader.time_step_total
        difficulties = torch.zeros(n_frames, device=self.device)
        
        # Compute difficulty for each frame
        for i in range(n_frames):
            features = self._compute_frame_features(i)
            difficulties[i] = self._features_to_difficulty(features)
        
        # Normalize to [0, 1] range
        min_diff = difficulties.min()
        max_diff = difficulties.max()
        if max_diff > min_diff:
            difficulties = (difficulties - min_diff) / (max_diff - min_diff)
        else:
            difficulties = torch.ones_like(difficulties) * 0.5
        
        return difficulties
    
    def _compute_frame_features(self, frame_idx: int) -> Dict[str, float]:
        """Extract difficulty-relevant features for a single frame.
        
        Args:
            frame_idx: Index of motion frame
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        # Joint velocity magnitude
        if "velocity" in self.feature_list:
            joint_vel = self.motion_loader.joint_vel[frame_idx]
            features['joint_vel_norm'] = joint_vel.norm().item()
        
        # Joint acceleration (approximate from velocity difference)
        if "acceleration" in self.feature_list:
            if frame_idx < self.motion_loader.time_step_total - 1:
                joint_vel_next = self.motion_loader.joint_vel[frame_idx + 1]
                joint_vel_curr = self.motion_loader.joint_vel[frame_idx]
                dt = 1.0 / self.motion_loader.fps
                joint_acc = (joint_vel_next - joint_vel_curr) / dt
                features['joint_acc_norm'] = joint_acc.norm().item()
            else:
                features['joint_acc_norm'] = 0.0
        
        # Body linear velocity
        if "body_velocity" in self.feature_list:
            body_vel = self.motion_loader.body_lin_vel_w[frame_idx]
            features['body_vel_norm'] = body_vel.norm(dim=-1).mean().item()
        
        # Body angular velocity  
        if "angular_velocity" in self.feature_list:
            body_ang_vel = self.motion_loader.body_ang_vel_w[frame_idx]
            features['body_ang_vel_norm'] = body_ang_vel.norm(dim=-1).mean().item()
        
        # CoM height variance (stability indicator)
        if "com_height_variance" in self.feature_list:
            # Use root body as CoM proxy
            root_height = self.motion_loader.body_pos_w[frame_idx, 0, 2]
            # Compare to average height
            avg_height = self.motion_loader.body_pos_w[:, 0, 2].mean()
            features['com_height_var'] = abs(root_height - avg_height).item()
        
        # Contact switches (approximate from height changes)
        if "contact_switches" in self.feature_list:
            if frame_idx > 0:
                # Detect potential contact changes from foot height changes
                # This is a simplified heuristic
                features['contact_complexity'] = 1.0
            else:
                features['contact_complexity'] = 0.0
                
        return features
    
    def _features_to_difficulty(self, features: Dict[str, float]) -> float:
        """Map features to a single difficulty score.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Scalar difficulty score
        """
        # Default weights for different features
        weights = {
            'joint_vel_norm': 0.3,
            'joint_acc_norm': 0.2,
            'body_vel_norm': 0.2,
            'body_ang_vel_norm': 0.2,
            'com_height_var': 0.05,
            'contact_complexity': 0.05,
        }
        
        # Weighted sum of available features
        difficulty = 0.0
        total_weight = 0.0
        
        for feature_name, weight in weights.items():
            if feature_name in features:
                difficulty += weight * features[feature_name]
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            difficulty /= total_weight
            
        return difficulty
    
    def get_difficulties(self) -> torch.Tensor:
        """Get pre-computed difficulty scores.
        
        Returns:
            Difficulty scores for all frames [n_frames]
        """
        return self.difficulties
    
    def get_bin_difficulties(self, bin_count: int) -> torch.Tensor:
        """Get average difficulty per time bin.
        
        Args:
            bin_count: Number of bins to divide motion into
            
        Returns:
            Average difficulty per bin [bin_count]
        """
        n_frames = len(self.difficulties)
        bin_size = n_frames // bin_count
        bin_difficulties = torch.zeros(bin_count, device=self.device)
        
        for i in range(bin_count):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, n_frames)
            if end_idx > start_idx:
                bin_difficulties[i] = self.difficulties[start_idx:end_idx].mean()
            else:
                bin_difficulties[i] = self.difficulties[-1]
                
        return bin_difficulties


class CapabilityMatcher:
    """Efficient capability-difficulty matching for curriculum selection.
    
    Implements the Zone of Proximal Development (ZPD) concept to match
    tasks to learner capability.
    """
    
    def __init__(self, 
                 learning_stretch: float = 0.2,
                 min_score: float = 0.01,
                 zpd_sharpness: float = 5.0):
        """Initialize capability matcher.
        
        Args:
            learning_stretch: How much harder than current capability tasks should be
            min_score: Minimum score for any segment (avoid zero probabilities)
            zpd_sharpness: Sharpness of score decay outside ZPD
        """
        self.learning_stretch = learning_stretch
        self.min_score = min_score
        self.zpd_sharpness = zpd_sharpness
        
        # Cache for repeated computations
        self.score_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def compute_sampling_probs(self,
                               competences: torch.Tensor,
                               difficulties: torch.Tensor,
                               fallback_probs: Optional[torch.Tensor] = None,
                               blend_ratio: float = 0.0) -> torch.Tensor:
        """Compute CACL-aware sampling probabilities.
        
        Args:
            competences: Current competence levels [batch_size]
            difficulties: Difficulty scores for segments [n_segments]
            fallback_probs: Original sampling probabilities for blending [n_segments]
            blend_ratio: Ratio of original sampling to use (0=pure CACL, 1=pure original)
            
        Returns:
            Sampling probabilities [n_segments]
        """
        # For batch processing, we average competences
        # In practice, each env samples independently
        avg_competence = competences.mean()
        
        # Compute optimal difficulty range (ZPD)
        optimal_difficulty = avg_competence * (1 + self.learning_stretch)
        
        # Score all segments
        scores = self._score_segments(difficulties, avg_competence, optimal_difficulty)
        
        # Convert scores to probabilities
        cacl_probs = torch.softmax(scores, dim=-1)
        
        # Optional blending with fallback probabilities
        if fallback_probs is not None and blend_ratio > 0:
            final_probs = (1 - blend_ratio) * cacl_probs + blend_ratio * fallback_probs
        else:
            final_probs = cacl_probs
        
        # Ensure probabilities sum to 1 (handle numerical issues)
        final_probs = final_probs / final_probs.sum()
        
        return final_probs
    
    def _score_segments(self, 
                       difficulties: torch.Tensor,
                       competence: torch.Tensor,
                       optimal: torch.Tensor) -> torch.Tensor:
        """Score segments based on capability-difficulty alignment.
        
        Args:
            difficulties: Difficulty scores [n_segments]
            competence: Current competence level (scalar)
            optimal: Optimal difficulty level (scalar)
            
        Returns:
            Scores for each segment [n_segments]
        """
        scores = torch.zeros_like(difficulties)
        
        # Three zones: too easy, ZPD, too hard
        
        # Zone 1: Too easy (below competence)
        easy_mask = difficulties < competence
        if easy_mask.any():
            # Low score that decreases with distance from competence
            distance_from_competence = competence - difficulties[easy_mask]
            scores[easy_mask] = self.min_score + 0.1 * torch.exp(-distance_from_competence)
        
        # Zone 2: ZPD (between competence and optimal)
        zpd_mask = (difficulties >= competence) & (difficulties <= optimal)
        if zpd_mask.any():
            # High score within ZPD
            scores[zpd_mask] = 1.0
        
        # Zone 3: Too hard (above optimal)
        hard_mask = difficulties > optimal
        if hard_mask.any():
            # Exponentially decreasing score
            distance_from_optimal = difficulties[hard_mask] - optimal
            scores[hard_mask] = torch.exp(-self.zpd_sharpness * distance_from_optimal)
        
        # Add small epsilon to avoid log(0) in entropy calculations
        scores = scores + 1e-8
        
        return scores
    
    def compute_metrics(self,
                       competences: torch.Tensor,
                       difficulties: torch.Tensor,
                       selected_indices: torch.Tensor) -> Dict[str, float]:
        """Compute metrics for curriculum alignment.
        
        Args:
            competences: Competence levels [batch_size]
            difficulties: Difficulty scores [n_segments]
            selected_indices: Indices of selected segments [batch_size]
            
        Returns:
            Dictionary of metrics
        """
        selected_difficulties = difficulties[selected_indices]
        avg_competence = competences.mean()
        
        # Compute alignment score
        optimal = avg_competence * (1 + self.learning_stretch)
        in_zpd = (selected_difficulties >= competences) & (selected_difficulties <= optimal.expand_as(selected_difficulties))
        zpd_ratio = in_zpd.float().mean()
        
        # Average distance from optimal
        distance_from_optimal = (selected_difficulties - optimal.expand_as(selected_difficulties)).abs().mean()
        
        metrics = {
            'cacl_zpd_ratio': zpd_ratio.item(),
            'cacl_avg_competence': avg_competence.item(),
            'cacl_avg_selected_difficulty': selected_difficulties.mean().item(),
            'cacl_distance_from_optimal': distance_from_optimal.item(),
        }
        
        return metrics
    
    def reset_cache(self):
        """Reset the score cache and statistics."""
        self.score_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0