"""Improved difficulty estimation for CACL based on LAFAN1 motion analysis.

Key insights from data analysis:
1. Coordination complexity is the strongest predictor (0.774 correlation)
2. Difficulty variance matters more than peak difficulty (0.711 correlation)  
3. Aerial phases are important but not dominant (0.551 correlation)
4. Balance peaks are actually negatively correlated (-0.173) - sustained balance is harder

This module provides a physics-informed difficulty estimator that captures:
- Balance requirements (ZMP, CoM stability)
- Contact complexity (aerial phases, ground reaction forces)
- Coordination demands (multi-limb synchronization)
- Dynamic feasibility (torque/power requirements)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Tuple


class PhysicsInformedDifficultyEstimator:
    """Estimates motion difficulty using physics-based features that matter for robots."""
    
    def __init__(self, 
                 motion_loader,
                 robot_params: Optional[Dict] = None,
                 device: str = "cuda"):
        """
        Args:
            motion_loader: MotionLoader instance with motion data
            robot_params: Robot-specific parameters (mass, inertia, torque limits)
            device: Computation device
        """
        self.device = device
        self.motion_loader = motion_loader
        self.fps = motion_loader.fps
        self.dt = 1.0 / self.fps
        
        # Robot parameters (defaults for G1)
        if robot_params is None:
            self.robot_params = {
                'mass': 35.0,  # kg
                'com_height_nominal': 0.85,  # m
                'foot_separation': 0.3,  # m
                'max_joint_torque': 120.0,  # Nm (varies by joint)
                'max_joint_velocity': 10.0,  # rad/s
                'gravity': 9.81  # m/s^2
            }
        else:
            self.robot_params = robot_params
            
        # Pre-compute all difficulty features
        self.difficulty_features = self._compute_all_features()
        self.difficulties = self._features_to_difficulty_scores()
        
        # Store statistics
        self.mean_difficulty = self.difficulties.mean().item()
        self.std_difficulty = self.difficulties.std().item()
        
    def _compute_all_features(self) -> Dict[str, torch.Tensor]:
        """Compute comprehensive physics-based difficulty features."""
        n_frames = self.motion_loader.time_step_total
        
        features = {
            'balance': self._compute_balance_features(),
            'contact': self._compute_contact_features(),
            'coordination': self._compute_coordination_features(),
            'dynamics': self._compute_dynamic_features(),
            'robustness': self._compute_robustness_requirements()
        }
        
        return features
    
    def _compute_balance_features(self) -> torch.Tensor:
        """Compute balance-related difficulty features.
        
        Returns:
            Tensor [n_frames, n_balance_features]
        """
        n_frames = self.motion_loader.time_step_total
        balance_features = []
        
        for i in range(n_frames):
            # Get current state
            body_pos = self.motion_loader.body_pos_w[i]  # [n_bodies, 3]
            body_vel = self.motion_loader.body_lin_vel_w[i] if hasattr(self.motion_loader, 'body_lin_vel_w') else torch.zeros_like(body_pos)
            
            # 1. CoM height variation (normalized by nominal height)
            com_height = body_pos[0, 2]  # Assuming first body is pelvis/root
            com_height_var = abs(com_height - self.robot_params['com_height_nominal']) / self.robot_params['com_height_nominal']
            
            # 2. CoM velocity (horizontal) - faster = harder to balance
            com_vel_horizontal = torch.norm(body_vel[0, :2])
            
            # 3. Angular momentum (simplified - using body angular velocities)
            if hasattr(self.motion_loader, 'body_ang_vel_w'):
                angular_momentum = torch.norm(self.motion_loader.body_ang_vel_w[i].mean(dim=0))
            else:
                angular_momentum = torch.tensor(0.0)
            
            # 4. Support polygon estimation (from foot positions if available)
            # Assuming bodies include feet (simplified)
            if body_pos.shape[0] > 2:
                # Estimate support polygon area from foot spread
                foot_spread = torch.norm(body_pos[-2:, :2].max(dim=0)[0] - body_pos[-2:, :2].min(dim=0)[0])
                support_polygon_inv = 1.0 / (foot_spread + 0.1)  # Smaller polygon = harder
            else:
                support_polygon_inv = torch.tensor(1.0)
            
            # 5. ZMP margin estimate (simplified)
            # Real ZMP requires ground reaction forces, we approximate
            com_acc = torch.zeros(3) if i == 0 else (body_vel[0] - self.motion_loader.body_lin_vel_w[i-1][0]) / self.dt
            zmp_offset = torch.norm(com_acc[:2]) / self.robot_params['gravity']  # Normalized
            
            balance_features.append(torch.stack([
                com_height_var,
                com_vel_horizontal / 3.0,  # Normalize by reasonable max velocity
                angular_momentum / 10.0,  # Normalize
                support_polygon_inv,
                zmp_offset
            ]))
        
        return torch.stack(balance_features)
    
    def _compute_contact_features(self) -> torch.Tensor:
        """Compute contact-related complexity features.
        
        Returns:
            Tensor [n_frames, n_contact_features]
        """
        n_frames = self.motion_loader.time_step_total
        contact_features = []
        
        for i in range(n_frames):
            body_pos = self.motion_loader.body_pos_w[i]
            body_vel = self.motion_loader.body_lin_vel_w[i] if hasattr(self.motion_loader, 'body_lin_vel_w') else torch.zeros_like(body_pos)
            
            # 1. Vertical velocity (potential aerial phase)
            vertical_vel = body_vel[0, 2]
            aerial_indicator = torch.sigmoid((vertical_vel - 0.5) * 10)  # Smooth indicator
            
            # 2. Vertical acceleration (impact forces)
            if i > 0:
                prev_vel = self.motion_loader.body_lin_vel_w[i-1][0, 2] if hasattr(self.motion_loader, 'body_lin_vel_w') else 0
                vertical_acc = abs(vertical_vel - prev_vel) / self.dt
                impact_force = vertical_acc / (self.robot_params['gravity'] * 2)  # Normalized
            else:
                impact_force = torch.tensor(0.0)
            
            # 3. Ground contact transitions (estimated from height changes)
            if body_pos.shape[0] > 2:  # Have foot bodies
                foot_heights = body_pos[-2:, 2]  # Last two bodies as feet
                ground_threshold = 0.05
                potential_contact_switch = (foot_heights < ground_threshold).any()
                contact_transition = 1.0 if potential_contact_switch else 0.0
            else:
                contact_transition = 0.0
            
            # 4. Estimated ground reaction force requirement
            # F = m(g + a)
            total_acc = torch.norm(body_vel[0] - (self.motion_loader.body_lin_vel_w[i-1][0] if i > 0 else body_vel[0])) / self.dt
            grf_normalized = (self.robot_params['gravity'] + total_acc) / self.robot_params['gravity']
            
            # 5. Dynamic motion indicator (high jerk = dynamic contact changes)
            if i > 1:
                jerk = torch.norm(body_vel[0] - 2 * self.motion_loader.body_lin_vel_w[i-1][0] + self.motion_loader.body_lin_vel_w[i-2][0]) / (self.dt ** 2)
                dynamic_indicator = jerk / 100.0  # Normalize
            else:
                dynamic_indicator = torch.tensor(0.0)
            
            contact_features.append(torch.stack([
                aerial_indicator,
                impact_force,
                torch.tensor(contact_transition),
                grf_normalized,
                dynamic_indicator
            ]))
        
        return torch.stack(contact_features)
    
    def _compute_coordination_features(self) -> torch.Tensor:
        """Compute multi-limb coordination complexity.
        
        Returns:
            Tensor [n_frames, n_coordination_features]
        """
        n_frames = self.motion_loader.time_step_total
        coordination_features = []
        
        joint_pos = self.motion_loader.joint_pos
        joint_vel = self.motion_loader.joint_vel
        
        for i in range(n_frames):
            # 1. Number of simultaneously moving joints
            active_joints = (torch.abs(joint_vel[i]) > 0.1).sum() / joint_vel.shape[1]
            
            # 2. Joint velocity variance (high variance = complex coordination)
            joint_vel_variance = joint_vel[i].var() if joint_vel[i].numel() > 0 else torch.tensor(0.0)
            
            # 3. Phase difference between limbs (using FFT for phase analysis)
            if i > 30:  # Need window for frequency analysis
                # Separate left/right leg joints (assuming first 6 joints per leg)
                if joint_vel.shape[1] >= 12:
                    left_leg = joint_vel[i-30:i, :6].mean(dim=1)
                    right_leg = joint_vel[i-30:i, 6:12].mean(dim=1)
                    
                    # Cross-correlation for phase difference
                    correlation = torch.corrcoef(torch.stack([left_leg, right_leg]))[0, 1]
                    phase_complexity = 1.0 - abs(correlation)  # Anti-phase or independent = complex
                else:
                    phase_complexity = torch.tensor(0.5)
            else:
                phase_complexity = torch.tensor(0.5)
            
            # 4. Upper/lower body coordination
            if joint_vel.shape[1] >= 15:
                lower_body_vel = torch.norm(joint_vel[i, :12])
                upper_body_vel = torch.norm(joint_vel[i, 12:])
                coordination_ratio = min(lower_body_vel, upper_body_vel) / (max(lower_body_vel, upper_body_vel) + 0.01)
            else:
                coordination_ratio = torch.tensor(0.5)
            
            # 5. Movement complexity (entropy of joint velocities)
            vel_distribution = torch.abs(joint_vel[i]) / (torch.abs(joint_vel[i]).sum() + 1e-6)
            movement_entropy = -(vel_distribution * torch.log(vel_distribution + 1e-6)).sum()
            movement_entropy_normalized = movement_entropy / np.log(joint_vel.shape[1])  # Normalize by max entropy
            
            coordination_features.append(torch.stack([
                active_joints,
                joint_vel_variance / 10.0,  # Normalize
                phase_complexity,
                coordination_ratio,
                movement_entropy_normalized
            ]))
        
        return torch.stack(coordination_features)
    
    def _compute_dynamic_features(self) -> torch.Tensor:
        """Compute dynamic feasibility features (torque/power requirements).
        
        Returns:
            Tensor [n_frames, n_dynamic_features]
        """
        n_frames = self.motion_loader.time_step_total
        dynamic_features = []
        
        joint_pos = self.motion_loader.joint_pos
        joint_vel = self.motion_loader.joint_vel
        
        for i in range(n_frames):
            # 1. Joint velocity magnitude (normalized by limits)
            vel_ratio = torch.abs(joint_vel[i]).max() / self.robot_params['max_joint_velocity']
            
            # 2. Joint acceleration (torque proxy)
            if i > 0:
                joint_acc = (joint_vel[i] - joint_vel[i-1]) / self.dt
                acc_magnitude = torch.norm(joint_acc)
                # Rough torque estimate: τ ≈ I * α (simplified)
                torque_estimate = acc_magnitude / self.robot_params['max_joint_torque']
            else:
                torque_estimate = torch.tensor(0.0)
            
            # 3. Mechanical power requirement
            # P = τ * ω
            if i > 0:
                power_estimate = torch.abs(joint_vel[i]).max() * acc_magnitude
                power_normalized = power_estimate / (self.robot_params['max_joint_torque'] * self.robot_params['max_joint_velocity'])
            else:
                power_normalized = torch.tensor(0.0)
            
            # 4. Kinetic energy
            kinetic_energy = 0.5 * self.robot_params['mass'] * torch.norm(self.motion_loader.body_lin_vel_w[i][0])**2 if hasattr(self.motion_loader, 'body_lin_vel_w') else torch.tensor(0.0)
            ke_normalized = kinetic_energy / (0.5 * self.robot_params['mass'] * 9.0)  # Normalize by energy at 3 m/s
            
            # 5. Dynamic consistency (smoothness)
            if i > 1:
                jerk = torch.norm(joint_vel[i] - 2*joint_vel[i-1] + joint_vel[i-2]) / (self.dt**2)
                smoothness = 1.0 / (1.0 + jerk/100.0)  # Smoother = easier
            else:
                smoothness = torch.tensor(1.0)
            
            dynamic_features.append(torch.stack([
                vel_ratio,
                torque_estimate,
                power_normalized,
                ke_normalized,
                1.0 - smoothness  # Invert so higher = harder
            ]))
        
        return torch.stack(dynamic_features)
    
    def _compute_robustness_requirements(self) -> torch.Tensor:
        """Compute robustness requirements (how much margin for error).
        
        Returns:
            Tensor [n_frames, n_robustness_features]
        """
        n_frames = self.motion_loader.time_step_total
        robustness_features = []
        
        for i in range(n_frames):
            # 1. Distance from joint limits
            joint_pos = self.motion_loader.joint_pos[i]
            # Assume ±2.0 rad as typical limits
            joint_limit_margin = 1.0 - torch.abs(joint_pos).max() / 2.0
            joint_limit_difficulty = 1.0 - joint_limit_margin  # Closer to limits = harder
            
            # 2. Recovery time available (based on velocity)
            if hasattr(self.motion_loader, 'body_lin_vel_w'):
                recovery_time = self.robot_params['com_height_nominal'] / (torch.abs(self.motion_loader.body_lin_vel_w[i][0, 2]) + 0.1)
                recovery_difficulty = 1.0 / (1.0 + recovery_time)
            else:
                recovery_difficulty = torch.tensor(0.0)
            
            # 3. Motion reversibility (can errors be corrected)
            if i < n_frames - 1:
                # High acceleration = hard to reverse
                acc = torch.norm(self.motion_loader.joint_vel[i+1] - self.motion_loader.joint_vel[i]) / self.dt
                reversibility_difficulty = acc / 100.0
            else:
                reversibility_difficulty = torch.tensor(0.0)
            
            # 4. Stability margin (simplified)
            if hasattr(self.motion_loader, 'body_ang_vel_w'):
                angular_rate = torch.norm(self.motion_loader.body_ang_vel_w[i][0])
                stability_margin = 1.0 / (1.0 + angular_rate)
                stability_difficulty = 1.0 - stability_margin
            else:
                stability_difficulty = torch.tensor(0.5)
            
            # 5. Task precision requirement (faster motion = less precise)
            velocity_magnitude = torch.norm(self.motion_loader.joint_vel[i])
            precision_difficulty = velocity_magnitude / 10.0
            
            robustness_features.append(torch.stack([
                joint_limit_difficulty,
                recovery_difficulty,
                reversibility_difficulty,
                stability_difficulty,
                precision_difficulty
            ]))
        
        return torch.stack(robustness_features)
    
    def _features_to_difficulty_scores(self) -> torch.Tensor:
        """Combine all features into a single difficulty score per frame.
        
        Based on correlations from LAFAN1 analysis:
        - Coordination: 0.774 correlation → 30% weight
        - Variance over time: 0.711 correlation → considered in temporal smoothing
        - Aerial/contact: 0.551 correlation → 20% weight
        - Balance: varies → 25% weight
        - Dynamics: important for robots → 25% weight
        
        Returns:
            Tensor [n_frames] with difficulty scores in [0, 1]
        """
        # Get feature tensors
        balance = self.difficulty_features['balance']
        contact = self.difficulty_features['contact']
        coordination = self.difficulty_features['coordination']
        dynamics = self.difficulty_features['dynamics']
        robustness = self.difficulty_features['robustness']
        
        # Normalize each feature group to [0, 1]
        balance_score = balance.mean(dim=1)
        balance_score = (balance_score - balance_score.min()) / (balance_score.max() - balance_score.min() + 1e-6)
        
        contact_score = contact.mean(dim=1)
        contact_score = (contact_score - contact_score.min()) / (contact_score.max() - contact_score.min() + 1e-6)
        
        coordination_score = coordination.mean(dim=1)
        coordination_score = (coordination_score - coordination_score.min()) / (coordination_score.max() - coordination_score.min() + 1e-6)
        
        dynamics_score = dynamics.mean(dim=1)
        dynamics_score = (dynamics_score - dynamics_score.min()) / (dynamics_score.max() - dynamics_score.min() + 1e-6)
        
        robustness_score = robustness.mean(dim=1)
        robustness_score = (robustness_score - robustness_score.min()) / (robustness_score.max() - robustness_score.min() + 1e-6)
        
        # Weighted combination based on correlation analysis
        overall_difficulty = (
            coordination_score * 0.30 +  # Strongest predictor
            balance_score * 0.25 +        # Important for robots
            dynamics_score * 0.20 +        # Feasibility matters
            contact_score * 0.15 +         # Moderate importance
            robustness_score * 0.10       # Safety margin
        )
        
        # Apply temporal smoothing (difficulty variance matters)
        # Use a rolling window to ensure gradual difficulty progression
        window_size = min(30, len(overall_difficulty) // 10)
        if window_size > 1:
            kernel = torch.ones(window_size) / window_size
            kernel = kernel.to(overall_difficulty.device)
            # Pad for convolution
            padded = torch.nn.functional.pad(overall_difficulty.unsqueeze(0).unsqueeze(0), 
                                            (window_size//2, window_size//2), mode='reflect')
            smoothed = torch.nn.functional.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0))
            overall_difficulty = 0.7 * overall_difficulty + 0.3 * smoothed.squeeze()
        
        # Final normalization to [0, 1]
        overall_difficulty = (overall_difficulty - overall_difficulty.min()) / (overall_difficulty.max() - overall_difficulty.min() + 1e-6)
        
        return overall_difficulty
    
    def get_difficulties(self) -> torch.Tensor:
        """Get pre-computed difficulty scores.
        
        Returns:
            Difficulty scores for all frames [n_frames] in [0, 1]
        """
        return self.difficulties
    
    def get_bin_difficulties(self, bin_count: int) -> torch.Tensor:
        """Get average difficulty per time bin for efficient sampling.
        
        Args:
            bin_count: Number of bins to divide motion into
            
        Returns:
            Average difficulty per bin [bin_count]
        """
        n_frames = len(self.difficulties)
        bin_size = max(1, n_frames // bin_count)
        bin_difficulties = torch.zeros(bin_count, device=self.device)
        
        for i in range(bin_count):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, n_frames)
            if end_idx > start_idx:
                # Use max difficulty in bin rather than mean
                # This ensures we don't underestimate difficulty
                bin_difficulties[i] = self.difficulties[start_idx:end_idx].max()
            else:
                bin_difficulties[i] = self.difficulties[-1]
                
        return bin_difficulties
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get relative importance of each feature category.
        
        Useful for debugging and understanding what makes motions difficult.
        """
        importance = {}
        
        for category in ['balance', 'contact', 'coordination', 'dynamics', 'robustness']:
            features = self.difficulty_features[category]
            # Compute variance contribution
            feature_variance = features.mean(dim=1).var()
            importance[category] = feature_variance.item()
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
            
        return importance


def compare_difficulty_estimators(motion_loader):
    """Compare simple vs physics-informed difficulty estimation."""
    
    # Simple estimator (current implementation)
    from cacl_components import DifficultyEstimator
    simple_estimator = DifficultyEstimator(motion_loader)
    simple_difficulties = simple_estimator.get_difficulties()
    
    # Physics-informed estimator
    physics_estimator = PhysicsInformedDifficultyEstimator(motion_loader)
    physics_difficulties = physics_estimator.get_difficulties()
    
    # Get feature importance
    importance = physics_estimator.get_feature_importance()
    
    print("\n" + "="*60)
    print("DIFFICULTY ESTIMATION COMPARISON")
    print("="*60)
    
    print(f"\nSimple Estimator:")
    print(f"  Mean difficulty: {simple_difficulties.mean():.3f}")
    print(f"  Std difficulty: {simple_difficulties.std():.3f}")
    print(f"  Based on: velocity, acceleration norms only")
    
    print(f"\nPhysics-Informed Estimator:")
    print(f"  Mean difficulty: {physics_difficulties.mean():.3f}")
    print(f"  Std difficulty: {physics_difficulties.std():.3f}")
    print(f"  Feature importance:")
    for feature, imp in importance.items():
        print(f"    {feature}: {imp:.2%}")
    
    # Correlation between estimators
    if len(simple_difficulties) == len(physics_difficulties):
        correlation = torch.corrcoef(torch.stack([simple_difficulties, physics_difficulties]))[0, 1]
        print(f"\nCorrelation between estimators: {correlation:.3f}")
    
    return simple_difficulties, physics_difficulties