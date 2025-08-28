#!/usr/bin/env python3
"""Analyze LAFAN1 motion dataset to understand difficulty characteristics for CACL.

This script analyzes motion files to extract meaningful difficulty features that
capture what makes motions challenging for humanoid robots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple

class MotionDifficultyAnalyzer:
    """Analyzes motion difficulty from multiple perspectives."""
    
    def __init__(self, motion_file: str, fps: int = 30):
        """Load and process motion data."""
        self.motion_file = motion_file
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Load motion data
        # Format: x,y,z, qw,qx,qy,qz, joint_angles[29]
        self.data = np.loadtxt(motion_file, delimiter=',')
        self.n_frames = self.data.shape[0]
        self.duration = self.n_frames / fps
        
        # Parse data
        self.base_pos = self.data[:, :3]  # x, y, z
        self.base_quat = self.data[:, 3:7]  # qw, qx, qy, qz
        self.joint_angles = self.data[:, 7:]  # 29 joint angles
        
        print(f"Loaded {Path(motion_file).name}: {self.n_frames} frames, {self.duration:.2f} seconds")
        
    def compute_velocities(self) -> Dict[str, np.ndarray]:
        """Compute linear and angular velocities."""
        # Linear velocity of base
        base_vel = np.gradient(self.base_pos, self.dt, axis=0)
        
        # Joint velocities
        joint_vel = np.gradient(self.joint_angles, self.dt, axis=0)
        
        return {
            'base_vel': base_vel,
            'joint_vel': joint_vel,
            'base_speed': np.linalg.norm(base_vel, axis=1),
            'max_joint_vel': np.max(np.abs(joint_vel), axis=1)
        }
    
    def compute_accelerations(self) -> Dict[str, np.ndarray]:
        """Compute linear and angular accelerations."""
        velocities = self.compute_velocities()
        
        # Linear acceleration
        base_acc = np.gradient(velocities['base_vel'], self.dt, axis=0)
        
        # Joint accelerations  
        joint_acc = np.gradient(velocities['joint_vel'], self.dt, axis=0)
        
        # Jerk (rate of change of acceleration)
        base_jerk = np.gradient(base_acc, self.dt, axis=0)
        
        return {
            'base_acc': base_acc,
            'joint_acc': joint_acc,
            'base_jerk': base_jerk,
            'base_acc_mag': np.linalg.norm(base_acc, axis=1),
            'max_joint_acc': np.max(np.abs(joint_acc), axis=1),
            'base_jerk_mag': np.linalg.norm(base_jerk, axis=1)
        }
    
    def compute_balance_difficulty(self) -> Dict[str, np.ndarray]:
        """Compute balance-related difficulty metrics."""
        velocities = self.compute_velocities()
        accelerations = self.compute_accelerations()
        
        # CoM height variation (proxy for squatting, jumping)
        com_height = self.base_pos[:, 2]
        com_height_var = np.abs(com_height - np.mean(com_height))
        
        # Estimate angular momentum (simplified)
        # High angular velocity + acceleration = harder to balance
        angular_difficulty = velocities['max_joint_vel'] * accelerations['max_joint_acc']
        
        # Estimate support polygon changes
        # Large lateral movement = potential stance width changes
        lateral_vel = np.abs(velocities['base_vel'][:, 1])  # y-axis
        
        # Vertical acceleration (jumping, landing impacts)
        vertical_acc = np.abs(accelerations['base_acc'][:, 2])
        
        return {
            'com_height_var': com_height_var,
            'angular_difficulty': angular_difficulty,
            'lateral_vel': lateral_vel,
            'vertical_acc': vertical_acc,
            'balance_score': com_height_var * 0.3 + angular_difficulty * 0.3 + 
                           lateral_vel * 0.2 + vertical_acc * 0.2
        }
    
    def compute_contact_complexity(self) -> Dict[str, np.ndarray]:
        """Estimate contact complexity from motion characteristics."""
        velocities = self.compute_velocities()
        accelerations = self.compute_accelerations()
        
        # Detect potential aerial phases (high vertical velocity + acceleration)
        vertical_vel = velocities['base_vel'][:, 2]
        vertical_acc = accelerations['base_acc'][:, 2]
        
        # Potential jump/flight phases
        potential_aerial = (vertical_vel > 0.5) | (vertical_acc > 9.8)
        
        # Detect quick direction changes (high jerk)
        direction_changes = accelerations['base_jerk_mag']
        
        # Estimate foot contact switches from base motion pattern
        # High frequency oscillations in height = stepping
        height_freq = np.abs(np.fft.fft(self.base_pos[:, 2]))[:self.n_frames//2]
        step_frequency = np.argmax(height_freq[1:]) + 1  # Dominant frequency
        
        return {
            'aerial_phases': potential_aerial.astype(float),
            'direction_changes': direction_changes,
            'step_frequency': np.ones(self.n_frames) * step_frequency,
            'contact_score': potential_aerial.astype(float) * 0.4 + 
                           direction_changes / np.max(direction_changes + 1e-6) * 0.6
        }
    
    def compute_coordination_complexity(self) -> Dict[str, np.ndarray]:
        """Compute multi-limb coordination complexity."""
        velocities = self.compute_velocities()
        
        joint_vel = velocities['joint_vel']
        
        # Separate limbs (approximate based on joint indices)
        # Assuming: legs[0:12], torso[12:15], arms[15:29]
        leg_joints = joint_vel[:, :12]
        torso_joints = joint_vel[:, 12:15]
        arm_joints = joint_vel[:, 15:] if joint_vel.shape[1] > 15 else np.zeros((self.n_frames, 1))
        
        # Compute correlation between limbs (low correlation = high coordination needed)
        coordination_scores = []
        for i in range(self.n_frames):
            if i > 10:  # Need window for correlation
                # Cross-correlation between leg and arm motion
                leg_pattern = leg_joints[i-10:i].flatten()
                arm_pattern = arm_joints[i-10:i].flatten()
                
                if len(arm_pattern) > 0 and len(leg_pattern) > 0:
                    # Ensure same length for correlation
                    min_len = min(len(leg_pattern), len(arm_pattern))
                    leg_pattern = leg_pattern[:min_len]
                    arm_pattern = arm_pattern[:min_len]
                    
                    if np.std(leg_pattern) > 0.01 and np.std(arm_pattern) > 0.01:
                        correlation = np.corrcoef(leg_pattern, arm_pattern)[0, 1]
                        # Low correlation means independent movement = harder
                        coordination_scores.append(1.0 - abs(correlation))
                    else:
                        coordination_scores.append(0.5)
                else:
                    coordination_scores.append(0.5)
            else:
                coordination_scores.append(0.5)
        
        # Count simultaneously moving joints
        active_joints = (np.abs(joint_vel) > 0.1).sum(axis=1)
        
        return {
            'coordination_score': np.array(coordination_scores),
            'active_joints': active_joints,
            'complexity_score': np.array(coordination_scores) * 0.5 + 
                              active_joints / joint_vel.shape[1] * 0.5
        }
    
    def compute_overall_difficulty(self) -> Dict[str, np.ndarray]:
        """Compute comprehensive difficulty score."""
        # Get all sub-scores
        velocities = self.compute_velocities()
        accelerations = self.compute_accelerations()
        balance = self.compute_balance_difficulty()
        contact = self.compute_contact_complexity()
        coordination = self.compute_coordination_complexity()
        
        # Normalize and combine
        overall_difficulty = (
            balance['balance_score'] / (np.max(balance['balance_score']) + 1e-6) * 0.25 +
            contact['contact_score'] / (np.max(contact['contact_score']) + 1e-6) * 0.25 +
            coordination['complexity_score'] / (np.max(coordination['complexity_score']) + 1e-6) * 0.25 +
            accelerations['base_jerk_mag'] / (np.max(accelerations['base_jerk_mag']) + 1e-6) * 0.25
        )
        
        return {
            'overall_difficulty': overall_difficulty,
            'balance': balance,
            'contact': contact,
            'coordination': coordination,
            'velocities': velocities,
            'accelerations': accelerations
        }
    
    def visualize_difficulty(self, save_path: str = None):
        """Create visualization of difficulty analysis."""
        difficulty = self.compute_overall_difficulty()
        
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        time = np.arange(self.n_frames) / self.fps
        
        # Overall difficulty
        axes[0].plot(time, difficulty['overall_difficulty'], 'k-', linewidth=2)
        axes[0].fill_between(time, 0, difficulty['overall_difficulty'], alpha=0.3)
        axes[0].set_ylabel('Overall\nDifficulty')
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3)
        
        # Balance difficulty
        axes[1].plot(time, difficulty['balance']['balance_score'] / 
                    (np.max(difficulty['balance']['balance_score']) + 1e-6), 'b-')
        axes[1].set_ylabel('Balance\nDifficulty')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        
        # Contact complexity
        axes[2].plot(time, difficulty['contact']['contact_score'] / 
                    (np.max(difficulty['contact']['contact_score']) + 1e-6), 'g-')
        axes[2].set_ylabel('Contact\nComplexity')
        axes[2].set_ylim([0, 1])
        axes[2].grid(True, alpha=0.3)
        
        # Coordination complexity
        axes[3].plot(time, difficulty['coordination']['complexity_score'] / 
                    (np.max(difficulty['coordination']['complexity_score']) + 1e-6), 'r-')
        axes[3].set_ylabel('Coordination\nComplexity')
        axes[3].set_ylim([0, 1])
        axes[3].grid(True, alpha=0.3)
        
        # Velocity profile
        axes[4].plot(time, difficulty['velocities']['base_speed'], 'c-', label='Base Speed')
        axes[4].set_ylabel('Speed (m/s)')
        axes[4].set_xlabel('Time (s)')
        axes[4].grid(True, alpha=0.3)
        axes[4].legend()
        
        plt.suptitle(f'Motion Difficulty Analysis: {Path(self.motion_file).stem}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def get_statistics(self) -> Dict[str, float]:
        """Get summary statistics of motion difficulty."""
        difficulty = self.compute_overall_difficulty()
        
        stats = {
            'mean_difficulty': np.mean(difficulty['overall_difficulty']),
            'max_difficulty': np.max(difficulty['overall_difficulty']),
            'difficulty_variance': np.var(difficulty['overall_difficulty']),
            'peak_velocity': np.max(difficulty['velocities']['base_speed']),
            'peak_acceleration': np.max(difficulty['accelerations']['base_acc_mag']),
            'aerial_ratio': np.mean(difficulty['contact']['aerial_phases']),
            'coordination_mean': np.mean(difficulty['coordination']['complexity_score']),
            'balance_peaks': np.sum(difficulty['balance']['balance_score'] > 
                                  np.percentile(difficulty['balance']['balance_score'], 90))
        }
        
        return stats


def analyze_motion_dataset(dataset_path: str = "/tmp/lafan1_dataset/g1"):
    """Analyze multiple motions to understand difficulty patterns."""
    
    motion_files = list(Path(dataset_path).glob("*.csv"))[:10]  # Analyze first 10
    
    all_stats = []
    motion_names = []
    
    for motion_file in motion_files:
        print(f"\nAnalyzing {motion_file.name}...")
        analyzer = MotionDifficultyAnalyzer(str(motion_file))
        
        stats = analyzer.get_statistics()
        stats['name'] = motion_file.stem
        all_stats.append(stats)
        motion_names.append(motion_file.stem)
        
        # Create detailed visualization for interesting motions
        if any(keyword in motion_file.stem.lower() for keyword in ['dance', 'jump', 'fall']):
            analyzer.visualize_difficulty(f"/tmp/{motion_file.stem}_difficulty.png")
    
    # Create comparison DataFrame
    df = pd.DataFrame(all_stats)
    df = df.set_index('name')
    
    # Print statistics
    print("\n" + "="*80)
    print("MOTION DIFFICULTY STATISTICS")
    print("="*80)
    print(df.round(3))
    
    # Identify patterns
    print("\n" + "="*80)
    print("KEY INSIGHTS FOR CACL DIFFICULTY ESTIMATION")
    print("="*80)
    
    # Find easiest and hardest motions
    easiest = df.loc[df['mean_difficulty'].idxmin()]
    hardest = df.loc[df['mean_difficulty'].idxmax()]
    
    print(f"\nEasiest motion: {easiest.name}")
    print(f"  - Mean difficulty: {easiest['mean_difficulty']:.3f}")
    print(f"  - Peak velocity: {easiest['peak_velocity']:.3f} m/s")
    
    print(f"\nHardest motion: {hardest.name}")
    print(f"  - Mean difficulty: {hardest['mean_difficulty']:.3f}")
    print(f"  - Peak velocity: {hardest['peak_velocity']:.3f} m/s")
    print(f"  - Aerial ratio: {hardest['aerial_ratio']:.3f}")
    
    # Correlation analysis
    print("\nDifficulty Correlations:")
    correlations = df.corr()['mean_difficulty'].sort_values(ascending=False)
    for metric, corr in correlations.items():
        if metric != 'mean_difficulty':
            print(f"  {metric}: {corr:.3f}")
    
    return df


if __name__ == "__main__":
    # Analyze dataset
    results = analyze_motion_dataset()
    
    # Save results
    results.to_csv("/tmp/motion_difficulty_analysis.csv")
    print("\nResults saved to /tmp/motion_difficulty_analysis.csv")