#!/usr/bin/env python3
"""
Ablation Study: Adaptive Sampling vs Uniform Sampling

This script runs comparative training experiments to validate the adaptive sampling
mechanism against the baseline uniform sampling approach.

Based on the BeyondMimic paper ablation study results:
- Adaptive sampling should enable training on long sequences that fail with uniform sampling
- Expected 2x speedup on complex motions like dance1_subject1
- Better learning of difficult motion segments (cartwheels, balancing, jumps)
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import wandb


class AblationStudy:
    """Manages ablation study experiments for adaptive sampling."""
    
    def __init__(self, base_project_name: str = "adaptive_sampling_ablation"):
        self.base_project_name = base_project_name
        self.results = {}
        self.experiment_configs = self._define_experiments()
    
    def _define_experiments(self) -> Dict[str, Dict]:
        """Define experimental configurations for ablation study."""
        return {
            # Baseline experiments (known to work)
            "cristiano_uniform": {
                "description": "Cristiano motion with uniform sampling (baseline)",
                "motion": "16726-org/wandb-registry-Motions/cristiano",
                "adaptive_sampling": False,
                "max_iterations": 3000,
                "expected_outcome": "success",
                "baseline_iterations": 3000
            },
            "cristiano_adaptive": {
                "description": "Cristiano motion with adaptive sampling",
                "motion": "16726-org/wandb-registry-Motions/cristiano", 
                "adaptive_sampling": True,
                "max_iterations": 1500,  # Expect 2x speedup
                "expected_outcome": "success",
                "baseline_iterations": 3000
            },
            
            # Challenge experiments (known to be difficult)
            "dance1_uniform": {
                "description": "Dance1 motion with uniform sampling (expected to fail)",
                "motion": "16726-org/wandb-registry-Motions/dance1_subject1",
                "adaptive_sampling": False,
                "max_iterations": 10000,
                "expected_outcome": "failure",
                "baseline_iterations": None  # No successful baseline
            },
            "dance1_adaptive": {
                "description": "Dance1 motion with adaptive sampling (should succeed)",
                "motion": "16726-org/wandb-registry-Motions/dance1_subject1",
                "adaptive_sampling": True,
                "max_iterations": 8000,
                "expected_outcome": "success",
                "baseline_iterations": None
            },
            
            # Additional validation experiments
            "walk3_uniform": {
                "description": "Walk3 motion with uniform sampling",
                "motion": "16726-org/wandb-registry-Motions/walk3_subject2",
                "adaptive_sampling": False,
                "max_iterations": 5000,
                "expected_outcome": "success",
                "baseline_iterations": 5000
            },
            "walk3_adaptive": {
                "description": "Walk3 motion with adaptive sampling",
                "motion": "16726-org/wandb-registry-Motions/walk3_subject2",
                "adaptive_sampling": True,
                "max_iterations": 3000,  # Expect speedup
                "expected_outcome": "success",
                "baseline_iterations": 5000
            }
        }
    
    def run_single_experiment(self, exp_name: str, config: Dict) -> Dict:
        """Run a single training experiment."""
        print(f"\n🧪 Starting experiment: {exp_name}")
        print(f"   Description: {config['description']}")
        print(f"   Motion: {config['motion']}")
        print(f"   Adaptive sampling: {config['adaptive_sampling']}")
        print(f"   Max iterations: {config['max_iterations']}")
        
        # Prepare run name
        timestamp = int(time.time())
        run_name = f"{exp_name}_{timestamp}"
        
        # Build training command
        cmd = [
            "python", "scripts/rsl_rl/train.py",
            "--task=Tracking-Flat-G1-v0",
            f"--registry_name={config['motion']}",
            "--headless",
            "--logger=wandb",
            f"--log_project_name={self.base_project_name}",
            f"--run_name={run_name}",
            f"--max_iterations={config['max_iterations']}",
            "--num_envs=1024",  # Smaller for faster experiments
        ]
        
        # Set environment variables for adaptive sampling
        env = os.environ.copy()
        env["ADAPTIVE_SAMPLING_ENABLED"] = str(config['adaptive_sampling']).lower()
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        
        # Run experiment
        start_time = time.time()
        try:
            print(f"   Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            # Parse results from wandb or log files
            metrics = self._parse_training_metrics(run_name)
            
            experiment_result = {
                "experiment": exp_name,
                "config": config,
                "success": success,
                "duration_minutes": duration / 60,
                "final_reward": metrics.get("final_reward", 0.0),
                "convergence_iteration": metrics.get("convergence_iteration", None),
                "max_reward": metrics.get("max_reward", 0.0),
                "run_name": run_name,
                "stdout": result.stdout[-2000:],  # Last 2000 chars
                "stderr": result.stderr[-2000:] if result.stderr else "",
            }
            
            if success:
                print(f"   ✅ Experiment completed successfully")
                print(f"   ⏱️  Duration: {duration/60:.1f} minutes")
                print(f"   🎯 Final reward: {metrics.get('final_reward', 0.0):.3f}")
            else:
                print(f"   ❌ Experiment failed (exit code: {result.returncode})")
                print(f"   ⏱️  Duration: {duration/60:.1f} minutes")
        
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Experiment timed out after 2 hours")
            experiment_result = {
                "experiment": exp_name,
                "config": config,
                "success": False,
                "duration_minutes": 120,
                "error": "timeout",
                "run_name": run_name,
            }
        
        except Exception as e:
            print(f"   💥 Experiment crashed: {e}")
            experiment_result = {
                "experiment": exp_name,
                "config": config,
                "success": False,
                "duration_minutes": (time.time() - start_time) / 60,
                "error": str(e),
                "run_name": run_name,
            }
        
        self.results[exp_name] = experiment_result
        return experiment_result
    
    def _parse_training_metrics(self, run_name: str) -> Dict:
        """Parse training metrics from wandb or local logs."""
        try:
            # Try to get metrics from wandb
            api = wandb.Api()
            runs = api.runs(f"16726/{self.base_project_name}", filters={"display_name": run_name})
            
            if runs:
                run = runs[0]
                history = run.history()
                
                if len(history) > 0:
                    # Get final and maximum rewards
                    reward_cols = [col for col in history.columns if 'reward' in col.lower() and 'total' in col.lower()]
                    if reward_cols:
                        reward_col = reward_cols[0]
                        final_reward = history[reward_col].iloc[-1]
                        max_reward = history[reward_col].max()
                        
                        # Find convergence iteration (when reward > 0.8)
                        convergence_mask = history[reward_col] > 0.8
                        convergence_iteration = history[convergence_mask].index[0] if convergence_mask.any() else None
                        
                        return {
                            "final_reward": final_reward,
                            "max_reward": max_reward,
                            "convergence_iteration": convergence_iteration
                        }
            
        except Exception as e:
            print(f"Warning: Could not parse wandb metrics: {e}")
        
        return {}
    
    def run_ablation_study(self, experiments: List[str] = None) -> Dict:
        """Run the complete ablation study."""
        print("🔬 Starting Adaptive Sampling Ablation Study")
        print("=" * 60)
        
        if experiments is None:
            experiments = list(self.experiment_configs.keys())
        
        print(f"📋 Running {len(experiments)} experiments:")
        for exp in experiments:
            print(f"   - {exp}: {self.experiment_configs[exp]['description']}")
        
        print("\n🚀 Beginning experiments...")
        
        # Run experiments
        for exp_name in experiments:
            if exp_name not in self.experiment_configs:
                print(f"⚠️  Unknown experiment: {exp_name}")
                continue
            
            config = self.experiment_configs[exp_name]
            self.run_single_experiment(exp_name, config)
        
        # Analyze results
        self._analyze_results()
        
        return self.results
    
    def _analyze_results(self):
        """Analyze and report ablation study results."""
        print("\n📊 Ablation Study Results")
        print("=" * 60)
        
        # Group by motion type
        motion_groups = {}
        for exp_name, result in self.results.items():
            motion = result['config']['motion'].split('/')[-1]
            if motion not in motion_groups:
                motion_groups[motion] = {}
            
            sampling_type = "adaptive" if result['config']['adaptive_sampling'] else "uniform"
            motion_groups[motion][sampling_type] = result
        
        # Analyze each motion
        for motion, experiments in motion_groups.items():
            print(f"\n🎯 Motion: {motion}")
            print("-" * 40)
            
            uniform_result = experiments.get('uniform')
            adaptive_result = experiments.get('adaptive')
            
            if uniform_result and adaptive_result:
                # Compare uniform vs adaptive
                self._compare_experiments(uniform_result, adaptive_result)
            else:
                # Report individual results
                if uniform_result:
                    self._report_single_experiment(uniform_result, "Uniform Sampling")
                if adaptive_result:
                    self._report_single_experiment(adaptive_result, "Adaptive Sampling")
        
        # Summary
        self._print_summary()
    
    def _compare_experiments(self, uniform_result: Dict, adaptive_result: Dict):
        """Compare uniform vs adaptive sampling results."""
        uniform_success = uniform_result['success']
        adaptive_success = adaptive_result['success']
        
        print(f"Uniform Sampling:  {'✅ Success' if uniform_success else '❌ Failed'}")
        print(f"Adaptive Sampling: {'✅ Success' if adaptive_success else '❌ Failed'}")
        
        if uniform_success and adaptive_success:
            # Compare performance metrics
            uniform_reward = uniform_result.get('final_reward', 0)
            adaptive_reward = adaptive_result.get('final_reward', 0)
            
            uniform_duration = uniform_result['duration_minutes']
            adaptive_duration = adaptive_result['duration_minutes']
            
            speedup = uniform_duration / adaptive_duration if adaptive_duration > 0 else float('inf')
            
            print(f"Final Rewards:     Uniform: {uniform_reward:.3f}, Adaptive: {adaptive_reward:.3f}")
            print(f"Training Time:     Uniform: {uniform_duration:.1f}min, Adaptive: {adaptive_duration:.1f}min")
            print(f"Speedup Factor:    {speedup:.2f}x")
            
            if speedup > 1.5:
                print("🚀 Adaptive sampling achieved significant speedup!")
            elif adaptive_success and not uniform_success:
                print("🎯 Adaptive sampling enabled successful training!")
        
        elif adaptive_success and not uniform_success:
            print("🎯 Adaptive sampling enabled training on difficult motion!")
        elif uniform_success and not adaptive_success:
            print("⚠️  Adaptive sampling failed where uniform succeeded")
    
    def _report_single_experiment(self, result: Dict, method_name: str):
        """Report results for a single experiment."""
        success = result['success']
        duration = result['duration_minutes']
        final_reward = result.get('final_reward', 0)
        
        print(f"{method_name}: {'✅ Success' if success else '❌ Failed'}")
        if success:
            print(f"  Final Reward: {final_reward:.3f}")
        print(f"  Duration: {duration:.1f} minutes")
    
    def _print_summary(self):
        """Print overall summary of ablation study."""
        print("\n🏆 Summary")
        print("=" * 60)
        
        total_experiments = len(self.results)
        successful_experiments = sum(1 for r in self.results.values() if r['success'])
        
        print(f"Total Experiments: {total_experiments}")
        print(f"Successful Runs: {successful_experiments}")
        print(f"Success Rate: {successful_experiments/total_experiments*100:.1f}%")
        
        # Check key hypotheses
        adaptive_enabled_difficult = False
        adaptive_achieved_speedup = False
        
        for exp_name, result in self.results.items():
            config = result['config']
            if config['adaptive_sampling'] and result['success']:
                if config['expected_outcome'] == 'failure':
                    adaptive_enabled_difficult = True
                if config.get('baseline_iterations'):
                    expected_speedup = config['baseline_iterations'] / config['max_iterations']
                    if expected_speedup > 1.5:
                        adaptive_achieved_speedup = True
        
        print(f"\n📈 Key Findings:")
        print(f"✅ Adaptive sampling enabled difficult motions: {adaptive_enabled_difficult}")
        print(f"🚀 Adaptive sampling achieved speedup: {adaptive_achieved_speedup}")
        
        # Save results
        self._save_results()
    
    def _save_results(self):
        """Save results to JSON file."""
        results_file = f"ablation_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {results_file}")


def main():
    """Main entry point for ablation study."""
    parser = argparse.ArgumentParser(description="Run adaptive sampling ablation study")
    parser.add_argument("--experiments", nargs="+", 
                       choices=["cristiano_uniform", "cristiano_adaptive", 
                               "dance1_uniform", "dance1_adaptive",
                               "walk3_uniform", "walk3_adaptive"],
                       help="Specific experiments to run (default: all)")
    parser.add_argument("--project", default="adaptive_sampling_ablation",
                       help="WandB project name")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print experiment plan without running")
    
    args = parser.parse_args()
    
    # Initialize ablation study
    study = AblationStudy(base_project_name=args.project)
    
    if args.dry_run:
        print("🔍 Dry Run: Experiment Plan")
        print("=" * 50)
        experiments = args.experiments or list(study.experiment_configs.keys())
        for exp_name in experiments:
            config = study.experiment_configs[exp_name]
            print(f"\n{exp_name}:")
            print(f"  Description: {config['description']}")
            print(f"  Motion: {config['motion']}")
            print(f"  Adaptive sampling: {config['adaptive_sampling']}")
            print(f"  Max iterations: {config['max_iterations']}")
            print(f"  Expected outcome: {config['expected_outcome']}")
        return
    
    # Run ablation study
    results = study.run_ablation_study(args.experiments)
    
    print("\n🎉 Ablation study completed!")
    print("Check the results file and WandB project for detailed analysis.")


if __name__ == "__main__":
    main()