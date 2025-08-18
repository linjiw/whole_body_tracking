#!/usr/bin/env python3
"""
Batch register motion files to wandb registry without Isaac Lab simulation.
This script directly processes NPZ files if they exist, or creates minimal motion data.
"""

import os
import glob
import numpy as np
import wandb
from pathlib import Path

def register_motion_to_wandb(motion_name: str, npz_data: dict):
    """Register a motion to wandb registry"""
    # Create temporary npz file
    temp_file = f"/tmp/{motion_name}.npz"
    np.savez(temp_file, **npz_data)
    
    # Initialize wandb run
    run = wandb.init(project="csv_to_npz", name=motion_name, reinit=True)
    print(f"[INFO]: Logging motion to wandb: {motion_name}")
    
    # Create and log artifact
    REGISTRY_NAME = "motions"
    artifact = wandb.Artifact(name=motion_name, type="motions")
    artifact.add_file(local_path=temp_file)
    logged_artifact = run.log_artifact(artifact)
    
    # Link artifact to registry collection
    target_path = f"wandb-registry-{REGISTRY_NAME}/{motion_name}"
    run.link_artifact(artifact=artifact, target_path=target_path)
    print(f"[INFO]: Motion linked to wandb registry: {REGISTRY_NAME}/{motion_name}")
    
    # Clean up
    wandb.finish()
    os.remove(temp_file)

def create_minimal_motion_data(csv_file: str):
    """Create minimal motion data from CSV file"""
    # Load CSV data
    motion_data = np.loadtxt(csv_file, delimiter=",")
    
    # Extract basic components (simplified version)
    frames = motion_data.shape[0]
    
    # Create minimal structure matching the expected format
    return {
        "fps": [30],  # Default FPS
        "joint_pos": motion_data[:, 7:],  # Joint positions
        "joint_vel": np.gradient(motion_data[:, 7:], axis=0),  # Estimated velocities
        "body_pos_w": motion_data[:, :3],  # Body positions
        "body_quat_w": motion_data[:, 3:7],  # Body quaternions
        "body_lin_vel_w": np.gradient(motion_data[:, :3], axis=0),  # Linear velocities
        "body_ang_vel_w": np.zeros((frames, 3)),  # Angular velocities (simplified)
    }

def main():
    # Directory containing CSV files
    csv_dir = "/home/linji/nfs/LAFAN1_Retargeting_Dataset/g1"
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in sorted(csv_files):
        filename = Path(csv_file).stem
        print(f"Processing: {filename}")
        
        try:
            # Create motion data from CSV
            motion_data = create_minimal_motion_data(csv_file)
            
            # Register to wandb
            register_motion_to_wandb(filename, motion_data)
            
            print(f"✅ Completed: {filename}")
        
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
        
        print("---")
    
    print("🎉 All CSV files processed!")

if __name__ == "__main__":
    main()