#!/usr/bin/env python3
"""
Script to update model parameters for reproducibility.
This script sets a fixed random_state in all ensemble.json files.
"""

import os
import json
import glob

def update_model_params(directory='data/csv/model_params'):
    """Update model parameter files to set fixed random_state."""
    # Find all ensemble.json files
    ensemble_files = glob.glob(f"{directory}/**/ensemble.json", recursive=True)
    
    if not ensemble_files:
        print(f"No ensemble.json files found in {directory}")
        return
    
    updated_count = 0
    
    for file_path in ensemble_files:
        try:
            # Read existing config
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Update random_state in all models
            modified = False
            if 'models' in config:
                for model in config['models']:
                    if 'params' in model and 'random_state' in model['params']:
                        if model['params']['random_state'] is None:
                            model['params']['random_state'] = 42
                            modified = True
            
            # Save updated config if modified
            if modified:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                updated_count += 1
                print(f"Updated {file_path}")
        
        except Exception as e:
            print(f"Error updating {file_path}: {e}")
    
    print(f"\nUpdated {updated_count} out of {len(ensemble_files)} ensemble.json files.")

if __name__ == "__main__":
    update_model_params()
    print("\nModel parameter update completed.")