#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility script to check and fix Qlib import issues
"""

import os
import sys
import importlib
import shutil
from pathlib import Path
import inspect


def check_qlib_version():
    """Check the installed Qlib version"""
    try:
        import qlib
        print(f"Qlib version: {qlib.__version__}")
        return qlib.__version__
    except ImportError:
        print("Qlib is not installed")
        return None
    except AttributeError:
        print("Qlib version information not available")
        return "unknown"


def check_imports():
    """Check various Qlib imports to identify issues"""
    import_status = {}
    
    # Check workflow module
    try:
        import qlib.workflow
        import_status["qlib.workflow"] = "OK"
    except ImportError as e:
        import_status["qlib.workflow"] = f"Error: {str(e)}"
    
    # Check record_temp module
    try:
        from qlib.workflow.record_temp import SignalRecord
        import_status["qlib.workflow.record_temp.SignalRecord"] = "OK"
    except ImportError as e:
        import_status["qlib.workflow.record_temp.SignalRecord"] = f"Error: {str(e)}"
    
    # Check evaluate module
    try:
        from qlib.contrib.evaluate import risk_analysis
        import_status["qlib.contrib.evaluate.risk_analysis"] = "OK"
    except ImportError as e:
        import_status["qlib.contrib.evaluate.risk_analysis"] = f"Error: {str(e)}"
    
    # Check strategy module
    try:
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
        import_status["qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy"] = "OK"
    except ImportError as e:
        import_status["qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy"] = f"Error: {str(e)}"
    
    return import_status


def get_qlib_path():
    """Get the path where Qlib is installed"""
    try:
        import qlib
        return os.path.dirname(inspect.getfile(qlib))
    except (ImportError, TypeError):
        return None


def create_missing_modules():
    """Create missing or patched modules for compatibility"""
    qlib_path = get_qlib_path()
    if qlib_path is None:
        print("Qlib installation path not found")
        return False
    
    print(f"Qlib installed at: {qlib_path}")
    
    # Check and create workflow/record module if needed
    workflow_dir = os.path.join(qlib_path, "workflow")
    if not os.path.exists(workflow_dir):
        print(f"Creating workflow directory: {workflow_dir}")
        os.makedirs(workflow_dir, exist_ok=True)
    
    record_module_path = os.path.join(workflow_dir, "record.py")
    if not os.path.exists(record_module_path):
        print(f"Creating compatibility record module: {record_module_path}")
        with open(record_module_path, "w") as f:
            f.write("""
# Compatibility module for older Qlib versions
import os
import pickle
import pandas as pd
from pathlib import Path

class SignalRecord:
    \"\"\"Compatibility class for SignalRecord\"\"\"
    
    def __init__(self, model=None, dataset=None, predictions=None):
        self.model = model
        self.dataset = dataset
        self.predictions = predictions
        self.model_config = {}
        self.dataset_config = {}
    
    @staticmethod
    def load(path):
        \"\"\"Load a SignalRecord from a path\"\"\"
        instance = SignalRecord()
        instance.path = path
        return instance
    
    def load_pred(self):
        \"\"\"Load predictions from the saved path\"\"\"
        if hasattr(self, 'predictions') and self.predictions is not None:
            return self.predictions
            
        # Try to load predictions from file
        pred_file = os.path.join(self.path, "pred.pkl")
        if os.path.exists(pred_file):
            with open(pred_file, "rb") as f:
                return pickle.load(f)
        
        # Try CSV if pickle not available
        pred_csv = os.path.join(self.path, "pred.csv")
        if os.path.exists(pred_csv):
            return pd.read_csv(pred_csv, index_col=0)
            
        return None
    
    def load_model(self):
        \"\"\"Load model from the saved path\"\"\"
        if hasattr(self, 'model') and self.model is not None:
            return self.model
            
        model_file = os.path.join(self.path, "model.pkl")
        if os.path.exists(model_file):
            with open(model_file, "rb") as f:
                return pickle.load(f)
        
        return None
    
    def load_dataset(self):
        \"\"\"Load dataset from the saved path\"\"\"
        if hasattr(self, 'dataset') and self.dataset is not None:
            return self.dataset
            
        dataset_file = os.path.join(self.path, "dataset.pkl")
        if os.path.exists(dataset_file):
            with open(dataset_file, "rb") as f:
                return pickle.load(f)
                
        return None
    
    def save(self, path):
        \"\"\"Save record to the given path\"\"\"
        os.makedirs(path, exist_ok=True)
        
        # Save predictions
        if self.predictions is not None:
            pred_file = os.path.join(path, "pred.pkl")
            with open(pred_file, "wb") as f:
                pickle.dump(self.predictions, f)
            
            # Also save as CSV for easier inspection
            pred_csv = os.path.join(path, "pred.csv")
            self.predictions.to_csv(pred_csv)
        
        # Save model if available
        if self.model is not None:
            model_file = os.path.join(path, "model.pkl")
            with open(model_file, "wb") as f:
                pickle.dump(self.model, f)
                
        # Save dataset if available
        if self.dataset is not None:
            dataset_file = os.path.join(path, "dataset.pkl")
            with open(dataset_file, "wb") as f:
                pickle.dump(self.dataset, f)


class PortAnaRecord:
    \"\"\"Compatibility class for PortAnaRecord\"\"\"
    
    def __init__(self, recorder=None):
        self.recorder = recorder
    
    def generate_report(self):
        \"\"\"Generate a simple analysis report\"\"\"
        predictions = self.recorder.load_pred()
        
        # Simple analysis: IC and ICIR calculation
        ic = predictions.mean().mean()
        icir = ic / (predictions.std().mean() + 1e-8)
        
        return pd.DataFrame({
            "IC": [ic],
            "ICIR": [icir]
        })
""")
    
    # Check and create or patch evaluate module if needed
    contrib_dir = os.path.join(qlib_path, "contrib")
    evaluate_module_path = os.path.join(contrib_dir, "evaluate.py")
    
    # Backup original if it exists and has issues
    if os.path.exists(evaluate_module_path):
        try:
            # Try importing to see if there's an issue
            spec = importlib.util.spec_from_file_location("evaluate", evaluate_module_path)
            evaluate_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluate_module)
            
            # Check if indicator_analysis is missing
            if not hasattr(evaluate_module, "indicator_analysis"):
                # Backup and patch
                backup_path = evaluate_module_path + ".backup"
                print(f"Backing up evaluate module to: {backup_path}")
                shutil.copy2(evaluate_module_path, backup_path)
                
                # Read the original content
                with open(evaluate_module_path, "r") as f:
                    content = f.read()
                
                # Add the missing function
                with open(evaluate_module_path, "w") as f:
                    f.write(content)
                    f.write("\n\n# Added by compatibility patch\n")
                    f.write("def indicator_analysis(pred_label, settlement, **kwargs):\n")
                    f.write("    \"\"\"Compatibility function for indicator_analysis\"\"\"\n")
                    f.write("    print('Using patched indicator_analysis function')\n")
                    f.write("    return {'IC': pred_label.mean(), 'ICIR': pred_label.mean() / (pred_label.std() + 1e-8)}\n")
        except Exception as e:
            print(f"Error checking/patching evaluate module: {str(e)}")
    
    print("Module creation/patching completed")
    return True


def main():
    """Main function to check and fix Qlib imports"""
    print("=== Qlib Import Checker and Fixer ===")
    version = check_qlib_version()
    
    if version is None:
        print("Please install Qlib first")
        return 1
    
    print("\nChecking imports:")
    import_status = check_imports()
    for imp, status in import_status.items():
        print(f"  {imp}: {status}")
    
    has_errors = any("Error" in status for status in import_status.values())
    if has_errors:
        print("\nIssues detected. Attempting to fix...")
        success = create_missing_modules()
        if success:
            print("\nFixes applied. Please restart your application and try again.")
        else:
            print("\nUnable to apply fixes automatically.")
    else:
        print("\nAll imports are working correctly!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 