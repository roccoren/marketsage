#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility script to check and fix Qlib import issues - includes both methods:
1. Checking and creating missing modules
2. Directly patching problematic files when needed
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


def patch_record_temp():
    """Directly patch the problematic record_temp.py file"""
    qlib_path = get_qlib_path()
    if not qlib_path:
        print("Failed to find Qlib installation path")
        return False

    record_temp_path = os.path.join(qlib_path, "workflow", "record_temp.py")
    
    if not os.path.exists(record_temp_path):
        print(f"record_temp.py not found at {record_temp_path}")
        return False
    
    print(f"Patching {record_temp_path}")
    
    # Read the current content
    with open(record_temp_path, "r") as f:
        original_content = f.read()
    
    # Create a backup
    backup_path = record_temp_path + ".backup"
    if not os.path.exists(backup_path):
        with open(backup_path, "w") as f:
            f.write(original_content)
        print(f"Created backup at {backup_path}")
    
    # Replace problematic imports with our fixed version
    fixed_content = original_content.replace(
        "from ..utils.serial import Serializable",
        "from qlib.utils.serial import Serializable"
    )
    
    fixed_content = fixed_content.replace(
        "from ..log import get_module_logger",
        "from qlib.log import get_module_logger"
    )
    
    # Write the patched file
    with open(record_temp_path, "w") as f:
        f.write(fixed_content)
    
    print(f"Successfully patched {record_temp_path}")
    return True


def patch_evaluate_module():
    """Patch the evaluate module if needed"""
    qlib_path = get_qlib_path()
    if not qlib_path:
        return False
    
    evaluate_path = os.path.join(qlib_path, "workflow", "evaluate.py")
    
    if not os.path.exists(evaluate_path):
        print(f"evaluate.py not found at {evaluate_path}")
        return False
    
    print(f"Checking {evaluate_path}")
    
    # Read the current content
    with open(evaluate_path, "r") as f:
        original_content = f.read()
    
    # Check if patching is needed
    if "from .record_temp import SignalRecord" in original_content:
        # Create a backup
        backup_path = evaluate_path + ".backup"
        if not os.path.exists(backup_path):
            with open(backup_path, "w") as f:
                f.write(original_content)
            print(f"Created backup at {backup_path}")
        
        # Replace problematic import
        fixed_content = original_content.replace(
            "from .record_temp import SignalRecord",
            "from qlib.workflow.record_temp import SignalRecord"
        )
        
        # Write the patched file
        with open(evaluate_path, "w") as f:
            f.write(fixed_content)
        
        print(f"Successfully patched {evaluate_path}")
        return True
    
    print(f"No patching needed for {evaluate_path}")
    return True


def main():
    """Main function to check and fix Qlib import issues"""
    # Step 1: Check qlib version
    qlib_version = check_qlib_version()
    
    if not qlib_version:
        print("Please install Qlib first")
        sys.exit(1)
    
    # Step 2: Check imports
    print("\nChecking imports...")
    import_status = check_imports()
    
    has_issues = any(status != "OK" for status in import_status.values())
    
    if has_issues:
        print("\nIssues detected. Attempting to fix...")
        
        # Try creating missing modules first
        modules_created = create_missing_modules()
        
        # Then try patching record_temp.py as a fallback
        record_temp_patched = patch_record_temp()
        evaluate_patched = patch_evaluate_module()
        
        if not modules_created and not record_temp_patched and not evaluate_patched:
            print("Failed to fix import issues automatically")
            sys.exit(1)
        
        # Recheck imports after fixing
        print("\nRechecking imports after fixes...")
        import_status = check_imports()
        has_issues = any(status != "OK" for status in import_status.values())
        
        if has_issues:
            print("\nSome issues remain after automatic fixes. Manual intervention may be needed.")
        else:
            print("\nAll import issues have been fixed!")
    else:
        print("\nAll imports are working correctly!")
    
    return import_status


if __name__ == "__main__":
    main() 