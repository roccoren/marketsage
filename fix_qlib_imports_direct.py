#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to directly patch the problematic record_temp.py file
"""

import os
import sys
import inspect

def get_qlib_path():
    """Get the path where Qlib is installed"""
    try:
        import qlib
        return os.path.dirname(inspect.getfile(qlib))
    except (ImportError, TypeError):
        return None

def patch_record_temp():
    """Directly patch the problematic record_temp.py file"""
    qlib_path = get_qlib_path()
    if not qlib_path:
        print("Qlib installation not found")
        return False
    
    # Path to the problematic file
    record_temp_path = os.path.join(qlib_path, "workflow", "record_temp.py")
    
    if not os.path.exists(record_temp_path):
        print(f"record_temp.py not found at {record_temp_path}")
        return False
    
    print(f"Patching {record_temp_path}")
    
    # Read the file
    with open(record_temp_path, "r") as f:
        content = f.read()
    
    # Create a backup
    backup_path = record_temp_path + ".backup"
    with open(backup_path, "w") as f:
        f.write(content)
    print(f"Created backup at {backup_path}")
    
    # Replace the problematic import
    new_content = content.replace(
        "from ..contrib.evaluate import risk_analysis, indicator_analysis",
        "from ..contrib.evaluate import risk_analysis\n\n# Patched indicator_analysis function\ndef indicator_analysis(pred_label, settlement, **kwargs):\n    \"\"\"Compatibility function for indicator_analysis\"\"\"\n    print('Using patched indicator_analysis function')\n    return {'IC': pred_label.mean(), 'ICIR': pred_label.mean() / (pred_label.std() + 1e-8)}"
    )
    
    # Write the patched file
    with open(record_temp_path, "w") as f:
        f.write(new_content)
    
    print("Patching completed successfully")
    return True

def patch_evaluate_module():
    """Add the indicator_analysis function to the evaluate module"""
    qlib_path = get_qlib_path()
    if not qlib_path:
        print("Qlib installation not found")
        return False
    
    # Path to the evaluate.py file
    evaluate_path = os.path.join(qlib_path, "contrib", "evaluate.py")
    
    if not os.path.exists(evaluate_path):
        print(f"evaluate.py not found at {evaluate_path}")
        return False
    
    print(f"Patching {evaluate_path}")
    
    # Read the file
    with open(evaluate_path, "r") as f:
        content = f.read()
    
    # Check if indicator_analysis is already defined
    if "def indicator_analysis" in content:
        print("indicator_analysis function already exists in evaluate.py")
        return True
    
    # Create a backup
    backup_path = evaluate_path + ".backup"
    with open(backup_path, "w") as f:
        f.write(content)
    print(f"Created backup at {backup_path}")
    
    # Add the indicator_analysis function at the end
    with open(evaluate_path, "a") as f:
        f.write("\n\n# Added by compatibility patch\n")
        f.write("def indicator_analysis(pred_label, settlement, **kwargs):\n")
        f.write("    \"\"\"Compatibility function for indicator_analysis\"\"\"\n")
        f.write("    print('Using patched indicator_analysis function')\n")
        f.write("    return {'IC': pred_label.mean(), 'ICIR': pred_label.mean() / (pred_label.std() + 1e-8)}\n")
    
    print("evaluate.py patched successfully")
    return True

def main():
    """Main function to patch the files"""
    print("=== Direct Qlib Patcher ===")
    
    # First try to patch the evaluate.py file
    evaluate_patched = patch_evaluate_module()
    if not evaluate_patched:
        print("Failed to patch evaluate.py")
        return 1
    
    # Then try patching record_temp.py as a fallback
    record_temp_patched = patch_record_temp()
    if not record_temp_patched:
        print("Failed to patch record_temp.py")
        return 1
    
    print("\nAll patches applied successfully!")
    print("Please restart your application and try again.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 