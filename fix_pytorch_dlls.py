#!/usr/bin/env python3
"""
Workaround for PyTorch DLL loading issue on Windows with AppControl policy

This script attempts to configure Python to work around Windows DLL blockers.
"""

import os
import sys
import subprocess

def unblock_dlls():
    """Unblock PyTorch DLLs on Windows"""
    torch_path = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib')
    
    if not os.path.exists(torch_path):
        print(f"Torch library path not found: {torch_path}")
        return False
    
    print(f"Unblocking DLLs in: {torch_path}")
    
    # PowerShell command to unblock files
    dlls = [f for f in os.listdir(torch_path) if f.endswith('.dll')]
    
    for dll in dlls:
        dll_path = os.path.join(torch_path, dll)
        try:
            # Use PowerShell to unblock the DLL
            subprocess.run([
                'powershell', '-Command',
                f"Unblock-File -Path '{dll_path}'"
            ], check=True, capture_output=True)
            print(f"  ✓ Unblocked: {dll}")
        except Exception as e:
            print(f"  ✗ Failed to unblock {dll}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("PyTorch DLL Unlocker")
    print("=" * 50)
    
    if unblock_dlls():
        print("\n✓ DLLs unblocked successfully!")
        print("\nTrying to import torch...")
        try:
            import torch
            print(f"✓ PyTorch {torch.__version__} is now working!")
        except Exception as e:
            print(f"✗ Still having issues: {e}")
            sys.exit(1)
    else:
        print("\n✗ Failed to unblock DLLs")
        print("\nAlternative: Run in PowerShell as Administrator:")
        print("  Get-ChildItem -Path 'C:\\Users\\Ethan\\AppData\\Local\\Python\\pythoncore-3.15-64\\Lib\\site-packages\\torch\\lib\\*.dll' | Unblock-File")
        sys.exit(1)
