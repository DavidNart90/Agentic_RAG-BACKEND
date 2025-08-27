"""
Cleanup script for the TrackRealties AI Platform.

This script walks through the project directory structure and removes:
1. Python bytecode files (.pyc)
2. __pycache__ directories

Usage:
    python cleanup.py

This helps keep the project clean and can resolve issues related to stale bytecode files,
especially after switching branches or making significant code changes.
"""

import os
import shutil

def clean_project():
    """
    Clean Python bytecode files and __pycache__ directories from the project.
    
    Walks through the entire project directory structure and:
    1. Removes all .pyc files (Python bytecode)
    2. Removes all __pycache__ directories
    
    Returns:
        tuple: (int, int) - Count of removed files and directories
    """
    files_removed = 0
    dirs_removed = 0
    
    for root, dirs, files in os.walk("."):
        # Remove .pyc files
        for name in files:
            if name.endswith(".pyc"):
                try:
                    os.remove(os.path.join(root, name))
                    files_removed += 1
                except Exception as e:
                    print(f"Error removing {os.path.join(root, name)}: {e}")
        
        # Remove __pycache__ directories
        for name in dirs:
            if name == "__pycache__":
                try:
                    shutil.rmtree(os.path.join(root, name))
                    dirs_removed += 1
                except Exception as e:
                    print(f"Error removing {os.path.join(root, name)}: {e}")
    
    return files_removed, dirs_removed

if __name__ == "__main__":
    files_removed, dirs_removed = clean_project()
    print(f"Project cleaned successfully: {files_removed} .pyc files and {dirs_removed} __pycache__ directories removed.")
