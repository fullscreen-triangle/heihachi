#!/usr/bin/env python3
"""
Setup script for Heihachi audio analysis framework.

This script installs the Heihachi framework with all user experience enhancements,
including command-line tools, shell completions, and required dependencies.
"""

import os
import sys
import shutil
import argparse
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional

# Determine project root
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define dependency groups
dependencies = {
    "core": [
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
        "psutil>=5.9.0"
    ],
    "interactive": [
        "inquirer>=2.10.0",
        "tabulate>=0.9.0",
        "curses; platform_system!='Windows'"
    ],
    "gpu": [
        "torch>=1.10.0",
        "numba>=0.55.0"
    ],
    "dev": [
        "pytest>=7.0.0",
        "pylint>=2.13.0",
        "black>=22.3.0",
        "mypy>=0.950"
    ]
}

def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Setup Heihachi audio analysis framework')
    
    parser.add_argument('--install-dir', type=str,
                       help='Installation directory (default: system site-packages)')
    
    parser.add_argument('--dev', action='store_true',
                       help='Install development dependencies')
    
    parser.add_argument('--no-gpu', action='store_true',
                       help='Skip GPU acceleration dependencies')
    
    parser.add_argument('--no-interactive', action='store_true',
                       help='Skip interactive mode dependencies')
    
    parser.add_argument('--shell-completion', action='store_true',
                       help='Install shell completion scripts')
    
    parser.add_argument('--no-confirm', action='store_true',
                       help='Skip confirmation prompts')
    
    parser.add_argument('--venv', action='store_true',
                       help='Create and use a virtual environment')
    
    parser.add_argument('--venv-dir', type=str, default='.venv',
                       help='Virtual environment directory (default: .venv)')
    
    return parser.parse_args()

def create_virtual_environment(venv_dir: str) -> bool:
    """Create a virtual environment.
    
    Args:
        venv_dir: Virtual environment directory
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Creating virtual environment in {venv_dir}...")
    
    try:
        # Check if venv module is available
        import venv
        venv.create(venv_dir, with_pip=True)
        return True
    except ImportError:
        try:
            # Try using virtualenv if venv is not available
            result = subprocess.run(
                ["virtualenv", venv_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Error: Failed to create virtual environment.")
            print("Please install the 'venv' module or 'virtualenv' package.")
            return False

def activate_virtual_environment(venv_dir: str) -> bool:
    """Activate a virtual environment.
    
    Args:
        venv_dir: Virtual environment directory
        
    Returns:
        True if successful, False otherwise
    """
    venv_path = Path(venv_dir).resolve()
    
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate"
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        python_path = venv_path / "bin" / "python"
    
    if not python_path.exists():
        print(f"Error: Python executable not found in {python_path}")
        return False
    
    # Update sys.executable to use the virtual environment Python
    sys.executable = str(python_path)
    
    print(f"Activated virtual environment in {venv_dir}")
    print(f"To activate manually, run: source {activate_script}")
    
    return True

def install_dependencies(dependency_groups: List[str], no_confirm: bool = False) -> bool:
    """Install dependencies.
    
    Args:
        dependency_groups: List of dependency groups to install
        no_confirm: Skip confirmation prompt
        
    Returns:
        True if successful, False otherwise
    """
    # Collect dependencies from specified groups
    deps_to_install = []
    for group in dependency_groups:
        if group in dependencies:
            deps_to_install.extend(dependencies[group])
    
    if not deps_to_install:
        print("No dependencies to install.")
        return True
    
    # Prompt for confirmation
    if not no_confirm:
        print("\nWill install the following dependencies:")
        for dep in deps_to_install:
            print(f"  - {dep}")
        
        response = input("\nProceed with installation? [Y/n] ")
        if response.lower() not in ("", "y", "yes"):
            print("Installation aborted.")
            return False
    
    # Install dependencies
    print("\nInstalling dependencies...")
    
    try:
        # Use the current Python executable (which may be in a virtual environment)
        cmd = [sys.executable, "-m", "pip", "install"] + deps_to_install
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("Dependencies installed successfully.")
        return True
    
    except subprocess.SubprocessError as e:
        print(f"Error installing dependencies: {e}")
        print("You may need to install them manually:")
        print(f"  pip install {' '.join(deps_to_install)}")
        return False

def install_package(install_dir: Optional[str] = None, no_confirm: bool = False) -> bool:
    """Install the Heihachi package.
    
    Args:
        install_dir: Installation directory
        no_confirm: Skip confirmation prompt
        
    Returns:
        True if successful, False otherwise
    """
    # Prompt for confirmation
    if not no_confirm:
        print("\nReady to install Heihachi framework.")
        response = input("Proceed with installation? [Y/n] ")
        if response.lower() not in ("", "y", "yes"):
            print("Installation aborted.")
            return False
    
    print("\nInstalling Heihachi framework...")
    
    try:
        # Use the current Python executable (which may be in a virtual environment)
        cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
        
        if install_dir:
            cmd.extend(["--target", install_dir])
        
        # Run from project root
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=project_root
        )
        
        print("Heihachi framework installed successfully.")
        return True
    
    except subprocess.SubprocessError as e:
        print(f"Error installing package: {e}")
        print("You may need to install it manually:")
        if install_dir:
            print(f"  pip install -e . --target {install_dir}")
        else:
            print("  pip install -e .")
        return False

def install_shell_completions(no_confirm: bool = False) -> bool:
    """Install shell completion scripts.
    
    Args:
        no_confirm: Skip confirmation prompt
        
    Returns:
        True if successful, False otherwise
    """
    # Check if completion script generator exists
    completion_script = project_root / "scripts" / "completion.py"
    
    if not completion_script.exists():
        print("Error: Shell completion script generator not found.")
        return False
    
    # Prompt for confirmation
    if not no_confirm:
        print("\nWill install shell completion scripts.")
        response = input("Proceed with installation? [Y/n] ")
        if response.lower() not in ("", "y", "yes"):
            print("Shell completion installation aborted.")
            return False
    
    print("\nInstalling shell completion scripts...")
    
    try:
        # Create completions directory
        completions_dir = project_root / "completions"
        completions_dir.mkdir(exist_ok=True)
        
        # Run completion script generator
        cmd = [sys.executable, str(completion_script), "--output-dir", str(completions_dir)]
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print output
        print(result.stdout)
        
        return True
    
    except subprocess.SubprocessError as e:
        print(f"Error installing shell completions: {e}")
        return False

def create_command_symlink(no_confirm: bool = False) -> bool:
    """Create a symlink for the 'heihachi' command.
    
    Args:
        no_confirm: Skip confirmation prompt
        
    Returns:
        True if successful, False otherwise
    """
    # Determine target locations based on platform
    if platform.system() == "Windows":
        # On Windows, we need to create a batch file
        bin_dir = Path(os.path.dirname(sys.executable))
        symlink_path = bin_dir / "heihachi.bat"
        module_path = project_root / "src" / "cli" / "enhanced_cli.py"
        
        if not no_confirm:
            print(f"\nWill create command shortcut at {symlink_path}")
            response = input("Proceed? [Y/n] ")
            if response.lower() not in ("", "y", "yes"):
                print("Command shortcut creation aborted.")
                return False
        
        try:
            # Create a batch file
            with open(symlink_path, "w") as f:
                f.write(f'@echo off\r\n"{sys.executable}" "{module_path}" %*\r\n')
            
            print(f"Command shortcut created at {symlink_path}")
            return True
        
        except Exception as e:
            print(f"Error creating command shortcut: {e}")
            return False
    
    else:
        # On Unix-like systems, we can create a symlink
        if os.geteuid() == 0:  # Running as root
            bin_dir = Path("/usr/local/bin")
        else:
            # Use ~/.local/bin for non-root users
            bin_dir = Path.home() / ".local" / "bin"
            bin_dir.mkdir(exist_ok=True, parents=True)
        
        symlink_path = bin_dir / "heihachi"
        module_path = project_root / "src" / "cli" / "enhanced_cli.py"
        
        if not no_confirm:
            print(f"\nWill create command symlink at {symlink_path}")
            response = input("Proceed? [Y/n] ")
            if response.lower() not in ("", "y", "yes"):
                print("Command symlink creation aborted.")
                return False
        
        try:
            # Create a wrapper script
            wrapper_path = project_root / "scripts" / "heihachi"
            
            with open(wrapper_path, "w") as f:
                f.write(f"""#!/bin/sh
{sys.executable} {module_path} "$@"
""")
            
            # Make it executable
            os.chmod(wrapper_path, 0o755)
            
            # Create the symlink
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            
            symlink_path.symlink_to(wrapper_path)
            
            print(f"Command symlink created at {symlink_path}")
            
            # Add to PATH if needed
            if bin_dir == Path.home() / ".local" / "bin":
                print(f"\nMake sure {bin_dir} is in your PATH.")
                print("You may need to add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):")
                print(f'  export PATH="$PATH:{bin_dir}"')
            
            return True
        
        except Exception as e:
            print(f"Error creating command symlink: {e}")
            print("\nYou may need to create it manually:")
            print(f'  ln -s "{module_path}" "{symlink_path}"')
            print("  chmod +x {symlink_path}")
            return False

def main():
    """Main function."""
    args = parse_args()
    
    print("Heihachi Audio Analysis Framework Setup")
    print("======================================")
    
    # Create virtual environment if requested
    if args.venv:
        if not create_virtual_environment(args.venv_dir):
            return 1
        
        if not activate_virtual_environment(args.venv_dir):
            print("Failed to activate virtual environment. Continuing with system Python...")
    
    # Determine dependency groups to install
    dependency_groups = ["core"]
    
    if not args.no_interactive:
        dependency_groups.append("interactive")
    
    if not args.no_gpu:
        dependency_groups.append("gpu")
    
    if args.dev:
        dependency_groups.append("dev")
    
    # Install dependencies
    if not install_dependencies(dependency_groups, args.no_confirm):
        return 1
    
    # Install package
    if not install_package(args.install_dir, args.no_confirm):
        return 1
    
    # Install shell completions if requested
    if args.shell_completion:
        if not install_shell_completions(args.no_confirm):
            print("Failed to install shell completions.")
    
    # Create command symlink
    if not create_command_symlink(args.no_confirm):
        print("Failed to create command symlink.")
    
    print("\nHeihachi framework setup complete!")
    
    if args.venv:
        print(f"\nTo use Heihachi, activate the virtual environment:")
        if platform.system() == "Windows":
            print(f"  {args.venv_dir}\\Scripts\\activate")
        else:
            print(f"  source {args.venv_dir}/bin/activate")
    
    print("\nTry running:")
    print("  heihachi --help")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 