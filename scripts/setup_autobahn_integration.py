#!/usr/bin/env python3
"""
Heihachi Autobahn Integration Setup Script
Installs and configures the Autobahn integration for delegated probabilistic reasoning
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
import yaml
import json

def run_command(command, check=True, shell=False):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            check=check, 
            capture_output=True, 
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command) if isinstance(command, list) else command}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required for Autobahn integration")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")

def check_rust_installation():
    """Check if Rust is installed for the core engine"""
    try:
        result = run_command(["rustc", "--version"])
        print(f"✓ Rust detected: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Rust not found. Installing Rust...")
        return install_rust()

def install_rust():
    """Install Rust using rustup"""
    try:
        # Download and run rustup installer
        if os.name == 'nt':  # Windows
            installer_url = "https://win.rustup.rs/x86_64"
            installer_name = "rustup-init.exe"
        else:  # Unix-like
            run_command(["curl", "--proto", "=https", "--tlsv1.2", "-sSf", "https://sh.rustup.rs", "|", "sh", "-s", "--", "-y"], shell=True)
            
        # Source the cargo environment
        cargo_env = os.path.expanduser("~/.cargo/env")
        if os.path.exists(cargo_env):
            os.system(f"source {cargo_env}")
            
        print("✓ Rust installed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to install Rust: {e}")
        return False

def install_dependencies(requirements_file, description):
    """Install Python dependencies from requirements file"""
    if not os.path.exists(requirements_file):
        print(f"⚠ {requirements_file} not found, skipping {description}")
        return
        
    print(f"Installing {description}...")
    result = run_command([
        sys.executable, "-m", "pip", "install", "-r", requirements_file
    ], check=False)
    
    if result.returncode == 0:
        print(f"✓ {description} installed successfully")
    else:
        print(f"⚠ Some {description} may have failed to install")

def setup_autobahn_config():
    """Set up Autobahn integration configuration"""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # Check if autobahn_integration.yaml exists
    autobahn_config = config_dir / "autobahn_integration.yaml"
    if autobahn_config.exists():
        print("✓ Autobahn integration config already exists")
    else:
        print("⚠ Autobahn integration config not found")
        print("Please ensure configs/autobahn_integration.yaml is properly configured")
    
    # Check if fire_interface.yaml exists
    fire_config = config_dir / "fire_interface.yaml"
    if fire_config.exists():
        print("✓ Fire interface config already exists")
    else:
        print("⚠ Fire interface config not found")
        print("Please ensure configs/fire_interface.yaml is properly configured")

def setup_rust_bridge():
    """Set up the Rust bridge for Autobahn integration"""
    print("Setting up Rust bridge for Autobahn integration...")
    
    # Check if Cargo.toml exists for Rust components
    if os.path.exists("Cargo.toml"):
        print("Building Rust components...")
        result = run_command(["cargo", "build", "--release"], check=False)
        if result.returncode == 0:
            print("✓ Rust components built successfully")
        else:
            print("⚠ Rust build encountered issues")
    else:
        print("⚠ Cargo.toml not found, skipping Rust build")

def create_autobahn_service_check():
    """Create a service check script for Autobahn connectivity"""
    check_script = Path("scripts/check_autobahn_service.py")
    check_script.parent.mkdir(exist_ok=True)
    
    check_content = '''#!/usr/bin/env python3
"""
Autobahn Service Connectivity Check
Verifies that the Autobahn service is running and accessible
"""

import requests
import sys
import yaml
from pathlib import Path

def load_config():
    """Load Autobahn configuration"""
    config_path = Path("configs/autobahn_integration.yaml")
    if not config_path.exists():
        print("Error: autobahn_integration.yaml not found")
        return None
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_autobahn_service(config):
    """Check if Autobahn service is accessible"""
    host = config['autobahn']['host']
    port = config['autobahn']['port']
    timeout = config['autobahn']['timeout'] / 1000  # Convert to seconds
    
    try:
        url = f"http://{host}:{port}/health"
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            print(f"✓ Autobahn service is accessible at {host}:{port}")
            return True
        else:
            print(f"✗ Autobahn service returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to Autobahn service: {e}")
        return False

def main():
    config = load_config()
    if not config:
        sys.exit(1)
        
    if check_autobahn_service(config):
        print("Autobahn integration is ready!")
        sys.exit(0)
    else:
        print("Autobahn service is not accessible. Please ensure it's running.")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open(check_script, 'w') as f:
        f.write(check_content)
    
    # Make script executable on Unix-like systems
    if os.name != 'nt':
        os.chmod(check_script, 0o755)
    
    print(f"✓ Created Autobahn service check script: {check_script}")

def setup_development_environment():
    """Set up development environment for Autobahn integration"""
    print("Setting up development environment...")
    
    # Create necessary directories
    dirs_to_create = [
        "logs/autobahn",
        "cache/autobahn", 
        "data/fire_patterns",
        "data/consciousness_logs",
        "tests/autobahn_integration"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Development directories created")

def main():
    parser = argparse.ArgumentParser(description="Set up Heihachi Autobahn Integration")
    parser.add_argument("--skip-rust", action="store_true", help="Skip Rust installation and build")
    parser.add_argument("--skip-deps", action="store_true", help="Skip Python dependency installation")
    parser.add_argument("--dev", action="store_true", help="Set up development environment")
    parser.add_argument("--check-only", action="store_true", help="Only check service connectivity")
    
    args = parser.parse_args()
    
    print("🧠 Heihachi Autobahn Integration Setup")
    print("=" * 50)
    
    if args.check_only:
        # Just run the service check
        try:
            subprocess.run([sys.executable, "scripts/check_autobahn_service.py"], check=True)
        except subprocess.CalledProcessError:
            sys.exit(1)
        return
    
    # Check Python version
    check_python_version()
    
    # Check/install Rust if needed
    if not args.skip_rust:
        check_rust_installation()
    
    # Install Python dependencies
    if not args.skip_deps:
        install_dependencies("requirements-autobahn.txt", "Autobahn integration dependencies")
        install_dependencies("requirements-fire-interface.txt", "Fire interface dependencies")
    
    # Set up configurations
    setup_autobahn_config()
    
    # Set up Rust bridge
    if not args.skip_rust:
        setup_rust_bridge()
    
    # Create service check script
    create_autobahn_service_check()
    
    # Set up development environment
    if args.dev:
        setup_development_environment()
    
    print("\n" + "=" * 50)
    print("🎉 Autobahn Integration Setup Complete!")
    print("\nNext steps:")
    print("1. Ensure Autobahn service is running")
    print("2. Run: python scripts/check_autobahn_service.py")
    print("3. Test integration: python examples/autobahn_delegation_demo.py")
    print("4. Start Heihachi with Autobahn: heihachi start --with-autobahn")

if __name__ == "__main__":
    main() 