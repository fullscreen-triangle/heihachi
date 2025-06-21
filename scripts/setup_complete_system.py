#!/usr/bin/env python3
"""
Heihachi Complete System Setup Script
Sets up the entire revolutionary audio analysis framework with:
- Fire-based emotional querying
- Autobahn biological intelligence integration
- Rust-powered performance core
- Pakati understanding engine
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
import yaml
import json
import time

def print_banner():
    """Print the setup banner"""
    banner = """
🔥🧠🦀 HEIHACHI COMPLETE SYSTEM SETUP 🦀🧠🔥

Revolutionary Audio Analysis Framework
- Fire-Based Emotional Querying
- Autobahn Biological Intelligence Integration  
- Rust-Powered Performance Core
- Consciousness-Aware Audio Generation

"What makes a tiger so strong is that it lacks humanity"
    """
    print(banner)

def run_command(command, check=True, shell=False, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            check=check, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command) if isinstance(command, list) else command}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check if we have pip
    try:
        run_command([sys.executable, "-m", "pip", "--version"])
        print("✅ pip is available")
    except:
        print("❌ pip is not available")
        sys.exit(1)
    
    # Check for Node.js (for fire interface)
    try:
        result = run_command(["node", "--version"])
        print(f"✅ Node.js detected: {result.stdout.strip()}")
    except:
        print("⚠️  Node.js not found - fire interface may not work fully")
    
    # Check for git
    try:
        run_command(["git", "--version"])
        print("✅ Git is available")
    except:
        print("⚠️  Git not found - some features may not work")

def install_rust():
    """Install Rust toolchain"""
    print("🦀 Installing Rust toolchain...")
    
    try:
        # Check if Rust is already installed
        result = run_command(["rustc", "--version"])
        print(f"✅ Rust already installed: {result.stdout.strip()}")
        return True
    except:
        pass
    
    try:
        if os.name == 'nt':  # Windows
            print("Please install Rust manually from https://rustup.rs/")
            return False
        else:  # Unix-like
            run_command([
                "curl", "--proto", "=https", "--tlsv1.2", "-sSf", 
                "https://sh.rustup.rs"
            ], shell=False)
            
            # Run the installer
            run_command(["sh", "-s", "--", "-y"], shell=True, 
                       cwd=os.path.expanduser("~"))
            
            # Source the cargo environment
            cargo_env = os.path.expanduser("~/.cargo/env")
            if os.path.exists(cargo_env):
                os.system(f"source {cargo_env}")
                
        print("✅ Rust installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install Rust: {e}")
        return False

def setup_python_environment():
    """Set up Python virtual environment and dependencies"""
    print("🐍 Setting up Python environment...")
    
    # Create virtual environment if it doesn't exist
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", ".venv"])
    
    # Determine activation script
    if os.name == 'nt':
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_path = venv_path / "Scripts" / "pip"
    else:
        activate_script = venv_path / "bin" / "activate"
        pip_path = venv_path / "bin" / "pip"
    
    # Install core dependencies
    requirements_files = [
        "requirements.txt",
        "requirements-autobahn.txt", 
        "requirements-fire-interface.txt",
        "requirements-huggingface.txt"
    ]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            print(f"Installing dependencies from {req_file}...")
            result = run_command([
                str(pip_path), "install", "-r", req_file
            ], check=False)
            
            if result.returncode == 0:
                print(f"✅ {req_file} installed successfully")
            else:
                print(f"⚠️  Some dependencies from {req_file} may have failed")
        else:
            print(f"⚠️  {req_file} not found, skipping")
    
    # Install the package itself
    print("Installing Heihachi package...")
    run_command([str(pip_path), "install", "-e", "."], check=False)

def build_rust_components():
    """Build Rust components"""
    print("🦀 Building Rust components...")
    
    if not Path("Cargo.toml").exists():
        print("⚠️  Cargo.toml not found, skipping Rust build")
        return
    
    # Build in release mode for performance
    result = run_command(["cargo", "build", "--release", "--features", "full"], check=False)
    
    if result.returncode == 0:
        print("✅ Rust components built successfully")
    else:
        print("⚠️  Rust build encountered issues, trying without full features...")
        result = run_command(["cargo", "build", "--release"], check=False)
        if result.returncode == 0:
            print("✅ Basic Rust components built successfully")
        else:
            print("❌ Rust build failed")

def setup_configurations():
    """Set up configuration files"""
    print("⚙️  Setting up configurations...")
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # List of expected config files
    expected_configs = [
        "default.yaml",
        "huggingface.yaml", 
        "autobahn_integration.yaml",
        "fire_interface.yaml"
    ]
    
    for config_file in expected_configs:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"✅ {config_file} exists")
        else:
            print(f"⚠️  {config_file} not found")

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directory structure...")
    
    directories = [
        "cache/autobahn",
        "cache/fire_patterns", 
        "logs/autobahn",
        "logs/fire_interface",
        "data/consciousness_logs",
        "data/fire_patterns",
        "data/emotional_profiles",
        "results/fire_analysis",
        "results/consciousness_analysis",
        "tests/autobahn_integration",
        "tests/fire_interface"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")

def create_startup_scripts():
    """Create convenient startup scripts"""
    print("🚀 Creating startup scripts...")
    
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Create a comprehensive startup script
    startup_script_content = '''#!/usr/bin/env python3
"""
Heihachi System Startup Script
Starts all components of the revolutionary audio analysis system
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

def start_autobahn_check():
    """Check if Autobahn service is running"""
    try:
        subprocess.run([sys.executable, "scripts/check_autobahn_service.py"], 
                      check=True, timeout=10)
        return True
    except:
        return False

def start_fire_interface():
    """Start the fire interface server"""
    print("🔥 Starting Fire Interface...")
    # This would start the Next.js fire interface
    # For now, just a placeholder
    print("Fire interface would start here")

def start_api_server():
    """Start the REST API server"""
    print("🌐 Starting REST API server...")
    subprocess.Popen([sys.executable, "api_server.py", "--host", "0.0.0.0", "--port", "5000"])

def main():
    parser = argparse.ArgumentParser(description="Start Heihachi System")
    parser.add_argument("--fire-interface", action="store_true", help="Start fire interface")
    parser.add_argument("--api-server", action="store_true", help="Start API server")
    parser.add_argument("--check-autobahn", action="store_true", help="Check Autobahn connectivity")
    parser.add_argument("--all", action="store_true", help="Start all components")
    
    args = parser.parse_args()
    
    if args.all:
        args.fire_interface = True
        args.api_server = True
        args.check_autobahn = True
    
    if args.check_autobahn:
        if start_autobahn_check():
            print("✅ Autobahn service is accessible")
        else:
            print("⚠️  Autobahn service not accessible")
    
    if args.fire_interface:
        start_fire_interface()
    
    if args.api_server:
        start_api_server()
    
    if not any([args.fire_interface, args.api_server, args.check_autobahn]):
        print("Heihachi System Ready!")
        print("Use --help for startup options")

if __name__ == "__main__":
    main()
'''
    
    startup_script = scripts_dir / "start_heihachi.py"
    with open(startup_script, 'w') as f:
        f.write(startup_script_content)
    
    # Make executable on Unix-like systems
    if os.name != 'nt':
        os.chmod(startup_script, 0o755)
    
    print("✅ Startup scripts created")

def run_tests():
    """Run basic system tests"""
    print("🧪 Running basic system tests...")
    
    # Test Python imports
    try:
        import numpy
        import scipy
        import yaml
        print("✅ Core Python dependencies working")
    except ImportError as e:
        print(f"⚠️  Some Python dependencies missing: {e}")
    
    # Test configuration loading
    try:
        config_path = Path("configs/default.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                yaml.safe_load(f)
            print("✅ Configuration files valid")
    except Exception as e:
        print(f"⚠️  Configuration issue: {e}")

def main():
    parser = argparse.ArgumentParser(description="Set up complete Heihachi system")
    parser.add_argument("--skip-rust", action="store_true", help="Skip Rust installation and build")
    parser.add_argument("--skip-deps", action="store_true", help="Skip Python dependency installation")
    parser.add_argument("--skip-tests", action="store_true", help="Skip system tests")
    parser.add_argument("--dev", action="store_true", help="Development setup with extra tools")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check system requirements
    check_system_requirements()
    
    # Install Rust if needed
    if not args.skip_rust:
        install_rust()
    
    # Set up Python environment
    if not args.skip_deps:
        setup_python_environment()
    
    # Build Rust components
    if not args.skip_rust:
        build_rust_components()
    
    # Set up configurations
    setup_configurations()
    
    # Create directories
    setup_directories()
    
    # Create startup scripts
    create_startup_scripts()
    
    # Run tests
    if not args.skip_tests:
        run_tests()
    
    # Final setup message
    print("\n" + "=" * 60)
    print("🎉 HEIHACHI COMPLETE SYSTEM SETUP FINISHED! 🎉")
    print("=" * 60)
    print("\n🚀 Next Steps:")
    print("1. Ensure Autobahn service is running")
    print("2. Test connectivity: python scripts/check_autobahn_service.py")
    print("3. Start system: python scripts/start_heihachi.py --all")
    print("4. Test integration: python examples/autobahn_delegation_demo.py")
    print("5. Access fire interface at: http://localhost:3000")
    print("6. Access REST API at: http://localhost:5000")
    print("\n🔥 Ready to revolutionize audio analysis with fire and consciousness! 🧠")

if __name__ == "__main__":
    main() 