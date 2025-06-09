#!/usr/bin/env python3
"""
Heihachi API Server

Standalone script to run the Heihachi REST API server.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

try:
    from src.api.app import create_app
    from src.api.config import DevelopmentConfig, ProductionConfig
except ImportError as e:
    print(f"Error importing Heihachi modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

def setup_logging(log_level):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('heihachi_api.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Heihachi REST API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')
    parser.add_argument('--config-path', help='Path to Heihachi config file')
    parser.add_argument('--upload-dir', default='uploads', help='Directory for uploaded files')
    parser.add_argument('--results-dir', default='results', help='Directory for results')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set environment variables for configuration
    if args.config_path:
        os.environ['HEIHACHI_CONFIG_PATH'] = args.config_path
    if args.upload_dir:
        os.environ['UPLOAD_FOLDER'] = args.upload_dir
    if args.results_dir:
        os.environ['RESULTS_FOLDER'] = args.results_dir
    
    # Create upload and results directories
    os.makedirs(args.upload_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Choose configuration
    if args.production:
        config_class = ProductionConfig
        logger.info("Starting Heihachi API in production mode")
    else:
        config_class = DevelopmentConfig
        logger.info("Starting Heihachi API in development mode")
    
    try:
        # Create Flask app
        app = create_app(config_class)
        
        # Log startup information
        logger.info(f"Heihachi API Server starting on {args.host}:{args.port}")
        logger.info(f"Upload directory: {args.upload_dir}")
        logger.info(f"Results directory: {args.results_dir}")
        logger.info(f"Debug mode: {args.debug or not args.production}")
        
        # Start the server
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug and not args.production,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down Heihachi API server...")
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 