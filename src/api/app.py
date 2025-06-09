"""
Heihachi REST API Application

Flask application for providing REST API endpoints for audio analysis.
"""

import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from .routes import api_bp
from .config import APIConfig

logger = logging.getLogger(__name__)


def create_app(config_class=APIConfig):
    """
    Create and configure the Flask application.
    
    Args:
        config_class: Configuration class to use
    
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Enable CORS for all domains on all routes
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    
    # Error handlers
    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(e):
        return jsonify({
            'error': 'File too large',
            'message': f'Maximum file size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'
        }), 413
    
    @app.errorhandler(404)
    def handle_not_found(e):
        return jsonify({
            'error': 'Not found',
            'message': 'The requested endpoint was not found'
        }), 404
    
    @app.errorhandler(500)
    def handle_internal_error(e):
        logger.error(f"Internal server error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'heihachi-api',
            'version': '1.0.0'
        })
    
    # API info endpoint
    @app.route('/api')
    def api_info():
        return jsonify({
            'service': 'Heihachi Audio Analysis API',
            'version': '1.0.0',
            'endpoints': {
                'analyze': '/api/v1/analyze',
                'batch_analyze': '/api/v1/batch-analyze',
                'extract_features': '/api/v1/features',
                'detect_beats': '/api/v1/beats',
                'analyze_drums': '/api/v1/drums',
                'separate_stems': '/api/v1/stems',
                'semantic_analyze': '/api/v1/semantic/analyze',
                'semantic_search': '/api/v1/semantic/search',
                'semantic_emotions': '/api/v1/semantic/emotions',
                'semantic_text': '/api/v1/semantic/text-analysis',
                'semantic_stats': '/api/v1/semantic/stats',
                'jobs': '/api/v1/jobs/{job_id}',
                'health': '/health'
            },
            'features': [
                'Audio analysis with neural processing',
                'Emotional feature mapping',
                'Semantic search and indexing',
                'HuggingFace AI model integration',
                'Asynchronous processing',
                'Batch operations',
                'Real-time beat detection',
                'Drum pattern analysis',
                'Stem separation'
            ],
            'documentation': 'https://github.com/fullscreen-triangle/heihachi#api-reference'
        })
    
    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Heihachi API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug) 