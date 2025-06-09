"""
Heihachi REST API Routes

This module contains all the API endpoints for the Heihachi audio analysis system.
"""

import os
import uuid
import time
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify, current_app, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from ..core.audio_processor import AudioProcessor
from ..huggingface.feature_extractor import FeatureExtractor
from ..huggingface.beat_detector import BeatDetector
from ..huggingface.drum_analyzer import DrumAnalyzer
from ..huggingface.stem_separator import StemSeparator
from ..semantic import SemanticSearch, SemanticAnalyzer, EmotionalFeatureMapper
from .config import APIConfig
from .job_manager import JobManager

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# Initialize job manager
job_manager = JobManager()

# Initialize semantic components
semantic_search = SemanticSearch()
semantic_analyzer = SemanticAnalyzer()
emotion_mapper = EmotionalFeatureMapper()


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return APIConfig.allowed_file(filename)


def save_uploaded_file(file):
    """Save uploaded file and return the file path"""
    if not file or file.filename == '':
        raise ValueError("No file provided")
    
    if not allowed_file(file.filename):
        raise ValueError(f"File type not allowed. Supported formats: {', '.join(APIConfig.ALLOWED_EXTENSIONS)}")
    
    # Create upload directory if it doesn't exist
    os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Generate unique filename
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
    
    # Save file
    file.save(file_path)
    logger.info(f"File saved: {file_path}")
    
    return file_path


@api_bp.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze_audio():
    """
    Analyze audio file with full Heihachi processing pipeline
    
    Returns:
        JSON response with analysis results or job ID for async processing
    """
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_path = save_uploaded_file(file)
        
        # Get processing options
        async_processing = request.form.get('async', 'false').lower() == 'true'
        config_path = request.form.get('config', current_app.config['HEIHACHI_CONFIG_PATH'])
        
        if async_processing:
            # Create job for async processing
            job_id = job_manager.create_job(
                job_type='analyze',
                file_path=file_path,
                config_path=config_path
            )
            
            return jsonify({
                'job_id': job_id,
                'status': 'processing',
                'message': 'Analysis started. Use /api/v1/jobs/{job_id} to check status.'
            }), 202
        else:
            # Process synchronously
            processor = AudioProcessor(config_path=config_path)
            results = processor.process_file(file_path)
            
            # Clean up uploaded file
            os.remove(file_path)
            
            return jsonify({
                'status': 'completed',
                'results': results,
                'processing_time': results.get('processing_time', 0)
            })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in analyze_audio: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/features', methods=['POST'])
@limiter.limit("20 per minute")
def extract_features():
    """
    Extract audio features using HuggingFace models
    
    Returns:
        JSON response with extracted features
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_path = save_uploaded_file(file)
        
        # Get model preference
        model_name = request.form.get('model', 'microsoft/BEATs-base')
        
        # Extract features
        extractor = FeatureExtractor(model=model_name)
        features = extractor.extract(audio_path=file_path)
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({
            'status': 'completed',
            'features': features,
            'model': model_name
        })
    
    except Exception as e:
        logger.error(f"Error in extract_features: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/beats', methods=['POST'])
@limiter.limit("20 per minute")
def detect_beats():
    """
    Detect beats and tempo in audio file
    
    Returns:
        JSON response with beat detection results
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_path = save_uploaded_file(file)
        
        # Detect beats
        detector = BeatDetector()
        beats = detector.detect(audio_path=file_path)
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({
            'status': 'completed',
            'beats': beats
        })
    
    except Exception as e:
        logger.error(f"Error in detect_beats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/drums', methods=['POST'])
@limiter.limit("10 per minute")
def analyze_drums():
    """
    Analyze drum patterns and hits in audio file
    
    Returns:
        JSON response with drum analysis results
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_path = save_uploaded_file(file)
        
        # Get options
        visualize = request.form.get('visualize', 'false').lower() == 'true'
        
        # Analyze drums
        analyzer = DrumAnalyzer()
        results = analyzer.analyze(audio_path=file_path, visualize=visualize)
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({
            'status': 'completed',
            'drum_analysis': results
        })
    
    except Exception as e:
        logger.error(f"Error in analyze_drums: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/stems', methods=['POST'])
@limiter.limit("5 per minute")
def separate_stems():
    """
    Separate audio into stems (drums, bass, vocals, other)
    
    Returns:
        JSON response with stem separation results
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_path = save_uploaded_file(file)
        
        # Get options
        save_stems = request.form.get('save_stems', 'false').lower() == 'true'
        output_format = request.form.get('format', 'wav')
        
        # Separate stems
        separator = StemSeparator()
        stems = separator.separate(
            audio_path=file_path,
            save_stems=save_stems,
            output_format=output_format
        )
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({
            'status': 'completed',
            'stems': stems
        })
    
    except Exception as e:
        logger.error(f"Error in separate_stems: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/batch-analyze', methods=['POST'])
@limiter.limit("2 per minute")
def batch_analyze():
    """
    Analyze multiple audio files in batch
    
    Returns:
        JSON response with batch job ID
    """
    try:
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files provided'}), 400
        
        if len(files) > 10:  # Limit batch size
            return jsonify({'error': 'Maximum 10 files per batch'}), 400
        
        # Save all files
        file_paths = []
        for file in files:
            if file.filename != '':
                file_path = save_uploaded_file(file)
                file_paths.append(file_path)
        
        # Create batch job
        job_id = job_manager.create_batch_job(
            job_type='batch_analyze',
            file_paths=file_paths,
            config_path=request.form.get('config', current_app.config['HEIHACHI_CONFIG_PATH'])
        )
        
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'file_count': len(file_paths),
            'message': 'Batch analysis started. Use /api/v1/jobs/{job_id} to check status.'
        }), 202
    
    except Exception as e:
        logger.error(f"Error in batch_analyze: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """
    Get status and results of a processing job
    
    Args:
        job_id: UUID of the job
    
    Returns:
        JSON response with job status and results
    """
    try:
        job = job_manager.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(job)
    
    except Exception as e:
        logger.error(f"Error in get_job_status: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/jobs/<job_id>/results', methods=['GET'])
def download_job_results(job_id):
    """
    Download job results as a file
    
    Args:
        job_id: UUID of the job
    
    Returns:
        File download or JSON error response
    """
    try:
        job = job_manager.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        if job['status'] != 'completed':
            return jsonify({'error': 'Job not completed yet'}), 400
        
        # Return results file if available
        results_file = job.get('results_file')
        if results_file and os.path.exists(results_file):
            return send_file(results_file, as_attachment=True)
        else:
            return jsonify({'error': 'Results file not found'}), 404
    
    except Exception as e:
        logger.error(f"Error in download_job_results: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/jobs', methods=['GET'])
def list_jobs():
    """
    List all jobs (with pagination)
    
    Returns:
        JSON response with list of jobs
    """
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)  # Max 100 per page
        status_filter = request.args.get('status')
        
        jobs = job_manager.list_jobs(page=page, per_page=per_page, status=status_filter)
        
        return jsonify(jobs)
    
    except Exception as e:
        logger.error(f"Error in list_jobs: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/semantic/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def semantic_analyze():
    """
    Perform semantic analysis on audio file including emotional mapping
    
    Returns:
        JSON response with semantic analysis results
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_path = save_uploaded_file(file)
        
        # Get options
        include_emotions = request.form.get('include_emotions', 'true').lower() == 'true'
        include_search_indexing = request.form.get('index_for_search', 'false').lower() == 'true'
        
        # Process with full pipeline first
        processor = AudioProcessor()
        analysis_result = processor.process_file(file_path)
        
        semantic_results = {}
        
        if include_emotions:
            # Map to emotions
            emotions = emotion_mapper.map_features_to_emotions(analysis_result)
            semantic_results['emotions'] = emotions
        
        if include_search_indexing:
            # Index for semantic search
            track_info = {
                'title': request.form.get('title', os.path.basename(file_path)),
                'artist': request.form.get('artist', 'Unknown Artist'),
                'path': file_path
            }
            
            track_id = f"{track_info['artist']}_{track_info['title']}"
            success = semantic_search.index_track(track_id, track_info, analysis_result)
            semantic_results['indexed'] = success
            semantic_results['track_id'] = track_id if success else None
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({
            'status': 'completed',
            'semantic_analysis': semantic_results,
            'full_analysis': analysis_result
        })
    
    except Exception as e:
        logger.error(f"Error in semantic_analyze: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/semantic/search', methods=['POST'])
@limiter.limit("20 per minute")
def semantic_search_tracks():
    """
    Search indexed tracks using semantic queries
    
    Returns:
        JSON response with search results
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        top_k = data.get('top_k', 5)
        enhance_query = data.get('enhance_query', True)
        
        # Perform semantic search
        results = semantic_search.search(query, top_k=top_k, enhance_query=enhance_query)
        
        return jsonify({
            'status': 'completed',
            'query': query,
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Error in semantic_search_tracks: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/semantic/emotions', methods=['POST'])
@limiter.limit("20 per minute")
def analyze_emotions():
    """
    Extract emotional features from audio analysis
    
    Returns:
        JSON response with emotional analysis
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_path = save_uploaded_file(file)
        
        # Process audio
        processor = AudioProcessor()
        analysis_result = processor.process_file(file_path)
        
        # Map to emotions
        emotions = emotion_mapper.map_features_to_emotions(analysis_result)
        
        # Generate descriptions
        emotion_descriptions = {}
        for emotion, value in emotions.items():
            emotion_descriptions[emotion] = emotion_mapper.describe_emotion(emotion, value)
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({
            'status': 'completed',
            'emotions': emotions,
            'descriptions': emotion_descriptions,
            'summary': {
                'dominant_emotion': max(emotions.items(), key=lambda x: x[1]),
                'overall_energy': emotions.get('energy', 0),
                'overall_mood': 'positive' if emotions.get('euphoria', 0) > emotions.get('melancholy', 0) else 'melancholic'
            }
        })
    
    except Exception as e:
        logger.error(f"Error in analyze_emotions: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/semantic/text-analysis', methods=['POST'])
@limiter.limit("30 per minute")
def analyze_text():
    """
    Analyze text description for sentiment and categorization
    
    Returns:
        JSON response with text analysis
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        
        # Analyze text
        analysis = semantic_analyzer.analyze_text(text)
        
        return jsonify({
            'status': 'completed',
            'text': text,
            'analysis': analysis
        })
    
    except Exception as e:
        logger.error(f"Error in analyze_text: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.route('/semantic/stats', methods=['GET'])
def get_semantic_stats():
    """
    Get statistics about the semantic search index
    
    Returns:
        JSON response with semantic search statistics
    """
    try:
        stats = semantic_search.get_stats()
        
        return jsonify({
            'status': 'completed',
            'stats': stats
        })
    
    except Exception as e:
        logger.error(f"Error in get_semantic_stats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api_bp.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit errors"""
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': str(e.description)
    }), 429 