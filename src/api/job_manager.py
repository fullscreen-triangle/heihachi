"""
Job Manager for Heihachi API

Handles asynchronous processing jobs, job status tracking, and result management.
"""

import os
import json
import uuid
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Job data structure"""
    id: str
    type: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    config_path: Optional[str] = None
    results: Optional[Dict] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    
    def to_dict(self):
        """Convert job to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif key == 'status':
                data[key] = value.value
        return data


class JobManager:
    """Manages processing jobs for the API"""
    
    def __init__(self, max_concurrent_jobs: int = 5):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.jobs: Dict[str, Job] = {}
        self.active_jobs: List[str] = []
        self.job_lock = threading.Lock()
        self.worker_thread = None
        self.should_stop = False
        
        # Start job worker thread
        self.start_worker()
    
    def start_worker(self):
        """Start the background worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Job worker thread started")
    
    def _worker_loop(self):
        """Main worker loop for processing jobs"""
        while not self.should_stop:
            try:
                self._process_pending_jobs()
                time.sleep(1)  # Check for new jobs every second
            except Exception as e:
                logger.error(f"Error in worker loop: {str(e)}")
                time.sleep(5)  # Wait longer if there's an error
    
    def _process_pending_jobs(self):
        """Process pending jobs if there's capacity"""
        with self.job_lock:
            # Remove completed jobs from active list
            self.active_jobs = [job_id for job_id in self.active_jobs 
                              if self.jobs[job_id].status == JobStatus.PROCESSING]
            
            # Find pending jobs
            pending_jobs = [job_id for job_id, job in self.jobs.items() 
                          if job.status == JobStatus.PENDING]
            
            # Start new jobs if we have capacity
            available_slots = self.max_concurrent_jobs - len(self.active_jobs)
            
            for job_id in pending_jobs[:available_slots]:
                self._start_job(job_id)
    
    def _start_job(self, job_id: str):
        """Start processing a specific job"""
        job = self.jobs[job_id]
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now()
        self.active_jobs.append(job_id)
        
        # Start job in separate thread
        job_thread = threading.Thread(target=self._execute_job, args=(job_id,))
        job_thread.start()
        
        logger.info(f"Started job {job_id} (type: {job.type})")
    
    def _execute_job(self, job_id: str):
        """Execute a specific job"""
        job = self.jobs[job_id]
        
        try:
            # Import here to avoid circular imports
            from ..core.audio_processor import AudioProcessor
            from ..huggingface.feature_extractor import FeatureExtractor
            from ..huggingface.beat_detector import BeatDetector
            from ..huggingface.drum_analyzer import DrumAnalyzer
            from ..huggingface.stem_separator import StemSeparator
            
            if job.type == 'analyze':
                # Full audio analysis
                processor = AudioProcessor(config_path=job.config_path)
                results = processor.process_file(job.file_path)
                job.results = results
                
            elif job.type == 'batch_analyze':
                # Batch analysis
                processor = AudioProcessor(config_path=job.config_path)
                results = []
                total_files = len(job.file_paths)
                
                for i, file_path in enumerate(job.file_paths):
                    try:
                        file_results = processor.process_file(file_path)
                        results.append({
                            'file_path': file_path,
                            'results': file_results,
                            'status': 'completed'
                        })
                    except Exception as e:
                        results.append({
                            'file_path': file_path,
                            'error': str(e),
                            'status': 'failed'
                        })
                    
                    # Update progress
                    job.progress = (i + 1) / total_files
                
                job.results = {
                    'batch_results': results,
                    'total_files': total_files,
                    'successful': len([r for r in results if r['status'] == 'completed']),
                    'failed': len([r for r in results if r['status'] == 'failed'])
                }
            
            elif job.type == 'extract_features':
                # Feature extraction
                extractor = FeatureExtractor()
                features = extractor.extract(audio_path=job.file_path)
                job.results = {'features': features}
            
            elif job.type == 'detect_beats':
                # Beat detection
                detector = BeatDetector()
                beats = detector.detect(audio_path=job.file_path)
                job.results = {'beats': beats}
            
            elif job.type == 'analyze_drums':
                # Drum analysis
                analyzer = DrumAnalyzer()
                results = analyzer.analyze(audio_path=job.file_path)
                job.results = {'drum_analysis': results}
            
            elif job.type == 'separate_stems':
                # Stem separation
                separator = StemSeparator()
                stems = separator.separate(audio_path=job.file_path)
                job.results = {'stems': stems}
            
            else:
                raise ValueError(f"Unknown job type: {job.type}")
            
            # Mark job as completed
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0
            
            # Clean up uploaded files
            self._cleanup_job_files(job)
            
            logger.info(f"Completed job {job_id}")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            # Clean up uploaded files even on failure
            self._cleanup_job_files(job)
            
            logger.error(f"Failed job {job_id}: {str(e)}")
    
    def _cleanup_job_files(self, job: Job):
        """Clean up uploaded files after job completion"""
        try:
            if job.file_path and os.path.exists(job.file_path):
                os.remove(job.file_path)
            
            if job.file_paths:
                for file_path in job.file_paths:
                    if os.path.exists(file_path):
                        os.remove(file_path)
        except Exception as e:
            logger.warning(f"Error cleaning up files for job {job.id}: {str(e)}")
    
    def create_job(self, job_type: str, file_path: str, config_path: str = None) -> str:
        """Create a new job for single file processing"""
        job_id = str(uuid.uuid4())
        
        job = Job(
            id=job_id,
            type=job_type,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            file_path=file_path,
            config_path=config_path
        )
        
        with self.job_lock:
            self.jobs[job_id] = job
        
        logger.info(f"Created job {job_id} (type: {job_type})")
        return job_id
    
    def create_batch_job(self, job_type: str, file_paths: List[str], config_path: str = None) -> str:
        """Create a new batch job for multiple files"""
        job_id = str(uuid.uuid4())
        
        job = Job(
            id=job_id,
            type=job_type,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            file_paths=file_paths,
            config_path=config_path
        )
        
        with self.job_lock:
            self.jobs[job_id] = job
        
        logger.info(f"Created batch job {job_id} (type: {job_type}, files: {len(file_paths)})")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status and results"""
        with self.job_lock:
            job = self.jobs.get(job_id)
            if job:
                return job.to_dict()
            return None
    
    def list_jobs(self, page: int = 1, per_page: int = 20, status: str = None) -> Dict:
        """List jobs with pagination"""
        with self.job_lock:
            jobs_list = list(self.jobs.values())
            
            # Filter by status if specified
            if status:
                try:
                    status_enum = JobStatus(status)
                    jobs_list = [job for job in jobs_list if job.status == status_enum]
                except ValueError:
                    pass  # Invalid status, ignore filter
            
            # Sort by creation date (newest first)
            jobs_list.sort(key=lambda x: x.created_at, reverse=True)
            
            # Pagination
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            page_jobs = jobs_list[start_idx:end_idx]
            
            return {
                'jobs': [job.to_dict() for job in page_jobs],
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': len(jobs_list),
                    'pages': (len(jobs_list) + per_page - 1) // per_page
                }
            }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job (only if it's pending)"""
        with self.job_lock:
            job = self.jobs.get(job_id)
            if job and job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                self._cleanup_job_files(job)
                logger.info(f"Cancelled job {job_id}")
                return True
            return False
    
    def cleanup_old_jobs(self, retention_days: int = 7):
        """Clean up old completed jobs"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        with self.job_lock:
            old_jobs = [
                job_id for job_id, job in self.jobs.items()
                if job.completed_at and job.completed_at < cutoff_date
            ]
            
            for job_id in old_jobs:
                del self.jobs[job_id]
            
            if old_jobs:
                logger.info(f"Cleaned up {len(old_jobs)} old jobs")
    
    def stop(self):
        """Stop the job manager"""
        self.should_stop = True
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Job manager stopped") 