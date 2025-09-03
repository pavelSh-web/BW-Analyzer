import os
import tempfile
import time
from typing import Dict, Any, Optional
import librosa
from celery import Task

from .celery_app import celery_app
from .main import get_audio_tags, analyze_audio_features


class AudioAnalysisTask(Task):
    """Base class for audio analysis tasks"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Error handling"""
        print(f"Task {task_id} failed: {exc}")
        
    def on_success(self, retval, task_id, args, kwargs):
        """Success handling"""
        print(f"Task {task_id} completed successfully")


@celery_app.task(base=AudioAnalysisTask, bind=True)
def analyze_audio_file(self, file_data: bytes, filename: str, top_tags_per_group: int = 5) -> Dict[str, Any]:
    """
    Asynchronous audio file analysis
    
    Args:
        file_data: Binary audio file data
        filename: File name
        top_tags_per_group: Number of tags per group
        
    Returns:
        Analysis result
    """
    task_id = self.request.id
    t0 = time.time()
    temp_path = None
    
    try:
        # Update status
        self.update_state(state='PROCESSING', meta={'stage': 'saving_file'})
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(filename)[1] or ".wav", 
            delete=False
        ) as tmp:
            tmp.write(file_data)
            temp_path = tmp.name
        
        # Update status
        self.update_state(state='PROCESSING', meta={'stage': 'loading_audio'})
        
        # Load audio
        y, sr = librosa.load(temp_path, sr=None, mono=True)
        
        # Update status
        self.update_state(state='PROCESSING', meta={'stage': 'analyzing_tags'})
        
        # Get PANNs tags by groups
        tags = get_audio_tags(temp_path, topk_per_group=top_tags_per_group)
        
        # Update status
        self.update_state(state='PROCESSING', meta={'stage': 'analyzing_features'})
        
        # Analyze musical characteristics
        features = analyze_audio_features(y, sr)
        
        # Track duration
        duration = float(len(y) / sr)
        
        result = {
            "task_id": task_id,
            "filename": filename,
            "duration_seconds": duration,
            "sample_rate": int(sr),
            "tags": tags,
            "musical_features": features,
            "elapsed_sec": round(time.time() - t0, 3),
            "status": "completed"
        }
        
        return result
        
    except Exception as e:
        # In case of error return error information
        error_result = {
            "task_id": task_id,
            "filename": filename,
            "error": str(e),
            "elapsed_sec": round(time.time() - t0, 3),
            "status": "failed"
        }
        raise self.retry(countdown=60, max_retries=3, exc=e)
        
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass


@celery_app.task
def get_analysis_status(task_id: str) -> Dict[str, Any]:
    """
    Get analysis task status
    
    Args:
        task_id: Task ID
        
    Returns:
        Task status
    """
    result = celery_app.AsyncResult(task_id)
    
    if result.state == 'PENDING':
        return {'status': 'pending', 'message': 'Task is waiting to be processed'}
    elif result.state == 'PROCESSING':
        return {
            'status': 'processing',
            'stage': result.info.get('stage', 'unknown'),
            'message': f"Task is being processed: {result.info.get('stage', 'unknown')}"
        }
    elif result.state == 'SUCCESS':
        return {'status': 'completed', 'result': result.result}
    elif result.state == 'FAILURE':
        return {
            'status': 'failed',
            'error': str(result.info),
            'message': 'Task failed to complete'
        }
    else:
        return {'status': result.state, 'message': f'Task state: {result.state}'}
