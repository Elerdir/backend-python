import threading
from typing import Any


class JobService:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "message": "Queued",
                "generated_filenames": [],
                "source_filename": None,
                "prompt": "",
                "negative_prompt": "",
                "subject": "",
                "hair": "",
                "clothing": "",
                "environment": "",
                "style": "",
                "seed": None,
                "model_id": None,
                "error": None,
            }

    def update_job(self, job_id: str, **kwargs) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id].update(kwargs)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None