import threading
import time
from collections import deque
from typing import Any, Callable


JobRunner = Callable[[str, dict[str, Any]], None]


class JobService:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._current_job_id: str | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._cancel_flags: dict[str, bool] = {}

    def create_job(
        self,
        job_id: str,
        *,
        prompt: str = "",
        negative_prompt: str = "",
        subject: str = "",
        hair: str = "",
        clothing: str = "",
        environment: str = "",
        style: str = "",
        seed: int | None = None,
        model_id: str | None = None,
        source_filename: str | None = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 12,
        guidance_scale: float = 6.5,
        num_images: int = 1,
    ) -> None:
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "message": "Queued",
                "generated_filenames": [],
                "source_filename": source_filename,
                "prompt": prompt,
                "original_prompt": None,
                "prompt_source_language": None,
                "prompt_was_translated": False,
                "negative_prompt": negative_prompt,
                "subject": subject,
                "hair": hair,
                "clothing": clothing,
                "environment": environment,
                "style": style,
                "seed": seed,
                "model_id": model_id,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images": num_images,
                "error": None,
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
            }
            self._cancel_flags[job_id] = False

    def enqueue_job(self, job_id: str) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job '{job_id}' does not exist.")

            if job_id == self._current_job_id or job_id in self._queue:
                return

            if self._jobs[job_id]["status"] == "cancelled":
                return

            self._queue.append(job_id)
            self._jobs[job_id]["status"] = "queued"
            self._jobs[job_id]["progress"] = 0
            self._jobs[job_id]["message"] = "Queued"

    def start_worker(self, runner: JobRunner) -> None:
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return

            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                args=(runner,),
                daemon=True,
                name="job-service-worker",
            )
            self._worker_thread.start()

    def stop_worker(self) -> None:
        self._stop_event.set()

        with self._lock:
            worker = self._worker_thread

        if worker and worker.is_alive():
            worker.join(timeout=2)

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False

            job = self._jobs[job_id]

            if job["status"] in ("completed", "failed", "cancelled"):
                return False

            if job_id in self._queue:
                self._queue.remove(job_id)
                job["status"] = "cancelled"
                job["progress"] = 100
                job["message"] = "Cancelled"
                job["finished_at"] = time.time()
                self._cancel_flags[job_id] = True
                return True

            if job_id == self._current_job_id:
                self._cancel_flags[job_id] = True
                job["message"] = "Cancelling..."
                return True

            return False

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return self._cancel_flags.get(job_id, False)

    def _worker_loop(self, runner: JobRunner) -> None:
        while not self._stop_event.is_set():
            job_id: str | None = None

            with self._lock:
                if self._current_job_id is None and self._queue:
                    candidate_job_id = self._queue.popleft()

                    if candidate_job_id in self._jobs:
                        job = self._jobs[candidate_job_id]

                        if job["status"] == "cancelled":
                            job["finished_at"] = job["finished_at"] or time.time()
                        else:
                            job_id = candidate_job_id
                            self._current_job_id = job_id
                            job["status"] = "running"
                            job["progress"] = 5
                            job["message"] = "Starting job"
                            job["started_at"] = time.time()

            if job_id is None:
                time.sleep(0.1)
                continue

            try:
                job = self.get_job(job_id)
                if job is None:
                    continue

                if self.is_cancelled(job_id):
                    raise RuntimeError("Job cancelled")

                runner(job_id, job)

                with self._lock:
                    if job_id in self._jobs and self._jobs[job_id]["status"] not in ("failed", "cancelled"):
                        self._jobs[job_id]["status"] = "completed"
                        self._jobs[job_id]["progress"] = 100
                        self._jobs[job_id]["message"] = "Completed"
                        self._jobs[job_id]["finished_at"] = time.time()

            except Exception as exc:
                with self._lock:
                    if job_id in self._jobs:
                        if self._cancel_flags.get(job_id, False) or "cancelled" in str(exc).lower():
                            self._jobs[job_id]["status"] = "cancelled"
                            self._jobs[job_id]["progress"] = 100
                            self._jobs[job_id]["message"] = "Cancelled by user"
                            self._jobs[job_id]["error"] = None
                        else:
                            self._jobs[job_id]["status"] = "failed"
                            self._jobs[job_id]["progress"] = 100
                            self._jobs[job_id]["message"] = "Failed"
                            self._jobs[job_id]["error"] = str(exc)

                        self._jobs[job_id]["finished_at"] = time.time()

            finally:
                with self._lock:
                    if self._current_job_id == job_id:
                        self._current_job_id = None

    def update_job(self, job_id: str, **kwargs) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id].update(kwargs)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

            result = dict(job)
            # generated_filenames is a mutable list – copy it so callers
            # cannot accidentally mutate the internal state.
            result["generated_filenames"] = list(job["generated_filenames"])

            if job_id == self._current_job_id:
                result["queue_position"] = 0
            else:
                try:
                    result["queue_position"] = list(self._queue).index(job_id) + 1
                except ValueError:
                    result["queue_position"] = None

            return result

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [dict(job) for job in self._jobs.values()]

        def sort_key(item: dict[str, Any]):
            started_at = item.get("started_at")
            finished_at = item.get("finished_at")
            return (
                0 if item.get("status") in ("running", "queued") else 1,
                -(started_at or 0),
                -(finished_at or 0),
            )

        jobs.sort(key=sort_key)
        return jobs

    def cleanup_old_jobs(self, max_age_seconds: int = 3600) -> int:
        now = time.time()
        removed = 0

        with self._lock:
            removable_job_ids: list[str] = []

            for job_id, job in self._jobs.items():
                if job_id == self._current_job_id:
                    continue
                if job_id in self._queue:
                    continue
                if job["status"] not in ("completed", "failed", "cancelled"):
                    continue

                finished_at = job.get("finished_at")
                if finished_at is None:
                    continue

                if now - finished_at > max_age_seconds:
                    removable_job_ids.append(job_id)

            for job_id in removable_job_ids:
                self._jobs.pop(job_id, None)
                self._cancel_flags.pop(job_id, None)
                removed += 1

        return removed

