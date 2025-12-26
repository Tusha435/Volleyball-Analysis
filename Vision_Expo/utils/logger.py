"""
Error tracking and reporting system for debugging tracking failures.
"""
import json
from datetime import datetime


class ErrorLogger:
    """
    Collects and serializes tracking errors to JSON for analysis.

    Attributes:
        enabled: Whether logging is active
        errors: List of accumulated error records
    """

    def __init__(self, enabled=True):
        """
        Initialize the error logger.

        Args:
            enabled: Enable or disable logging
        """
        self.enabled = enabled
        self.errors = []

    def log(self, frame_id, error_type, severity, description, **metadata):
        """
        Record an error event with metadata.

        Args:
            frame_id: Frame number where error occurred
            error_type: Category of error (e.g., 'detection_failure')
            severity: Error severity level
            description: Human-readable error description
            **metadata: Additional context as key-value pairs
        """
        if not self.enabled:
            return

        self.errors.append({
            "frame": int(frame_id),
            "type": str(error_type),
            "severity": str(severity),
            "description": str(description),
            "metadata": self._sanitize(metadata)
        })

    def save(self, path):
        """
        Write collected errors to a JSON file.

        Args:
            path: Output file path for the error report
        """
        if not self.enabled:
            return

        data = {
            "timestamp": datetime.now().isoformat(),
            "total_errors": int(len(self.errors)),
            "errors_by_type": {},
            "errors": self.errors
        }

        for err in self.errors:
            t = err["type"]
            data["errors_by_type"][t] = data["errors_by_type"].get(t, 0) + 1

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n[ERROR REPORT] Saved to {path}")

    def _sanitize(self, obj):
        """
        Convert complex Python objects to JSON-serializable types.

        Args:
            obj: Object to sanitize (can be nested)

        Returns:
            JSON-compatible version of the object
        """
        import numpy as np

        if obj is None:
            return None
        if isinstance(obj, (int, float, str, bool)):
            return obj
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, np.ndarray):
            return self._sanitize(obj.tolist())

        return str(obj)
