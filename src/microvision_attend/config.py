from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings with conservative defaults for low-resource devices."""

    camera_index: int = 0
    detection_confidence: float = 0.5
    match_threshold: float = 0.5
    attendance_cooldown_seconds: float = 30.0
    input_size: int = 112
    embeddings_path: Path = Path("data/embeddings.json")
    attendance_path: Path = Path("data/attendance.csv")
    model_path: Path = Path("models/mobilefacenet.tflite")
    detector_model_path: Path = Path("models/blaze_face_short_range.tflite")
    landmark_model_path: Path = Path("models/face_landmarker.task")
