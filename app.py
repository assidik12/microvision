from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support running `python app.py` directly from a source checkout that uses
# the standard `src/` layout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from microvision_attend.attendance import AttendanceWriter
from microvision_attend.camera import FaceDetector, crop_face, frames
from microvision_attend.config import AppConfig
from microvision_attend.embedder import TFLiteEmbedder
from microvision_attend.matcher import EmbeddingRegistry


def run(config: AppConfig) -> None:
    import cv2

    registry = EmbeddingRegistry.load(config.embeddings_path)
    embedder = TFLiteEmbedder(str(config.model_path), config.input_size)
    attendance = AttendanceWriter(config.attendance_path, config.attendance_cooldown_seconds)
    detector = FaceDetector(
        config.detection_confidence,
        str(config.detector_model_path),
        str(config.landmark_model_path),
    )

    try:
        for frame in frames(config.camera_index):
            for observation in detector.detect_observations(frame):
                box = observation.box
                embedding = embedder(
                    crop_face(frame, box, config.input_size, observation.landmarks)
                )
                match = registry.match(embedding, config.match_threshold)
                x, y, width, height = box

                # Visual differentiation: Green for recognized, Amber/Red for unknown
                is_recognized = match.name is not None
                box_color = (0, 220, 0) if is_recognized else (0, 140, 255)
                name_display = match.name if is_recognized else "Unknown"
                label = f"{name_display} ({match.score:.2f})"

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + width, y + height), box_color, 2)

                # Draw filled label background badge for maximum contrast & readability
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                badge_y = max(y, text_h + baseline + 6)
                cv2.rectangle(
                    frame,
                    (x, badge_y - text_h - baseline - 4),
                    (x + text_w + 8, badge_y),
                    box_color,
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (x + 4, badge_y - baseline - 2),
                    font,
                    font_scale,
                    (0, 0, 0) if is_recognized else (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

                if match.name:
                    attendance.record(match.name)
            cv2.imshow("microvision-attend", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local face attendance.")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run(AppConfig(camera_index=args.camera))
