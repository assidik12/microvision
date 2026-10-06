"""Interactive Face Data Collector with Real-time MediaPipe Quality Check.

Captures aligned face crops (112x112) directly from camera, checking:
1. Face presence & confidence
2. Eye landmarks detection (for alignment quality)
3. Minimum face bounding box size (prevents low-resolution captures)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microvision_attend.camera import FaceDetector, crop_face


def collect_faces(
    name: str,
    output_dir: Path,
    camera_index: int = 0,
    target_count: int = 20,
    capture_interval: float = 0.35,
    min_box_size: int = 80,
    confidence: float = 0.6,
) -> int:
    person_dir = output_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(confidence=confidence)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    count = 0
    last_capture_time = 0.0

    print(f"\n[Data Collector] Collecting {target_count} face samples for: {name!r}")
    print("Petunjuk:")
    print("  - Gerakkan kepala perlahan (depan, sedikit miring kiri/kanan/atas/bawah)")
    print("  - Berikan variasi ekspresi natural")
    print("  - Tekan 'c' untuk capture manual, atau diam sejenak untuk auto-capture berkala")
    print("  - Tekan 'q' untuk berhenti lebih awal\n")

    try:
        while count < target_count:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Failed to read camera frame.")

            observations = detector.detect_observations(frame)
            now = time.time()
            valid_observation = None

            if observations:
                # Ambil wajah terbesar di frame
                largest = max(observations, key=lambda item: item.box[2] * item.box[3])
                x, y, w, h = largest.box

                # Validasi ukuran wajah
                if w >= min_box_size and h >= min_box_size:
                    valid_observation = largest
                    color = (0, 220, 0)
                    status_text = f"Quality OK ({w}x{h})"
                else:
                    color = (0, 165, 255)
                    status_text = f"Too far / small ({w}x{h})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    status_text,
                    (x, max(25, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # Auto capture dengan interval waktu
            key = cv2.waitKey(1) & 0xFF
            auto_ready = valid_observation and (now - last_capture_time >= capture_interval)
            manual_trigger = valid_observation and (key == ord("c"))

            if auto_ready or manual_trigger:
                count += 1
                last_capture_time = now
                aligned_crop = crop_face(
                    frame,
                    valid_observation.box,
                    size=112,
                    landmarks=valid_observation.landmarks,
                )
                save_path = person_dir / f"{name}_{count:03d}.jpg"
                cv2.imwrite(str(save_path), cv2.cvtColor(aligned_crop, cv2.COLOR_RGB2BGR))
                print(f"[{count}/{target_count}] Captured & aligned: {save_path.name}")

            # HUD Display
            progress_text = f"Identity: {name} | Progress: {count}/{target_count}"
            cv2.putText(
                frame,
                progress_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2,
            )
            cv2.imshow("MicroVision - Face Data Collector", frame)

            if key == ord("q"):
                print("[Abort] Pengambilan sampel dihentikan oleh pengguna.")
                break

    finally:
        detector.close()
        capture.release()
        cv2.destroyAllWindows()

    print(f"\n[Selesai] Berhasil menyimpan {count} foto ke folder: {person_dir}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect aligned face images for edge registration.")
    parser.add_argument("name", help="Identity name (e.g. 'Ahmad Sofi Sidik').")
    parser.add_argument("--output-dir", type=Path, default=Path("data/datasets"), help="Base dataset directory.")
    parser.add_argument("--count", type=int, default=15, help="Number of face samples to collect.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument("--interval", type=float, default=0.4, help="Auto-capture interval (seconds).")
    parser.add_argument("--confidence", type=float, default=0.6, help="Detector confidence threshold.")
    args = parser.parse_args()

    collect_faces(
        name=args.name,
        output_dir=args.output_dir,
        camera_index=args.camera,
        target_count=args.count,
        capture_interval=args.interval,
        confidence=args.confidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
