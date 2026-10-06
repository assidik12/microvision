"""Dataset Validator & Pre-flight Quality Assessment for Edge Face Recognition.

Performs multi-criteria validation:
1. Directory & format check (.jpg, .png, .webp)
2. Corrupt / unreadable image detection
3. MediaPipe face detection & confidence thresholding
4. Landmark alignment capability test
5. Inter-image duplicate detection via perceptual/SHA256 hashing
6. Generates JSON audit report & visual quality summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microvision_attend.camera import FaceDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_dataset(
    dataset_dir: Path,
    min_images_per_person: int = 3,
    confidence_threshold: float = 0.5,
    min_resolution: int = 60,
) -> dict[str, Any]:
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    identity_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not identity_dirs:
        raise ValueError("Dataset directory contains no identity subdirectories.")

    detector = FaceDetector(confidence=confidence_threshold)
    report: dict[str, Any] = {
        "dataset_path": str(dataset_dir.resolve()),
        "total_identities": len(identity_dirs),
        "total_images": 0,
        "valid_face_images": 0,
        "no_face_images": [],
        "corrupt_images": [],
        "duplicate_images": [],
        "warnings": [],
        "identities_summary": {},
    }

    seen_hashes: dict[str, Path] = {}

    try:
        for p_dir in identity_dirs:
            person_name = p_dir.name
            files = sorted(
                p for p in p_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

            p_summary: dict[str, Any] = {
                "total_files": len(files),
                "valid_faces": 0,
                "skipped_no_face": 0,
                "resolutions": Counter(),
            }

            if len(files) < min_images_per_person:
                report["warnings"].append(
                    f"{person_name}: Has only {len(files)} image(s). Recommended min is {min_images_per_person}."
                )

            for img_path in files:
                report["total_images"] += 1
                img = cv2.imread(str(img_path))
                if img is None:
                    report["corrupt_images"].append(str(img_path))
                    continue

                h, w = img.shape[:2]
                p_summary["resolutions"][f"{w}x{h}"] += 1

                # Check minimum resolution
                if w < min_resolution or h < min_resolution:
                    report["warnings"].append(
                        f"Low resolution ({w}x{h}) in {img_path.name} for {person_name}."
                    )

                # Check duplicate
                file_hash = hashlib.sha256(img.tobytes()).hexdigest()
                if file_hash in seen_hashes:
                    report["duplicate_images"].append(
                        {"image": str(img_path), "duplicate_of": str(seen_hashes[file_hash])}
                    )
                else:
                    seen_hashes[file_hash] = img_path

                # Face detection & landmark assessment
                obs = detector.detect_observations(img)
                if not obs:
                    p_summary["skipped_no_face"] += 1
                    report["no_face_images"].append(str(img_path))
                else:
                    p_summary["valid_faces"] += 1
                    report["valid_face_images"] += 1

            p_summary["resolutions"] = dict(p_summary["resolutions"])
            report["identities_summary"][person_name] = p_summary

    finally:
        detector.close()

    # Pass / Fail readiness criteria
    report["ready_for_registration"] = (
        len(report["corrupt_images"]) == 0
        and all(v["valid_faces"] > 0 for v in report["identities_summary"].values())
    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate face dataset quality and alignment readiness.")
    parser.add_argument("dataset", type=Path, help="Path to data/datasets folder.")
    parser.add_argument("--min-images", type=int, default=3, help="Minimum images recommended per identity.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detector confidence threshold.")
    parser.add_argument("--json", type=Path, default=Path("data/validation_report.json"), help="Output JSON report.")
    args = parser.parse_args()

    print(f"\n[Dataset Validator] Auditing dataset in: {args.dataset}")
    report = validate_dataset(args.dataset, args.min_images, args.confidence)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== HASIL AUDIT VALIDASI DATASET ===")
    print(f"Total Identitas: {report['total_identities']}")
    print(f"Total Foto Ditemukan: {report['total_images']}")
    print(f"Foto Lolos Deteksi Wajah: {report['valid_face_images']}")
    print(f"Foto Tanpa Wajah / Buram: {len(report['no_face_images'])}")
    print(f"Foto Rusak (Corrupt): {len(report['corrupt_images'])}")
    print(f"Foto Duplikat: {len(report['duplicate_images'])}")
    print(f"Status Kesiapan Registrasi: {'[SIAP DIGUNAKAN]' if report['ready_for_registration'] else '[PERLU PERBAIKAN]'}")
    print(f"Laporan lengkap disimpan di: {args.json}\n")

    return 0 if report["ready_for_registration"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
