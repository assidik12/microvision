"""Real-time face recognition with MediaPipe alignment and a TFLite embedder.

The supplied TFLite model must be a face-embedding model (for example
MobileFaceNet), not a classifier. Reference images are aligned and averaged
into one normalized prototype per identity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microvision_attend.camera import FaceDetector, crop_face
from microvision_attend.embedder import TFLiteEmbedder
from microvision_attend.matcher import EmbeddingRegistry

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read reference image: {path}")
    return image


def _largest_observation(detector: FaceDetector, image: np.ndarray):
    observations = detector.detect_observations(image)
    if not observations:
        raise ValueError("No face passed the detection confidence threshold")
    return max(observations, key=lambda item: item.box[2] * item.box[3])


def register_identity(
    name: str,
    image_paths: list[Path],
    detector: FaceDetector,
    embedder: TFLiteEmbedder,
    registry_path: Path,
) -> bool:
    vectors: list[np.ndarray] = []
    skipped = 0
    for image_path in image_paths:
        try:
            image = _read_image(image_path)
            observation = _largest_observation(detector, image)
            face = crop_face(image, observation.box, 112, observation.landmarks)
            vectors.append(embedder(face))
        except ValueError as err:
            skipped += 1
            print(f"  [Notice] Skipping image {image_path.name} for {name}: {err}")
            continue

    if not vectors:
        print(f"  [Warning] No valid face found across all {len(image_paths)} images for {name!r}!")
        return False

    prototype = np.mean(vectors, axis=0)
    prototype /= np.linalg.norm(prototype)

    payload = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else []
    payload = [item for item in payload if str(item.get("name")) != name]
    payload.append({"name": name, "embedding": prototype.astype(float).tolist()})
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Registered {name!r} from {len(vectors)} valid face image(s) (skipped {skipped}).")
    return True


def register_dataset(
    dataset_path: Path,
    detector: FaceDetector,
    embedder: TFLiteEmbedder,
    registry_path: Path,
) -> None:
    if not dataset_path.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_path}")
    identity_dirs = sorted(path for path in dataset_path.iterdir() if path.is_dir())
    if not identity_dirs:
        raise ValueError("Dataset must contain one subdirectory per identity.")

    registered_count = 0
    for identity_dir in identity_dirs:
        image_paths = sorted(
            path
            for path in identity_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            print(f"Skipping {identity_dir.name!r}: no supported image files.")
            continue
        if register_identity(
            identity_dir.name,
            image_paths,
            detector,
            embedder,
            registry_path,
        ):
            registered_count += 1
    print(f"\n[Summary] Successfully registered {registered_count}/{len(identity_dirs)} identities into {registry_path}.")


def run_camera(
    camera_index: int,
    detector: FaceDetector,
    embedder: TFLiteEmbedder,
    registry: EmbeddingRegistry,
    threshold: float,
) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera frame read failed")
            for observation in detector.detect_observations(frame):
                embedding = embedder(
                    crop_face(frame, observation.box, 112, observation.landmarks)
                )
                match = registry.match(embedding, threshold)
                name = match.name or "Unknown"
                color = (0, 200, 0) if match.name else (0, 140, 255)
                x, y, width, height = observation.box
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(
                    frame,
                    f"{name} {match.score:.3f}",
                    (x, max(25, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("TFLite Face Recognition - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def visualize_embeddings(registry_path: Path, output_image: Path | None = None) -> None:
    """Visualize inter-class cosine similarity matrix and embedding metrics using Matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise RuntimeError("Install matplotlib to use visualization.") from err

    if not registry_path.is_file():
        raise FileNotFoundError(f"Registry file not found at {registry_path}")

    data = json.loads(registry_path.read_text(encoding="utf-8"))
    if not data:
        print("Registry is empty, nothing to visualize.")
        return

    names = [item["name"] for item in data]
    embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)

    # Normalize vectors just in case
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.where(norms > 0, embeddings / norms, embeddings)

    # Compute pairwise cosine similarity matrix
    sim_matrix = embeddings @ embeddings.T

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])

    # 1. Heatmap
    ax_heatmap = fig.add_subplot(gs[0, :])
    cax = ax_heatmap.imshow(sim_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(cax, ax=ax_heatmap, label="Cosine Similarity")

    ax_heatmap.set_xticks(range(len(names)))
    ax_heatmap.set_yticks(range(len(names)))
    ax_heatmap.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax_heatmap.set_yticklabels(names, fontsize=9)
    ax_heatmap.set_title("MobileFaceNet Face Embedding Similarity Matrix (ArcFace)", fontsize=13, pad=12)

    # Annotate values if small class size
    if len(names) <= 20:
        for i in range(len(names)):
            for j in range(len(names)):
                val = sim_matrix[i, j]
                color = "black" if val > 0.7 else "white"
                ax_heatmap.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    # 2. Distribution histogram of off-diagonal (inter-class) similarities
    ax_hist = fig.add_subplot(gs[1, 0])
    mask = ~np.eye(len(names), dtype=bool)
    if np.any(mask):
        inter_sims = sim_matrix[mask]
        ax_hist.hist(inter_sims, bins=20, color="#2b5c8f", edgecolor="black", alpha=0.8)
        mean_val = float(np.mean(inter_sims))
        max_val = float(np.max(inter_sims))
        ax_hist.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
        ax_hist.axvline(0.5, color="green", linestyle=":", label="Default Threshold: 0.50")
        ax_hist.set_xlabel("Inter-Class Cosine Similarity")
        ax_hist.set_ylabel("Pair Count")
        ax_hist.set_title("Inter-Class Separation Distribution", fontsize=10)
        ax_hist.legend(fontsize=8)
        ax_hist.grid(alpha=0.3)

    # 3. Summary text card
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    stats_text = (
        f"Registry Stats:\n\n"
        f"- Identities: {len(names)}\n"
        f"- Vector Dim: {embeddings.shape[1]}-D\n"
        f"- Avg Inter-sim: {np.mean(sim_matrix[mask]):.3f}\n"
        f"- Max Cross-sim: {np.max(sim_matrix[mask]):.3f}\n"
        f"- Min Cross-sim: {np.min(sim_matrix[mask]):.3f}\n"
    )
    ax_text.text(0.1, 0.5, stats_text, fontsize=10, va="center", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#b0c4de"))

    fig.tight_layout()

    out = output_image or Path("models/embeddings_similarity.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"\n[Visualization] Similarity chart saved to: {out}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="MobileFaceNet/TFLite webcam recognition.")
    parser.add_argument("--model", type=Path, default=Path("models/mobilefacenet.tflite"), help="TFLite embedding model.")
    parser.add_argument("--registry", type=Path, default=Path("data/embeddings.json"))
    parser.add_argument("--detector-model", type=Path, default=Path("models/blaze_face_short_range.tflite"))
    parser.add_argument("--landmarker-model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--register-name", help="Register/update this identity instead of opening webcam.")
    parser.add_argument("--reference-images", type=Path, nargs="+")
    parser.add_argument(
        "--register-dataset",
        type=Path,
        help="Register every identity subdirectory in this dataset.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize embeddings cosine similarity matrix and inter-class distribution using Matplotlib.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("models/embeddings_similarity.png"),
        help="Path to save the generated Matplotlib chart.",
    )
    args = parser.parse_args()

    if args.visualize:
        visualize_embeddings(args.registry, args.plot_output)
        return 0

    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if not -1.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between -1 and 1")
    if not args.model.is_file():
        parser.error(
            f"TFLite model not found: {args.model}. "
            "Copy a MobileFaceNet embedding model to that path or pass --model with its actual path."
        )
    if bool(args.register_name) != bool(args.reference_images):
        parser.error("--register-name and --reference-images must be supplied together")
    if args.register_name and args.register_dataset:
        parser.error("Use either --register-name or --register-dataset, not both")

    detector = FaceDetector(
        args.confidence, str(args.detector_model), str(args.landmarker_model)
    )
    try:
        embedder = TFLiteEmbedder(str(args.model), input_size=112)
        if args.register_name:
            register_identity(
                args.register_name,
                args.reference_images,
                detector,
                embedder,
                args.registry,
            )
        elif args.register_dataset:
            register_dataset(args.register_dataset, detector, embedder, args.registry)
        else:
            registry = EmbeddingRegistry.load(args.registry)
            run_camera(args.camera, detector, embedder, registry, args.threshold)
    finally:
        detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
