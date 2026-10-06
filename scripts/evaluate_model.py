"""Face Recognition Performance & Separation Evaluator (Edge Benchmark).

Evaluates:
1. Intra-class variance (consistency across photos of the same person)
2. Inter-class separation margin (confusion between different people)
3. Precision, Recall, and Accuracy across varying Cosine thresholds (ROC / Threshold curve)
4. Generates comprehensive Matplotlib evaluation report:
   - Heatmap of inter-class similarities
   - Similarity distribution curves (Same person vs Different people)
   - Threshold vs False Accept Rate (FAR) & False Reject Rate (FRR)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microvision_attend.camera import FaceDetector, crop_face
from microvision_attend.embedder import TFLiteEmbedder

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def evaluate_recognition_pipeline(
    dataset_dir: Path,
    model_path: Path,
    output_plot: Path,
    samples_per_person: int = 10,
    confidence: float = 0.5,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise RuntimeError("Install matplotlib to generate evaluation charts.") from err

    detector = FaceDetector(confidence=confidence)
    embedder = TFLiteEmbedder(str(model_path), input_size=112)

    identity_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not identity_dirs:
        raise ValueError("No subdirectories found in dataset.")

    embeddings_by_id: dict[str, list[np.ndarray]] = {}

    print(f"\n[Evaluator] Extracting MobileFaceNet embeddings from dataset...")
    try:
        for p_dir in identity_dirs:
            person = p_dir.name
            files = sorted(
                p for p in p_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )[:samples_per_person]

            vectors = []
            for f in files:
                import cv2
                img = cv2.imread(str(f))
                if img is None:
                    continue
                obs = detector.detect_observations(img)
                if not obs:
                    continue
                largest = max(obs, key=lambda item: item.box[2] * item.box[3])
                crop = crop_face(img, largest.box, 112, largest.landmarks)
                vectors.append(embedder(crop))

            if len(vectors) >= 2:
                embeddings_by_id[person] = vectors
                print(f"  • {person}: {len(vectors)} valid test embeddings extracted.")
    finally:
        detector.close()

    names = list(embeddings_by_id.keys())
    if len(names) < 2:
        raise ValueError("Need at least 2 identities with >= 2 images each for benchmark.")

    # 1. Compute Intra-class similarities (same person)
    same_person_scores: list[float] = []
    prototypes: list[np.ndarray] = []

    for name in names:
        vecs = np.array(embeddings_by_id[name])
        proto = np.mean(vecs, axis=0)
        proto /= np.linalg.norm(proto)
        prototypes.append(proto)

        # Pairwise inner similarity
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                same_person_scores.append(float(np.dot(vecs[i], vecs[j])))

    # 2. Compute Inter-class similarities (different people prototypes)
    proto_matrix = np.array(prototypes)
    inter_matrix = proto_matrix @ proto_matrix.T
    mask = ~np.eye(len(names), dtype=bool)
    diff_person_scores = inter_matrix[mask].tolist()

    same_arr = np.array(same_person_scores)
    diff_arr = np.array(diff_person_scores)

    # 3. Compute FAR and FRR curves over thresholds [0.1 to 0.9]
    thresholds = np.linspace(0.1, 0.9, 81)
    far_list, frr_list, f1_list = [], [], []

    for th in thresholds:
        # False Accept: different person with score >= th
        far = np.mean(diff_arr >= th) if len(diff_arr) else 0.0
        # False Reject: same person with score < th
        frr = np.mean(same_arr < th) if len(same_arr) else 0.0
        far_list.append(far)
        frr_list.append(frr)
        
        tp = np.sum(same_arr >= th)
        fp = np.sum(diff_arr >= th)
        fn = np.sum(same_arr < th)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

    best_idx = int(np.argmax(f1_list))
    best_th = float(thresholds[best_idx])
    best_f1 = float(f1_list[best_idx])

    # 4. Generate Matplotlib Comprehensive Report
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # A: Similarity Heatmap
    im = axes[0, 0].imshow(inter_matrix, cmap="coolwarm", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=axes[0, 0], label="Cosine Similarity")
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_yticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0, 0].set_yticklabels(names, fontsize=8)
    axes[0, 0].set_title("Prototype Cross-Similarity Matrix", fontsize=11, fontweight="bold")

    # B: Distribution Histogram
    axes[0, 1].hist(same_arr, bins=25, alpha=0.6, color="green", density=True, label="Same Identity (Intra-class)")
    axes[0, 1].hist(diff_arr, bins=25, alpha=0.6, color="red", density=True, label="Different Identities (Inter-class)")
    axes[0, 1].axvline(best_th, color="blue", linestyle="--", linewidth=2, label=f"Optimal Threshold: {best_th:.2f}")
    axes[0, 1].set_xlabel("Cosine Similarity")
    axes[0, 1].set_ylabel("Probability Density")
    axes[0, 1].set_title("Score Distribution: Intra vs Inter Class", fontsize=11, fontweight="bold")
    axes[0, 1].legend(loc="upper left", fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # C: FAR & FRR vs Threshold
    axes[1, 0].plot(thresholds, far_list, label="FAR (False Acceptance)", color="red", linewidth=2)
    axes[1, 0].plot(thresholds, frr_list, label="FRR (False Rejection)", color="orange", linewidth=2)
    axes[1, 0].axvline(best_th, color="blue", linestyle=":", label=f"EER/Optimal: {best_th:.2f}")
    axes[1, 0].set_xlabel("Decision Threshold")
    axes[1, 0].set_ylabel("Error Rate")
    axes[1, 0].set_title("FAR & FRR Curve (Trade-off)", fontsize=11, fontweight="bold")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    # D: Metric Summary Text Card
    axes[1, 1].axis("off")
    summary_card = (
        f"MicroVision-Attend Model Evaluation Report\n"
        f"-----------------------------------------\n"
        f"• Evaluated Identities    : {len(names)}\n"
        f"• Feature Extractor       : MobileFaceNet (192-D)\n"
        f"• Mean Intra-Class Sim    : {float(np.mean(same_arr)):.3f}\n"
        f"• Mean Inter-Class Sim    : {float(np.mean(diff_arr)):.3f}\n"
        f"• Separation Margin       : {float(np.mean(same_arr) - np.mean(diff_arr)):.3f}\n"
        f"-----------------------------------------\n"
        f"• Recommended Threshold   : {best_th:.2f}\n"
        f"• FAR at Threshold        : {far_list[best_idx]*100:.2f}%\n"
        f"• FRR at Threshold        : {frr_list[best_idx]*100:.2f}%\n"
        f"• Best Verification F1    : {best_f1*100:.2f}%\n"
    )
    axes[1, 1].text(0.05, 0.5, summary_card, fontsize=10, family="monospace", va="center",
                     bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#6c757d"))

    fig.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)
    print(f"\n[Selesai] Grafik evaluasi performa disimpan di: {output_plot}")

    return {
        "identities": len(names),
        "mean_intra_sim": float(np.mean(same_arr)),
        "mean_inter_sim": float(np.mean(diff_arr)),
        "separation_margin": float(np.mean(same_arr) - np.mean(diff_arr)),
        "recommended_threshold": best_th,
        "best_f1": best_f1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MobileFaceNet verification performance.")
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets"), help="Validation dataset directory.")
    parser.add_argument("--model", type=Path, default=Path("models/mobilefacenet.tflite"), help="MobileFaceNet TFLite model.")
    parser.add_argument("--samples", type=int, default=8, help="Sample images per identity to evaluate.")
    parser.add_argument("--plot-output", type=Path, default=Path("models/evaluation_report.png"), help="Chart output path.")
    args = parser.parse_args()

    results = evaluate_recognition_pipeline(
        dataset_dir=args.dataset,
        model_path=args.model,
        output_plot=args.plot_output,
        samples_per_person=args.samples,
    )
    print("\n=== RINGKASAN BENCHMARK EVALUASI ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
