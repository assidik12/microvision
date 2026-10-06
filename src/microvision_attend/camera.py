from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np


class FaceObservation:
    def __init__(
        self,
        box: tuple[int, int, int, int],
        confidence: float,
        landmarks: np.ndarray | None,
    ) -> None:
        self.box = box
        self.confidence = confidence
        self.landmarks = landmarks


class FaceDetector:
    """Reusable MediaPipe detector with optional eye landmarks for alignment."""

    def __init__(
        self,
        confidence: float,
        model_path: str | None = None,
        landmark_model_path: str | None = None,
    ) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError("Install opencv-python and mediapipe first.") from error
        self._cv2: Any = cv2
        self._mp = mp
        self._is_tasks = False
        self._min_confidence = confidence
        self._landmarker: Any = None
        self._landmarker_is_tasks = False

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=confidence
            )
        else:
            from pathlib import Path
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            mpath = model_path or "models/blaze_face_short_range.tflite"
            if not Path(mpath).is_file():
                import urllib.request
                Path(mpath).parent.mkdir(parents=True, exist_ok=True)
                url = (
                    "https://storage.googleapis.com/mediapipe-models/face_detector/"
                    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
                )
                urllib.request.urlretrieve(url, mpath)

            base_opts = mp_python.BaseOptions(model_asset_path=str(mpath))
            options = vision.FaceDetectorOptions(
                base_options=base_opts,
                min_detection_confidence=confidence,
            )
            self._detector = vision.FaceDetector.create_from_options(options)
            self._is_tasks = True
            landmark_path = landmark_model_path or "models/face_landmarker.task"
            if not Path(landmark_path).is_file():
                import urllib.request
                Path(landmark_path).parent.mkdir(parents=True, exist_ok=True)
                landmark_url = (
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                    "face_landmarker/float16/latest/face_landmarker.task"
                )
                urllib.request.urlretrieve(landmark_url, landmark_path)
            landmark_base = mp_python.BaseOptions(model_asset_path=str(landmark_path))
            landmark_options = vision.FaceLandmarkerOptions(
                base_options=landmark_base,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=10,
                min_face_detection_confidence=confidence,
                min_face_presence_confidence=confidence,
                min_tracking_confidence=confidence,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(landmark_options)
            self._landmarker_is_tasks = True
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._landmarker = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=confidence,
                min_tracking_confidence=confidence,
            )

    def detect_observations(self, frame_bgr: np.ndarray) -> list[FaceObservation]:
        """Return high-confidence detections and normalized face landmarks."""
        height, width = frame_bgr.shape[:2]
        detections: list[tuple[tuple[int, int, int, int], float]] = []

        if self._is_tasks:
            rgb_frame = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            result = self._detector.detect(mp_image)
            for detection in result.detections or []:
                confidence = float(detection.categories[0].score)
                if confidence < self._min_confidence:
                    continue
                bb = detection.bounding_box
                x = max(0, int(bb.origin_x))
                y = max(0, int(bb.origin_y))
                right = min(width, x + int(bb.width))
                bottom = min(height, y + int(bb.height))
                if right > x and bottom > y:
                    detections.append(((x, y, right - x, bottom - y), confidence))
        else:
            result = self._detector.process(
                self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
            )
            for detection in result.detections or []:
                confidence = float(detection.score[0])
                if confidence < self._min_confidence:
                    continue
                relative = detection.location_data.relative_bounding_box
                x = max(0, int(relative.xmin * width))
                y = max(0, int(relative.ymin * height))
                right = min(width, int((relative.xmin + relative.width) * width))
                bottom = min(height, int((relative.ymin + relative.height) * height))
                if right > x and bottom > y:
                    detections.append(((x, y, right - x, bottom - y), confidence))

        landmarks = self._detect_landmarks(frame_bgr)
        observations: list[FaceObservation] = []
        for box, confidence in detections:
            points = self._landmarks_for_box(landmarks, box, width, height)
            observations.append(FaceObservation(box, confidence, points))
        return observations

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Compatibility helper returning only high-confidence bounding boxes."""
        return [observation.box for observation in self.detect_observations(frame_bgr)]

    def _detect_landmarks(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        if self._landmarker is None:
            return []
        rgb_frame = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        if self._landmarker_is_tasks:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            result = self._landmarker.detect(mp_image)
            faces = result.face_landmarks or []
            return [
                np.asarray([(landmark.x, landmark.y) for landmark in face], dtype=np.float32)
                for face in faces
            ]
        else:
            result = self._landmarker.process(rgb_frame)
            faces = result.multi_face_landmarks or []
        return [
            np.asarray([(landmark.x, landmark.y) for landmark in face.landmark], dtype=np.float32)
            for face in faces
        ]

    @staticmethod
    def _landmarks_for_box(
        faces: list[np.ndarray],
        box: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> np.ndarray | None:
        if not faces:
            return None
        x, y, box_width, box_height = box
        center = np.array([x + box_width / 2, y + box_height / 2])
        face = min(
            faces,
            key=lambda points: np.linalg.norm(
                points[[33, 263]].mean(axis=0) * [width, height] - center
            ),
        )
        return face * [width, height]

    def close(self) -> None:
        self._detector.close()
        if self._landmarker is not None:
            self._landmarker.close()


def detect_faces(frame_bgr: np.ndarray, confidence: float) -> list[tuple[int, int, int, int]]:
    """Compatibility helper for one-shot detection."""
    detector = FaceDetector(confidence)
    try:
        return detector.detect(frame_bgr)
    finally:
        detector.close()


def crop_face(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    size: int,
    landmarks: np.ndarray | None = None,
) -> np.ndarray:
    import cv2

    if landmarks is not None and len(landmarks) > 263:
        left_eye, right_eye = landmarks[33], landmarks[263]
        center = (left_eye + right_eye) / 2.0
        angle = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
        distance = float(np.linalg.norm(right_eye - left_eye))
        if distance > 1.0:
            target_center = np.array([size / 2, size * 0.43], dtype=np.float32)
            target_eye_distance = size * 0.36
            matrix = cv2.getRotationMatrix2D(tuple(center), angle, target_eye_distance / distance)
            matrix[:, 2] += target_center - matrix[:, :2] @ center
            aligned = cv2.warpAffine(
                frame_bgr, matrix, (size, size), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

    x, y, width, height = box
    crop = frame_bgr[y : y + height, x : x + width]
    return cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB)


def frames(camera_index: int) -> Iterator[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open camera index {camera_index}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera frame read failed")
            yield frame
    finally:
        capture.release()
