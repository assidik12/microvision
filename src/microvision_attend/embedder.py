from __future__ import annotations

from typing import Any

import numpy as np


class TFLiteEmbedder:
    """TFLite embedding inference with float and quantized tensor support."""

    def __init__(self, model_path: str, input_size: int = 112) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            except (ImportError, AttributeError):
                try:
                    from tensorflow.lite.python.interpreter import Interpreter
                except ImportError as error:
                    raise RuntimeError(
                        "Install tflite-runtime on edge Linux or tensorflow on Windows."
                    ) from error
        self._interpreter: Any = Interpreter(model_path=model_path, num_threads=2)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        self._input_size = input_size

    def __call__(self, face_rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(face_rgb)
        if image.shape[:2] != (self._input_size, self._input_size):
            raise ValueError("face crop must match the configured model input size")
        tensor = self._preprocess(image)
        self._interpreter.set_tensor(self._input["index"], tensor)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output["index"])
        scale, zero_point = self._output["quantization"]
        if scale:
            output = (output.astype(np.float32) - zero_point) * scale
        vector = np.asarray(output).reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        input_dtype = self._input["dtype"]
        if input_dtype == np.float32:
            return ((image.astype(np.float32) / 127.5) - 1.0)[None, ...]
        scale, zero_point = self._input["quantization"]
        if not scale:
            raise ValueError("quantized model input is missing scale")
        tensor = np.round(image.astype(np.float32) / scale + zero_point)
        info = np.iinfo(input_dtype)
        return np.clip(tensor, info.min, info.max).astype(input_dtype)[None, ...]
