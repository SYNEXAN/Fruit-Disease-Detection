"""
Fruit Disease Detection - Inference Module
Uses Pre-trained & Fine-tuned EfficientNetB0
Test Accuracy: 88.61%, Average F1-score: 88.48%
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from PIL import Image
import cv2

# Import class definitions
from classes import FRUIT_CLASSES, FRUIT_VN_MAP, DISEASE_INFO, get_pathogen_info

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

IMG_SIZE = (224, 224)

# Global variables for caching loaded model
_MODEL = None
_MODEL_TYPE = None  # 'keras' or 'tflite'
_TFLITE_INTERPRETER = None
_TFLITE_DETAILS = None
_TOP_GRAD_FN = None
_LAST_MODEL_ID = None


def _get_top_grad_fn(model):
    """Compile and cache tf.function for top layers gradient computation."""
    global _TOP_GRAD_FN, _LAST_MODEL_ID
    if _TOP_GRAD_FN is not None and _LAST_MODEL_ID == id(model):
        return _TOP_GRAD_FN

    import tensorflow as tf
    top_layers = model.layers[1:]

    @tf.function
    def compute_top_grads(conv_outputs):
        with tf.GradientTape() as tape:
            tape.watch(conv_outputs)
            x = conv_outputs
            for layer in top_layers:
                x = layer(x, training=False)
            preds = x
            top_class = preds[:, tf.argmax(preds[0])]
        grads = tape.gradient(top_class, conv_outputs)
        return grads, preds

    _TOP_GRAD_FN = compute_top_grads
    _LAST_MODEL_ID = id(model)
    return _TOP_GRAD_FN


def resolve_model_path() -> Optional[Path]:
    """Search for available EfficientNetB0 model files (prioritizing 88% TFLite baseline)."""
    candidate_paths = [
        MODEL_DIR / "EfficientNetB0_model.tflite",
        MODEL_DIR / "EfficientNetB0.tflite",
        BASE_DIR / "EfficientNetB0.tflite",
        MODEL_DIR / "EfficientNetB0_final.keras",
        MODEL_DIR / "EfficientNetB0_best.keras",
        MODEL_DIR / "EfficientNetB0.keras",
        MODEL_DIR / "EfficientNetB0_final_finetuned.keras",
        MODEL_DIR / "best_model.keras",
        BASE_DIR / "EfficientNetB0_final.keras"
    ]
    for p in candidate_paths:
        if p.is_file():
            return p
    return None


def get_model():
    """Load and cache the model."""
    global _MODEL, _MODEL_TYPE, _TFLITE_INTERPRETER, _TFLITE_DETAILS
    if _MODEL is not None or _TFLITE_INTERPRETER is not None:
        return _MODEL or _TFLITE_INTERPRETER

    model_path = resolve_model_path()
    if not model_path:
        print("[Inference] Warning: Model file not yet found. Running in mock/standby mode.")
        return None

    print(f"[Inference] Loading model from: {model_path}")
    if model_path.suffix == ".tflite":
        interpreter = None
        # Try ai_edge_litert (new official TFLite), tflite_runtime, then tensorflow
        try:
            from ai_edge_litert.interpreter import Interpreter
            interpreter = Interpreter(model_path=str(model_path))
        except Exception:
            try:
                import tflite_runtime.interpreter as tflite
                interpreter = tflite.Interpreter(model_path=str(model_path))
            except Exception:
                try:
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=str(model_path))
                except Exception as e:
                    raise RuntimeError(f"Could not load TFLite runtime: {e}")

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        _TFLITE_INTERPRETER = interpreter
        _TFLITE_DETAILS = (input_details, output_details)
        _MODEL_TYPE = 'tflite'
        return _TFLITE_INTERPRETER

    else:
        # Load Keras model
        import tensorflow as tf
        _MODEL = tf.keras.models.load_model(str(model_path), compile=False)
        _MODEL_TYPE = 'keras'
        return _MODEL


def preprocess_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    return_full_res: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load and preprocess image for EfficientNetB0.
    Returns:
        (processed_input_tensor, original_rgb_uint8_resized) or
        (processed_input_tensor, original_rgb_uint8_resized, original_rgb_full_res)
    """
    if isinstance(image, (str, Path)):
        img_pil = Image.open(str(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            img_pil = Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 4:
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
        else:
            img_pil = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        img_pil = image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image)}")

    orig_full_rgb = np.array(img_pil, dtype=np.uint8)
    img_resized = img_pil.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)

    # EfficientNet preprocessing: rescale / zero-center as per tf.keras
    try:
        import tensorflow as tf
        processed = tf.keras.applications.efficientnet.preprocess_input(img_array.copy())
    except Exception:
        # Fallback to standard EfficientNet preprocessing: pass-through or norm
        processed = img_array

    input_tensor = np.expand_dims(processed, axis=0)
    orig_rgb_uint8 = np.array(img_resized, dtype=np.uint8)

    if return_full_res:
        return input_tensor, orig_rgb_uint8, orig_full_rgb
    return input_tensor, orig_rgb_uint8


def make_gradcam_heatmap_and_preds(
    img_array: np.ndarray,
    model,
    last_conv_layer_name: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate Grad-CAM heatmap AND prediction distribution in a SINGLE combined forward pass.
    Avoids duplicate forward pass and reduces latency significantly.
    Returns:
        (heatmap, predictions_vector)
    """
    import tensorflow as tf

    preds = None
    heatmap = None

    # 1. If Sequential with base feature extractor (EfficientNetB0 backbone)
    if hasattr(model, 'layers') and len(model.layers) > 1 and isinstance(model.layers[0], tf.keras.Model):
        base_model = model.layers[0]
        try:
            conv_outputs = base_model(img_array, training=False)
            grad_fn = _get_top_grad_fn(model)
            grads, predictions = grad_fn(conv_outputs)

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs_0 = conv_outputs[0]
            hm = conv_outputs_0 @ pooled_grads[..., tf.newaxis]
            hm = tf.squeeze(hm)
            hm = tf.maximum(hm, 0) / (tf.math.reduce_max(hm) + 1e-10)
            heatmap = hm.numpy()
            preds = predictions.numpy()[0]
            return heatmap, preds
        except Exception as e:
            print(f"[Grad-CAM] Sequential tape error: {e}")

    # 2. Standard Functional model
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() or isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            last_conv_layer_name = 'top_conv'

    try:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            top_class_channel = predictions[:, tf.argmax(predictions[0])]

        grads = tape.gradient(top_class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        hm = conv_outputs @ pooled_grads[..., tf.newaxis]
        hm = tf.squeeze(hm)
        hm = tf.maximum(hm, 0) / (tf.math.reduce_max(hm) + 1e-10)
        heatmap = hm.numpy()
        preds = predictions.numpy()[0]
        return heatmap, preds
    except Exception as e:
        print(f"[Grad-CAM] Error generating heatmap: {e}")
        return None, None


def make_gradcam_heatmap(img_array: np.ndarray, model, last_conv_layer_name: Optional[str] = None) -> Optional[np.ndarray]:
    """Generate Grad-CAM heatmap using TensorFlow GradientTape (wrapper)."""
    heatmap, _ = make_gradcam_heatmap_and_preds(img_array, model, last_conv_layer_name)
    return heatmap


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay 2D heatmap on original RGB image."""
    h, w, _ = original_img.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(original_img, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)
    return blended


def generate_tflite_cam(
    interpreter,
    input_tensor: np.ndarray,
    orig_rgb: np.ndarray,
    top_class: int,
    base_score: float,
    is_healthy: bool = False
) -> np.ndarray:
    """
    Generate Class Activation Map (CAM) for TFLite via fruit-masked perturbation.
    Accurately isolates lesions on the fruit surface and prevents background spill.
    """
    input_data = input_tensor[0]
    h, w, c = input_data.shape

    # 1. Compute robust fruit foreground mask
    # Background in agricultural datasets is typically at the corners/borders
    orig_resized = cv2.resize(orig_rgb, (w, h))
    corners = np.concatenate([
        orig_resized[:20, :20].reshape(-1, 3),
        orig_resized[:20, -20:].reshape(-1, 3),
        orig_resized[-20:, :20].reshape(-1, 3),
        orig_resized[-20:, -20:].reshape(-1, 3)
    ], axis=0)
    bg_color = np.median(corners, axis=0)
    diff_from_bg = np.linalg.norm(orig_resized.astype(np.float32) - bg_color, axis=-1)

    # Soft foreground mask
    fruit_mask = cv2.GaussianBlur(diff_from_bg, (15, 15), 0)
    thresh = np.percentile(fruit_mask, 28)
    fruit_mask = (fruit_mask > thresh).astype(np.float32)
    fruit_mask = cv2.GaussianBlur(fruit_mask, (21, 21), 0)

    # 2. Fruit-focused overlapping patch grid
    patch_size = 48
    stride = 24
    coords = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Only test patches that intersect the fruit body
            if np.mean(fruit_mask[y:y+patch_size, x:x+patch_size]) > 0.08:
                coords.append((y, x))

    if not coords:
        # Fallback grid if mask is too restrictive
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                coords.append((y, x))

    batch_size = len(coords)
    interpreter.resize_tensor_input(0, [batch_size, h, w, c])
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    batch_inputs = np.repeat(input_tensor, batch_size, axis=0)
    mean_val = np.mean(input_data, axis=(0, 1))

    # Smooth 2D Gaussian window for patches (avoids harsh rectangular box edges)
    gy, gx = np.meshgrid(np.linspace(-1, 1, patch_size), np.linspace(-1, 1, patch_size))
    g_mask = np.exp(-(gx**2 + gy**2) / 0.6)[:, :, np.newaxis]

    for i, (y, x) in enumerate(coords):
        patch = batch_inputs[i, y:y+patch_size, x:x+patch_size, :]
        batch_inputs[i, y:y+patch_size, x:x+patch_size, :] = patch * (1.0 - g_mask) + mean_val * g_mask

    interpreter.set_tensor(in_idx, batch_inputs)
    interpreter.invoke()
    batch_preds = interpreter.get_tensor(out_idx)

    # Restore interpreter shape
    interpreter.resize_tensor_input(0, [1, h, w, c])
    interpreter.allocate_tensors()

    # Accumulate drops weighted by Gaussian window
    heatmap = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    for i, (y, x) in enumerate(coords):
        drop = max(0.0, base_score - float(batch_preds[i, top_class]))
        heatmap[y:y+patch_size, x:x+patch_size] += drop * g_mask[:, :, 0]
        weight_map[y:y+patch_size, x:x+patch_size] += g_mask[:, :, 0]

    weight_map[weight_map == 0] = 1.0
    heatmap /= weight_map

    # Strictly suppress background spill
    heatmap *= fruit_mask

    # Subtract median noise floor inside fruit
    fruit_pixels = heatmap[fruit_mask > 0.4]
    if len(fruit_pixels) > 0:
        baseline = np.percentile(fruit_pixels, 45)
        heatmap = np.maximum(0.0, heatmap - baseline)

    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    return overlay_heatmap(orig_rgb, heatmap, alpha=0.45)


def predict_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    conf_threshold: float = 0.05,
    generate_cam: bool = True
) -> Dict[str, Any]:
    """
    Perform end-to-end inference on a single image.
    Returns:
        dict with keys: 'predictions', 'best', 'gradcam_image', 'disease_info'
    """
    model = get_model()
    input_tensor, orig_rgb, orig_full_rgb = preprocess_image(image, return_full_res=True)

    if model is None:
        # Standby mock response if model weight is currently downloading
        return {
            "predictions": [
                {"class": "APPLE_Rot_Apple", "vn_name": FRUIT_VN_MAP["APPLE_Rot_Apple"], "confidence": 0.8861, "percentage": "88.6%"}
            ],
            "best": {"class": "APPLE_Rot_Apple", "vn_name": FRUIT_VN_MAP["APPLE_Rot_Apple"], "confidence": 0.8861},
            "gradcam_image": orig_full_rgb,
            "disease_info": DISEASE_INFO.get("APPLE_Rot_Apple", {})
        }

    # Inference
    if _MODEL_TYPE == 'tflite':
        input_details, output_details = _TFLITE_DETAILS
        _TFLITE_INTERPRETER.set_tensor(input_details[0]['index'], input_tensor.astype(input_details[0]['dtype']))
        _TFLITE_INTERPRETER.invoke()
        preds = _TFLITE_INTERPRETER.get_tensor(output_details[0]['index'])[0]
        gradcam_img = orig_full_rgb
        if generate_cam:
            try:
                top_class = int(np.argmax(preds))
                class_name = FRUIT_CLASSES[top_class] if top_class < len(FRUIT_CLASSES) else ""
                is_healthy = "healthy" in class_name.lower()
                gradcam_img = generate_tflite_cam(
                    _TFLITE_INTERPRETER, input_tensor, orig_full_rgb, top_class, float(preds[top_class]), is_healthy=is_healthy
                )
            except Exception as cam_err:
                print(f"[Grad-CAM TFLite] Warning: {cam_err}")
    else:
        # Keras model inference - Optimized Unified Pass
        preds = None
        gradcam_img = orig_full_rgb
        if generate_cam:
            try:
                heatmap, preds = make_gradcam_heatmap_and_preds(input_tensor, model)
                if heatmap is not None:
                    gradcam_img = overlay_heatmap(orig_full_rgb, heatmap, alpha=0.45)
            except Exception as e:
                print(f"[Grad-CAM] Single-pass error: {e}")

        # Fallback if preds wasn't generated in Grad-CAM (e.g. generate_cam=False or tape failure)
        if preds is None:
            preds = model(input_tensor, training=False).numpy()[0]

    # Top predictions
    sorted_indices = np.argsort(preds)[::-1]
    predictions = []
    for idx in sorted_indices:
        score = float(preds[idx])
        if score < conf_threshold and len(predictions) >= 3:
            break
        class_name = FRUIT_CLASSES[idx] if idx < len(FRUIT_CLASSES) else f"Class_{idx}"
        predictions.append({
            "class": class_name,
            "vn_name": FRUIT_VN_MAP.get(class_name, class_name),
            "confidence": score,
            "percentage": f"{score * 100:.2f}%"
        })

    best_pred = predictions[0] if predictions else {"class": "Unknown", "vn_name": "Không xác định", "confidence": 0.0, "percentage": "0.0%"}
    info = DISEASE_INFO.get(best_pred["class"], {
        "status": "info",
        "fruit": "Trái cây",
        "severity": "Không xác định",
        "description": "Không có thông tin mô tả chi tiết.",
        "treatment": "Kiểm tra thêm điều kiện chăm sóc vườn cây."
    })
    pathogen = get_pathogen_info(best_pred["class"])

    return {
        "predictions": predictions[:5],
        "best": best_pred,
        "gradcam_image": gradcam_img,
        "disease_info": info,
        "pathogen_info": pathogen
    }


def warmup_model():
    """
    Pre-warm the model by executing one dummy inference and Grad-CAM run.
    Eliminates cold-start loading and TensorFlow JIT graph compilation delays.
    """
    model = get_model()
    if model is not None:
        try:
            dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            _ = predict_image(dummy_img, generate_cam=True)
            print("[Inference] Model pre-warming completed successfully.")
        except Exception as e:
            print(f"[Inference] Pre-warming notice: {e}")
