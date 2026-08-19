from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from paddleocr import PaddleOCR


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OUTPUT_ROOT = Path("output")
PROCESSED_DIR = OUTPUT_ROOT / "processed"
JSON_DIR = OUTPUT_ROOT / "json"
VISUALISATION_DIR = OUTPUT_ROOT / "visualisations"

MIN_CONFIDENCE = 0.50


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# PaddleOCR model
# ---------------------------------------------------------------------

_ocr_model: PaddleOCR | None = None


def get_ocr_model() -> PaddleOCR:
    """
    Initialize PaddleOCR only when it is first required.

    The same model instance is reused for later invoice-processing requests.

    Returns:
        Initialized PaddleOCR model.
    """
    global _ocr_model

    if _ocr_model is None:
        logger.info("Initializing PaddleOCR model...")

        _ocr_model = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        logger.info("PaddleOCR model initialized successfully.")

    return _ocr_model


# ---------------------------------------------------------------------
# OpenCV preprocessing
# ---------------------------------------------------------------------

def preprocess_invoice(image: np.ndarray) -> np.ndarray:
    """
    Apply conservative OpenCV preprocessing to an invoice image.

    The preprocessing is intentionally light because aggressive thresholding
    may remove thin characters, decimal points, and table borders.

    Args:
        image: Original BGR image loaded by OpenCV.

    Returns:
        Preprocessed BGR image suitable for PaddleOCR.

    Raises:
        ValueError: If the supplied image is empty.
    """
    if image is None or image.size == 0:
        raise ValueError("The supplied image is empty.")

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    denoised = cv2.fastNlMeansDenoising(
        grayscale,
        None,
        h=7,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(denoised)

    processed = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2BGR,
    )

    return processed


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def create_output_directories() -> None:
    """Create all required backend output directories."""
    for directory in (
        PROCESSED_DIR,
        JSON_DIR,
        VISUALISATION_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def convert_bbox_to_integer_list(
    polygon: Any,
) -> list[list[int]]:
    """
    Convert a PaddleOCR polygon into a JSON-safe integer list.

    Expected form:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    Args:
        polygon: Bounding-box polygon returned by PaddleOCR.

    Returns:
        Four-point integer bounding box.

    Raises:
        ValueError: If the polygon shape is invalid.
    """
    polygon_array = np.asarray(polygon)

    if polygon_array.shape != (4, 2):
        raise ValueError(
            f"Expected bounding box shape (4, 2), "
            f"but received {polygon_array.shape}."
        )

    return [
        [
            int(round(float(x))),
            int(round(float(y))),
        ]
        for x, y in polygon_array
    ]


def extract_result_dictionary(
    result_object: Any,
) -> dict[str, Any]:
    """
    Extract the serializable dictionary from a PaddleOCR result object.

    PaddleOCR 3.x commonly exposes the output through the `json`
    property and may store the actual result under a top-level `res` key.

    Args:
        result_object: One PaddleOCR prediction result.

    Returns:
        Dictionary containing recognized text, scores, and polygons.

    Raises:
        TypeError: If PaddleOCR returns an unexpected representation.
    """
    result_json = result_object.json

    if callable(result_json):
        result_json = result_json()

    if not isinstance(result_json, dict):
        raise TypeError(
            "PaddleOCR returned an unexpected result representation."
        )

    if "res" in result_json:
        result_json = result_json["res"]

    if not isinstance(result_json, dict):
        raise TypeError(
            "PaddleOCR result data is not a dictionary."
        )

    return result_json


def save_processed_image(
    processed_image: np.ndarray,
    image_path: Path,
) -> Path:
    """
    Save the OpenCV-preprocessed invoice image.

    Args:
        processed_image: Preprocessed image.
        image_path: Original invoice image path.

    Returns:
        Path of the saved processed image.

    Raises:
        IOError: If the image cannot be saved.
    """
    processed_path = (
        PROCESSED_DIR
        / f"{image_path.stem}_processed{image_path.suffix}"
    )

    if not cv2.imwrite(
        str(processed_path),
        processed_image,
    ):
        raise IOError(
            f"Could not save processed image to: {processed_path}"
        )

    return processed_path


def save_json_output(
    output: list[dict[str, Any]],
    image_path: Path,
) -> Path:
    """
    Save OCR output as a backend JSON file.

    Args:
        output: OCR result.
        image_path: Original invoice image path.

    Returns:
        Path of the saved JSON file.
    """
    json_path = JSON_DIR / f"{image_path.stem}_ocr.json"

    with json_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            output,
            json_file,
            indent=2,
            ensure_ascii=False,
        )

    return json_path


def draw_ocr_boxes(
    original_image: np.ndarray,
    extracted_data: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Draw OCR polygons and confidence scores on the original image.

    Args:
        original_image: Original BGR invoice image.
        extracted_data: Extracted OCR regions.
        output_path: Path for the saved visualization.

    Raises:
        IOError: If the visualization cannot be saved.
    """
    visualisation = original_image.copy()

    for item in extracted_data:
        points = np.asarray(
            item["bbox"],
            dtype=np.int32,
        )

        cv2.polylines(
            visualisation,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        top_left_x = int(points[:, 0].min())
        top_left_y = max(
            int(points[:, 1].min()) - 8,
            20,
        )

        label = f'{item["confidence"]:.2f}'

        cv2.putText(
            visualisation,
            label,
            (top_left_x, top_left_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(
        str(output_path),
        visualisation,
    ):
        raise IOError(
            f"Could not save visualisation to: {output_path}"
        )


# ---------------------------------------------------------------------
# OCR pipeline
# ---------------------------------------------------------------------

def run_invoice_ocr(
    image_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Run OpenCV preprocessing and PaddleOCR on one invoice image.

    Args:
        image_path: Path to one uploaded invoice image.

    Returns:
        A list containing one invoice OCR result dictionary.

    Raises:
        FileNotFoundError: If the invoice does not exist.
        ValueError: If the path is invalid or the image cannot be read.
        IOError: If an output artifact cannot be saved.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Invoice image does not exist: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Invoice path is not a file: {image_path}"
        )

    create_output_directories()

    logger.info(
        "Reading invoice: %s",
        image_path,
    )

    original_image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if original_image is None:
        raise ValueError(
            f"OpenCV could not read the image: {image_path}"
        )

    image_height, image_width = original_image.shape[:2]

    logger.info(
        "Original image dimensions: width=%d, height=%d",
        image_width,
        image_height,
    )

    logger.info(
        "Applying OpenCV preprocessing..."
    )

    processed_image = preprocess_invoice(
        original_image
    )

    processed_path = save_processed_image(
        processed_image=processed_image,
        image_path=image_path,
    )

    logger.info(
        "Processed image saved to: %s",
        processed_path,
    )

    logger.info(
        "Running PaddleOCR..."
    )

    ocr_model = get_ocr_model()

    prediction_results = ocr_model.predict(
        processed_image
    )

    extracted_data: list[dict[str, Any]] = []

    for prediction_result in prediction_results:
        result = extract_result_dictionary(
            prediction_result
        )

        texts = result.get(
            "rec_texts",
            [],
        )

        scores = result.get(
            "rec_scores",
            [],
        )

        polygons = result.get(
            "rec_polys",
            [],
        )

        if not (
            len(texts)
            == len(scores)
            == len(polygons)
        ):
            raise ValueError(
                "PaddleOCR returned mismatched numbers of texts, "
                "scores, and bounding boxes."
            )

        for text, confidence, polygon in zip(
            texts,
            scores,
            polygons,
        ):
            clean_text = str(text).strip()
            confidence_value = float(confidence)

            if not clean_text:
                continue

            if confidence_value < MIN_CONFIDENCE:
                continue

            extracted_data.append(
                {
                    "text": clean_text,
                    "confidence": round(
                        confidence_value,
                        4,
                    ),
                    "bbox": convert_bbox_to_integer_list(
                        polygon
                    ),
                }
            )

    invoice_result = {
        "file_name": image_path.name,
        "img_width": int(image_width),
        "img_height": int(image_height),
        "ocr_status": "completed",
        "text_region_count": len(extracted_data),
        "extracted_data": extracted_data,
    }

    final_output = [
        invoice_result,
    ]

    json_path = save_json_output(
        output=final_output,
        image_path=image_path,
    )

    visualisation_path = (
        VISUALISATION_DIR
        / f"{image_path.stem}_bbox{image_path.suffix}"
    )

    draw_ocr_boxes(
        original_image=original_image,
        extracted_data=extracted_data,
        output_path=visualisation_path,
    )

    logger.info(
        "OCR completed. Extracted %d text regions.",
        len(extracted_data),
    )

    logger.info(
        "JSON output saved to: %s",
        json_path,
    )

    logger.info(
        "Bounding-box image saved to: %s",
        visualisation_path,
    )

    return final_output