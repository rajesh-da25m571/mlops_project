# python -m venv .venv
# .venv\Scripts\activate
# python -m pip install --upgrade pip
# pip install paddlepaddle
# pip install paddleocr
# pip install opencv-python numpy

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from paddleocr import PaddleOCR


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Folder containing all invoice images.
INPUT_FOLDER = Path(
    r"C:\Users\yabara\Downloads\batch_1\batch_1\batch1_1"
)


# Supported image formats.
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# Output folders.
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
    Initialize PaddleOCR only once.

    The same PaddleOCR model is reused for all invoice images.
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

        logger.info(
            "PaddleOCR model initialized successfully."
        )

    return _ocr_model


# ---------------------------------------------------------------------
# OpenCV preprocessing
# ---------------------------------------------------------------------

def preprocess_invoice(
    image: np.ndarray,
) -> np.ndarray:
    """
    Apply conservative OpenCV preprocessing.

    The preprocessing is intentionally light because aggressive thresholding
    can remove thin characters, decimal points and table borders.

    Args:
        image: Original BGR image loaded by OpenCV.

    Returns:
        Preprocessed BGR image suitable for PaddleOCR.
    """
    if image is None or image.size == 0:
        raise ValueError(
            "The supplied image is empty."
        )

    # Convert to grayscale.
    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Reduce small amounts of image noise while preserving text edges.
    denoised = cv2.fastNlMeansDenoising(
        grayscale,
        None,
        h=7,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # Improve local contrast.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(
        denoised
    )

    # Convert back to BGR.
    processed = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2BGR,
    )

    return processed


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def create_output_directories() -> None:
    """
    Create all output directories if they do not already exist.
    """
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
    Convert a PaddleOCR polygon into a JSON-safe
    four-point integer list.

    Expected form:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    polygon_array = np.asarray(
        polygon
    )

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
    Obtain the dictionary stored inside a PaddleOCR Result object.

    PaddleOCR 3.x commonly exposes the serialisable result
    through result_object.json, generally under a top-level
    `res` key.
    """
    result_json = result_object.json

    if callable(result_json):
        result_json = result_json()

    if not isinstance(
        result_json,
        dict,
    ):
        raise TypeError(
            "PaddleOCR returned an unexpected result representation."
        )

    if "res" in result_json:
        result_json = result_json["res"]

    if not isinstance(
        result_json,
        dict,
    ):
        raise TypeError(
            "PaddleOCR result data is not a dictionary."
        )

    return result_json


def draw_ocr_boxes(
    original_image: np.ndarray,
    extracted_data: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Draw recognised polygons and confidence values
    on the original image.
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

        top_left_x = int(
            points[:, 0].min()
        )

        top_left_y = max(
            int(points[:, 1].min()) - 8,
            20,
        )

        label = (
            f'{item["confidence"]:.2f}'
        )

        cv2.putText(
            visualisation,
            label,
            (
                top_left_x,
                top_left_y,
            ),
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
    image_path: Path,
) -> list[dict[str, Any]]:
    """
    Run OpenCV preprocessing and PaddleOCR on one invoice image.

    Args:
        image_path: Path to one invoice image.

    Returns:
        A list containing one invoice result dictionary.
    """
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

    image_height, image_width = (
        original_image.shape[:2]
    )

    logger.info(
        "Original image dimensions: width=%d, height=%d",
        image_width,
        image_height,
    )


    # -----------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------

    processed_image = preprocess_invoice(
        original_image
    )


    # Keep exactly the same image filename as the input.
    #
    # Example:
    # Input:
    # batch1-0001.jpg
    #
    # Output:
    # processed/batch1-0001.jpg

    processed_path = (
        PROCESSED_DIR
        / image_path.name
    )

    if not cv2.imwrite(
        str(processed_path),
        processed_image,
    ):
        raise IOError(
            f"Could not save processed image to: {processed_path}"
        )

    logger.info(
        "Processed image saved to: %s",
        processed_path,
    )


    # -----------------------------------------------------------------
    # PaddleOCR
    # -----------------------------------------------------------------

    # Get the OCR model.
    #
    # It is initialized only once and reused
    # for all remaining images.
    ocr = get_ocr_model()

    logger.info(
        "Running PaddleOCR..."
    )

    prediction_results = ocr.predict(
        processed_image
    )


    extracted_data: list[
        dict[str, Any]
    ] = []


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
                "scores and bounding boxes."
            )


        for text, confidence, polygon in zip(
            texts,
            scores,
            polygons,
        ):

            clean_text = str(
                text
            ).strip()

            confidence_value = float(
                confidence
            )


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


    # -----------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------

    invoice_result = {

        # Original input filename is preserved.
        "file_name": image_path.name,

        "img_width": int(
            image_width
        ),

        "img_height": int(
            image_height
        ),

        "extracted_data": extracted_data,
    }


    final_output = [
        invoice_result
    ]


    # -----------------------------------------------------------------
    # JSON output
    # -----------------------------------------------------------------

    # Keep the same base name as the image.
    #
    # Input:
    # batch1-0001.jpg
    #
    # JSON:
    # batch1-0001.json

    json_path = (
        JSON_DIR
        / f"{image_path.stem}.json"
    )


    with json_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:

        json.dump(
            final_output,
            json_file,
            indent=2,
            ensure_ascii=False,
        )


    # -----------------------------------------------------------------
    # Visualisation output
    # -----------------------------------------------------------------

    # Keep exactly the same filename as the input.
    #
    # Input:
    # batch1-0001.jpg
    #
    # Visualisation:
    # batch1-0001.jpg

    visualisation_path = (
        VISUALISATION_DIR
        / image_path.name
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
        "JSON output: %s",
        json_path,
    )

    logger.info(
        "Bounding-box image: %s",
        visualisation_path,
    )


    return final_output


# ---------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------

def process_all_invoices(
    folder_path: Path,
) -> tuple[int, int, int]:
    """
    Process every supported invoice image inside the input folder.

    Args:
        folder_path: Folder containing invoice images.

    Returns:
        Tuple containing:
            total files,
            successful files,
            failed files.
    """

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {folder_path}"
        )


    if not folder_path.is_dir():
        raise ValueError(
            f"Input path is not a folder: {folder_path}"
        )


    # -----------------------------------------------------------------
    # Find all supported images
    # -----------------------------------------------------------------

    image_files = sorted(
        [
            file_path

            for file_path in folder_path.iterdir()

            if (
                file_path.is_file()

                and file_path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ],

        key=lambda path: path.name.lower(),
    )


    total_files = len(
        image_files
    )


    if total_files == 0:

        logger.warning(
            "No supported image files found in: %s",
            folder_path,
        )

        return (
            0,
            0,
            0,
        )


    logger.info(
        "============================================================"
    )

    logger.info(
        "Input folder: %s",
        folder_path,
    )

    logger.info(
        "Found %d invoice image(s).",
        total_files,
    )

    logger.info(
        "============================================================"
    )


    successful_files = 0

    failed_files = 0


    # -----------------------------------------------------------------
    # Process images one by one
    # -----------------------------------------------------------------

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        logger.info(
            ""
        )

        logger.info(
            "------------------------------------------------------------"
        )

        logger.info(
            "Processing file %d of %d",
            index,
            total_files,
        )

        logger.info(
            "File name: %s",
            image_path.name,
        )

        logger.info(
            "------------------------------------------------------------"
        )


        try:

            output = run_invoice_ocr(
                image_path
            )


            successful_files += 1


            logger.info(
                "SUCCESS: %s",
                image_path.name,
            )


            # Same JSON result that the original
            # single-file program printed.
            print(
                json.dumps(
                    output,
                    indent=2,
                    ensure_ascii=False,
                )
            )


        except Exception as error:

            failed_files += 1


            logger.exception(
                "FAILED: %s | Error: %s",
                image_path.name,
                error,
            )


            # Continue automatically with the next image.
            continue


    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    logger.info(
        ""
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "BATCH PROCESSING COMPLETED"
    )

    logger.info(
        "Total files      : %d",
        total_files,
    )

    logger.info(
        "Successful files : %d",
        successful_files,
    )

    logger.info(
        "Failed files     : %d",
        failed_files,
    )

    logger.info(
        "============================================================"
    )


    return (
        total_files,
        successful_files,
        failed_files,
    )


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

def main() -> int:
    """
    Application entry point.

    Process all invoice images from INPUT_FOLDER.
    """

    try:

        (
            total_files,
            successful_files,
            failed_files,
        ) = process_all_invoices(
            INPUT_FOLDER
        )


        if total_files == 0:

            logger.warning(
                "Nothing was processed."
            )

            return 0


        if failed_files > 0:

            logger.warning(
                "%d file(s) failed during processing.",
                failed_files,
            )


        logger.info(
            "Program finished successfully."
        )


        return 0


    except Exception as error:

        logger.exception(
            "Invoice OCR batch processing failed: %s",
            error,
        )

        return 1


if __name__ == "__main__":

    sys.exit(
        main()
    )