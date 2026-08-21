from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
from confluent_kafka import Consumer, KafkaError

from src.ocr_pipeline import run_invoice_ocr


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

INVOICE_TOPIC = "invoice-uploaded"

CONSUMER_GROUP_ID = "invoice-ocr-workers"


# ---------------------------------------------------------------------
# MLflow Configuration
# ---------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

MLFLOW_EXPERIMENT_NAME = "invoice-ocr-processing"


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configure MLflow
# ---------------------------------------------------------------------

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    MLFLOW_EXPERIMENT_NAME
)


# ---------------------------------------------------------------------
# Kafka Consumer
# ---------------------------------------------------------------------

def create_kafka_consumer() -> Consumer:
    """
    Create the Kafka consumer used by the OCR worker.

    Returns:
        Configured Kafka Consumer.
    """

    consumer = Consumer(
        {
            "bootstrap.servers": (
                KAFKA_BOOTSTRAP_SERVERS
            ),

            "group.id": (
                CONSUMER_GROUP_ID
            ),

            "auto.offset.reset": "earliest",

            # Commit only after OCR and MLflow
            # processing complete successfully.
            "enable.auto.commit": False,
        }
    )

    return consumer


# ---------------------------------------------------------------------
# Process Kafka invoice message
# ---------------------------------------------------------------------

def process_invoice_message(
    payload: dict,
) -> None:
    """
    Process one invoice event received from Kafka.

    The flow is:

    1. Validate Kafka message
    2. Start MLflow run
    3. Run OCR pipeline
    4. Log parameters
    5. Log metrics
    6. Log artifacts
    7. Finish MLflow run

    Args:
        payload:
            Kafka message converted from JSON.
    """

    invoice_id = payload.get(
        "invoice_id"
    )

    image_path_value = payload.get(
        "image_path"
    )

    original_filename = payload.get(
        "original_filename"
    )

    if not invoice_id:

        raise ValueError(
            "Kafka message does not contain invoice_id."
        )

    if not image_path_value:

        raise ValueError(
            "Kafka message does not contain image_path."
        )

    image_path = Path(
        image_path_value
    )

    logger.info(
        "----------------------------------------"
    )

    logger.info(
        "Received invoice from Kafka."
    )

    logger.info(
        "Invoice ID: %s",
        invoice_id,
    )

    logger.info(
        "Original filename: %s",
        original_filename,
    )

    logger.info(
        "Image path: %s",
        image_path,
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Invoice image not found: {image_path}"
        )


    # -------------------------------------------------------------
    # Start MLflow run
    # -------------------------------------------------------------

    logger.info(
        "Starting MLflow run..."
    )

    with mlflow.start_run(
        run_name=f"invoice_{invoice_id}"
    ) as mlflow_run:

        logger.info(
            "MLflow Run ID: %s",
            mlflow_run.info.run_id,
        )


        # ---------------------------------------------------------
        # Log MLflow parameters
        # ---------------------------------------------------------

        mlflow.log_param(
            "invoice_id",
            invoice_id,
        )

        mlflow.log_param(
            "original_filename",
            original_filename,
        )

        mlflow.log_param(
            "ocr_engine",
            "PaddleOCR",
        )

        mlflow.log_param(
            "preprocessing",
            "OpenCV",
        )

        mlflow.log_param(
            "processing_mode",
            "Kafka Consumer",
        )


        # ---------------------------------------------------------
        # Run OCR
        # ---------------------------------------------------------

        logger.info(
            "Starting OCR processing..."
        )

        ocr_result = (
            run_invoice_ocr(
                image_path
            )
        )

        logger.info(
            "OCR processing completed successfully."
        )


        # ---------------------------------------------------------
        # Log OCR metrics
        # ---------------------------------------------------------

        text_region_count = (
            ocr_result[
                "text_region_count"
            ]
        )

        average_confidence = (
            ocr_result[
                "average_confidence"
            ]
        )

        mlflow.log_metric(
            "text_region_count",
            text_region_count,
        )

        mlflow.log_metric(
            "average_ocr_confidence",
            average_confidence,
        )

        logger.info(
            "Logged MLflow metrics."
        )

        logger.info(
            "Text region count: %s",
            text_region_count,
        )

        logger.info(
            "Average OCR confidence: %.4f",
            average_confidence,
        )


        # ---------------------------------------------------------
        # Log bounding-box visualization artifact
        # ---------------------------------------------------------

        visualisation_path = Path(
            ocr_result[
                "visualisation_path"
            ]
        )

        if not visualisation_path.exists():

            raise FileNotFoundError(
                (
                    "Bounding-box visualization "
                    f"not found: {visualisation_path}"
                )
            )

        mlflow.log_artifact(
            str(
                visualisation_path
            ),
            artifact_path="visualisations",
        )

        logger.info(
            (
                "Bounding-box visualization "
                "logged to MLflow: %s"
            ),
            visualisation_path,
        )


        # ---------------------------------------------------------
        # Log OCR JSON artifact
        # ---------------------------------------------------------

        json_path = Path(
            ocr_result[
                "json_path"
            ]
        )

        if not json_path.exists():

            raise FileNotFoundError(
                (
                    "OCR JSON artifact "
                    f"not found: {json_path}"
                )
            )

        mlflow.log_artifact(
            str(
                json_path
            ),
            artifact_path="ocr_json",
        )

        logger.info(
            "OCR JSON logged to MLflow: %s",
            json_path,
        )


        # ---------------------------------------------------------
        # Optional processed image artifact
        # ---------------------------------------------------------

        processed_image_path = Path(
            ocr_result[
                "processed_image_path"
            ]
        )

        if processed_image_path.exists():

            mlflow.log_artifact(
                str(
                    processed_image_path
                ),
                artifact_path="processed_images",
            )

            logger.info(
                (
                    "Processed image logged "
                    "to MLflow: %s"
                ),
                processed_image_path,
            )


        # ---------------------------------------------------------
        # Log OCR status as tag
        # ---------------------------------------------------------

        mlflow.set_tag(
            "pipeline_stage",
            "ocr",
        )

        mlflow.set_tag(
            "invoice_processing_status",
            "completed",
        )

        mlflow.set_tag(
            "source",
            "kafka",
        )


        # ---------------------------------------------------------
        # Log OCR output in application logs
        # ---------------------------------------------------------

        logger.info(
            "Extracted invoice output: %s",
            ocr_result[
                "ocr_output"
            ],
        )

        logger.info(
            "MLflow logging completed successfully."
        )


# ---------------------------------------------------------------------
# Consumer loop
# ---------------------------------------------------------------------

def consume_invoices() -> None:
    """
    Continuously listen for invoice events
    and process them using the OCR pipeline.
    """

    consumer = create_kafka_consumer()

    consumer.subscribe(
        [
            INVOICE_TOPIC,
        ]
    )

    logger.info(
        "Invoice OCR consumer started."
    )

    logger.info(
        "Kafka broker: %s",
        KAFKA_BOOTSTRAP_SERVERS,
    )

    logger.info(
        "Listening to topic: %s",
        INVOICE_TOPIC,
    )

    logger.info(
        "Consumer group: %s",
        CONSUMER_GROUP_ID,
    )

    logger.info(
        "MLflow Tracking URI: %s",
        MLFLOW_TRACKING_URI,
    )

    logger.info(
        "MLflow Experiment: %s",
        MLFLOW_EXPERIMENT_NAME,
    )

    try:

        while True:

            message = consumer.poll(
                timeout=1.0
            )

            if message is None:
                continue

            if message.error():

                if (
                    message.error().code()
                    == KafkaError._PARTITION_EOF
                ):

                    continue

                logger.error(
                    "Kafka consumer error: %s",
                    message.error(),
                )

                continue

            try:

                message_value = (
                    message.value()
                )

                if message_value is None:

                    raise ValueError(
                        "Received empty Kafka message."
                    )

                payload = json.loads(
                    message_value.decode(
                        "utf-8"
                    )
                )

                if not isinstance(
                    payload,
                    dict,
                ):

                    raise ValueError(
                        "Kafka payload must be a JSON object."
                    )


                # -------------------------------------------------
                # Run OCR and MLflow tracking
                # -------------------------------------------------

                process_invoice_message(
                    payload
                )


                # -------------------------------------------------
                # Commit Kafka offset only after
                # OCR + MLflow logging succeed
                # -------------------------------------------------

                consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                logger.info(
                    "Kafka message committed successfully."
                )


            except Exception:

                logger.exception(
                    (
                        "Invoice processing failed. "
                        "Kafka message was not committed."
                    )
                )


    except KeyboardInterrupt:

        logger.info(
            "Stopping invoice OCR consumer..."
        )


    finally:

        consumer.close()

        logger.info(
            "Kafka consumer closed."
        )


# ---------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    consume_invoices()