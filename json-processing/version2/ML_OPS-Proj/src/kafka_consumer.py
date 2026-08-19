from __future__ import annotations

import json
import logging
from pathlib import Path

from confluent_kafka import Consumer, KafkaError

from src.ocr_pipeline import run_invoice_ocr


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

INVOICE_TOPIC = "invoice-uploaded"

CONSUMER_GROUP_ID = "invoice-ocr-workers"


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

            # We commit only after OCR processing
            # completes successfully.
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
    # Run existing OCR pipeline
    # -------------------------------------------------------------

    logger.info(
        "Starting OCR processing..."
    )

    ocr_output = run_invoice_ocr(
        image_path
    )

    logger.info(
        "OCR processing completed successfully."
    )

    logger.info(
        "Extracted invoice output: %s",
        ocr_output,
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
                # Run OCR
                # -------------------------------------------------

                process_invoice_message(
                    payload
                )

                # -------------------------------------------------
                # Commit Kafka offset only after successful OCR
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