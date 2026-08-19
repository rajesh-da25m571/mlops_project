from __future__ import annotations

import json
import logging

from confluent_kafka import Producer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

INVOICE_TOPIC = "invoice-uploaded"


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Kafka Producer
# ---------------------------------------------------------------------

_producer: Producer | None = None


def get_kafka_producer() -> Producer:
    """
    Create and reuse a Kafka producer instance.

    Returns:
        Configured Kafka Producer.
    """

    global _producer

    if _producer is None:

        logger.info(
            "Initializing Kafka producer..."
        )

        _producer = Producer(
            {
                "bootstrap.servers": (
                    KAFKA_BOOTSTRAP_SERVERS
                ),
            }
        )

        logger.info(
            "Kafka producer initialized."
        )

    return _producer


# ---------------------------------------------------------------------
# Delivery callback
# ---------------------------------------------------------------------

def delivery_report(
    error,
    message,
) -> None:
    """
    Callback executed when Kafka confirms whether
    a message was delivered.

    Args:
        error:
            Kafka delivery error, if any.

        message:
            Delivered Kafka message.
    """

    if error is not None:

        logger.error(
            "Kafka message delivery failed: %s",
            error,
        )

        return

    logger.info(
        (
            "Kafka message delivered successfully. "
            "topic=%s partition=%s offset=%s"
        ),
        message.topic(),
        message.partition(),
        message.offset(),
    )


# ---------------------------------------------------------------------
# Publish invoice
# ---------------------------------------------------------------------

def publish_invoice(
    invoice_id: str,
    image_path: str,
    original_filename: str,
) -> None:
    """
    Publish an invoice-processing event to Kafka.

    The actual invoice image is stored on disk.
    Kafka only receives metadata containing the
    location of the uploaded image.

    Args:
        invoice_id:
            Unique invoice identifier.

        image_path:
            Path of the stored invoice image.

        original_filename:
            Original filename uploaded by the user.
    """

    producer = get_kafka_producer()

    message = {
        "invoice_id": invoice_id,
        "image_path": image_path,
        "original_filename": original_filename,
    }

    message_json = json.dumps(
        message
    )

    logger.info(
        "Publishing invoice %s to Kafka topic %s",
        invoice_id,
        INVOICE_TOPIC,
    )

    producer.produce(
        topic=INVOICE_TOPIC,
        key=invoice_id,
        value=message_json,
        callback=delivery_report,
    )

    # Trigger delivery callbacks.
    producer.poll(0)

    # For this simple project/demo, wait until the
    # message has been handed to Kafka before returning.
    remaining_messages = producer.flush(
        timeout=10
    )

    if remaining_messages > 0:

        raise RuntimeError(
            (
                "Kafka could not deliver "
                f"{remaining_messages} message(s)."
            )
        )

    logger.info(
        "Invoice event published successfully."
    )