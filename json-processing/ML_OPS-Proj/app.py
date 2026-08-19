from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.kafka_producer import publish_invoice


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

UPLOAD_DIR = Path("uploads")

ALLOWED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Invoice Verification System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        .main-heading {
            font-size: 2.5rem;
            font-weight: 750;
            margin-bottom: 0.3rem;
        }

        .sub-heading {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .info-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.2rem;
        }

        .status-card {
            background-color: #f0fdf4;
            border: 1px solid #86efac;
            border-radius: 14px;
            padding: 1.2rem;
            margin-top: 1rem;
        }

        .pending-card {
            background-color: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 14px;
            padding: 1.2rem;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def create_upload_directory() -> None:
    """
    Create the uploads directory if it does not already exist.
    """
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_uploaded_invoice(uploaded_file) -> Path:
    """
    Save an uploaded invoice using a unique backend filename.

    Args:
        uploaded_file:
            Streamlit UploadedFile object.

    Returns:
        Path of the saved invoice.
    """

    create_upload_directory()

    original_suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if original_suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported invoice format."
        )

    unique_filename = (
        f"{uuid4().hex}_{Path(uploaded_file.name).name}"
    )

    saved_path = (
        UPLOAD_DIR
        / unique_filename
    )

    saved_path.write_bytes(
        uploaded_file.getbuffer()
    )

    return saved_path


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:

    st.title(
        "Invoice AI"
    )

    st.markdown(
        """
        ### Current workflow

        1. Upload invoice  
        2. Save invoice securely  
        3. Publish invoice event to Kafka  
        4. Kafka consumer receives invoice  
        5. Apply OpenCV preprocessing  
        6. Extract text using PaddleOCR  
        7. Generate backend JSON  
        """
    )

    st.divider()

    st.caption(
        "MLOps Course End-Term Project"
    )


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

st.markdown(
    '<div class="main-heading">'
    'Intelligent Invoice Processor'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-heading">'
    'Upload an invoice image for automated document processing.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------------

uploaded_file = st.file_uploader(
    label="Upload invoice",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
    accept_multiple_files=False,
    help=(
        "Upload one invoice image in PNG, JPG, "
        "or JPEG format."
    ),
)


# ---------------------------------------------------------------------
# No file uploaded
# ---------------------------------------------------------------------

if uploaded_file is None:

    st.info(
        "Upload an invoice image to begin processing."
    )


# ---------------------------------------------------------------------
# File uploaded
# ---------------------------------------------------------------------

else:

    # -------------------------------------------------------------
    # Validate uploaded image
    # -------------------------------------------------------------

    try:

        uploaded_file.seek(0)

        invoice_image = Image.open(
            uploaded_file
        )

        invoice_image.verify()

        uploaded_file.seek(0)

        invoice_image = Image.open(
            uploaded_file
        )

    except UnidentifiedImageError:

        st.error(
            "The uploaded file is not a valid image."
        )

        st.stop()

    except Exception as error:

        st.error(
            f"Unable to read the uploaded invoice: {error}"
        )

        st.stop()


    # -------------------------------------------------------------
    # Invoice preview and file details
    # -------------------------------------------------------------

    preview_column, details_column = st.columns(
        [2, 1],
        gap="large",
    )

    with preview_column:

        st.subheader(
            "Invoice Preview"
        )

        st.image(
            invoice_image,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    with details_column:

        st.subheader(
            "File Details"
        )

        file_size_kb = (
            uploaded_file.size
            / 1024
        )

        st.markdown(
            f"""
            <div class="info-card">

                <strong>File name</strong><br>
                {uploaded_file.name}

                <br><br>

                <strong>File type</strong><br>
                {uploaded_file.type}

                <br><br>

                <strong>File size</strong><br>
                {file_size_kb:.2f} KB

                <br><br>

                <strong>Current status</strong><br>
                Ready for submission

            </div>
            """,
            unsafe_allow_html=True,
        )


    st.divider()


    # -------------------------------------------------------------
    # Process invoice button
    # -------------------------------------------------------------

    process_button = st.button(
        label="Process Invoice",
        type="primary",
        use_container_width=True,
    )


    # -------------------------------------------------------------
    # Kafka producer processing
    # -------------------------------------------------------------

    if process_button:

        saved_invoice_path: Path | None = None

        try:

            with st.status(
                "Submitting invoice for processing...",
                expanded=True,
            ) as processing_status:

                # -------------------------------------------------
                # Step 1: Save invoice
                # -------------------------------------------------

                st.write(
                    "Saving invoice to backend storage..."
                )

                uploaded_file.seek(0)

                saved_invoice_path = (
                    save_uploaded_invoice(
                        uploaded_file
                    )
                )


                # -------------------------------------------------
                # Step 2: Generate invoice ID
                # -------------------------------------------------

                invoice_id = (
                    saved_invoice_path.stem
                )


                st.write(
                    f"Invoice stored at: "
                    f"{saved_invoice_path}"
                )


                # -------------------------------------------------
                # Step 3: Publish Kafka message
                # -------------------------------------------------

                st.write(
                    "Publishing invoice event to Kafka..."
                )

                publish_invoice(
                    invoice_id=invoice_id,
                    image_path=str(
                        saved_invoice_path
                    ),
                    original_filename=(
                        uploaded_file.name
                    ),
                )


                # -------------------------------------------------
                # Kafka submission completed
                # -------------------------------------------------

                processing_status.update(
                    label=(
                        "Invoice submitted successfully"
                    ),
                    state="complete",
                    expanded=False,
                )


            # -----------------------------------------------------
            # Display queue status
            # -----------------------------------------------------

            st.markdown(
                f"""
                <div class="pending-card">

                    <strong>
                        Invoice submitted successfully
                    </strong>

                    <br><br>

                    <strong>Invoice ID:</strong><br>
                    {invoice_id}

                    <br><br>

                    <strong>Queue Status:</strong>
                    Submitted to Kafka

                    <br>

                    <strong>OCR Status:</strong>
                    Pending

                    <br>

                    <strong>Next Stage:</strong>
                    Waiting for OCR consumer

                </div>
                """,
                unsafe_allow_html=True,
            )


        except Exception as error:

            st.error(
                "Invoice submission failed."
            )

            st.exception(
                error
            )