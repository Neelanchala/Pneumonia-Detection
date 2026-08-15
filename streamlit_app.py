import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import uuid
from datetime import datetime

from database import (
    create_database,
    save_report,
    get_all_reports,
    search_reports,
    get_report,
    get_statistics,
    delete_report
)

from report_generator import generate_report


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "my_cnn_best.keras"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

HEATMAP_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "heatmaps"
)

OVERLAY_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "overlays"
)

REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "reports"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(OVERLAY_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# ============================================================
# CREATE DATABASE
# ============================================================

create_database()


# ============================================================
# SESSION ID
# ============================================================
# Each visitor gets a unique session ID.
# This prevents one visitor from seeing another visitor's
# patient history and dashboard records.

if "session_id" not in st.session_state:

    st.session_state.session_id = uuid.uuid4().hex


SESSION_ID = st.session_state.session_id


# ============================================================
# LOAD MODEL ONCE
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading AI model..."):

    model = load_model()


CLASS_NAMES = [
    "Normal",
    "Pneumonia"
]


# ============================================================
# PREDICTION
# ============================================================

def predict_with_model(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image = tf.keras.utils.img_to_array(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    class_index = int(
        np.argmax(prediction[0])
    )

    confidence = float(
        np.max(prediction[0])
    )

    predicted_class = CLASS_NAMES[
        class_index
    ]

    return predicted_class, confidence


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    image_path,
    heatmap_path,
    overlay_path
):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # FIND MOBILENETV2
    # --------------------------------------------------------

    mobilenet = None

    for layer in model.layers:

        if "mobilenetv2" in layer.name.lower():

            mobilenet = layer

            break

    if mobilenet is None:

        raise ValueError(
            "MobileNetV2 layer was not found in the model."
        )

    # --------------------------------------------------------
    # FIND TARGET CONVOLUTION LAYER
    # --------------------------------------------------------

    try:

        target_layer = mobilenet.get_layer(
            "block_16_project"
        )

    except Exception:

        target_layer = None

        for layer in reversed(
            mobilenet.layers
        ):

            if len(layer.output.shape) == 4:

                target_layer = layer

                break

        if target_layer is None:

            raise ValueError(
                "Could not find a suitable Grad-CAM layer."
            )

    # --------------------------------------------------------
    # APPLY LAYERS BEFORE MOBILENETV2
    # --------------------------------------------------------

    mobilenet_index = model.layers.index(
        mobilenet
    )

    processed_image = image_batch

    for layer in model.layers[
        :mobilenet_index
    ]:

        if isinstance(
            layer,
            tf.keras.layers.InputLayer
        ):

            continue

        processed_image = layer(
            processed_image
        )

    # --------------------------------------------------------
    # FEATURE MODEL
    # --------------------------------------------------------

    feature_model = tf.keras.models.Model(
        inputs=mobilenet.input,
        outputs=[
            target_layer.output,
            mobilenet.output
        ]
    )

    # --------------------------------------------------------
    # GRADIENT TAPE
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, mobilenet_output = feature_model(
            processed_image,
            training=False
        )

        x = mobilenet_output

        # ----------------------------------------------------
        # APPLY LAYERS AFTER MOBILENETV2
        # ----------------------------------------------------

        for layer in model.layers[
            mobilenet_index + 1:
        ]:

            if isinstance(
                layer,
                tf.keras.layers.InputLayer
            ):

                continue

            if isinstance(
                layer,
                tf.keras.layers.Dropout
            ):

                x = layer(
                    x,
                    training=False
                )

            else:

                x = layer(
                    x
                )

        predictions = x

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            0,
            predicted_class
        ]

    # --------------------------------------------------------
    # CALCULATE GRADIENTS
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = (
        pooled_gradients.numpy()
    )

    conv_outputs = (
        conv_outputs.numpy()
    )

    # --------------------------------------------------------
    # WEIGHT FEATURE MAPS
    # --------------------------------------------------------

    for i in range(
        conv_outputs.shape[-1]
    ):

        conv_outputs[:, :, i] *= (
            pooled_gradients[i]
        )

    # --------------------------------------------------------
    # CREATE HEATMAP
    # --------------------------------------------------------

    heatmap = np.mean(
        conv_outputs,
        axis=-1
    )

    heatmap = np.maximum(
        heatmap,
        0
    )

    if np.max(heatmap) > 0:

        heatmap /= np.max(
            heatmap
        )

    heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # --------------------------------------------------------
    # APPLY COLOR MAP
    # --------------------------------------------------------

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # SAVE HEATMAP
    # --------------------------------------------------------

    cv2.imwrite(
        heatmap_path,
        colored_heatmap
    )

    # --------------------------------------------------------
    # CREATE OVERLAY
    # --------------------------------------------------------

    original = cv2.imread(
        image_path
    )

    original = cv2.resize(
        original,
        (224, 224)
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        colored_heatmap,
        0.4,
        0
    )

    # --------------------------------------------------------
    # SAVE OVERLAY
    # --------------------------------------------------------

    cv2.imwrite(
        overlay_path,
        overlay
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🩺 AI Pneumonia Detection"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔬 Analyze X-ray",
        "📋 Patient History",
        "📊 Dashboard"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "🩺 AI Pneumonia Detection System"
    )

    st.write(
        "AI-assisted chest X-ray analysis using "
        "a CNN model and Grad-CAM explainability."
    )

    st.info(
        "This project is intended for educational "
        "and research purposes. It is not a medical "
        "diagnostic device."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(
            "🧠 CNN Prediction"
        )

        st.write(
            "Classifies chest X-rays as "
            "Normal or Pneumonia."
        )

    with col2:

        st.subheader(
            "🔥 Grad-CAM"
        )

        st.write(
            "Visualizes regions that contributed "
            "to the model prediction."
        )

    with col3:

        st.subheader(
            "📄 PDF Reports"
        )

        st.write(
            "Generates downloadable AI analysis reports."
        )

    st.divider()

    st.subheader(
        "System Features"
    )

    st.markdown(
        """
        - Chest X-ray upload
        - CNN-based prediction
        - Confidence score
        - Grad-CAM heatmap
        - Attention overlay
        - PDF report generation
        - Patient history
        - Searchable reports
        - Dashboard statistics
        - SQLite database
        """
    )


# ============================================================
# ANALYZE X-RAY
# ============================================================

elif page == "🔬 Analyze X-ray":

    st.title(
        "🔬 Analyze Chest X-ray"
    )

    st.subheader(
        "Patient Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        patient_name = st.text_input(
            "Patient Name"
        )

        patient_id = st.text_input(
            "Patient ID"
        )

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=25
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

        doctor_name = st.text_input(
            "Doctor Name"
        )

        hospital = st.text_input(
            "Hospital / Clinic"
        )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded X-ray",
            width="stretch"
        )

        if st.button(
            "🔍 Analyze X-ray",
            type="primary"
        ):

            unique_id = uuid.uuid4().hex

            # ------------------------------------------------
            # SAVE UPLOADED IMAGE
            # ------------------------------------------------

            image_extension = (
                uploaded_file.name
                .split(".")[-1]
                .lower()
            )

            image_filename = (
                unique_id
                + "_xray."
                + image_extension
            )

            image_path = os.path.join(
                UPLOAD_FOLDER,
                image_filename
            )

            with open(
                image_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            # ------------------------------------------------
            # HEATMAP PATH
            # ------------------------------------------------

            heatmap_path = os.path.join(
                HEATMAP_FOLDER,
                unique_id + "_heatmap.jpg"
            )

            # ------------------------------------------------
            # OVERLAY PATH
            # ------------------------------------------------

            overlay_path = os.path.join(
                OVERLAY_FOLDER,
                unique_id + "_overlay.jpg"
            )

            # ------------------------------------------------
            # PDF REPORT PATH
            # ------------------------------------------------

            report_filename = (
                "AI_Report_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                + "_"
                + unique_id[:6]
                + ".pdf"
            )

            report_path = os.path.join(
                REPORT_FOLDER,
                report_filename
            )

            try:

                # ============================================
                # CNN PREDICTION
                # ============================================

                with st.spinner(
                    "Running CNN prediction..."
                ):

                    prediction, confidence = (
                        predict_with_model(
                            image_path
                        )
                    )

                # ============================================
                # GRAD-CAM
                # ============================================

                with st.spinner(
                    "Generating Grad-CAM..."
                ):

                    generate_gradcam(
                        image_path,
                        heatmap_path,
                        overlay_path
                    )

                # ============================================
                # PDF REPORT
                # ============================================

                with st.spinner(
                    "Generating PDF report..."
                ):

                    generate_report(
                        output_path=report_path,
                        patient_name=patient_name,
                        patient_id=patient_id,
                        age=age,
                        gender=gender,
                        doctor_name=doctor_name,
                        hospital=hospital,
                        prediction=prediction,
                        confidence=confidence * 100,
                        original_image=image_path,
                        heatmap_image=heatmap_path,
                        overlay_image=overlay_path
                    )

                # ============================================
                # SAVE TO DATABASE
                # ============================================

                save_report(
                    SESSION_ID,
                    patient_name,
                    patient_id,
                    age,
                    gender,
                    doctor_name,
                    hospital,
                    prediction,
                    round(
                        confidence * 100,
                        2
                    ),
                    image_path,
                    heatmap_path,
                    overlay_path,
                    report_path
                )

                st.success(
                    "Analysis completed successfully."
                )

                # ============================================
                # RESULT
                # ============================================

                st.divider()

                st.subheader(
                    "🧠 AI Result"
                )

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    if prediction.lower() == "normal":

                        st.success(
                            "Prediction: NORMAL"
                        )

                    else:

                        st.error(
                            "Prediction: PNEUMONIA"
                        )

                with result_col2:

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                # ============================================
                # VISUALIZATIONS
                # ============================================

                st.subheader(
                    "🔥 Grad-CAM Visualization"
                )

                img1, img2, img3 = st.columns(3)

                with img1:

                    st.image(
                        image_path,
                        caption="Original X-ray"
                    )

                with img2:

                    st.image(
                        heatmap_path,
                        caption="Grad-CAM Heatmap"
                    )

                with img3:

                    st.image(
                        overlay_path,
                        caption="Attention Overlay"
                    )

                # ============================================
                # PDF DOWNLOAD
                # ============================================

                st.subheader(
                    "📄 PDF Report"
                )

                with open(
                    report_path,
                    "rb"
                ) as pdf_file:

                    pdf_bytes = pdf_file.read()

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=report_filename,
                    mime="application/pdf"
                )

            except Exception as e:

                st.error(
                    "An error occurred during analysis."
                )

                st.exception(e)


# ============================================================
# PATIENT HISTORY
# ============================================================

elif page == "📋 Patient History":

    st.title(
        "📋 Patient History"
    )

    st.info(
        "Your history is private to this browser session. "
        "Other visitors cannot see these records."
    )

    search = st.text_input(
        "Search by patient name or patient ID"
    )

    # --------------------------------------------------------
    # SEARCH CURRENT SESSION ONLY
    # --------------------------------------------------------

    if search.strip():

        reports = search_reports(
            SESSION_ID,
            search.strip()
        )

    else:

        reports = get_all_reports(
            SESSION_ID
        )

    st.write(
        f"Total records: {len(reports)}"
    )

    # --------------------------------------------------------
    # DISPLAY REPORTS
    # --------------------------------------------------------

    for report in reports:

        report_id = report["id"]

        with st.expander(
            f"{report['patient_name']} — "
            f"{report['prediction']} — "
            f"{report['created_at']}"
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**Patient ID:** "
                    f"{report['patient_id']}"
                )

                st.write(
                    f"**Age:** "
                    f"{report['age']}"
                )

                st.write(
                    f"**Gender:** "
                    f"{report['gender']}"
                )

                st.write(
                    f"**Doctor:** "
                    f"{report['doctor_name']}"
                )

            with col2:

                st.write(
                    f"**Hospital:** "
                    f"{report['hospital']}"
                )

                st.write(
                    f"**Prediction:** "
                    f"{report['prediction']}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{report['confidence']:.2f}%"
                )

            # ------------------------------------------------
            # ORIGINAL X-RAY
            # ------------------------------------------------

            if os.path.exists(
                report["original_image"]
            ):

                st.image(
                    report["original_image"],
                    caption="Original X-ray"
                )

            # ------------------------------------------------
            # GRAD-CAM OVERLAY
            # ------------------------------------------------

            if os.path.exists(
                report["overlay_image"]
            ):

                st.image(
                    report["overlay_image"],
                    caption="Grad-CAM Overlay"
                )

            # ------------------------------------------------
            # PDF DOWNLOAD
            # ------------------------------------------------

            if os.path.exists(
                report["report_path"]
            ):

                with open(
                    report["report_path"],
                    "rb"
                ) as f:

                    st.download_button(
                        "⬇️ Download Report",
                        data=f.read(),
                        file_name=os.path.basename(
                            report["report_path"]
                        ),
                        mime="application/pdf",
                        key=f"download_{report_id}"
                    )

            # ------------------------------------------------
            # DELETE REPORT
            # ------------------------------------------------

            if st.button(
                "🗑️ Delete Report",
                key=f"delete_{report_id}"
            ):

                delete_report(
                    SESSION_ID,
                    report_id
                )

                st.success(
                    "Report deleted."
                )

                st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Dashboard":

    st.title(
        "📊 Dashboard"
    )

    st.info(
        "Dashboard statistics are private to this browser session."
    )

    # --------------------------------------------------------
    # GET CURRENT SESSION STATISTICS
    # --------------------------------------------------------

    stats = get_statistics(
        SESSION_ID
    )

    # --------------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Reports",
            stats["total"]
        )

    with col2:

        st.metric(
            "Normal",
            stats["normal"]
        )

    with col3:

        st.metric(
            "Pneumonia",
            stats["pneumonia"]
        )

    with col4:

        st.metric(
            "Average Confidence",
            f"{stats['average']:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    st.subheader(
        "📈 Prediction Analysis"
    )

    chart_col1, chart_col2 = st.columns(2)

    # ========================================================
    # PIE CHART
    # ========================================================

    with chart_col1:

        st.markdown(
            "### 🥧 Prediction Distribution"
        )

        normal_count = int(
            stats["normal"]
        )

        pneumonia_count = int(
            stats["pneumonia"]
        )

        pie_data = {

            "labels": [
                "Normal",
                "Pneumonia"
            ],

            "values": [
                normal_count,
                pneumonia_count
            ]
        }

        pie_chart = {

            "mark": {
                "type": "arc",
                "innerRadius": 0
            },

            "encoding": {

                "theta": {
                    "field": "values",
                    "type": "quantitative"
                },

                "color": {

                    "field": "labels",

                    "type": "nominal",

                    "scale": {

                        "domain": [
                            "Normal",
                            "Pneumonia"
                        ],

                        "range": [
                            "#2E8B57",
                            "#D9534F"
                        ]
                    },

                    "legend": {
                        "title": "Prediction"
                    }
                },

                "tooltip": [

                    {
                        "field": "labels",
                        "type": "nominal",
                        "title": "Prediction"
                    },

                    {
                        "field": "values",
                        "type": "quantitative",
                        "title": "Reports"
                    }
                ]
            }
        }

        # ----------------------------------------------------
        # HANDLE EMPTY DATASET
        # ----------------------------------------------------

        if stats["total"] > 0:

            st.vega_lite_chart(
                pie_data,
                pie_chart,
                use_container_width=True
            )

        else:

            st.info(
                "Analyze an X-ray to generate the prediction chart."
            )

    # ========================================================
    # CONFIDENCE CHART
    # ========================================================

    with chart_col2:

        st.markdown(
            "### 🎯 Confidence Summary"
        )

        confidence_data = {

            "Metric": [
                "Average Confidence"
            ],

            "Confidence": [
                float(stats["average"])
            ]
        }

        confidence_chart = {

            "mark": "bar",

            "encoding": {

                "x": {

                    "field": "Metric",

                    "type": "nominal",

                    "title": ""
                },

                "y": {

                    "field": "Confidence",

                    "type": "quantitative",

                    "title": "Confidence (%)",

                    "scale": {

                        "domain": [
                            0,
                            100
                        ]
                    }
                },

                "tooltip": [

                    {

                        "field": "Confidence",

                        "type": "quantitative",

                        "title": "Confidence (%)"
                    }
                ]
            }
        }

        if stats["total"] > 0:

            st.vega_lite_chart(
                confidence_data,
                confidence_chart,
                use_container_width=True
            )

        else:

            st.info(
                "Analyze an X-ray to generate the confidence chart."
            )

    st.divider()

    # ========================================================
    # RECENT REPORTS
    # ========================================================

    st.subheader(
        "📋 Recent Reports"
    )

    reports = get_all_reports(
        SESSION_ID
    )

    if reports:

        recent_data = []

        for report in reports[:10]:

            recent_data.append({

                "Patient": report[
                    "patient_name"
                ],

                "Patient ID": report[
                    "patient_id"
                ],

                "Prediction": report[
                    "prediction"
                ],

                "Confidence": (
                    f"{report['confidence']:.2f}%"
                ),

                "Date": report[
                    "created_at"
                ]
            })

        st.dataframe(
            recent_data,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No reports available yet."
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.sidebar.divider()

st.sidebar.warning(
    "Educational/research project only. "
    "Not a medical diagnostic device."
)
