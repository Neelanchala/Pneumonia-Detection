from flask import Flask, render_template, request, send_file, redirect
import os
from datetime import datetime

from predict import predict_image
from gradcam import generate_gradcam
from report_generator import generate_report

from database import (
    create_database,
    save_report,
    get_all_reports,
    search_reports,
    get_report,
    get_statistics,
    delete_report
)


app = Flask(__name__)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
OVERLAY_FOLDER = "static/overlays"
REPORT_FOLDER = "reports"


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(OVERLAY_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# Create database when application starts
create_database()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# UPLOAD AND ANALYZE X-RAY
# ============================================================

@app.route("/upload", methods=["POST"])
def upload():

    # --------------------------------------------------------
    # Check uploaded file
    # --------------------------------------------------------

    if "image" not in request.files:

        return "No file selected"


    file = request.files["image"]


    if file.filename == "":

        return "No file selected"


    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    image_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(image_path)


    # --------------------------------------------------------
    # Patient information
    # --------------------------------------------------------

    patient_name = request.form.get(
        "patient_name",
        ""
    )

    patient_id = request.form.get(
        "patient_id",
        ""
    )

    age = request.form.get(
        "age",
        ""
    )

    gender = request.form.get(
        "gender",
        ""
    )

    doctor_name = request.form.get(
        "doctor_name",
        ""
    )

    hospital = request.form.get(
        "hospital",
        ""
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction, confidence = predict_image(
        image_path
    )


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    heatmap_path = os.path.join(
        HEATMAP_FOLDER,
        "heatmap.jpg"
    )

    overlay_path = os.path.join(
        OVERLAY_FOLDER,
        "overlay.jpg"
    )


    generate_gradcam(
        image_path,
        heatmap_path,
        overlay_path
    )


    # --------------------------------------------------------
    # Generate PDF report
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_filename = (
        f"AI_Report_{timestamp}.pdf"
    )

    report_path = os.path.join(
        REPORT_FOLDER,
        report_filename
    )


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


    # --------------------------------------------------------
    # Save report information to database
    # --------------------------------------------------------

    save_report(

        patient_name,

        patient_id,

        age,

        gender,

        doctor_name,

        hospital,

        prediction,

        round(confidence * 100, 2),

        image_path,

        heatmap_path,

        overlay_path,

        report_path

    )


    # --------------------------------------------------------
    # Show result page
    # --------------------------------------------------------

    return render_template(

        "result.html",

        prediction=prediction,

        confidence=round(
            confidence * 100,
            2
        ),

        patient_name=patient_name,

        patient_id=patient_id,

        age=age,

        gender=gender,

        doctor_name=doctor_name,

        hospital=hospital,

        original=image_path,

        heatmap=heatmap_path,

        overlay=overlay_path,

        report=report_filename

    )


# ============================================================
# DOWNLOAD PDF REPORT
# ============================================================

@app.route("/download-report/<filename>")
def download_report(filename):

    report_path = os.path.join(
        REPORT_FOLDER,
        filename
    )


    if not os.path.exists(report_path):

        return "Report not found"


    return send_file(

        report_path,

        as_attachment=True

    )


# ============================================================
# PATIENT HISTORY
# ============================================================

@app.route("/history")
def history():

    keyword = request.args.get(
        "search",
        ""
    ).strip()


    if keyword:

        reports = search_reports(
            keyword
        )

    else:

        reports = get_all_reports()


    return render_template(

        "history.html",

        reports=reports,

        search=keyword

    )


# ============================================================
# REPORT DETAILS
# ============================================================

@app.route("/report/<int:report_id>")
def report_details(report_id):

    report = get_report(
        report_id
    )


    if report is None:

        return "Report not found"


    return render_template(

        "report_details.html",

        report=report

    )


# ============================================================
# DELETE REPORT
# ============================================================

@app.route("/delete/<int:report_id>")
def delete(report_id):

    delete_report(
        report_id
    )


    return redirect(
        "/history"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    stats = get_statistics()

    reports = get_all_reports()[:5]


    return render_template(

        "dashboard.html",

        stats=stats,

        reports=reports

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )