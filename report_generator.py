from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch

from datetime import datetime
import os


styles = getSampleStyleSheet()

title_style = styles["Heading1"]
title_style.alignment = TA_CENTER
title_style.textColor = HexColor("#1E3A8A")

heading_style = styles["Heading2"]
heading_style.textColor = HexColor("#1E3A8A")

normal_style = styles["BodyText"]


def generate_report(
    output_path,
    patient_name,
    patient_id,
    age,
    gender,
    doctor_name,
    hospital,
    prediction,
    confidence,
    original_image,
    heatmap_image,
    overlay_image
):

    doc = SimpleDocTemplate(
        output_path,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    elements = []

    # =====================================================
    # TITLE
    # =====================================================

    elements.append(
        Paragraph(
            "AI Pneumonia Diagnostic Report",
            title_style
        )
    )

    elements.append(Spacer(1, 20))

    # =====================================================
    # REPORT DETAILS
    # =====================================================

    report_table = Table([
        ["Report ID", datetime.now().strftime("AI-%Y%m%d-%H%M%S")],
        ["Report Date", datetime.now().strftime("%d-%m-%Y %H:%M")]
    ], colWidths=[130, 300])

    report_table.setStyle(TableStyle([

        ("BACKGROUND", (0,0), (0,-1), HexColor("#E8F0FE")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8)

    ]))

    elements.append(report_table)
    elements.append(Spacer(1,20))

    # =====================================================
    # PATIENT INFORMATION
    # =====================================================

    elements.append(
        Paragraph("Patient Information", heading_style)
    )

    patient_table = Table([

        ["Patient Name", patient_name],
        ["Patient ID", patient_id],
        ["Age", age],
        ["Gender", gender],
        ["Doctor", doctor_name if doctor_name else "-"],
        ["Hospital", hospital if hospital else "-"]

    ], colWidths=[140,320])

    patient_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(0,-1),HexColor("#E8F0FE")),

        ("GRID",(0,0),(-1,-1),0.5,colors.grey),

        ("BOTTOMPADDING",(0,0),(-1,-1),8)

    ]))

    elements.append(patient_table)

    elements.append(Spacer(1,20))

    # =====================================================
    # AI RESULT
    # =====================================================

    elements.append(
        Paragraph("AI Analysis", heading_style)
    )

    result_color = "#16A34A"

    if prediction.lower() != "normal":
        result_color = "#DC2626"

    prediction_table = Table([

                [
            "Prediction",
            Paragraph(
                f"<font color='{result_color}'><b>{prediction}</b></font>",
                normal_style
            )
        ],

        ["Confidence",
         f"{confidence:.2f}%"]

    ], colWidths=[140,320])

    prediction_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(0,-1),HexColor("#F5F9FF")),

        ("GRID",(0,0),(-1,-1),0.5,colors.grey),

        ("BOTTOMPADDING",(0,0),(-1,-1),8)

    ]))

    elements.append(prediction_table)

    elements.append(Spacer(1,20))

    # =====================================================
    # INTERPRETATION
    # =====================================================

    if prediction.lower() == "normal":

        interpretation = (
            "The uploaded chest X-ray does not demonstrate "
            "significant radiological findings suggestive of "
            "pneumonia according to the trained CNN model."
        )

        recommendation = (
            "Clinical correlation is recommended. If symptoms "
            "persist, further medical evaluation should be "
            "considered."
        )

    else:

        interpretation = (
            "The CNN model detected radiological findings "
            "suggestive of pneumonia. The highlighted Grad-CAM "
            "regions indicate the areas that contributed most "
            "to the prediction."
        )

        recommendation = (
            "Immediate clinical evaluation by a qualified "
            "physician is recommended."
        )

    elements.append(
        Paragraph("AI Interpretation", heading_style)
    )

    elements.append(
        Paragraph(interpretation, normal_style)
    )

    elements.append(Spacer(1,12))

    elements.append(
        Paragraph("Clinical Recommendation", heading_style)
    )

    elements.append(
        Paragraph(recommendation, normal_style)
    )

    elements.append(Spacer(1,20))

    # =====================================================
    # VISUALIZATION
    # =====================================================

    elements.append(
        Paragraph("Grad-CAM Visualization", heading_style)
    )

    image_row = []

    for path in [original_image, heatmap_image, overlay_image]:

        if os.path.exists(path):

            image_row.append(
                Image(path, width=1.8*inch, height=1.8*inch)
            )

        else:

            image_row.append(
                Paragraph("Image Missing", normal_style)
            )

    caption_row = [
        "Original X-ray",
        "Grad-CAM Heatmap",
        "Overlay"
    ]

    image_table = Table(
        [image_row, caption_row],
        colWidths=[180,180,180]
    )

    image_table.setStyle(TableStyle([

        ("GRID",(0,0),(-1,-1),0.5,colors.grey),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),

        ("BOTTOMPADDING",(0,0),(-1,-1),10),

        ("TOPPADDING",(0,0),(-1,-1),10)

    ]))

    elements.append(image_table)

    elements.append(Spacer(1,25))

    # =====================================================
    # DISCLAIMER
    # =====================================================

    elements.append(
        Paragraph("Disclaimer", heading_style)
    )

    elements.append(
        Paragraph(
            "This report has been generated using an Artificial "
            "Intelligence (CNN + Grad-CAM) model. The results are "
            "intended to assist healthcare professionals and should "
            "not be considered as a final medical diagnosis. "
            "Clinical examination and professional judgement remain "
            "essential.",
            normal_style
        )
    )

    elements.append(Spacer(1,30))

    # =====================================================
    # FOOTER
    # =====================================================

    footer = Table([

        ["Generated by AI Pneumonia Detection System"]

    ])

    footer.setStyle(TableStyle([

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("TEXTCOLOR",(0,0),(-1,-1),colors.grey),

        ("FONTSIZE",(0,0),(-1,-1),9)

    ]))

    elements.append(footer)

    # =====================================================
    # BUILD
    # =====================================================

    doc.build(elements)