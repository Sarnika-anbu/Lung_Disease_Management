"""PDF report generator for the Lung Disease Management System.

Produces a 9-section clinical PDF report using reportlab.

Requirements: 12.1, 12.2, 12.3
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.models.types import EncounterData, RiskTier


class ReportGenerator:
    """Generates a structured 9-section clinical PDF report.

    Uses ``reportlab==4.0`` to produce a downloadable PDF containing:
      1. Patient header with risk badge
      2. Primary diagnosis with confidence bar
      3. Full class probability distribution
      4. Grad-CAM spectrogram image
      5. Risk stratification metrics grid
      6. Progression timeline
      7. Management recommendations
      8. Model quality metadata
      9. Disclaimer footer (includes ICBHI score + training dataset name)

    Args:
        None

    Example::

        generator = ReportGenerator()
        pdf_bytes = generator.generate(encounter)
    """

    def generate(self, encounter: EncounterData) -> bytes:
        """Generate the clinical PDF report and return raw bytes.

        Args:
            encounter: Fully populated :class:`~src.models.types.EncounterData`
                containing prediction, risk, recommendations, and progression data.

        Returns:
            Raw PDF bytes with Content-Type ``application/pdf``.

        Requirements: 12.1, 12.2, 12.3
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )
        styles = getSampleStyleSheet()
        story: list = []

        # Helper styles
        h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14, spaceAfter=4)
        h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, spaceAfter=2)
        normal = styles["Normal"]
        small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8)

        # ------------------------------------------------------------------
        # Section 1: Patient header with risk badge
        # ------------------------------------------------------------------
        story.append(Paragraph("SECTION_1_PATIENT_HEADER", h1))
        tier_colours = {
            RiskTier.LOW: "#27ae60",
            RiskTier.MEDIUM: "#e67e22",
            RiskTier.HIGH: "#e74c3c",
        }
        badge_colour = tier_colours.get(encounter.risk_result.tier, "#7f8c8d")
        badge_html = (
            f'<font color="{badge_colour}"><b>[{encounter.risk_result.tier.value} RISK]</b></font>'
        )
        patient_data = [
            ["Patient Name", encounter.patient_name, "Patient ID", encounter.patient_id],
            ["Age", f"{encounter.age} years", "Sex", encounter.sex],
            ["BMI", f"{encounter.bmi:.1f} kg/m\u00b2", "Pack-Years", f"{encounter.smoking_pack_years:.1f}"],
            ["Recording Location", encounter.recording_location, "Risk Tier", encounter.risk_result.tier.value],
        ]
        patient_table = Table(patient_data, colWidths=[45 * mm, 55 * mm, 45 * mm, 35 * mm])
        patient_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
            ])
        )
        story.append(patient_table)
        story.append(Paragraph(badge_html, normal))
        story.append(Spacer(1, 6 * mm))

        # ------------------------------------------------------------------
        # Section 2: Primary diagnosis with confidence bar
        # ------------------------------------------------------------------
        story.append(Paragraph("SECTION_2_DIAGNOSIS", h1))
        confidence_pct = encounter.prediction.confidence * 100
        bar_filled = int(confidence_pct / 5)  # out of 20 blocks
        bar = "\u2588" * bar_filled + "\u2591" * (20 - bar_filled)
        story.append(Paragraph(f"<b>Predicted Diagnosis:</b> {encounter.prediction.disease_class.value}", h2))
        story.append(Paragraph(f"<b>Confidence:</b> {confidence_pct:.1f}%", normal))
        story.append(Paragraph(f"<font face='Courier'>{bar}</font>", normal))
        story.append(Spacer(1, 6 * mm))

        # ------------------------------------------------------------------
        # Section 3: Full class probability distribution
        # ------------------------------------------------------------------
        story.append(Paragraph("SECTION_3_PROBABILITIES", h1))
        prob_rows = [["Disease Class", "Probability", "Uncertainty"]]
        for cls_name, prob in encounter.prediction.probabilities.items():
            unc = encounter.prediction.uncertainty.get(cls_name, 0.0)
            prob_rows.append([cls_name, f"{prob:.4f}", f"{unc:.4f}"])
        prob_table = Table(prob_rows, colWidths=[70 * mm, 40 * mm, 40 * mm])
        prob_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
            ])
        )
        story.append(prob_table)
        story.append(Spacer(1, 6 * mm))

        # ------------------------------------------------------------------
        # Section 4: Grad-CAM spectrogram image
        # ------------------------------------------------------------------
        story.append(Paragraph("SECTION_4_GRADCAM", h1))
        try:
            from PIL import Image as PILImage
            img_data = base64.b64decode(encounter.spectrogram_base64)
            img_buffer = io.BytesIO(img_data)
            # Eagerly validate and load the image via Pillow so any
            # decode errors surface here inside the try/except rather
            # than later inside doc.build().
            pil_img = PILImage.open(img_buffer)
            pil_img.load()
            # Convert to RGB for reportlab compatibility
            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")
            # Write the validated (possibly converted) image to a fresh buffer
            validated_buf = io.BytesIO()
            pil_img.save(validated_buf, format="PNG")
            validated_buf.seek(0)
            rl_img = RLImage(validated_buf, width=120 * mm, height=60 * mm)
            story.append(rl_img)
        except Exception:
            story.append(Paragraph("[Grad-CAM image unavailable]", normal))
        story.append(Spacer(1, 6 * mm))

        # ------------------------------------------------------------------
        # Section 5: Risk stratification metrics grid
        # ------------------------------------------------------------------
        story.append(Paragraph("SECTION_5_RISK_METRICS", h1))
        risk_data = [
            ["Metric", "Value"],
            ["Risk Tier", encounter.risk_result.tier.value],
            ["Risk Score", f"{encounter.risk_result.score:.1f} / 100"],
            ["Model Confidence", f"{encounter.prediction.confidence:.1%}"],
            ["Mean Uncertainty", f"{max(encounter.prediction.uncertainty.values()):.4f}"],
        ]
        risk_table = Table(risk_data, colWidths=[80 * mm, 100 * mm])
        risk_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2980b9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
            ])
        )
        story.append(risk_table)
        story.append(Spacer(1, 6 * mm))

        # ------------------------------------------------------------------
        # Section 6: Progression timeline
        # ------------------------------------------------------------------
        story.append(Paragraph("SECTION_6_PROGRESSION", h1))
        prog_rows = [["Outcome", "3-Month", "6-Month", "12-Month"]]
        outcomes = ["stable", "mild_progression", "moderate_progression", "severe_progression"]
        for outcome in outcomes:
            row = [
                outcome.replace("_", " ").title(),
                f"{encounter.progression.month_3.get(outcome, 0.0):.2f}",
                f"{encounter.progression.month_6.get(outcome, 0.0):.2f}",
                f"{encounter.progression.month_12.get(outcome, 0.0):.2f}",
            ]
            prog_rows.append(row)
        prog_table = Table(prog_rows, colWidths=[70 * mm, 35 * mm, 35 * mm, 35 * mm])
        prog_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8e44ad")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
            ])
        )
        story.append(prog_table)
        story.append(Spacer(1, 6 * mm))

        # ------------------------------------------------------------------
        # Section 7: Management recommendations list
        # ------------------------------------------------------------------
        story.append(Paragraph("SECTION_7_RECOMMENDATIONS", h1))
        for rec in encounter.recommendations:
            story.append(Paragraph(f"{rec.icon} <b>{rec.text}</b>", normal))
            story.append(Paragraph(f"  {rec.sub_text}", small))
            story.append(Paragraph(f"  <i>Source: {rec.source}</i>", small))
            story.append(Spacer(1, 2 * mm))
        story.append(Spacer(1, 4 * mm))

        # ------------------------------------------------------------------
        # Section 8: Model quality metadata
        # ------------------------------------------------------------------
        story.append(Paragraph("SECTION_8_MODEL_QUALITY", h1))
        quality_data = [
            ["Metric", "Value"],
            ["ICBHI Score (test set)", f"{encounter.model_icbhi_score:.4f}"],
            ["Training Dataset", encounter.training_dataset],
        ]
        quality_table = Table(quality_data, colWidths=[80 * mm, 100 * mm])
        quality_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16a085")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
            ])
        )
        story.append(quality_table)
        story.append(Spacer(1, 6 * mm))

        # ------------------------------------------------------------------
        # Section 9: Disclaimer footer
        # ------------------------------------------------------------------
        story.append(Paragraph("SECTION_9_DISCLAIMER", h1))
        disclaimer_text = (
            f"DISCLAIMER: This report was generated by an AI system trained on the "
            f"{encounter.training_dataset} dataset. "
            f"The model achieved an ICBHI Score of {encounter.model_icbhi_score:.4f} on the "
            f"patient-independent test split. "
            f"This report is intended for clinical support only and does not replace "
            f"professional medical judgment. All findings must be verified by a qualified "
            f"healthcare professional before clinical decisions are made."
        )
        story.append(Paragraph(disclaimer_text, small))

        doc.build(story)
        return buffer.getvalue()
