import io
import datetime
from typing import List, Dict, Optional, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for 'Page X of Y' dynamic footers and header branding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header rule & title
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(40, letter[1] - 38, letter[0] - 40, letter[1] - 38)

        self.drawString(40, letter[1] - 32, "DocuVault AI — Executive Document Intelligence Brief")
        gen_time = datetime.datetime.now().strftime("%B %d, %Y")
        self.drawRightString(letter[0] - 40, letter[1] - 32, gen_time)

        # Footer rule & page numbering
        self.line(40, 42, letter[0] - 40, 42)
        self.drawString(40, 30, "Confidential • Generated via DocuVault AI Retrieval-Augmented Generation Service")

        self.drawRightString(letter[0] - 40, 30, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


class ReportService:
    @staticmethod
    def generate_executive_pdf(
        documents_meta: List[Dict[str, Any]],
        summary_text: Optional[str] = None,
        qa_history: Optional[List[Dict[str, Any]]] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        total_chunks: int = 0,
    ) -> bytes:
        """
        Compiles document metadata, executive summary, and query audit trail
        into a publication-quality PDF report.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=40,
            rightMargin=40,
            topMargin=54,
            bottomMargin=54,
        )

        styles = getSampleStyleSheet()
        primary_color = colors.HexColor("#1E3A8A")  # Deep Navy
        secondary_color = colors.HexColor("#2563EB")  # Royal Blue
        text_dark = colors.HexColor("#0F172A")  # Slate 900
        text_muted = colors.HexColor("#475569")  # Slate 600
        bg_light = colors.HexColor("#F8FAFC")  # Slate 50
        border_color = colors.HexColor("#E2E8F0")

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=primary_color,
            spaceAfter=4,
        )

        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=text_muted,
            spaceAfter=14,
        )

        h2_style = ParagraphStyle(
            "Heading2Custom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=primary_color,
            spaceBefore=14,
            spaceAfter=8,
        )

        body_style = ParagraphStyle(
            "BodyDark",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14.5,
            textColor=text_dark,
            spaceAfter=6,
        )

        callout_style = ParagraphStyle(
            "CalloutText",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=13,
            textColor=text_muted,
        )

        story = []

        # Document Header
        story.append(Paragraph("DocuVault AI: Executive Intelligence Brief", title_style))

        story.append(
            Paragraph(
                f"Multi-Document Semantic Synthesis • Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                subtitle_style,
            )
        )
        story.append(HRFlowable(width="100%", thickness=1.5, color=secondary_color, spaceBefore=0, spaceAfter=14))

        # 1. Corpus Statistics Summary Box
        story.append(Paragraph("1. Indexed Corpus Overview", h2_style))
        total_pages = sum(d.get("pages", 0) for d in documents_meta)
        table_data = [
            [
                Paragraph("<b>Total Documents</b>", body_style),
                Paragraph(str(len(documents_meta)), body_style),
                Paragraph("<b>Embedding Model</b>", body_style),
                Paragraph(embedding_model, body_style),
            ],
            [
                Paragraph("<b>Total Pages</b>", body_style),
                Paragraph(str(total_pages), body_style),
                Paragraph("<b>Vector Chunks</b>", body_style),
                Paragraph(str(total_chunks), body_style),
            ],
        ]

        if documents_meta:
            doc_names = ", ".join([d.get("filename", "") for d in documents_meta])
            table_data.append(
                [
                    Paragraph("<b>Source Files</b>", body_style),
                    Paragraph(doc_names[:120] + ("..." if len(doc_names) > 120 else ""), body_style),
                    Paragraph("<b>Vector Engine</b>", body_style),
                    Paragraph("FAISS In-Memory Index", body_style),
                ]
            )

        stat_table = Table(table_data, colWidths=[1.4 * inch, 2.0 * inch, 1.4 * inch, 2.2 * inch])
        stat_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), bg_light),
                    ("BOX", (0, 0), (-1, -1), 0.75, border_color),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(stat_table)
        story.append(Spacer(1, 12))

        # 2. Executive Synthesis Section
        if summary_text and summary_text.strip():
            story.append(Paragraph("2. Executive Document Synthesis", h2_style))
            # Clean markdown markers for plain report display
            clean_summary = summary_text.replace("### ", "").replace("## ", "").replace("**", "")
            for paragraph in clean_summary.split("\n\n"):
                p_clean = paragraph.strip()
                if p_clean:
                    story.append(Paragraph(p_clean.replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 14))

        # 3. Interactive Query & Evidence Audit Trail
        if qa_history:
            story.append(Paragraph("3. Interactive Query & Evidence Audit Trail", h2_style))
            story.append(
                Paragraph(
                    "The following queries and responses were generated using dense vector similarity matching and grounded LLM reasoning:",
                    callout_style,
                )
            )
            story.append(Spacer(1, 8))

            for idx, item in enumerate(qa_history, start=1):
                q = item.get("question", "")
                ans = item.get("answer", "")
                conf = item.get("confidence_label", "🟢 High Confidence")
                sources = item.get("sources", [])

                qa_block = []
                qa_block.append(Paragraph(f"<b>Query #{idx}:</b> {q}", body_style))
                qa_block.append(Paragraph(f"<b>Confidence:</b> {conf}", body_style))
                qa_block.append(Spacer(1, 4))
                qa_block.append(Paragraph(f"<b>Grounded Synthesis:</b><br/>{ans.replace(chr(10), '<br/>')}", body_style))

                if sources:
                    qa_block.append(Spacer(1, 4))
                    source_str = " | ".join(
                        [f"<b>{s.get('document_name', 'Doc')}</b> (p. {s.get('page_number', 1)})" for s in sources[:3]]
                    )
                    qa_block.append(Paragraph(f"<b>Primary Citations:</b> {source_str}", callout_style))

                card_table = Table([[qa_block]], colWidths=[7.0 * inch])
                card_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
                            ("TOPPADDING", (0, 0), (-1, -1), 8),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                            ("LEFTPADDING", (0, 0), (-1, -1), 10),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ]
                    )
                )
                story.append(KeepTogether([card_table, Spacer(1, 10)]))

        doc.build(story, canvasmaker=NumberedCanvas)
        buffer.seek(0)
        return buffer.getvalue()


report_service = ReportService()
