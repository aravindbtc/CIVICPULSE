from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import openpyxl
import pandas as pd
import os

def generate_pdf_report(df):
    path = "reports/report.pdf"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = canvas.Canvas(path, pagesize=letter)
    c.drawString(100, 750, "e-Consultation Report")
    y = 700
    for _, row in df.iterrows():
        c.drawString(100, y, f"Comment: {row['original_comment'][:100]}...")
        c.drawString(100, y-20, f"Sentiment: {row['sentiment']}")
        y -= 40
        if y < 100:
            c.showPage()
            y = 750
    c.save()
    return path

def generate_excel_report(df):
    path = "reports/report.xlsx"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="Comments")
        summary = pd.DataFrame({"Overall Summary": ["Placeholder"]})
        summary.to_excel(writer, sheet_name="Summary")
    return path