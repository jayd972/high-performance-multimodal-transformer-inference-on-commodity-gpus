"""
Convert the final report from Markdown to PDF.
Uses markdown + xhtml2pdf (pure Python, no external tools needed).
"""

import os, sys
import markdown
from xhtml2pdf import pisa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

INPUT_MD = os.path.join(RESULTS_DIR, "week14_final", "report_final.md")
OUTPUT_PDF = os.path.join(RESULTS_DIR, "week14_final", "report_final.pdf")

CSS = """
@page {
    size: letter;
    margin: 1in 0.9in;
    @frame footer {
        -pdf-frame-content: footerContent;
        bottom: 0.4in;
        margin-left: 0.9in;
        margin-right: 0.9in;
        height: 0.4in;
    }
}

body {
    font-family: "Times New Roman", Times, serif;
    font-size: 11pt;
    line-height: 1.45;
    color: #1a1a1a;
}

h1 {
    font-size: 18pt;
    text-align: center;
    margin-top: 0;
    margin-bottom: 6pt;
    color: #111;
    border-bottom: none;
}

h2 {
    font-size: 14pt;
    margin-top: 18pt;
    margin-bottom: 8pt;
    color: #222;
    border-bottom: 1px solid #ccc;
    padding-bottom: 3pt;
}

h3 {
    font-size: 12pt;
    margin-top: 14pt;
    margin-bottom: 6pt;
    color: #333;
}

h4 {
    font-size: 11pt;
    margin-top: 10pt;
    margin-bottom: 4pt;
    font-style: italic;
    color: #444;
}

p {
    margin-top: 4pt;
    margin-bottom: 6pt;
    text-align: justify;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8pt;
    margin-bottom: 10pt;
    font-size: 9.5pt;
}

th {
    background-color: #2c3e50;
    color: white;
    padding: 5pt 6pt;
    text-align: center;
    font-weight: bold;
    border: 1px solid #2c3e50;
}

td {
    padding: 4pt 6pt;
    border: 1px solid #ddd;
    text-align: center;
}

tr:nth-child(even) td {
    background-color: #f8f9fa;
}

strong {
    color: #1a1a1a;
}

em {
    color: #555;
}

code {
    font-family: "Courier New", monospace;
    font-size: 9pt;
    background-color: #f4f4f4;
    padding: 1pt 3pt;
    border-radius: 2pt;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 14pt 0;
}

ul, ol {
    margin-top: 4pt;
    margin-bottom: 6pt;
    padding-left: 20pt;
}

li {
    margin-bottom: 3pt;
}

blockquote {
    border-left: 3pt solid #2c3e50;
    padding-left: 10pt;
    margin-left: 0;
    color: #555;
    font-style: italic;
}
"""

def convert():
    print(f"Reading: {INPUT_MD}")
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert markdown to HTML
    extensions = ["tables", "fenced_code", "toc", "sane_lists"]
    html_body = markdown.markdown(md_text, extensions=extensions)

    # Wrap in full HTML document
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>{CSS}</style>
</head>
<body>
{html_body}
<div id="footerContent">
    <p style="font-size: 8pt; color: #999; text-align: center; margin: 0;">
        Darji &amp; Patel — San Jose State University, Spring 2026
    </p>
</div>
</body>
</html>"""

    print(f"Converting to PDF...")
    with open(OUTPUT_PDF, "wb") as pdf_file:
        status = pisa.CreatePDF(html, dest=pdf_file)

    if status.err:
        print(f"ERROR: PDF conversion had {status.err} errors")
        return False

    size_kb = os.path.getsize(OUTPUT_PDF) / 1024
    print(f"PDF saved: {OUTPUT_PDF}")
    print(f"Size: {size_kb:.0f} KB")
    return True


if __name__ == "__main__":
    success = convert()
    sys.exit(0 if success else 1)
