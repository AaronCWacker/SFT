import gradio as gr
from pathlib import Path
import datetime
import re
import os
import shutil
import fitz  # PyMuPDF
from PIL import Image
from collections import defaultdict
import io
from pypdf import PdfWriter

# Imports for new formats
from docx import Document
from docx.shared import Inches
import openpyxl

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, BaseDocTemplate, Frame, PageTemplate, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter, A4, legal, landscape
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Configuration & Setup ---
CWD = Path.cwd()
LAYOUTS = {
    "A4 Portrait": {"size": A4},
    "A4 Landscape": {"size": landscape(A4)},
    "Letter Portrait": {"size": letter},
    "Letter Landscape": {"size": landscape(letter)},
}
OUTPUT_DIR = CWD / "generated_outputs"
PREVIEW_DIR = CWD / "previews"
FONT_DIR = CWD

# Create necessary directories
OUTPUT_DIR.mkdir(exist_ok=True)
PREVIEW_DIR.mkdir(exist_ok=True)


# --- Font & Emoji Handling (for PDF) ---

def register_local_fonts():
    """Finds and registers all .ttf files from the application's base directory."""
    text_font_names, emoji_font_name = [], None
    font_files = list(FONT_DIR.glob("*.ttf"))
    print(f"Found {len(font_files)} .ttf files: {[f.name for f in font_files]}")

    for font_path in font_files:
        try:
            font_name = font_path.stem
            # Register the regular font
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
            # Also register a bold version, pointing to the same file. ReportLab's <b> tag will work.
            pdfmetrics.registerFont(TTFont(f"{font_name}-Bold", str(font_path))) 
            pdfmetrics.registerFontFamily(font_name, normal=font_name, bold=f"{font_name}-Bold")

            if "notocoloremoji-regular" in font_name.lower():
                emoji_font_name = font_name
            elif "notoemoji" not in font_name.lower():
                text_font_names.append(font_name)
        except Exception as e:
            print(f"Could not register font {font_path.name}: {e}")
    if not text_font_names: text_font_names.append('Helvetica')
    return sorted(text_font_names), emoji_font_name

def apply_emoji_font(text: str, emoji_font_name: str) -> str:
    """Wraps emoji characters in a <font> tag for ReportLab."""
    if not emoji_font_name: return text
    emoji_pattern = re.compile(f"([{re.escape(''.join(map(chr, range(0x1f600, 0x1f650))))}"
                               f"{re.escape(''.join(map(chr, range(0x1f300, 0x1f5ff))))}]+)")
    return emoji_pattern.sub(fr'<font name="{emoji_font_name}">\1</font>', text)


# --- Document Generation Engines ---

def create_pdf(md_content, font_name, emoji_font, pagesize, num_columns):
    """Generates a PDF file from markdown content."""
    md_buffer = io.BytesIO()
    story = markdown_to_story(md_content, font_name, emoji_font)
    if num_columns > 1:
        doc = BaseDocTemplate(md_buffer, pagesize=pagesize, leftMargin=0.5*inch, rightMargin=0.5*inch)
        frame_width = (doc.width / num_columns) - (num_columns - 1) * 0.1*inch
        frames = [Frame(doc.leftMargin + i * (frame_width + 0.2*inch), doc.bottomMargin, frame_width, doc.height) for i in range(num_columns)]
        doc.addPageTemplates([PageTemplate(id='MultiCol', frames=frames)])
    else:
        doc = SimpleDocTemplate(md_buffer, pagesize=pagesize)
    doc.build(story)
    return md_buffer

def create_docx(md_content):
    """Generates a DOCX file from markdown content."""
    document = Document()
    for line in md_content.split('\n'):
        if line.startswith('# '): document.add_heading(line[2:], level=1)
        elif line.startswith('## '): document.add_heading(line[3:], level=2)
        elif line.strip().startswith(('- ','* ')): document.add_paragraph(line.strip()[2:], style='List Bullet')
        else:
            p = document.add_paragraph()
            parts = re.split(r'(\*\*.*?\*\*)', line)
            for part in parts:
                if part.startswith('**') and part.endswith('**'): p.add_run(part[2:-2]).bold = True
                else: p.add_run(part)
    return document

def create_xlsx(md_content):
    """Generates an XLSX file, splitting content by H1 headers into columns."""
    workbook = openpyxl.Workbook(); sheet = workbook.active
    sections = re.split(r'\n# ', '\n' + md_content)
    if sections[0] == '': sections.pop(0)
    column_data = []
    for section in sections:
        lines = section.split('\n'); header = lines[0]
        content = [l.strip() for l in lines[1:] if l.strip()]
        column_data.append({'header': header, 'content': content})
    for c_idx, col in enumerate(column_data, 1):
        sheet.cell(row=1, column=c_idx, value=col['header'])
        for r_idx, line_content in enumerate(col['content'], 2):
            sheet.cell(row=r_idx, column=c_idx, value=line_content)
    return workbook

def markdown_to_story(markdown_text: str, font_name: str, emoji_font: str):
    """Converts markdown to a ReportLab story for PDF generation with enhanced styling."""
    styles = getSampleStyleSheet()
    # Use the bold variant of the selected font for headers
    bold_font = f"{font_name}-Bold" if font_name != "Helvetica" else "Helvetica-Bold"
    
    # Create styles with dynamic font sizes and bolding for headers
    style_normal = ParagraphStyle('BodyText', fontName=font_name, spaceAfter=6, fontSize=10)
    style_h1 = ParagraphStyle('h1', fontName=bold_font, spaceBefore=12, fontSize=24, leading=28)
    style_h2 = ParagraphStyle('h2', fontName=bold_font, spaceBefore=10, fontSize=18, leading=22)
    style_h3 = ParagraphStyle('h3', fontName=bold_font, spaceBefore=8, fontSize=14, leading=18)
    
    story, first_heading = [], True
    for line in markdown_text.split('\n'):
        content, style = line, style_normal
        
        # Determine the style based on markdown heading level
        if line.startswith("# "):
            if not first_heading: story.append(PageBreak())
            content, style, first_heading = line.lstrip('# '), style_h1, False
        elif line.startswith("## "):
            content, style = line.lstrip('## '), style_h2
        elif line.startswith("### "):
            content, style = line.lstrip('### '), style_h3
        
        # Apply bold tags and then apply emoji font wrapper
        formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
        final_content = apply_emoji_font(formatted_content, emoji_font)
        story.append(Paragraph(final_content, style))
        
    return story

def create_pdf_preview(pdf_path: Path):
    preview_path = PREVIEW_DIR / f"{pdf_path.stem}.png"
    try:
        doc = fitz.open(pdf_path); page = doc.load_page(0); pix = page.get_pixmap()
        pix.save(str(preview_path)); doc.close()
        return str(preview_path)
    except: return None

# --- Main API Function ---
def generate_outputs_api(files, output_formats, layouts, fonts, num_columns, page_w_mult, page_h_mult, progress=gr.Progress(track_tqdm=True)):
    if not files: raise gr.Error("Please upload at least one file.")
    if not output_formats: raise gr.Error("Please select at least one output format.")

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True); shutil.rmtree(PREVIEW_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(); PREVIEW_DIR.mkdir()

    # Consolidate all markdown content
    md_content = "\n".join([Path(f.name).read_text(encoding='utf-8') for f in files if Path(f.name).suffix.lower() == '.md'])

    log_updates, generated_files = "", []
    
    for format_choice in progress.tqdm(output_formats, desc="Generating Formats"):
        time_str = datetime.datetime.now().strftime('%m-%d-%a_%I%M%p').upper()
        
        if format_choice == "PDF":
            for layout_name in layouts:
                for font_name in fonts:
                    pagesize = LAYOUTS[layout_name]["size"]
                    final_pagesize = (pagesize[0] * page_w_mult, pagesize[1] * page_h_mult)
                    pdf_buffer = create_pdf(md_content, font_name, EMOJI_FONT_NAME, final_pagesize, num_columns)
                    filename = f"Document_{time_str}_{layout_name.replace(' ','-')}_{font_name}.pdf"
                    output_path = OUTPUT_DIR / filename
                    with open(output_path, "wb") as f: f.write(pdf_buffer.getvalue())
                    generated_files.append(output_path)
        
        elif format_choice == "DOCX":
            docx_doc = create_docx(md_content)
            filename = f"Document_{time_str}.docx"
            output_path = OUTPUT_DIR / filename
            docx_doc.save(output_path)
            generated_files.append(output_path)
        
        elif format_choice == "XLSX":
            xlsx_book = create_xlsx(md_content)
            filename = f"Outline_{time_str}.xlsx"
            output_path = OUTPUT_DIR / filename
            xlsx_book.save(output_path)
            generated_files.append(output_path)
            
    gallery_previews = [create_pdf_preview(p) for p in generated_files if p.suffix == '.pdf']
    final_gallery = [g for g in gallery_previews if g]
    
    return final_gallery, f"Generated {len(generated_files)} files.", [str(p) for p in generated_files]

# --- Gradio UI Definition ---
AVAILABLE_FONTS, EMOJI_FONT_NAME = register_local_fonts()
SAMPLE_MARKDOWN = "# Deities Guide\n\n- **Purpose**: Explore deities and their morals! \n- **Themes**: Justice ⚖️, faith 🙏\n\n# Arthurian Legends\n\n - **Merlin, Arthur**: Mentor 🧙, son 👑.\n - **Lesson**: Honor 🎖️ vs. betrayal 🗡️."
with open(CWD / "sample.md", "w", encoding="utf-8") as f: f.write(SAMPLE_MARKDOWN)

with gr.Blocks(theme=gr.themes.Soft(), title="Advanced Document Generator") as demo:
    gr.Markdown("# 📄 Advanced Document Generator (PDF, DOCX, XLSX)")
    gr.Markdown("Upload Markdown files to generate documents in multiple formats. `# Headers` create columns in XLSX and page breaks in multi-page PDFs.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Settings")
            uploaded_files = gr.File(label="Upload Markdown & Image Files", file_count="multiple", file_types=[".md", ".png", ".jpg"])
            
            output_formats = gr.CheckboxGroup(choices=["PDF", "DOCX", "XLSX"], label="Select Output Formats", value=["PDF"])
            
            with gr.Accordion("PDF Customization", open=True):
                with gr.Row():
                    page_w_mult_slider = gr.Slider(label="Page Width Multiplier", minimum=1, maximum=5, step=1, value=1)
                    page_h_mult_slider = gr.Slider(label="Page Height Multiplier", minimum=1, maximum=2, step=1, value=1)
                num_columns_slider = gr.Slider(label="Text Columns", minimum=1, maximum=4, step=1, value=1)
                selected_layouts = gr.CheckboxGroup(choices=list(LAYOUTS.keys()), label="Base Page Layout", value=["A4 Portrait"])
                selected_fonts = gr.CheckboxGroup(choices=AVAILABLE_FONTS, label="Text Font", value=[AVAILABLE_FONTS[0]] if AVAILABLE_FONTS else [])
            
            generate_btn = gr.Button("🚀 Generate Documents", variant="primary")
        
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Output Files")
            gallery_output = gr.Gallery(label="PDF Previews", show_label=False, elem_id="gallery", columns=3, height="auto", object_fit="contain")
            log_output = gr.Markdown(label="Generation Log", value="Ready...")
            downloadable_files_output = gr.Files(label="Download Generated Files")
            
    generate_btn.click(fn=generate_outputs_api, 
                       inputs=[uploaded_files, output_formats, selected_layouts, selected_fonts, num_columns_slider, page_w_mult_slider, page_h_mult_slider], 
                       outputs=[gallery_output, log_output, downloadable_files_output])

if __name__ == "__main__":
    demo.launch()
