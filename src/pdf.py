import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from typing import List, Optional
from datetime import datetime


def create_summary_pdf(
    output_dir: str,
    # Changed to a list, made optional
    output_pdf_path: str,
    video_name: str,
    summary_txt_filenames: Optional[List[str]] = None,
):
    """
    Creates a summary PDF from text content and image files in a specified directory.

    The PDF will contain:
    1. A title page.
    2. Contents of specified (or all found) .txt summary files.
    3. Each PNG image on a new page with its filename as a title.

    Args:
        output_dir: The directory containing the .txt summary file(s) and .png image files.
        summary_txt_filenames: An optional list of specific text filenames (e.g., ["summary1.txt", "summary2.txt"])
                               to include. If None or empty, all .txt files in output_dir will be included.
        output_pdf_path: The full path for the output PDF file (e.g., "report.pdf").
    """
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Custom style for titles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['h1'],
        fontSize=24,
        spaceAfter=24,
        alignment=TA_CENTER
    )

    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['h2'],
        fontSize=18,
        spaceAfter=12,
        alignment=TA_LEFT
    )

    subsection_title_style = ParagraphStyle(  # New style for multiple summary titles
        'SubsectionTitle',
        parent=styles['h3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        alignment=TA_LEFT
    )

    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 14  # Line spacing

    # --- 1. Add a Title Page ---
    report_title = "Cycling Analysis Report"
    pdf_base_name = os.path.splitext(os.path.basename(output_pdf_path))[0]

    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['h3']))
    story.append(Paragraph(
        f"Video: {video_name}" if video_name else "Video: Not specified", styles['h3']))
    # Include current location as well
    story.append(
        Paragraph("Location: Lützelbach, Hessen, Germany", styles['h3']))
    story.append(Spacer(1, 2 * inch))
    # Add a note about the report's purpose
    story.append(Paragraph(
        "This report provides a summary of cycling angle analysis, including key metrics and visual frames.",
        normal_style
    ))
    story.append(PageBreak())

    # --- 2. Add the Text Summaries ---
    text_files_to_include = []
    if summary_txt_filenames:
        text_files_to_include = [
            f for f in summary_txt_filenames if os.path.exists(os.path.join(output_dir, f))]
        if not text_files_to_include:
            print("Warning: None of the specified summary text files were found.")
    else:  # If no specific filenames are given, find all .txt files
        text_files_to_include = [f for f in os.listdir(
            output_dir) if f.lower().endswith('.txt')]
        text_files_to_include.sort()  # Sort them alphabetically

    if text_files_to_include:
        story.append(Paragraph("Analysis Summaries", section_title_style))
        story.append(Spacer(1, 0.2 * inch))

        for i, txt_filename in enumerate(text_files_to_include):
            txt_path = os.path.join(output_dir, txt_filename)
            try:
                story.append(Paragraph(
                    f"Summary: {os.path.splitext(txt_filename)[0].replace('_', ' ').title()}", subsection_title_style))
                story.append(Spacer(1, 0.1 * inch))
                with open(txt_path, 'r', encoding='utf-8') as f:  # Specify encoding for robustness
                    summary_content = f.read()
                    summary_content = summary_content.replace(
                        '\n\n', '<br/><br/>').replace('\n', '<br/>')
                    story.append(Paragraph(summary_content, normal_style))

                if i < len(text_files_to_include) - 1:  # Add space between summaries
                    story.append(Spacer(1, 0.5 * inch))

            except Exception as e:
                print(f"Error reading summary text file {txt_filename}: {e}")
                story.append(
                    Paragraph(f"<i>Error loading summary from {txt_filename}</i>", normal_style))
        story.append(PageBreak())  # Move to next page after all text summaries
    else:
        print(
            f"Warning: No summary text files found in {output_dir}. Skipping text summary section.")

    # --- 3. Add Images ---
    image_files = [f for f in os.listdir(
        output_dir) if f.lower().endswith('.png')]
    image_files.sort()  # Sort to ensure consistent order

    if image_files:
        story.append(
            Paragraph("Key Frames Visualizations", section_title_style))
        story.append(Spacer(1, 0.2 * inch))
        for img_filename in image_files:
            img_path = os.path.join(output_dir, img_filename)
            try:
                # Add image title
                # Make the title more readable from filename
                img_title = os.path.splitext(img_filename)[0].replace(
                    "_", " ").replace("frame", "Frame")
                story.append(Paragraph(f"Frame: {img_title}", styles['h3']))
                story.append(Spacer(1, 0.1 * inch))

                img = Image(img_path)
                # Calculate aspect ratio to fit image on page without distortion
                # Page width minus margins (1 inch on each side)
                max_width = A4[0] - 2 * inch
                # Page height minus margins and title space
                max_height = A4[1] - 3 * inch

                # Get original image dimensions
                img_width, img_height = img.drawWidth, img.drawHeight

                # Calculate scaling factor
                width_scale = max_width / img_width
                height_scale = max_height / img_height
                scale_factor = min(width_scale, height_scale)

                # Apply scaling
                img.drawWidth = img_width * scale_factor
                img.drawHeight = img_height * scale_factor

                story.append(img)
                story.append(Spacer(1, 0.2 * inch))  # Space after image
                # Start a new page for the next image
                story.append(PageBreak())
            except Exception as e:
                print(f"Error adding image {img_filename} to PDF: {e}")
                story.append(
                    Paragraph(f"<i>Could not load image: {img_filename}</i>", normal_style))
                # Still move to next page for robustness, even if image fails
                story.append(PageBreak())
    else:
        print(f"No PNG images found in {output_dir}. Skipping image section.")

    # Build the PDF
    try:
        doc.build(story)
        print(f"\nPDF summary successfully created at: {output_pdf_path}")
    except Exception as e:
        print(f"Error building PDF: {e}")
