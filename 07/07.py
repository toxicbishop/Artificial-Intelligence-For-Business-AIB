"""
=============================================================================
  AI FOR BUSINESS — Week 07
  Topic : Public relations, communications, and AI
  File  : 07.py

  Approach : AI-driven Presentation Creation
    1. Create the title slide.
    2. Add a content slide.
    3. Add an image slide (branding).

  Dependencies : python-pptx  (pip install python-pptx)
=============================================================================
"""

from pptx import Presentation
from pptx.util import Inches
import os

# Ensure we run from the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Ensure output directory exists
os.makedirs("Program 7/output", exist_ok=True)

# ============================================================
# Step 1: Create the title slide
# ============================================================
print("Step 1: Creating title slide...")

prs = Presentation()

# Create a title slide (layout 0)
slide_layout = prs.slide_layouts[0]  # Title Slide
slide = prs.slides.add_slide(slide_layout)

# Set title and subtitle for the title slide
title = slide.shapes.title
subtitle = slide.placeholders[1]

# Assign text to title and subtitle
title.text = "AI in Public Relations"
subtitle.text = "Automating PR and Branding with Python"

# Save the presentation to a file in the output folder
prs.save("Program 7/output/presentation.pptx")
print("Title slide created and saved to 'Program 7/output/presentation.pptx'")

# ============================================================
# Step 2: Add a content slide
# ============================================================
print("\nStep 2: Adding content slide...")

# Add a content slide (layout 1)
slide_layout = prs.slide_layouts[1]  # Title and Content Slide
slide = prs.slides.add_slide(slide_layout)

# Set title and content for the slide
title = slide.shapes.title
content = slide.shapes.placeholders[1]

# Assign text to title and content
title.text = "What is AI in PR?"
content.text = "AI can automate content creation, audience targeting, and media monitoring."

# Save the updated presentation with the content slide
prs.save("Program 7/output/presentation_with_content.pptx")
print("Content slide added and saved to 'Program 7/output/presentation_with_content.pptx'")

# ============================================================
# Step 3: Add an image slide
# ============================================================
print("\nStep 3: Adding image slide...")

# Reload the existing presentation with content
prs = Presentation("Program 7/output/presentation_with_content.pptx")

# Add a new slide (title and content layout)
slide_layout = prs.slide_layouts[1]
slide = prs.slides.add_slide(slide_layout)

# Set title for the slide
title = slide.shapes.title
title.text = "Branding Example"

# Add the image (properly sized and positioned)
img_path = "Program 7/assets/logo.png"

# Add image with fixed size and position
left = Inches(1)
top = Inches(2)
width = Inches(4)
height = Inches(1.5)

slide.shapes.add_picture(img_path, left, top, width=width, height=height)

# Save the updated presentation
prs.save("Program 7/output/presentation_with_image.pptx")
print("Image slide added and saved to 'Program 7/output/presentation_with_image.pptx'")

print("\n✅ All presentations created successfully in 'Program 7/output/' folder!")
print("Files generated:")
for f in os.listdir("Program 7/output"):
    print(f"  - {f}")
