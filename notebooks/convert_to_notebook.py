"""Convert train_classifier.py into a well-structured Colab .ipynb notebook."""
import json, re

# Read the source
with open("/home/sefasys/Desktop/Breast_Ultrasound_Dataset/train_classifier.py") as f:
    source = f.read()

# Split by section comments (─── N. TITLE ───)
sections = re.split(r'\n# ─── (\d+)\. ([A-Z &]+) ─+\n', source)

cells = []

# Title markdown cell
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🩺 Breast Ultrasound Classification\n",
        "## EfficientNet-B0 + Transfer Learning | Google Colab\n",
        "\n",
        "**Anti-Overfitting:** Transfer Learning, Dropout (0.4), Label Smoothing (0.1), ",
        "Weight Decay, Early Stopping, Online Augmentation (7 transform)\n",
        "\n",
        "**Eğitim Stratejisi:** 2 aşamalı — Stage 1: Head only → Stage 2: Fine-tune backbone"
    ]
})

# Drive mount cell
cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
    ],
    "execution_count": None,
    "outputs": []
})

# Now parse sections
# sections[0] is the docstring + imports before first section marker
# Then triplets: (number, title, code)
preamble = sections[0].strip()

# Add preamble (docstring + imports) as first code cell
# Remove the docstring part
preamble_lines = preamble.split('\n')
# Find where imports start
import_start = 0
for i, line in enumerate(preamble_lines):
    if line.startswith('import ') or line.startswith('from '):
        import_start = i
        break

imports_code = '\n'.join(preamble_lines[import_start:])

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. Imports"]
})
cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": imports_code.split('\n'),
    "execution_count": None,
    "outputs": []
})

# Process remaining sections as triplets
i = 1
while i < len(sections) - 2:
    num = sections[i]
    title = sections[i+1].strip()
    code = sections[i+2].strip()
    
    # Markdown header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {num}. {title.title()}"]
    })
    
    # For long sections, split into logical sub-cells
    # Split on class/function definitions
    if code.count('\ndef ') > 1 or code.count('\nclass ') > 0:
        # Split by top-level class/function
        parts = re.split(r'\n(?=(?:class |def )\w)', code)
        for part in parts:
            part = part.strip()
            if part:
                cells.append({
                    "cell_type": "code",
                    "metadata": {},
                    "source": [line + '\n' for line in part.split('\n')],
                    "execution_count": None,
                    "outputs": []
                })
    else:
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [line + '\n' for line in code.split('\n')],
            "execution_count": None,
            "outputs": []
        })
    
    i += 3

# Build notebook
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": cells
}

out_path = "/home/sefasys/Desktop/Breast_Ultrasound_Dataset/train_classifier.ipynb"
with open(out_path, 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook created: {out_path}")
print(f"   Total cells: {len(cells)}")
print(f"   Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
print(f"   Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
