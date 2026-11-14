"""
Constructs the complete metadata dictionary for all 85 steps of the notebook.
Each step has:
- title
- code_explanation (90-120 word continuous paragraph, simple language)
- output_explanation (90-120 word continuous paragraph, exact numbers)
- fig_info (optional, image file, caption, 100-120 word visual analysis)
- purpose (for cell index)
- artifact (for cell index)
"""

import json

steps = {}

# We will populate steps[1] to steps[85]
