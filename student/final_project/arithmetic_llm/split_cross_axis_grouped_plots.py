import os
from PIL import Image

INPUT_IMAGES = [
    "instructor/final_project/arithmetic_llm/018_proposal_cross_axis_plots/cross_axis_accuracy_grouped_bars.png",
    "instructor/final_project/arithmetic_llm/018_proposal_cross_axis_plots/cross_axis_accuracy_zoomed_adv_def.png"
]
OUTPUT_DIR = "student/final_project/adversarial_results/plots_cross_axis_separated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Each image has 3 panels: Numeric, Language, Politeness
# We'll crop them by thirds horizontally
for idx, img_path in enumerate(INPUT_IMAGES):
    img = Image.open(img_path)
    w, h = img.size
    panel_width = w // 3
    legend_height = 120  # Adjust if legend is taller/shorter
    legend_full = img.crop((0, 0, w, legend_height))
    panels = [
        (0, 0, panel_width, h),  # Numeric
        (panel_width, 0, 2*panel_width, h),  # Language
        (2*panel_width, 0, w, h)  # Politeness
    ]
    for pidx, box in enumerate(panels):
        cropped = img.crop(box)
        # Paste legend onto panel (resize legend to panel width if needed)
        panel_w, panel_h = cropped.size
        # Center the legend at its original width
        legend_w, legend_h = legend_full.size
        if panel_w >= legend_w:
            # Panel is wider than legend, paste legend centered
            offset_x = (panel_w - legend_w) // 2
            cropped.paste(legend_full, (offset_x, 0))
        else:
            # Panel is narrower, crop legend to panel width
            legend_cropped = legend_full.crop((legend_w//2 - panel_w//2, 0, legend_w//2 + panel_w//2, legend_h))
            cropped.paste(legend_cropped, (0, 0))
        out_path = os.path.join(OUTPUT_DIR, f"plot{idx*3+pidx+1}.png")
        cropped.save(out_path)
        print(out_path)
