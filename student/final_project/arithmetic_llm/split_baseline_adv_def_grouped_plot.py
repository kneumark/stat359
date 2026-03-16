import os
from PIL import Image

INPUT_IMAGE = "instructor/final_project/arithmetic_llm/018_proposal_cross_axis_plots/cross_axis_accuracy_grouped_bars.png"
OUTPUT_DIR = "student/final_project/adversarial_results/plots_baseline_adv_def_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = Image.open(INPUT_IMAGE)
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
    panel_w, panel_h = cropped.size
    # Center the legend at its original width
    legend_w, legend_h = legend_full.size
    if panel_w >= legend_w:
        offset_x = (panel_w - legend_w) // 2
        cropped.paste(legend_full, (offset_x, 0))
    else:
        legend_cropped = legend_full.crop((legend_w//2 - panel_w//2, 0, legend_w//2 + panel_w//2, legend_h))
        cropped.paste(legend_cropped, (0, 0))
    out_path = os.path.join(OUTPUT_DIR, f"plot{pidx+1}.png")
    cropped.save(out_path)
    print(out_path)
