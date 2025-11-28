# Converting HTML Visualizations to Images for LaTeX

LaTeX cannot directly include HTML files. Here are your options:

## ✅ Quick Fix: Use Existing PNGs (Already Done!)

Your repository already has PNG versions of all the visualizations you need:
- `figures/end_to_end/oos/*.png` - Greek surfaces
- `figures/final_results/*.png` - Price surfaces
- `figures/residual_heatmaps/pde_residual_coarse.png` - PDE residual

The draft has been updated to use these PNG files.

## 🔧 Future: Converting Plotly HTML to Static Images

If you need to convert new HTML files to images, here are several methods:

### Method 1: Using kaleido (Python - Best for Plotly)

```python
import plotly.graph_objects as go
from plotly.io import write_image

# If you have a Plotly figure object:
fig = go.Figure(...)
fig.write_image("output.png", width=1200, height=800, scale=2)

# Or load from HTML and re-export:
import plotly.io as pio
fig = pio.read_html("figure.html")
fig.write_image("output.png")
```

Install kaleido: `pip install kaleido`

### Method 2: Screenshot in Browser

1. Open the HTML file in a browser
2. Use browser screenshot tools:
   - **Chrome/Edge**: F12 → Ctrl+Shift+P → "Capture screenshot"
   - **Firefox**: Right-click → "Take Screenshot"
3. Save as PNG

### Method 3: Using Selenium (Automated)

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

driver.get(f'file://{abs_path_to_html}')
time.sleep(2)  # Wait for rendering
driver.save_screenshot('output.png')
driver.quit()
```

### Method 4: Using wkhtmltoimage (Command Line)

```bash
# Convert HTML to image
wkhtmltoimage figure.html figure.png

# With custom size
wkhtmltoimage --width 1200 --height 800 figure.html figure.png
```

Install: `sudo apt-get install wkhtmltopdf` (includes wkhtmltoimage)

### Method 5: Using Playwright (Modern alternative to Selenium)

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1200, 'height': 800})
    page.goto(f'file://{abs_path_to_html}')
    page.screenshot(path='output.png')
    browser.close()
```

## 📋 Recommended Workflow for Your Project

Since you're using Plotly to generate HTML files:

1. **Modify your visualization scripts** to also save PNG versions:

```python
# In your eval.py or visualization notebooks:
fig = px.imshow(...)

# Save both formats
fig.write_html("output.html")  # Interactive for exploration
fig.write_image("output.png", width=1200, height=900, scale=2)  # For LaTeX
```

2. **Update your evaluation script** (`src/eval.py`) to export PNG alongside HTML:

```python
# Around line 150+ where figures are saved
if fig_dir is not None:
    fig_path = fig_dir / f'{name}_surface.html'
    fig.write_html(str(fig_path))

    # Add PNG export:
    png_path = fig_dir / f'{name}_surface.png'
    fig.write_image(str(png_path), width=1200, height=900, scale=2)
```

## 🎨 Image Quality Tips for LaTeX

- **Resolution**: Use `scale=2` or higher for publication quality
- **Size**: 1200x900 px is good for most academic papers
- **Format**: PNG for plots with sharp lines, PDF for vector graphics
- **DPI**: Aim for 300 DPI minimum for print publications

## 📦 Adding Vector Graphics Support (Optional)

For even better quality, export as PDF instead of PNG:

```python
fig.write_image("output.pdf")  # Vector format, scales perfectly
```

Then in LaTeX:
```latex
\includegraphics[width=0.7\linewidth]{figures/output.pdf}
```

Vector PDFs are ideal for line plots and Greek surfaces!
