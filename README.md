# GEN-AI-POWERED-MEDICAL-REPORT-GENERATOR
# Gen AI Medical Image Report Generator (Demo - Template-based)

This is a **lightweight demo** of a "Gen AI powered medical image report generator" intended to run easily on **Windows (CPU)**.
It uses a simple image-statistics -> template approach (fast) rather than heavy ML models, so you can run it without GPUs.

## Files
- `app.py` : Streamlit app (UI)
- `model.py` : Simple image analyzer + template report generator
- `utils.py` : Helper functions
- `requirements.txt` : Python dependencies
- `sample_data/chest_xray.jpg` : Sample placeholder image
- `sample_data/reports/` : example reports folder

## Run (Windows)
1. Create and activate a Python virtual environment (recommended).
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the app:
```
streamlit run app.py
```

Open the URL shown by Streamlit in your browser, upload an image (or use the sample image), and click "Generate Report".

This demo is intentionally simple (option 1 you requested). Replace the `generate_report` function in `model.py` with a real ML pipeline later when ready.
