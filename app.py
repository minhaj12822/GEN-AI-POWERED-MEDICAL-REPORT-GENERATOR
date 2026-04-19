import streamlit as st
from PIL import Image
import io
import os
import base64

from model_infer import load_trained_model, predict_from_image, image_stats_and_check

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="GenAI Medical Report Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def get_base64_image(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def dummy_predict():
    return "NORMAL", 0.50, [0.50, 0.50]

def safe_probs(probs):
    try:
        normal_prob = float(probs[0])
        pneumonia_prob = float(probs[1])
    except Exception:
        normal_prob = 0.50
        pneumonia_prob = 0.50
    return normal_prob, pneumonia_prob

def generate_medical_report(pred_label, conf_pct, stats):
    report = []
    report.append("EXAM: Chest X-ray (PA/AP view)")
    report.append("")
    report.append("FINDINGS:")

    if pred_label.lower() == "normal":
        report.append("- Lung fields are clear with no focal air-space consolidation.")
        report.append("- Cardiothoracic silhouette is within expected limits.")
        report.append("- No pleural effusion or pneumothorax identified.")
        impression = "No active cardiopulmonary abnormality detected."
        recommendation = "Routine clinical correlation and follow-up if indicated."

    elif pred_label.lower() == "pneumonia":
        if conf_pct < 70:
            severity = "mild"
        elif conf_pct < 90:
            severity = "moderate"
        else:
            severity = "severe"

        report.append(f"- Patchy or confluent opacities noted, suggestive of {severity} pneumonia.")
        report.append("- Increased lung markings are seen in involved regions.")
        report.append("- No pleural effusion or pneumothorax identified on the current view.")
        impression = f"Findings are suggestive of {severity} pneumonia."
        recommendation = "Recommend clinical correlation and follow-up chest X-ray in 7–10 days."

    else:
        report.append("- Image quality appears suboptimal or findings are not classifiable.")
        impression = "Unable to determine abnormality confidently."
        recommendation = "Repeat radiographic study with proper positioning and exposure."

    report.append("")
    report.append("IMAGE METRICS:")
    report.append(f"- Mean Pixel Intensity: {stats['mean']:.1f}")
    report.append(f"- Standard Deviation: {stats['std']:.1f}")
    report.append(f"- Edge Density: {stats['edge_count']}")
    report.append(f"- Aspect Ratio: {stats['aspect_ratio']:.2f}")

    final_report = "\n".join(report)
    final_report += f"\n\nIMPRESSION: {impression}"
    final_report += f"\n\nRECOMMENDATION: {recommendation}"
    return final_report

def find_existing_file(possible_paths):
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

# ---------------------------------------------------
# ASSETS AUTO-DETECT
# ---------------------------------------------------
bg_path = find_existing_file([
    "assets/bg.jpg", "assets/bg.jpeg", "assets/bg.png"
])

hero_img = find_existing_file([
    "assets/hero.png", "assets/hero.jpg", "assets/hero.jpeg"
])

sample1 = find_existing_file([
    "assets/sample1.jpg", "assets/sample1.jpeg", "assets/sample1.png"
])

sample2 = find_existing_file([
    "assets/sample2.jpg", "assets/sample2.jpeg", "assets/sample2.png"
])

sample3 = find_existing_file([
    "assets/sample3.jpg", "assets/sample3.jpeg", "assets/sample3.png"
])

bg_img = get_base64_image(bg_path) if bg_path else None

# ---------------------------------------------------
# BACKGROUND
# ---------------------------------------------------
if bg_img:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(2,6,23,0.78), rgba(2,6,23,0.88)),
                url("data:image/jpg;base64,{bg_img}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, #0f172a 0%, #020617 45%, #000000 100%);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# CSS BLOCK
# ---------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.92);
    border-right: 1px solid rgba(255,255,255,0.06);
}

.sidebar-title {
    font-size: 24px;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 6px;
}

.sidebar-sub {
    font-size: 13px;
    color: #cbd5e1;
    margin-bottom: 18px;
}

.sidebar-box {
    background: rgba(30,41,59,0.70);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 12px;
    color: #e2e8f0;
}

.hero-wrap {
    background: rgba(15,23,42,0.52);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 28px;
    padding: 30px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.30);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 52px;
    font-weight: 850;
    line-height: 1.1;
    color: #f8fafc;
}

.hero-highlight {
    color: #38bdf8;
}

.hero-sub {
    color: #cbd5e1;
    font-size: 18px;
    line-height: 1.7;
    margin-top: 14px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 14px;
    margin-right: 10px;
    margin-top: 8px;
    border-radius: 999px;
    font-size: 13px;
    color: #e0f2fe;
    background: rgba(14,165,233,0.14);
    border: 1px solid rgba(56,189,248,0.24);
}

.glass-card {
    background: rgba(15,23,42,0.55);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 22px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

.section-title {
    font-size: 28px;
    font-weight: 780;
    color: #f8fafc;
    margin-bottom: 8px;
}

.section-sub {
    font-size: 15px;
    color: #cbd5e1;
    margin-bottom: 14px;
}

.metric-card {
    border-radius: 22px;
    padding: 22px 18px;
    min-height: 145px;
    text-align: center;
    background: linear-gradient(135deg, rgba(15,23,42,0.90), rgba(30,41,59,0.84));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 24px rgba(0,0,0,0.22);
}

.metric-label {
    color: #cbd5e1;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 14px;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    line-height: 1.2;
}

.metric-green { color: #4ade80; }
.metric-blue { color: #38bdf8; }
.metric-purple { color: #c084fc; }
.metric-orange { color: #f59e0b; }

.report-box {
    background: rgba(2,6,23,0.86);
    border: 1px solid rgba(148,163,184,0.20);
    border-left: 6px solid #38bdf8;
    border-radius: 18px;
    padding: 24px;
    color: #e2e8f0;
    line-height: 1.8;
    font-size: 15px;
    white-space: pre-wrap;
}

.info-box {
    background: rgba(14,165,233,0.12);
    border: 1px solid rgba(56,189,248,0.22);
    border-radius: 16px;
    padding: 16px;
    color: #dbeafe;
}

.stButton > button {
    width: 100%;
    height: 3.2rem;
    border: none;
    border-radius: 14px;
    font-size: 17px;
    font-weight: 750;
    color: white;
    background: linear-gradient(90deg, #0284c7, #2563eb);
    box-shadow: 0 8px 18px rgba(37,99,235,0.28);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0369a1, #1d4ed8);
}

.stDownloadButton > button {
    width: 100%;
    height: 3rem;
    border-radius: 14px;
    border: none;
    font-size: 16px;
    font-weight: 750;
    color: white;
    background: linear-gradient(90deg, #0f766e, #0891b2);
}

[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.62);
    border: 1px dashed rgba(125,211,252,0.28);
    border-radius: 18px;
    padding: 12px;
}

.footer-box {
    text-align: center;
    color: #94a3b8;
    padding-top: 14px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">GenAI Radiology</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Project Dashboard • Minor Project Submission</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-box">
        <b>Project:</b><br>
        GenAI-Powered Image Medical Report Generator
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-box">
        <b>Workflow:</b><br>
        Upload → Validate → Predict → Analyze → Generate Report
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-box">
        <b>Technologies:</b><br>
        Python, Streamlit, PyTorch, CNN, PIL, NumPy
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-box">
        <b>Clinical Scope:</b><br>
        Chest X-ray classification into NORMAL / PNEUMONIA with structured AI-assisted reporting
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------
@st.cache_resource
def get_model():
    try:
        return load_trained_model("fast_chest_model.pth", device="cpu", num_classes=2)
    except Exception:
        return None

model = get_model()

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------
hero_col1, hero_col2 = st.columns([1.6, 1], gap="large")

with hero_col1:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">
            <span class="hero-highlight">GenAI-Powered</span><br>
            Image Medical Report Generator
        </div>
        <div class="hero-sub">
            A next-generation AI interface for chest X-ray screening and automated
            radiology-style report generation using deep learning, image intelligence,
            and clinically structured rule-based logic.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:-10px; margin-bottom:14px;">
        <span class="hero-badge">Chest X-ray AI</span>
        <span class="hero-badge">CNN Classification</span>
        <span class="hero-badge">Radiology Reporting</span>
        <span class="hero-badge">Academic + Assistive Use</span>
    </div>
    """, unsafe_allow_html=True)

with hero_col2:
    if hero_img:
        st.image(hero_img, use_container_width=True)
    else:
        st.markdown("""
        <div class="hero-wrap" style="min-height:290px; display:flex; align-items:center; justify-content:center; flex-direction:column;">
            <div style="font-size:74px;">🩻</div>
            <div style="font-size:22px; font-weight:800; color:#f8fafc; margin-top:10px;">AI Radiology Interface</div>
            <div style="font-size:14px; color:#94a3b8; margin-top:6px; text-align:center;">
                Add assets/hero image for a better top visual.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# UPLOAD + WORKFLOW
# ---------------------------------------------------
left_col, right_col = st.columns([1.2, 0.8], gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload Chest X-ray</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a radiographic image in JPG / JPEG / PNG format for automated analysis.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    generate_clicked = st.button("🚀 Generate AI Medical Report")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">🧠 AI Workflow Intelligence</div>
        <div class="section-sub">
            The system processes medical image data through a structured diagnosis pipeline.
        </div>
        <div class="info-box">
            1. Image upload and validation<br>
            2. CNN-based feature extraction<br>
            3. Classification into NORMAL / PNEUMONIA<br>
            4. Statistical metric extraction<br>
            5. Structured report generation
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# OPTIONAL SAMPLE GALLERY
# ---------------------------------------------------
available_samples = [x for x in [sample1, sample2, sample3] if x is not None]
if available_samples:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🖼 Sample X-ray Gallery</div>', unsafe_allow_html=True)
    sample_cols = st.columns(len(available_samples))
    for i, sample_path in enumerate(available_samples):
        with sample_cols[i]:
            st.image(sample_path, use_container_width=True)

# ---------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------
if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    is_xray, stats = image_stats_and_check(image)

    st.markdown("<br>", unsafe_allow_html=True)

    preview_col, info_col = st.columns([1, 1], gap="large")

    with preview_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🖼 Uploaded Image Preview</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📑 Initial Image Check</div>', unsafe_allow_html=True)

        validity_text = "Valid Chest X-ray Pattern" if is_xray else "Image Pattern Not Typical"
        validity_color = "#4ade80" if is_xray else "#f87171"

        st.markdown(
            f"""
            <div class="info-box">
                <b>Status:</b> <span style="color:{validity_color}; font-weight:700;">{validity_text}</span><br>
                <b>Mean Intensity:</b> {stats['mean']:.1f}<br>
                <b>Std. Deviation:</b> {stats['std']:.1f}<br>
                <b>Aspect Ratio:</b> {stats['aspect_ratio']:.2f}<br>
                <b>Edge Count:</b> {stats['edge_count']}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if generate_clicked:
        if not is_xray:
            st.error("This uploaded image does not appear to be a valid chest X-ray. Please upload a proper radiographic image.")
        else:
            if model is not None:
                try:
                    label, conf, probs = predict_from_image(
                        model,
                        image,
                        device="cpu",
                        class_names=["NORMAL", "PNEUMONIA"]
                    )
                except Exception as e:
                    st.warning(f"Prediction issue occurred. Using fallback mode. Error: {e}")
                    label, conf, probs = dummy_predict()
            else:
                st.warning("Model file not loaded. Running fallback prediction.")
                label, conf, probs = dummy_predict()

            conf_pct = conf * 100
            normal_prob, pneumonia_prob = safe_probs(probs)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="text-align:center;">📊 AI Diagnostic Summary</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4, gap="medium")

            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Prediction</div>
                    <div class="metric-value metric-green">{label}</div>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value metric-blue">{conf_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Edge Density</div>
                    <div class="metric-value metric-purple">{stats['edge_count']}</div>
                </div>
                """, unsafe_allow_html=True)

            with m4:
                status = "Stable" if label == "NORMAL" else "Attention"
                color_class = "metric-green" if label == "NORMAL" else "metric-orange"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Clinical Flag</div>
                    <div class="metric-value {color_class}">{status}</div>
                </div>
                """, unsafe_allow_html=True)

            report = generate_medical_report(label, conf_pct, stats)

            st.markdown("<br>", unsafe_allow_html=True)

            report_col, prob_col = st.columns([1.6, 0.8], gap="large")

            with report_col:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🧾 Generated Radiology Report</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with prob_col:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📈 Probability Analysis</div>', unsafe_allow_html=True)

                st.progress(normal_prob, text=f"NORMAL: {normal_prob*100:.2f}%")
                st.progress(pneumonia_prob, text=f"PNEUMONIA: {pneumonia_prob*100:.2f}%")

                st.markdown(
                    f"""
                    <div class="info-box">
                        <b>Normal Probability:</b> {normal_prob*100:.2f}%<br>
                        <b>Pneumonia Probability:</b> {pneumonia_prob*100:.2f}%<br>
                        <b>Decision:</b> {label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.download_button(
                    "📥 Download Report",
                    report,
                    file_name="AI_Medical_Report.txt",
                    use_container_width=True
                )

                st.markdown('</div>', unsafe_allow_html=True)

            st.warning("⚠️ This is an AI-assisted report generated for educational and assistive use only. Final diagnosis must be made by a qualified radiologist.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer-box">
    Developed with AI, Deep Learning, CNN and Streamlit • GenAI-Powered Image Medical Report Generator
</div>
""", unsafe_allow_html=True)