# LeukoAnalyze Pro 🔬

**Advanced Hematology Analysis & Leukemia Diagnostic Screening Platform**

LeukoAnalyze Pro is a premium, AI-powered Streamlit dashboard designed to automate the screening of peripheral blood smear images. It integrates state-of-the-art computer vision models (YOLO & Vision Transformers) to detect, classify, and morphology-profile individual blood cells, culminating in a professional, clinical-grade diagnostic report.



## ✨ Core Features

*   **📤 Batch Processing Pipeline:** Drag-and-drop multiple high-resolution peripheral blood smear images for instantaneous batch inference.
*   **🎯 Cellular Detection (YOLOv8):** Accurately detects and draws bounding boxes around Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets.
*   **🧬 Morphology Classification (ViT):** Extracts individual WBCs and uses a custom Vision Transformer model to predict highly specific cellular attributes (e.g., Nucleus Shape, Chromatin Density, N:C Ratio).
*   **📊 Cohort Analytics:** Consolidates data from all images into beautiful, dark-mode Plotly charts, breaking down the cellular distribution and blast proliferation percentages.
*   **🧾 Official Clinical Report:** Automatically generates and exports a stylized, high-fidelity HTML Diagnostic Report designed for Electronic Health Records (EHR) attachment or immediate printing.

## 🛠️ Technology Stack

*   **Frontend / UI:** [Streamlit](https://streamlit.io/) with Custom Premium CSS injections.
*   **Object Detection:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (PyTorch)
*   **Image Classification:** [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
*   **Data Processing:** Pandas, NumPy
*   **Visualizations:** Plotly Express

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/LeukoAnalyze-Pro.git
   cd LeukoAnalyze-Pro
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model Weights:**
   Ensure you have the pre-trained weights for the pipeline:
   *   `best_TXL-PBC_model.pt` (YOLOv8 Weights)
   *   `vit_epoch_4.pth` (Vision Transformer Weights)
   *   `pbc_attr_v1_train.csv` (Attribute configuration mappings)
   
   *Place these files in the root directory or configure their paths in the dashboard sidebar.*

## 💻 Usage

Run the dashboard locally using Streamlit:

```bash
streamlit run app_v1.py
```

1. Navigate to `http://localhost:8501` in your browser.
2. In the **⚙️ Engine Settings** sidebar, verify the paths to your YOLO and ViT models.
3. Use the **📤 Import Batch** tab to upload peripheral blood smear images (`.png`, `.jpg`).
4. Click **🚀 PROCESS BATCH** and wait for the AI to finish analyzing.
5. Navigate through the subsequent tabs to view single-cell analysis, overall analytics, and download your Official Clinical Report.

## ⚠️ Disclaimer

**LeukoAnalyze Pro is an AI-assisted research and screening tool.** It is not intended to replace professional medical diagnosis, judgment, or standard hematopathology reviews. All findings—especially those flagging "Blast-like" cells—should be verified by a certified medical professional via manual slide review and/or bone marrow aspiration.

---
*Built as a Capstone Project for Advanced AI Leukemia Detection.*
