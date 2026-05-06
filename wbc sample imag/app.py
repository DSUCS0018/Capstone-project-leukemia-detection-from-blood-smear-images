# import streamlit as st
# from PIL import Image
# import torch
# import torch.nn as nn
# from torchvision import transforms
# import timm
# import pandas as pd
# import io
# import os
# import json
# import numpy as np
# import streamlit.components.v1 as components

# # -------------------------------------------------------
# # 1. Blast classifier with reasons
# # -------------------------------------------------------
# def classify_blast_with_reasons(row):
#     abnormal = 0
#     reasons = []

#     blast_nucleus = {"unsegmented-round", "unsegmented-band", "unsegmented-indented"}
#     if row["nucleus_shape"] in blast_nucleus:
#         abnormal += 1; reasons.append(f"nucleus_shape = {row['nucleus_shape']}")

#     if row["nuclear_cytoplasmic_ratio"] == "high":
#         abnormal += 1; reasons.append("nuclear_cytoplasmic_ratio = high")

#     if row["chromatin_density"] == "loosely":
#         abnormal += 1; reasons.append("chromatin_density = loosely")

#     if row["granularity"] == "no" or row["granule_type"] == "nil":
#         abnormal += 1; reasons.append(f"granules = {row['granularity']} / {row['granule_type']}")

#     if row["cell_size"] == "big":
#         abnormal += 1; reasons.append("cell_size = big")

#     status = "blast-like" if abnormal >= 2 else "normal"
#     return status, reasons


# # -------------------------------------------------------
# # 2. Load attribute mappings from CSV
# # -------------------------------------------------------
# # Load attributes CSV if present — otherwise keep None and show helpful UI
# CSV_PATH = "pbc_attr_v1_train.csv"
# if os.path.exists(CSV_PATH):
#     df = pd.read_csv(CSV_PATH)
# else:
#     df = None

# ATTRIBUTES = [
#     "cell_size","cell_shape","nucleus_shape","nuclear_cytoplasmic_ratio",
#     "chromatin_density","cytoplasm_vacuole","cytoplasm_texture",
#     "cytoplasm_colour","granule_type","granule_colour","granularity"
# ]

# if df is not None:
#     attr2idx = {a: {v:i for i,v in enumerate(sorted(df[a].astype(str).unique()))}
#                 for a in ATTRIBUTES}
#     idx2attr = {a: {i:v for v,i in attr2idx[a].items()} for a in ATTRIBUTES}

#     labels = sorted(df["label"].unique())
#     cls2idx = {c:i for i,c in enumerate(labels)}
#     idx2cls = {i:c for c,i in cls2idx.items()}
# else:
#     # fallbacks so app can still run when CSV is missing
#     attr2idx = {a: {"unknown": 0} for a in ATTRIBUTES}
#     idx2attr = {a: {0: "unknown"} for a in ATTRIBUTES}
#     labels = ["class0", "class1"]
#     cls2idx = {c:i for i,c in enumerate(labels)}
#     idx2cls = {i:c for c,i in cls2idx.items()}


# # -------------------------------------------------------
# # 3. Model definition (exact same as training)
# # -------------------------------------------------------
# class MultiTaskViT(nn.Module):
#     def __init__(self, backbone, embed_dim, n_classes, attr_cardinalities, dropout=0.2):
#         super().__init__()
#         self.backbone = backbone
#         self.dropout = nn.Dropout(dropout)
#         self.classifier = nn.Linear(embed_dim, n_classes)
#         self.attr_heads = nn.ModuleDict({a: nn.Linear(embed_dim, c)
#                                          for a, c in attr_cardinalities.items()})

#     def forward(self, x):
#         feats = self.backbone.forward_features(x)
#         cls_feat = feats[:,0]
#         cls_feat = self.dropout(cls_feat)
#         out_class = self.classifier(cls_feat)
#         out_attrs = {a: head(cls_feat) for a, head in self.attr_heads.items()}
#         return out_class, out_attrs


# # -------------------------------------------------------
# # 4. Load pretrained model
# # -------------------------------------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"


# @st.cache_resource
# def build_and_load_model(weights_path: str = "vit_epoch_4.pth"):
#     """Create model architecture and load weights if available. Returns model or None and an error message.

#     We wrap this in cache_resource to avoid repeated heavy initialization.
#     """
#     try:
#         backbone = timm.create_model("vit_base_patch16_224", pretrained=True)
#     except Exception as e:
#         return None, f"Failed to create backbone: {e}"

#     try:
#         embed_dim = backbone.head.in_features
#         backbone.reset_classifier(0)
#         backbone.to(device)

#         attr_cards = {a: len(attr2idx[a]) for a in ATTRIBUTES} if df is not None else {a: 2 for a in ATTRIBUTES}
#         model_instance = MultiTaskViT(backbone, embed_dim, len(cls2idx), attr_cards).to(device)

#         if os.path.exists(weights_path):
#             model_instance.load_state_dict(torch.load(weights_path, map_location=device))
#             model_instance.eval()
#             return model_instance, None
#         else:
#             return None, f"Weights not found at: {weights_path}"
#     except Exception as e:
#         return None, f"Error building/loading model: {e}"

# # Image transform
# tfm = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])


# # -------------------------------------------------------
# # 5. STREAMLIT UI
# # -------------------------------------------------------
# st.set_page_config(page_title="WBC Classifier", page_icon="🩸", layout="wide")

# header_left, header_right = st.columns([6, 2])
# with header_left:
#     st.markdown("# 🩸 WBC Classification & Blast Detection")
#     st.write("Upload a WBC image to classify cell type, predict morphological attributes, and flag blast-like morphology.")
# # with header_right:
# #     st.write("")
# #     st.image("https://raw.githubusercontent.com/streamlit/brand/main/streamlit-mark-black.png", width=80)

# # Sidebar: model path + status
# st.sidebar.header("Model & data")
# model_weights = st.sidebar.text_input("Model weights path", value="vit_epoch_4.pth")
# csv_status = "found" if df is not None else "missing"
# st.sidebar.write(f"Attributes CSV: {csv_status}")

# uploaded_file = st.file_uploader("Drop or choose a WBC image — detection runs automatically", type=["jpg", "jpeg", "png"], key="main_uploader")

# if uploaded_file is not None:
#     img = Image.open(uploaded_file).convert("RGB")

#     st.image(img, caption="Uploaded Image", width='stretch' \
#     '')

#     # Prepare input tensor
#     inp = tfm(img).unsqueeze(0).to(device)

#     # Build/load model on demand
#     model, model_err = build_and_load_model(model_weights)
#     if model is None:
#         st.error(f"Model unavailable: {model_err}")
#         st.info("If you'd like, upload your model weights to the project folder or update the path in the sidebar.")
#     else:
#         with torch.no_grad():
#             out_class, out_attrs = model(inp)

#         # Decode class + softmax probabilities
#         cls_logits = out_class.cpu().numpy().squeeze()
#         cls_probs = np.exp(cls_logits) / np.sum(np.exp(cls_logits))
#         top_k = sorted(list(enumerate(cls_probs)), key=lambda x: x[1], reverse=True)[:3]
#         cls_pred = idx2cls[int(np.argmax(cls_probs))]

#         # Decode attributes with probabilities if CSV available
#         pred_attr = {}
#         attr_conf = {}
#         for a, out in out_attrs.items():
#             arr = out.cpu().numpy().squeeze()
#             probs = np.exp(arr) / np.sum(np.exp(arr))
#             top_idx = int(np.argmax(probs))
#             if df is not None:
#                 pred_attr[a] = idx2attr[a][top_idx]
#             else:
#                 pred_attr[a] = str(top_idx)
#             attr_conf[a] = float(np.max(probs))

#         # Blast detection
#         blast_status, reasons = classify_blast_with_reasons(pred_attr)

#         # ---------------------------------------------------
#         # RESULTS DISPLAY
#         # ---------------------------------------------------
#         # RESULTS — layout into two columns: left = image/results, right = details
#         left_col, right_col = st.columns([3,2])
#         with left_col:
#             st.subheader(f"🔍 Predicted Class: **{cls_pred}**")
#             st.markdown("**Top predictions (class ± probability)**")
#             for i, p in top_k:
#                 name = idx2cls[int(i)]
#                 st.write(f"- {name}: {p:.2%}")

#             st.subheader(f"🧬 Blast Status: **{blast_status}**")

#             if blast_status == "blast-like":
#                 st.error(" ⚠ Blast-like morphology detected")
#                 with st.expander("View reasons", expanded=True):
#                     for r in reasons:
#                         st.write(f"- {r}")
#             else:
#                 st.success("Normal cell morphology detected")

#         with right_col:
#             st.markdown("### Predicted Morphological Attributes")
#             # Show attributes table with confidence
#             attr_rows = []
#             for k, v in pred_attr.items():
#                 attr_rows.append({"attribute": k, "prediction": v, "confidence": f"{attr_conf.get(k,0):.2%}"})
#             st.table(pd.DataFrame(attr_rows))

#         # Downloadable result JSON
#         results_summary = {
#             "predicted_class": cls_pred,
#             "class_probs": {idx2cls[i]: float(p) for i,p in enumerate(cls_probs)},
#             "attributes": pred_attr,
#             "attribute_confidences": attr_conf,
#             "blast_status": blast_status,
#             "reasons": reasons,
#         }
#         st.download_button("Download result (JSON)", data=json.dumps(results_summary, indent=2), file_name="wbc_prediction.json")

#Set-Location -LiteralPath 'C:\Users\madhu\Desktop\major project\wbc sample imag'; & 'C:\Users\madhu\AppData\Local\Programs\Python\Python313\python.exe' -m streamlit run app.py



import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm
import pandas as pd
import os
import json
import numpy as np

# -------------------------------------------------------
# 1. Blast classifier with reasons
# -------------------------------------------------------
def classify_blast_with_reasons(row):
    abnormal = 0
    reasons = []

    blast_nucleus = {"unsegmented-round", "unsegmented-band", "unsegmented-indented"}
    if row["nucleus_shape"] in blast_nucleus:
        abnormal += 1; reasons.append(f"nucleus_shape = {row['nucleus_shape']}")

    if row["nuclear_cytoplasmic_ratio"] == "high":
        abnormal += 1; reasons.append("nuclear_cytoplasmic_ratio = high")

    if row["chromatin_density"] == "loosely":
        abnormal += 1; reasons.append("chromatin_density = loosely")

    if row["granularity"] == "no" or row["granule_type"] == "nil":
        abnormal += 1; reasons.append(f"granules = {row['granularity']} / {row['granule_type']}")

    if row["cell_size"] == "big":
        abnormal += 1; reasons.append("cell_size = big")

    status = "blast-like" if abnormal >= 2 else "normal"
    return status, reasons


# -------------------------------------------------------
# 2. Load attribute mappings
# -------------------------------------------------------
CSV_PATH = "pbc_attr_v1_train.csv"
df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else None

ATTRIBUTES = [
    "cell_size","cell_shape","nucleus_shape","nuclear_cytoplasmic_ratio",
    "chromatin_density","cytoplasm_vacuole","cytoplasm_texture",
    "cytoplasm_colour","granule_type","granule_colour","granularity"
]

if df is not None:
    attr2idx = {a: {v:i for i,v in enumerate(sorted(df[a].astype(str).unique()))}
                for a in ATTRIBUTES}
    idx2attr = {a: {i:v for v,i in attr2idx[a].items()} for a in ATTRIBUTES}
    labels = sorted(df["label"].unique())
    cls2idx = {c:i for i,c in enumerate(labels)}
    idx2cls = {i:c for c,i in cls2idx.items()}
else:
    attr2idx = {a: {"unknown": 0} for a in ATTRIBUTES}
    idx2attr = {a: {0: "unknown"} for a in ATTRIBUTES}
    labels = ["class0", "class1"]
    cls2idx = {c:i for i,c in enumerate(labels)}
    idx2cls = {i:c for c,i in cls2idx.items()}


# -------------------------------------------------------
# 3. Model
# -------------------------------------------------------
class MultiTaskViT(nn.Module):
    def __init__(self, backbone, embed_dim, n_classes, attr_cardinalities):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.attr_heads = nn.ModuleDict(
            {a: nn.Linear(embed_dim, c) for a, c in attr_cardinalities.items()}
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        cls_feat = feats[:, 0]
        return (
            self.classifier(cls_feat),
            {a: h(cls_feat) for a, h in self.attr_heads.items()}
        )


# -------------------------------------------------------
# 4. Load model
# -------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model(weights_path):
    backbone = timm.create_model("vit_base_patch16_224", pretrained=True)
    embed_dim = backbone.head.in_features
    backbone.reset_classifier(0)
    backbone.to(device)

    attr_cards = {a: len(attr2idx[a]) for a in ATTRIBUTES}
    model = MultiTaskViT(backbone, embed_dim, len(cls2idx), attr_cards).to(device)

    if not os.path.exists(weights_path):
        return None

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# -------------------------------------------------------
# 5. Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="WBC Classifier", layout="wide")
st.title("WBC Classification & Blast Detection")

WBC_CONF_THRESHOLD = 0.5

model_weights = st.sidebar.text_input("Model weights path", "vit_epoch_4.pth")
model = load_model(model_weights)

uploaded_file = st.file_uploader("Upload WBC image", type=["jpg","jpeg","png"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_container_width=True)

    inp = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_class, out_attrs = model(inp)

    cls_probs = torch.softmax(out_class, dim=1).cpu().numpy()[0]
    max_conf = float(np.max(cls_probs))
    cls_pred = idx2cls[int(np.argmax(cls_probs))]

    # ---------- NO WBC GATE ----------
    if max_conf < WBC_CONF_THRESHOLD:
        st.error("No WBC found in the image")
        st.write(f"Max class confidence: {max_conf:.2f}")
        st.stop()
    # --------------------------------

    pred_attr, attr_conf = {}, {}
    for a, out in out_attrs.items():
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        pred_attr[a] = idx2attr[a][idx]
        attr_conf[a] = float(np.max(probs))

    blast_status, reasons = classify_blast_with_reasons(pred_attr)

    st.subheader(f"Predicted Class: {cls_pred}")
    st.subheader(f"Blast Status: {blast_status}")

    if reasons:
        for r in reasons:
            st.write("-", r)

    st.table(pd.DataFrame([
        {"attribute": k, "prediction": v, "confidence": f"{attr_conf[k]:.2%}"}
        for k,v in pred_attr.items()
    ]))

    st.download_button(
        "Download JSON",
        data=json.dumps({
            "class": cls_pred,
            "class_confidence": max_conf,
            "attributes": pred_attr,
            "blast_status": blast_status,
            "reasons": reasons
        }, indent=2),
        file_name="wbc_result.json"
    )
