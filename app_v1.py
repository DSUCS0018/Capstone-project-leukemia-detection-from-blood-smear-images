import base64
import io
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import timm
import torch
import torch.nn as nn
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

YOLO_RBC_CLASS_ID = 0
YOLO_WBC_CLASS_ID = 1
YOLO_PLATELET_CLASS_ID = 2

DEFAULT_YOLO_PATH = "./best_TXL-PBC_model.pt"
DEFAULT_VIT_PATH = "./vit_epoch_4.pth"
DEFAULT_CONF_THRESHOLD = 0.5
BLAST_THRESHOLD = 50.0

DEFAULT_ATTRS = [
    "cell_size",
    "cell_shape",
    "nucleus_shape",
    "nuclear_cytoplasmic_ratio",
    "chromatin_density",
    "cytoplasm_vacuole",
    "cytoplasm_texture",
    "cytoplasm_colour",
    "granule_type",
    "granule_colour",
    "granularity",
]


def resolve_existing_path(candidates: List[str]) -> str:
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return candidates[0] if candidates else ""


def _normalize_label(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def infer_yolo_class_ids(yolo_model):
    names = yolo_model.names if isinstance(yolo_model.names, dict) else {}

    rbc_id = None
    wbc_id = None
    platelet_id = None

    for class_id, raw_name in names.items():
        normalized = _normalize_label(raw_name)
        if rbc_id is None and ("rbc" in normalized or "redbloodcell" in normalized or normalized == "redcell"):
            rbc_id = int(class_id)
        if wbc_id is None and ("wbc" in normalized or "whitebloodcell" in normalized or "leukocyte" in normalized):
            wbc_id = int(class_id)
        if platelet_id is None and ("platelet" in normalized or "thrombocyte" in normalized):
            platelet_id = int(class_id)

    if rbc_id is None:
        rbc_id = YOLO_RBC_CLASS_ID
    if wbc_id is None:
        wbc_id = YOLO_WBC_CLASS_ID
    if platelet_id is None:
        platelet_id = YOLO_PLATELET_CLASS_ID

    return rbc_id, wbc_id, platelet_id


def classify_blast_from_attributes(attr_pred: Dict[str, str]) -> Tuple[str, List[str]]:
    abnormal = 0
    reasons: List[str] = []

    blast_nucleus = {"unsegmented-round", "unsegmented-band", "unsegmented-indented"}
    if attr_pred.get("nucleus_shape") in blast_nucleus:
        abnormal += 1
        reasons.append(f"nucleus_shape={attr_pred.get('nucleus_shape')}")

    if attr_pred.get("nuclear_cytoplasmic_ratio") == "high":
        abnormal += 1
        reasons.append("nuclear_cytoplasmic_ratio=high")

    if attr_pred.get("chromatin_density") == "loosely":
        abnormal += 1
        reasons.append("chromatin_density=loosely")

    if attr_pred.get("granularity") == "no" or attr_pred.get("granule_type") == "nil":
        abnormal += 1
        reasons.append(f"granules={attr_pred.get('granularity')}|{attr_pred.get('granule_type')}")

    if attr_pred.get("cell_size") == "big":
        abnormal += 1
        reasons.append("cell_size=big")

    return ("blast-like", reasons) if abnormal >= 2 else ("normal", reasons)


def format_blast_reason(reason: str) -> str:
    reason_map = {
        "predicted_by_blast_head": "Model identified blast-like pattern",
        "nuclear_cytoplasmic_ratio=high": "High nucleus-to-cytoplasm ratio",
        "chromatin_density=loosely": "Loose chromatin density",
        "cell_size=big": "Large cell size",
    }
    if reason in reason_map:
        return reason_map[reason]
    if reason.startswith("nucleus_shape="):
        return f"Nucleus shape: {reason.split('=', 1)[1]}"
    if reason.startswith("granules="):
        return "Reduced or absent granules"
    return reason.replace("_", " ").replace("=", ": ")


def collect_blast_reason_summary(wbc_predictions: List[Dict]) -> List[str]:
    reason_counts = Counter()
    for pred in wbc_predictions:
        if pred["blast_status"] != "blast-like":
            continue
        for reason in pred["blast_reasons"]:
            reason_counts[format_blast_reason(reason)] += 1
    return [f"{reason} ({count})" for reason, count in reason_counts.most_common()]


def extract_report(item: Dict) -> Dict:
    """Safely return the nested report dictionary for a result item."""
    return item.get("report", {})


def extract_counts(report: Dict) -> Dict:
    """Safely return cell-count data from a report."""
    return report.get("counts", {})


def get_wbc_predictions(item: Dict) -> List[Dict]:
    """Return the WBC predictions list for a processed item."""
    return item.get("wbc_predictions", [])


def get_blast_diagnosis(blast_percentage: float) -> str:
    """Map blast percentage to the app's final diagnosis label."""
    return "Blast-like" if blast_percentage >= BLAST_THRESHOLD else "Normal"


def compute_wbc_distribution(results_dict: Dict) -> Counter:
    """Aggregate WBC class counts across all processed images."""
    class_distribution = Counter()
    for item in results_dict.values():
        for prediction in get_wbc_predictions(item):
            wbc_class = prediction.get("wbc_class")
            if wbc_class:
                class_distribution[wbc_class] += 1
    return class_distribution


def compute_morphology_summary(results_dict: Dict) -> Counter:
    """Aggregate morphology findings from per-image blast summaries."""
    morphology_summary = Counter()
    for item in results_dict.values():
        report = extract_report(item)
        for finding in report.get("blast_reasons_summary", []):
            morphology_summary[finding] += 1
    return morphology_summary


def compute_global_metrics(results_dict: Dict) -> Dict:
    """Compute the shared aggregate metrics used across tabs."""
    total_rbc = 0
    total_platelets = 0
    total_wbc = 0
    total_blast = 0
    total_normal = 0
    blast_percentages = []
    diagnosis_counts = Counter()

    for item in results_dict.values():
        report = extract_report(item)
        counts = extract_counts(report)
        predictions = get_wbc_predictions(item)

        total_rbc += counts.get("RBC", 0)
        total_platelets += counts.get("Platelets", 0)
        total_wbc += len(predictions)
        total_blast += sum(1 for prediction in predictions if prediction.get("blast_status") == "blast-like")
        total_normal += sum(1 for prediction in predictions if prediction.get("blast_status") == "normal")

        blast_percentage = report.get("blast_percentage")
        if blast_percentage is not None:
            blast_percentages.append(float(blast_percentage))

        diagnosis = report.get("diagnosis")
        if diagnosis:
            diagnosis_counts[diagnosis] += 1

    dataset_blast_percentage = (total_blast / total_wbc * 100.0) if total_wbc else 0.0

    return {
        "total_images": len(results_dict),
        "total_rbc": total_rbc,
        "total_wbc": total_wbc,
        "total_platelets": total_platelets,
        "total_blast": total_blast,
        "total_normal": total_normal,
        "blast_percentage": dataset_blast_percentage,
        "average_blast_percentage": float(pd.Series(blast_percentages).mean()) if blast_percentages else 0.0,
        "diagnosis_counts": diagnosis_counts,
        "class_distribution": compute_wbc_distribution(results_dict),
        "morphology_summary": compute_morphology_summary(results_dict),
    }


def build_batch_overview_dataframe(results_dict: Dict) -> pd.DataFrame:
    """Create the overview table shown in the batch tab."""
    overview_rows = []
    for filename, item in results_dict.items():
        report = extract_report(item)
        counts = extract_counts(report)
        overview_rows.append(
            {
                "Thumbnail": create_image_thumbnail(item.get("original_image")),
                "Filename": filename,
                "RBC": counts.get("RBC", 0),
                "WBC": counts.get("WBC", 0),
                "Platelets": counts.get("Platelets", 0),
                "Blast %": f"{report.get('blast_percentage', 0.0):.1f}%",
                "Diagnosis": report.get("diagnosis", "Unknown"),
            }
        )
    return pd.DataFrame(overview_rows)


def build_batch_statistics_dataframe(results_dict: Dict) -> pd.DataFrame:
    """Create the audit table used by the final report export."""
    statistics_rows = []
    for filename, item in results_dict.items():
        report = extract_report(item)
        counts = extract_counts(report)
        statistics_rows.append(
            {
                "Filename": filename,
                "RBC": counts.get("RBC", 0),
                "WBC": counts.get("WBC", 0),
                "Platelets": counts.get("Platelets", 0),
                "Total_WBCs": len(get_wbc_predictions(item)),
                "Blast_Count": report.get("blast_count", 0),
                "Blast_%": report.get("blast_percentage", 0.0),
                "Diagnosis": report.get("diagnosis", "Unknown"),
                "Timestamp": report.get("timestamp", ""),
            }
        )
    return pd.DataFrame(statistics_rows)


class MultiTaskViT(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        n_classes: int,
        attr_cardinalities: Dict[str, int],
        has_blast_head: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.attr_heads = nn.ModuleDict({a: nn.Linear(embed_dim, c) for a, c in attr_cardinalities.items()})
        self.blast_head = nn.Linear(embed_dim, 1) if has_blast_head else None

    def forward(self, x: torch.Tensor):
        feats = self.backbone.forward_features(x)
        cls_feat = feats[:, 0]
        class_logits = self.classifier(cls_feat)
        attr_logits = {a: h(cls_feat) for a, h in self.attr_heads.items()}
        blast_logits = self.blast_head(cls_feat) if self.blast_head is not None else None
        return class_logits, attr_logits, blast_logits


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"], checkpoint
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return checkpoint["model_state_dict"], checkpoint
        return checkpoint, checkpoint
    raise ValueError("Unsupported checkpoint format")


def _load_csv_metadata(csv_path: str):
    if not csv_path or not Path(csv_path).exists():
        return None, None

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        return None, None

    class_names = sorted(df["label"].astype(str).unique().tolist())
    attr_value_map: Dict[str, List[str]] = {}
    for attr in DEFAULT_ATTRS:
        if attr in df.columns:
            attr_value_map[attr] = sorted(df[attr].astype(str).unique().tolist())

    return class_names, attr_value_map


@st.cache_resource
def load_yolo_model(model_path: str):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    return YOLO(model_path)


@st.cache_resource
def load_vit_model(model_path: str, csv_path: str, device_str: str):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"ViT model not found: {model_path}")

    device = torch.device(device_str)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict, raw_checkpoint = _extract_state_dict(checkpoint)

    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None:
        raise KeyError("Checkpoint missing classifier.weight")
    n_classes = int(classifier_weight.shape[0])

    attr_cardinalities: Dict[str, int] = {}
    for key, value in state_dict.items():
        if key.startswith("attr_heads.") and key.endswith(".weight"):
            attr_name = key.split(".")[1]
            attr_cardinalities[attr_name] = int(value.shape[0])

    if not attr_cardinalities:
        raise KeyError("Checkpoint missing attr_heads.*.weight")

    has_blast_head = "blast_head.weight" in state_dict

    backbone = timm.create_model("vit_base_patch16_224", pretrained=False)
    embed_dim = int(backbone.head.in_features)
    backbone.reset_classifier(0)

    model = MultiTaskViT(
        backbone=backbone,
        embed_dim=embed_dim,
        n_classes=n_classes,
        attr_cardinalities=attr_cardinalities,
        has_blast_head=has_blast_head,
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    csv_class_names, csv_attr_values = _load_csv_metadata(csv_path)

    ckpt_meta = raw_checkpoint if isinstance(raw_checkpoint, dict) else {}
    meta_class_names = ckpt_meta.get("class_names") if isinstance(ckpt_meta.get("class_names"), list) else None
    meta_attr_values = ckpt_meta.get("attribute_values") if isinstance(ckpt_meta.get("attribute_values"), dict) else None

    class_names = meta_class_names or csv_class_names
    if not class_names or len(class_names) != n_classes:
        class_names = [f"class_{i}" for i in range(n_classes)]

    attr_names = list(attr_cardinalities.keys())
    attr_values: Dict[str, List[str]] = {}
    for attr in attr_names:
        card = attr_cardinalities[attr]
        if meta_attr_values and attr in meta_attr_values and len(meta_attr_values[attr]) == card:
            attr_values[attr] = list(meta_attr_values[attr])
        elif csv_attr_values and attr in csv_attr_values and len(csv_attr_values[attr]) == card:
            attr_values[attr] = list(csv_attr_values[attr])
        else:
            attr_values[attr] = [f"{attr}_{i}" for i in range(card)]

    return {
        "model": model,
        "device": device,
        "class_names": class_names,
        "attr_names": attr_names,
        "attr_values": attr_values,
        "has_blast_head": has_blast_head,
        "attr_cardinalities": attr_cardinalities,
    }


def run_yolo_detection(
    image: Image.Image,
    yolo_model,
    rbc_class_id: int,
    wbc_class_id: int,
    platelet_class_id: int,
    conf_threshold: float,
):
    image_np = np.array(image)
    results = yolo_model.predict(source=image_np, save=False, verbose=False)
    if not results:
        raise RuntimeError("YOLO did not return results")

    result = results[0]
    names = yolo_model.names if isinstance(yolo_model.names, dict) else {}

    all_boxes = []
    counts = Counter()

    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        clss = result.boxes.cls.detach().cpu().numpy().astype(int)

        for idx in range(len(xyxy)):
            class_id = int(clss[idx])
            class_name = str(names.get(class_id, f"class_{class_id}"))
            conf = float(confs[idx])
            x1, y1, x2, y2 = [float(v) for v in xyxy[idx]]
            all_boxes.append(
                {
                    "box_id": idx + 1,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            counts[class_id] += 1

    wbc_boxes = [b for b in all_boxes if b["class_id"] == wbc_class_id and b["confidence"] >= conf_threshold]

    annotated = result.plot()
    if isinstance(annotated, np.ndarray) and annotated.ndim == 3 and annotated.shape[2] == 3:
        annotated = annotated[:, :, ::-1]
    annotated_image = Image.fromarray(annotated)

    def class_avg_confidence(class_id: int) -> float:
        confs = [b["confidence"] for b in all_boxes if b["class_id"] == class_id]
        return float(sum(confs) / len(confs)) if confs else 0.0

    return {
        "result": result,
        "annotated_image": annotated_image,
        "all_boxes": all_boxes,
        "wbc_boxes": wbc_boxes,
        "rbc_count": int(counts.get(rbc_class_id, 0)),
        "wbc_count": int(counts.get(wbc_class_id, 0)),
        "platelet_count": int(counts.get(platelet_class_id, 0)),
        "confidence": {
            "RBC": class_avg_confidence(rbc_class_id),
            "WBC": class_avg_confidence(wbc_class_id),
            "Platelets": class_avg_confidence(platelet_class_id),
        },
    }


def crop_wbc(image: Image.Image, boxes: List[Dict]):
    width, height = image.size
    crops = []

    for box in boxes:
        x1 = max(0, min(width - 1, int(round(box["x1"]))))
        y1 = max(0, min(height - 1, int(round(box["y1"]))))
        x2 = max(0, min(width, int(round(box["x2"]))))
        y2 = max(0, min(height, int(round(box["y2"]))))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        if crop.size[0] == 0 or crop.size[1] == 0:
            continue

        crops.append(
            {
                "wbc_id": len(crops) + 1,
                "bbox": [x1, y1, x2, y2],
                "det_confidence": float(box["confidence"]),
                "crop": crop,
            }
        )

    return crops


def preprocess_for_vit(crops: List[Dict]):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    tensors = []
    valid_crops = []
    for item in crops:
        try:
            tensors.append(transform(item["crop"]))
            valid_crops.append(item)
        except Exception:
            continue

    if not tensors:
        return None, []

    return torch.stack(tensors), valid_crops


def run_vit_inference(crops: List[Dict], vit_bundle: Dict):
    batch, valid_crops = preprocess_for_vit(crops)
    if batch is None or not valid_crops:
        return []

    model = vit_bundle["model"]
    device = vit_bundle["device"]
    class_names = vit_bundle["class_names"]
    attr_names = vit_bundle["attr_names"]
    attr_values = vit_bundle["attr_values"]
    has_blast_head = vit_bundle["has_blast_head"]

    batch = batch.to(device)

    predictions = []
    with torch.no_grad():
        class_logits, attr_logits, blast_logits = model(batch)
        class_probs = torch.softmax(class_logits, dim=1).cpu()

        for i, item in enumerate(valid_crops):
            class_prob = class_probs[i]
            class_idx = int(torch.argmax(class_prob).item())
            class_conf = float(class_prob[class_idx].item())
            class_name = class_names[class_idx]

            attr_pred = {}
            attr_conf = {}
            for attr in attr_names:
                probs = torch.softmax(attr_logits[attr][i], dim=0).cpu()
                attr_idx = int(torch.argmax(probs).item())
                attr_pred[attr] = attr_values[attr][attr_idx]
                attr_conf[attr] = float(probs[attr_idx].item())

            if has_blast_head and blast_logits is not None:
                blast_prob = float(torch.sigmoid(blast_logits[i]).item())
                blast_status = "blast-like" if blast_prob >= 0.5 else "normal"
                blast_reasons = ["predicted_by_blast_head"]
            else:
                blast_status, blast_reasons = classify_blast_from_attributes(attr_pred)
                blast_prob = 1.0 if blast_status == "blast-like" else 0.0

            predictions.append(
                {
                    "wbc_id": item["wbc_id"],
                    "bbox": item["bbox"],
                    "det_confidence": item["det_confidence"],
                    "wbc_class": class_name,
                    "wbc_class_confidence": class_conf,
                    "attributes": attr_pred,
                    "attribute_confidence": attr_conf,
                    "blast_status": blast_status,
                    "blast_probability": blast_prob,
                    "blast_reasons": blast_reasons,
                }
            )

    return predictions


def generate_report(image_name: str, yolo_summary: Dict, wbc_predictions: List[Dict]):
    """Build the report payload stored for each processed image."""
    total_wbc = len(wbc_predictions)
    blast_count = sum(1 for item in wbc_predictions if item["blast_status"] == "blast-like")
    blast_percentage = (blast_count / total_wbc * 100.0) if total_wbc > 0 else 0.0
    diagnosis = get_blast_diagnosis(blast_percentage)
    blast_reasons_summary = collect_blast_reason_summary(wbc_predictions)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_name": image_name,
        "counts": {
            "RBC": yolo_summary["rbc_count"],
            # Keep report-level WBC counts aligned with the cells that were
            # successfully cropped and classified, so all tabs show the same total.
            "WBC": total_wbc,
            "Platelets": yolo_summary["platelet_count"],
        },
        "detection_confidence": yolo_summary["confidence"],
        "detected_wbc_count": yolo_summary["wbc_count"],
        "blast_count": blast_count,
        "blast_percentage": round(blast_percentage, 2),
        "diagnosis": diagnosis,
        "blast_reasons_summary": blast_reasons_summary,
        "wbc_predictions": wbc_predictions,
    }


def _safe_open_image(file) -> Tuple[Image.Image, str]:
    try:
        img = Image.open(file).convert("RGB")
        name = getattr(file, "name", f"image_{int(time.time())}.png")
        return img, name
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {e}")


def process_batch(uploaded_files, yolo_model, vit_bundle, rbc_class_id, wbc_class_id, platelet_class_id, conf_threshold, max_images=50):
    results = {}
    total = min(len(uploaded_files), max_images)
    progress = st.progress(0)
    status_text = st.empty()

    for idx, uploaded in enumerate(uploaded_files[:max_images]):
        status_text.info(f"Processing image {idx+1} of {total}: {getattr(uploaded, 'name', 'unknown')}")
        try:
            image, name = _safe_open_image(uploaded)
        except Exception as exc:
            st.warning(f"Skipping file {getattr(uploaded,'name','unknown')}: {exc}")
            progress.progress(int((idx + 1) / total * 100))
            continue

        try:
            yolo_summary = run_yolo_detection(
                image=image,
                yolo_model=yolo_model,
                rbc_class_id=rbc_class_id,
                wbc_class_id=wbc_class_id,
                platelet_class_id=platelet_class_id,
                conf_threshold=conf_threshold,
            )

            crops = crop_wbc(image, yolo_summary["wbc_boxes"]) if yolo_summary else []

            if crops:
                wbc_predictions = run_vit_inference(crops, vit_bundle)
            else:
                wbc_predictions = []

            report = generate_report(name, yolo_summary, wbc_predictions)

            # attach image data URIs for display & thumbnails
            annotated_uri = pil_to_data_uri(yolo_summary.get("annotated_image")) if yolo_summary.get("annotated_image") is not None else ""
            original_uri = pil_to_data_uri(image)

            # store crop data URIs for download/preview
            for c in crops:
                c["data_uri"] = pil_to_data_uri(c["crop"])

            results[name] = {
                "image_name": name,
                "original_image": image,
                "original_uri": original_uri,
                "annotated_image": yolo_summary.get("annotated_image"),
                "annotated_uri": annotated_uri,
                "yolo_summary": yolo_summary,
                "crops": crops,
                "wbc_predictions": wbc_predictions,
                "report": report,
            }
        except Exception as exc:
            st.warning(f"Error processing {getattr(uploaded,'name','unknown')}: {exc}")

        progress.progress(int((idx + 1) / total * 100))

    status_text.empty()
    return results


def render_batch_summary_table(results_dict):
    if not results_dict:
        st.info("No processed images to summarize.")
        return

    rows = []
    for name, item in results_dict.items():
        rpt = item["report"]
        rows.append(
            {
                "filename": name,
                "total_wbc": rpt["counts"]["WBC"],
                "blast_percentage": rpt["blast_percentage"],
                "diagnosis": rpt["diagnosis"],
            }
        )

    df = pd.DataFrame(rows)
    st.markdown("### Batch Summary")
    st.dataframe(df)


def create_image_thumbnail(pil_image: Image.Image, size=(120, 90)) -> str:
    try:
        img = pil_image.copy()
        img.thumbnail(size)
        return pil_to_data_uri(img)
    except Exception:
        return ""


def render_batch_overview_tab(results_dict):
    """Render the overview tab for the current batch."""
    st.markdown("## 📋 Batch Statistics")
    if not results_dict:
        st.info("No processed results. Use Upload & Process to add images.")
        return

    global_metrics = compute_global_metrics(results_dict)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card_modern("Total Samples", "🧪", f"{global_metrics['total_images']}", 100, "accent")
    with c2:
        render_kpi_card_modern("Total WBCs", "🧬", f"{global_metrics['total_wbc']}", 100, "accent")
    with c3:
        render_kpi_card_modern("Avg Blast %", "📊", f"{global_metrics['average_blast_percentage']:.1f}%", 100, "accent")
    with c4:
        diag_text = "Mixed" if len(global_metrics['diagnosis_counts']) > 1 else list(global_metrics['diagnosis_counts'].keys())[0] if global_metrics['diagnosis_counts'] else "N/A"
        render_kpi_card_modern("Primary Status", "🩺", diag_text, 100, "accent")


    st.markdown("---")
    overview_df = build_batch_overview_dataframe(results_dict)
    st.dataframe(overview_df, use_container_width=True)

    st.markdown("---")
    # Export buttons
    csv_bytes, zip_bytes = export_batch_results(results_dict)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download Batch CSV", data=csv_bytes, file_name="batch_summary.csv", mime="text/csv")
    with c2:
        st.download_button("Download Batch ZIP", data=zip_bytes, file_name="batch_images.zip", mime="application/zip")


def render_single_image_detection_section(image_name, item, expanded=False):
    """Render one image block in the full detection results tab."""
    report = extract_report(item)
    counts = extract_counts(report)
    header = f"📊 {image_name}"
    with st.expander(header, expanded=expanded):
        st.markdown(f"**Timestamp:** {report.get('timestamp', '')}")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(item.get("original_image"), use_container_width=True, caption="Original")
        with c2:
            st.image(item.get("annotated_image"), use_container_width=True, caption="Annotated")
        st.markdown("---")
        st.markdown("**Metrics**")
        m1, m2, m3 = st.columns(3)
        m1.metric("RBC", counts.get("RBC", 0))
        m2.metric("WBC", counts.get("WBC", 0))
        m3.metric("Platelets", counts.get("Platelets", 0))

def render_empty_state():
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 350px; opacity: 0.6;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📭</div>
        <div style="font-size: 1.1rem; color: #94A3B8; font-weight: 500;">Awaiting Data</div>
        <div style="font-size: 0.9rem; color: #64748B;">Please import a batch of blood smear images to unlock this tab.</div>
    </div>
    """, unsafe_allow_html=True)


def render_full_detection_results_tab(results_dict):
    st.markdown("## 🔍 Full Detection Results")
    if not results_dict:
        render_empty_state()
        return

    metrics = compute_global_metrics(results_dict)
    
    # KPI grid for overall counts
    cols = st.columns(3)
    with cols[0]:
        render_kpi_card_modern("TOTAL RBCS", "🔴", str(metrics.get("total_rbc", 0)), 100.0, "#2563eb")
    with cols[1]:
        render_kpi_card_modern("TOTAL WBCS", "⚪", str(metrics.get("total_wbc", 0)), 100.0, "#7c3aed")
    with cols[2]:
        render_kpi_card_modern("TOTAL PLATELETS", "🔶", str(metrics.get("total_platelets", 0)), 100.0, "#ea580c")
        
    st.markdown("---")

    expand_all = st.checkbox("Expand all image sections", value=True)

    for name, item in results_dict.items():
        render_single_image_detection_section(name, item, expanded=expand_all)


def render_image_wbc_section(image_name, item, wbc_filter="All"):
    """Render all classified WBC crops for a single processed image."""
    predictions = get_wbc_predictions(item)
    
    if wbc_filter == "Blast-like":
        predictions = [p for p in predictions if p.get("blast_status") == "blast-like"]
    elif wbc_filter == "Normal":
        predictions = [p for p in predictions if p.get("blast_status") == "normal"]
        
    if not predictions and wbc_filter != "All":
        return
        
    st.markdown("---")
    blast_like_count = sum(1 for prediction in predictions if prediction.get("blast_status") == "blast-like")
    normal_count = sum(1 for prediction in predictions if prediction.get("blast_status") != "blast-like")

    header_html = f"""
    <div style="background: #1E293B; border-left: 4px solid #3B82F6; padding: 12px 16px; border-radius: 6px; margin-bottom: 24px; margin-top: 12px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div>
            <div style="font-family: 'DM Mono', monospace; font-size: 0.85rem; color: #94A3B8; margin-bottom: 2px;">SOURCE IMAGE</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC;">{image_name}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.95rem; color: #CBD5E1; margin-bottom: 2px;"><span style="color: #EF4444; font-weight: 700;">{blast_like_count}</span> Blast-like &nbsp;|&nbsp; <span style="color: #10B981; font-weight: 700;">{normal_count}</span> Normal</div>
            <div style="font-size: 0.85rem; color: #94A3B8; font-family: 'DM Mono', monospace;">Total WBCs: {len(predictions)}</div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    cols_per_row = 4
    for i in range(0, len(predictions), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(predictions):
                prediction = predictions[i + j]
                with col:
                    st.markdown("<div style='background: #111827; border: 1px solid #1E2D45; border-radius: 12px; padding: 16px; margin-bottom: 1.5rem; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>", unsafe_allow_html=True)
                    
                    status = prediction.get("blast_status", "normal")
                    if status == "blast-like":
                        badge = "<div style='background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); color: #EF4444; padding: 2px 8px; border-radius: 999px; font-size: 0.65rem; font-weight: 700;'>⚠️ BLAST-LIKE</div>"
                    else:
                        badge = "<div style='background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #10B981; padding: 2px 8px; border-radius: 999px; font-size: 0.65rem; font-weight: 700;'>✅ NORMAL</div>"
                        
                    st.markdown(f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'><span style='font-family: \"DM Mono\", monospace; color: #94A3B8; font-weight: 600; font-size: 0.85rem;'>WBC #{prediction['wbc_id']}</span>{badge}</div>", unsafe_allow_html=True)
                    
                    crop = next((crop_item for crop_item in item.get("crops", []) if crop_item["wbc_id"] == prediction["wbc_id"]), None)
                    if crop is not None:
                        st.image(crop["crop"], use_container_width=True)
                    
                    st.markdown(f"<div style='font-weight: 700; margin-top: 12px; margin-bottom: 4px; font-size: 1.05rem; color: #F1F5F9;'>{prediction['wbc_class']}</div>", unsafe_allow_html=True)
                    st.progress(prediction.get('wbc_class_confidence', 0.0))
                    
                    st.markdown("<div style='margin-top: 12px;'>", unsafe_allow_html=True)
                    with st.expander("Attributes"):
                        for attribute_name, attribute_value in prediction["attributes"].items():
                            st.write(f"- {attribute_name}: {attribute_value}")
                    if status == "blast-like":
                        with st.expander("Blast reasons"):
                            for reason in prediction["blast_reasons"]:
                                st.write(f"- {format_blast_reason(reason)}")
                    st.markdown("</div></div>", unsafe_allow_html=True)


def render_comprehensive_wbc_tab(results_dict):
    """Render the detailed WBC review tab."""
    st.markdown("## 🧬 Morphology Classification (ViT)")
    st.markdown("<p style='color: #94A3B8; font-size: 1rem; margin-top: -10px; margin-bottom: 24px;'>Review AI-generated classifications and morphology profiles for individually detected white blood cells.</p>", unsafe_allow_html=True)
    
    if not results_dict:
        render_empty_state()
        return

    st.markdown("<div style='margin-bottom: 16px; font-weight: 600; color: #CBD5E1;'>Filter WBCs by Diagnostic Status:</div>", unsafe_allow_html=True)
    wbc_filter = st.radio("Filter WBCs by Status", ["All", "Blast-like", "Normal"], key="wbc_filter", horizontal=True, label_visibility="collapsed")

    for name, item in results_dict.items():
        render_image_wbc_section(name, item, wbc_filter=wbc_filter)


def render_diagnosis_card(image_name, report):
    """Render one compact diagnosis summary card."""
    blast_percentage = report.get("blast_percentage", 0.0)
    is_blast = report.get("diagnosis") == "Blast-like"
    border = '2px solid #dc2626' if is_blast else '2px solid #059669'
    st.markdown(f"<div style='border:{border}; padding:12px; border-radius:8px'>", unsafe_allow_html=True)
    st.markdown(f"**{image_name}**")
    st.markdown(f"<div style='font-size:1.2rem; font-weight:700'>{report.get('diagnosis', 'Unknown')}</div>", unsafe_allow_html=True)
    st.markdown(f"Blast: {blast_percentage:.2f}% ({report.get('blast_count', 0)})")
    if report.get("blast_reasons_summary"):
        st.markdown("**Key findings:**")
        for finding in report["blast_reasons_summary"]:
            st.markdown(f"- {finding}")
    st.markdown("</div>", unsafe_allow_html=True)


def apply_dark_theme(fig):
    """Apply the clinical dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#94A3B8"),
        legend=dict(font=dict(color="#CBD5E1")),
        margin=dict(t=40, b=0, l=0, r=0),
        height=420,
    )
    return fig


def render_combined_diagnostic_tab(results_dict):
    """Render the aggregate diagnostic summary tab."""
    st.markdown("## 📊 Combined Diagnostic Report")
    if not results_dict:
        render_empty_state()
        return

    import plotly.express as px

    global_metrics = compute_global_metrics(results_dict)
    class_distribution = global_metrics["class_distribution"]

    st.markdown("### 🩸 Total Cell Count Summary")
    
    cell_k1, cell_k2, cell_k3 = st.columns(3)
    with cell_k1:
        render_kpi_card_modern("TOTAL RBCS", "🔴", str(global_metrics["total_rbc"]), 100.0, "#2563eb")
    with cell_k2:
        render_kpi_card_modern("TOTAL WBCS", "⚪", str(global_metrics["total_wbc"]), 100.0, "#7c3aed")
    with cell_k3:
        render_kpi_card_modern("TOTAL PLATELETS", "🔶", str(global_metrics["total_platelets"]), 100.0, "#ea580c")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔬 Aggregate WBC Analysis")
    
    wbc_k1, wbc_k2, wbc_k3 = st.columns(3)
    with wbc_k1:
        render_kpi_card_modern("TOTAL NORMAL WBCS", "✅", str(global_metrics["total_normal"]), 100.0, "#10B981")
    with wbc_k2:
        render_kpi_card_modern("TOTAL BLAST CELLS", "⚠️", str(global_metrics["total_blast"]), 100.0, "#EF4444")
    with wbc_k3:
        render_kpi_card_modern("BLAST RATE", "📉", f"{global_metrics['blast_percentage']:.2f}%", 100.0, "#F59E0B")

    st.markdown("<br>", unsafe_allow_html=True)
    
    chart_cols = st.columns(3)
    
    with chart_cols[0]:
        cell_counts_df = pd.DataFrame([
            {"Cell Type": "RBC", "Count": global_metrics["total_rbc"]},
            {"Cell Type": "WBC", "Count": global_metrics["total_wbc"]},
            {"Cell Type": "Platelets", "Count": global_metrics["total_platelets"]},
        ])
        cell_count_chart = px.pie(cell_counts_df, names="Cell Type", values="Count", title="Total Cell Distribution", hole=0.35, color_discrete_sequence=['#2563eb', '#7c3aed', '#ea580c'])
        cell_count_chart = apply_dark_theme(cell_count_chart)
        st.plotly_chart(cell_count_chart, use_container_width=True)
        
    with chart_cols[1]:
        if global_metrics["total_wbc"]:
            blast_normal_df = pd.DataFrame([
                {"Category": "Blast-like", "Count": global_metrics["total_blast"]},
                {"Category": "Normal", "Count": global_metrics["total_normal"]},
            ])
            blast_normal_chart = px.pie(blast_normal_df, names="Category", values="Count", title="Blast vs Normal Cells", hole=0.35, color_discrete_sequence=['#EF4444', '#10B981'])
            blast_normal_chart = apply_dark_theme(blast_normal_chart)
            st.plotly_chart(blast_normal_chart, use_container_width=True)
            
    with chart_cols[2]:
        if class_distribution:
            class_distribution_df = pd.DataFrame([
                {"WBC Class": wbc_class, "Count": count}
                for wbc_class, count in class_distribution.most_common()
            ])
            class_distribution_chart = px.pie(class_distribution_df, names="WBC Class", values="Count", title="WBC Sub-type Distribution", hole=0.35)
            class_distribution_chart = apply_dark_theme(class_distribution_chart)
            st.plotly_chart(class_distribution_chart, use_container_width=True)


def render_final_diagnostic_report(results_dict):
    """Render the clinician-friendly consolidated final report tab."""
    st.markdown("## 🧾 Official Diagnostic Report")
    if not results_dict:
        render_empty_state()
        return

    import datetime
    import uuid
    import io
    import pandas as pd
    from app_v1 import compute_global_metrics, BLAST_THRESHOLD, build_batch_statistics_dataframe

    global_metrics = compute_global_metrics(results_dict)
    class_distribution = global_metrics["class_distribution"]
    morphology_summary = global_metrics["morphology_summary"]
    blast_percentage = global_metrics["blast_percentage"]

    if blast_percentage >= BLAST_THRESHOLD:
        result_summary = "BLAST-LIKE (High Suspicion)"
        interpretation_text = "Findings suggest a strong likelihood of abnormal blast proliferation. Immediate hematopathologist review and bone marrow aspiration recommended."
        final_conclusion = "POSITIVE FOR BLAST-LIKE CELLS"
        banner_color = "rgba(239, 68, 68, 0.15)"
        border_color = "#EF4444"
        text_color = "#EF4444"
    elif blast_percentage == 0.0:
        result_summary = "NORMAL / BELOW THRESHOLD"
        interpretation_text = "No definitive evidence of abnormal blast proliferation is seen in the aggregated analysis. Routine follow-up suggested."
        final_conclusion = "NEGATIVE FOR BLAST-LIKE CELLS"
        banner_color = "rgba(16, 185, 129, 0.15)"
        border_color = "#10B981"
        text_color = "#10B981"
    else:
        result_summary = "NORMAL / BELOW THRESHOLD"
        interpretation_text = "Blast-like cells are present, but the overall aggregated burden remains below the diagnostic decision threshold."
        final_conclusion = "NEGATIVE FOR BLAST-LIKE CELLS"
        banner_color = "rgba(16, 185, 129, 0.15)"
        border_color = "#10B981"
        text_color = "#10B981"

    report_id = f"LKP-{uuid.uuid4().hex[:8].upper()}"
    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # The Premium Medical Report HTML
    report_html = f"""
    <div style="background: #111827; border: 1px solid #1E2D45; border-radius: 12px; padding: 40px; font-family: 'DM Sans', sans-serif; color: #F8FAFC; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
        <!-- Header -->
        <div style="border-bottom: 2px solid #2563EB; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: flex-end;">
            <div>
                <h1 style="margin: 0; color: #F8FAFC; font-size: 24px; font-weight: 800; letter-spacing: 1px;">LEUKOANALYZE PRO</h1>
                <div style="color: #94A3B8; font-family: 'DM Mono', monospace; font-size: 14px; margin-top: 4px;">Automated Peripheral Blood Smear Analysis</div>
            </div>
            <div style="text-align: right; color: #94A3B8; font-family: 'DM Mono', monospace; font-size: 13px;">
                <div><strong>Date:</strong> {report_date}</div>
                <div><strong>Report ID:</strong> {report_id}</div>
                <div><strong>Facility:</strong> General Hospital Hemepath</div>
            </div>
        </div>

        <!-- Primary Diagnosis Banner -->
        <div style="background: {banner_color}; border-left: 6px solid {border_color}; padding: 20px; border-radius: 4px; margin-bottom: 30px;">
            <div style="font-size: 12px; font-weight: 700; color: {text_color}; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px;">Final Automated Assessment</div>
            <div style="font-size: 22px; font-weight: 800; color: #F8FAFC; margin-bottom: 8px;">{final_conclusion}</div>
            <div style="font-size: 15px; color: #CBD5E1; line-height: 1.5;">{interpretation_text}</div>
            <div style="margin-top: 12px; font-family: 'DM Mono', monospace; font-size: 14px; color: {text_color};">Calculated Blast Burden: {blast_percentage:.2f}% (Threshold: {BLAST_THRESHOLD:.0f}%)</div>
        </div>

        <!-- Grid Layout for Metrics -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
            
            <!-- Left Column: Core Metrics -->
            <div>
                <h3 style="color: #60A5FA; font-size: 16px; border-bottom: 1px solid #1E2D45; padding-bottom: 8px; margin-bottom: 16px;">I. CORE METRICS</h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                    <tr style="border-bottom: 1px solid #1E2D45;"><td style="padding: 8px 0; color: #94A3B8;">Total Red Blood Cells (RBC)</td><td style="text-align: right; font-weight: 700;">{global_metrics['total_rbc']:,}</td></tr>
                    <tr style="border-bottom: 1px solid #1E2D45;"><td style="padding: 8px 0; color: #94A3B8;">Total White Blood Cells (WBC)</td><td style="text-align: right; font-weight: 700;">{global_metrics['total_wbc']:,}</td></tr>
                    <tr style="border-bottom: 1px solid #1E2D45;"><td style="padding: 8px 0; color: #94A3B8;">Total Platelets</td><td style="text-align: right; font-weight: 700;">{global_metrics['total_platelets']:,}</td></tr>
                    <tr style="border-bottom: 1px solid #1E2D45;"><td style="padding: 8px 0; color: #94A3B8;">Detected Blast-like Cells</td><td style="text-align: right; font-weight: 700; color: #EF4444;">{global_metrics['total_blast']:,}</td></tr>
                    <tr style="border-bottom: 1px solid #1E2D45;"><td style="padding: 8px 0; color: #94A3B8;">Normal WBCs</td><td style="text-align: right; font-weight: 700; color: #10B981;">{global_metrics['total_normal']:,}</td></tr>
                </table>
            </div>

            <!-- Right Column: Differential -->
            <div>
                <h3 style="color: #60A5FA; font-size: 16px; border-bottom: 1px solid #1E2D45; padding-bottom: 8px; margin-bottom: 16px;">II. WBC DIFFERENTIAL</h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
    """

    for cell_type in ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]:
        count = class_distribution.get(cell_type, 0)
        percentage = (count / global_metrics["total_wbc"] * 100) if global_metrics["total_wbc"] else 0.0
        report_html += f'<tr style="border-bottom: 1px solid #1E2D45;"><td style="padding: 8px 0; color: #94A3B8;">{cell_type}s</td><td style="text-align: right; font-weight: 700;">{percentage:.1f}%</td><td style="text-align: right; color: #64748B;">({count})</td></tr>'
        
    for cell_type, count in class_distribution.most_common():
        if cell_type not in ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]:
            percentage = (count / global_metrics["total_wbc"] * 100) if global_metrics["total_wbc"] else 0.0
            report_html += f'<tr style="border-bottom: 1px solid #1E2D45;"><td style="padding: 8px 0; color: #94A3B8;">{cell_type}</td><td style="text-align: right; font-weight: 700;">{percentage:.1f}%</td><td style="text-align: right; color: #64748B;">({count})</td></tr>'

    report_html += """
                </table>
            </div>
        </div>

        <!-- Morphological Findings -->
        <h3 style="color: #60A5FA; font-size: 16px; border-bottom: 1px solid #1E2D45; padding-bottom: 8px; margin-bottom: 16px;">III. MORPHOLOGICAL FINDINGS</h3>
        <ul style="font-size: 14px; color: #CBD5E1; line-height: 1.6; margin-bottom: 30px; padding-left: 20px;">
    """
    
    if morphology_summary:
        for finding, count in morphology_summary.most_common(5):
            report_html += f"<li>{finding} (Identified in {count} cells)</li>"
    else:
        report_html += "<li>No significant abnormal morphology identified in the aggregated AI findings.</li>"

    report_html += """
        </ul>

        <!-- Disclaimer -->
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px dashed #334155; font-size: 12px; color: #64748B; text-align: justify; line-height: 1.5;">
            <strong>DISCLAIMER:</strong> This document is generated by LeukoAnalyze Pro, an AI-assisted diagnostic screening tool. The results provided are for investigational and screening purposes only and do not constitute a definitive medical diagnosis. All findings, particularly the presence of blast-like cells, must be manually verified and validated by a board-certified hematopathologist or clinical specialist prior to clinical decision-making.
        </div>
        
        <div style="margin-top: 20px; display: flex; justify-content: space-between; font-size: 12px; color: #64748B;">
            <div>Signed: ___________________________</div>
            <div>Date: ___________________________</div>
        </div>
    </div>
    """

    import re
    cleaned_html = re.sub(r'^\s+', '', report_html, flags=re.MULTILINE)
    st.markdown(cleaned_html, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns([1, 1, 1])
    
    # Download buttons
    detailed_stats_df = build_batch_statistics_dataframe(results_dict)
    csv_buf = io.StringIO()
    detailed_stats_df.to_csv(csv_buf, index=False)
    
    with cols[0]:
        st.download_button(
            "📥 Download Batch CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"leukoanalyze_batch_{report_id}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with cols[1]:
        st.download_button(
            "📄 Export Report HTML",
            data=report_html.encode("utf-8"),
            file_name=f"diagnostic_report_{report_id}.html",
            mime="text/html",
            use_container_width=True
        )


def render_detection_results_tab(results_dict):
    """Thin wrapper for the full detection results tab."""
    render_full_detection_results_tab(results_dict)


def render_wbc_analysis_tab(results_dict):
    """Thin wrapper for the comprehensive WBC analysis tab."""
    render_comprehensive_wbc_tab(results_dict)


def render_enhanced_analytics_tab(results_dict):
    st.markdown("## 📈 Enhanced Aggregate Analytics")
    if not results_dict:
        st.info("No processed results.")
        return

    # Per-image blast percentage barh
    filenames = list(results_dict.keys())
    blast_percentages = [results_dict[f]['report']['blast_percentage'] for f in filenames]
    colors = ['#dc2626' if bp > 20 else '#059669' for bp in blast_percentages]
    fig, ax = plt.subplots(figsize=(10, max(3, len(filenames) * 0.4)))
    ax.barh(filenames, blast_percentages, color=colors)
    ax.axvline(20, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Blast Percentage (%)')
    ax.set_title('Blast Burden Comparison Across Samples')
    st.pyplot(fig)

    # Cell count comparison
    df_counts = pd.DataFrame([
        {
            'filename': name,
            'RBC': item['report']['counts']['RBC'],
            'WBC': item['report']['counts']['WBC'],
            'Platelets': item['report']['counts']['Platelets'],
        }
        for name, item in results_dict.items()
    ])
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    df_counts.set_index('filename').plot(kind='bar', ax=ax2)
    ax2.set_title('Cell Counts per Image')
    st.pyplot(fig2)

    # WBC class distribution stacked bar
    class_map = {}
    for name, item in results_dict.items():
        classes = [p['wbc_class'] for p in item.get('wbc_predictions', [])]
        cnts = Counter(classes)
        class_map[name] = cnts
    classes_all = sorted({c for cnt in class_map.values() for c in cnt.keys()})
    stacked_df = pd.DataFrame([{ 'filename': name, **{c: class_map[name].get(c,0) for c in classes_all}} for name in class_map.keys()]).set_index('filename')
    if not stacked_df.empty:
        fig3, ax3 = plt.subplots(figsize=(10,4))
        stacked_df.plot(kind='bar', stacked=True, ax=ax3)
        ax3.set_title('WBC Class Distribution per Image')
        st.pyplot(fig3)

    # Heatmap
    heatmap_data = []
    for name, item in results_dict.items():
        rpt = item['report']
        heatmap_data.append({
            'Image': name,
            'RBC': rpt['counts']['RBC'],
            'WBC': rpt['counts']['WBC'],
            'Platelets': rpt['counts']['Platelets'],
            'Blast %': rpt['blast_percentage']
        })
    heat_df = pd.DataFrame(heatmap_data).set_index('Image')
    fig4, ax4 = plt.subplots(figsize=(10, max(3, len(heat_df)*0.4)))
    sns.heatmap(heat_df, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax4)
    st.pyplot(fig4)

    # Detailed batch stats table
    stats_rows = []
    for name, item in results_dict.items():
        rpt = item['report']
        stats_rows.append({
            'Filename': name,
            'RBC': rpt['counts']['RBC'],
            'WBC': rpt['counts']['WBC'],
            'Platelets': rpt['counts']['Platelets'],
            'Total_WBCs': len(item.get('wbc_predictions', [])),
            'Blast_Count': rpt['blast_count'],
            'Blast_%': rpt['blast_percentage'],
            'Diagnosis': rpt['diagnosis'],
            'Timestamp': rpt.get('timestamp','')
        })
    stats_df = pd.DataFrame(stats_rows)
    st.markdown('### Detailed Batch Statistics')
    st.dataframe(stats_df)
    csv_buf = io.StringIO()
    stats_df.to_csv(csv_buf, index=False)
    st.download_button('Export Detailed CSV', data=csv_buf.getvalue().encode('utf-8'), file_name='detailed_batch_stats.csv', mime='text/csv')


def render_executive_summary(results_dict):
    total = len(results_dict)
    blast_like = sum(1 for it in results_dict.values() if it['report']['diagnosis'] == 'Blast-like')
    pct = (blast_like / total * 100.0) if total > 0 else 0.0
    # risk color
    if blast_like == 0:
        risk_color = '#059669'
        risk_text = 'LOW RISK'
    elif blast_like <= 2:
        risk_color = '#f59e0b'
        risk_text = 'MODERATE CONCERN'
    else:
        risk_color = '#dc2626'
        risk_text = 'HIGH PRIORITY'

    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {risk_color}, #1f2937); color: white; padding: 1.2rem; border-radius:12px;">
      <div style="font-size:1.1rem; font-weight:800;">🔬 BATCH ANALYSIS SUMMARY - {total} Samples Processed</div>
      <div style="margin-top:0.6rem; font-size:1.05rem;">⚠️ {blast_like} of {total} samples ({pct:.1f}%) show blast-like patterns</div>
      <div style="margin-top:0.5rem; opacity:0.95">Recommendation: Further clinical correlation and confirmatory tests</div>
      <div style="margin-top:0.6rem; font-size:0.85rem; opacity:0.9">Timestamp: {datetime.now().isoformat(timespec='seconds')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Critical KPI row
    normal_count = total - blast_like
    avg_blast = float(pd.Series([it['report']['blast_percentage'] for it in results_dict.values()]).mean()) if total>0 else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Samples', str(total), delta=f'{total} processed', delta_color='off')
    c2.metric('Normal Samples', f"{normal_count} ({normal_count/total*100 if total>0 else 0:.1f}%)", delta='Within threshold', delta_color='normal')
    c3.metric('Blast-like Samples', f"{blast_like} ({pct:.1f}%)", delta='Requires review', delta_color='inverse')
    c4.metric('Average Blast Burden', f"{avg_blast:.1f}%", delta=f"{avg_blast-20:.1f}% vs threshold")


def render_blast_comparison_chart(results_dict):
    filenames = list(results_dict.keys())
    blast_pct = [results_dict[f]['report']['blast_percentage'] for f in filenames]
    order = sorted(range(len(blast_pct)), key=lambda i: blast_pct[i], reverse=True)
    ordered_names = [filenames[i] for i in order]
    ordered_pct = [blast_pct[i] for i in order]
    colors = ['#dc2626' if p>20 else '#059669' for p in ordered_pct]

    fig, ax = plt.subplots(figsize=(12, max(4, len(filenames)*0.4)))
    bars = ax.barh(ordered_names, ordered_pct, color=colors, height=0.6)
    for i, (bar, pct) in enumerate(zip(bars, ordered_pct)):
        ax.text(pct + 1.5, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', va='center', fontweight='bold')
    ax.axvline(20, color='black', linestyle='--', linewidth=2.5)
    ax.set_xlim(0, max(110, max(ordered_pct)+10))
    ax.set_xlabel('Blast Percentage (%)')
    ax.set_title('Comparative Blast Burden Analysis Across All Samples')
    ax.grid(axis='x', alpha=0.2, linestyle=':')
    st.pyplot(fig)


def render_cell_count_comparison(results_dict):
    filenames = list(results_dict.keys())
    rbc = [results_dict[f]['report']['counts']['RBC'] for f in filenames]
    wbc = [results_dict[f]['report']['counts']['WBC'] for f in filenames]
    plate = [results_dict[f]['report']['counts']['Platelets'] for f in filenames]

    x = np.arange(len(filenames))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, rbc, width, label='RBC', color='#ef4444')
    ax.bar(x, wbc, width, label='WBC', color='#8b5cf6')
    ax.bar(x + width, plate, width, label='Platelets', color='#f97316')
    ax.set_xticks(x)
    ax.set_xticklabels([n.split('.')[0][:15] for n in filenames], rotation=45, ha='right')
    ax.set_ylabel('Cell Count')
    ax.set_title('Cell Count Distribution Across Samples')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    st.pyplot(fig)


def render_diagnosis_distribution(results_dict):
    counts = pd.Series([it['report']['diagnosis'] for it in results_dict.values()]).value_counts()
    labels = [f"{k} ({v})" for k, v in counts.items()]
    sizes = counts.values.tolist()
    colors = ['#059669' if 'Normal' in k else '#dc2626' for k in counts.index]
    fig, ax = plt.subplots(figsize=(6,6))
    explode = [0.1 if 'Blast-like' in lbl else 0 for lbl in counts.index]
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize':11,'fontweight':'bold'})
    ax.set_title('Overall Diagnosis Distribution')
    st.pyplot(fig)


def render_detailed_breakdown(results_dict):
    st.markdown('### 📊 Detailed Breakdown')
    normal = [n for n, it in results_dict.items() if it['report']['diagnosis']=='Normal']
    blast = [n for n, it in results_dict.items() if it['report']['diagnosis']=='Blast-like']
    st.markdown(f"**Normal Samples ({len(normal)}):**")
    for n in normal:
        st.markdown(f"- {n}")
    st.markdown(f"**Blast-like Samples ({len(blast)}):**")
    for n in blast:
        pct = results_dict[n]['report']['blast_percentage']
        st.markdown(f"- ⚠️ {n} ({pct:.1f}% blast)")


def render_metrics_heatmap(results_dict):
    rows = []
    for name, it in results_dict.items():
        rpt = it['report']
        rows.append({'Sample': name, 'RBC': rpt['counts']['RBC'], 'WBC': rpt['counts']['WBC'], 'Platelets': rpt['counts']['Platelets'], 'Total WBCs': len(it.get('wbc_predictions', [])), 'Blast Count': rpt['blast_count'], 'Blast %': rpt['blast_percentage']})
    df = pd.DataFrame(rows).set_index('Sample')
    fig, ax = plt.subplots(figsize=(12, max(4, len(df)*0.4)))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=0.5, cbar_kws={'label':'Value'}, ax=ax, vmin=0)
    ax.set_title('Comprehensive Metrics Heatmap - All Samples')
    st.pyplot(fig)


def render_common_patterns(results_dict):
    blast_names = [n for n,it in results_dict.items() if it['report']['diagnosis']=='Blast-like']
    st.markdown('### 🔬 Cross-Sample Morphological Analysis')
    st.markdown(f"**Samples with Blast-like Patterns: {len(blast_names)}**")
    # aggregate reasons
    counts = Counter()
    for n in blast_names:
        for r in results_dict[n]['report'].get('blast_reasons_summary', []):
            counts[r] += 1
    if counts:
        st.markdown('**Common Findings Across Abnormal Samples:**')
        for r, c in counts.most_common():
            st.markdown(f"- {r}: {c}/{len(blast_names)}")
    else:
        st.markdown('No consistent morphological indicators found across abnormal samples.')


def render_sample_by_sample_view(results_dict):
    st.markdown('### 📋 Detailed Sample Analysis')
    for name, data in results_dict.items():
        report = data['report']
        diagnosis = report['diagnosis']
        blast_pct = report['blast_percentage']
        border = '#dc2626' if diagnosis=='Blast-like' else '#059669'
        icon = '⚠️' if diagnosis=='Blast-like' else '✓'
        with st.expander(f"{icon} {name} - {diagnosis} ({blast_pct:.1f}% blast)", expanded=(diagnosis=='Blast-like')):
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.markdown('**Detection Summary**')
                st.metric('RBC', report['counts']['RBC'])
                st.metric('WBC', report['counts']['WBC'])
                st.metric('Platelets', report['counts']['Platelets'])
            with col2:
                st.markdown('**Images**')
                st.image(data['original_image'], caption='Original', width=240)
                st.image(data['annotated_image'], caption='Annotated', width=240)
            with col3:
                st.markdown('**Analysis**')
                st.markdown(f"**Diagnosis:** {diagnosis}")
                st.markdown(f"**Blast Burden:** {blast_pct:.1f}%")
                st.markdown(f"**Total WBCs Analyzed:** {report['counts']['WBC']}")
                if diagnosis=='Blast-like':
                    st.markdown('**Key Findings:**')
                    for reason in report.get('blast_reasons_summary', []):
                        st.markdown(f"- {reason}")


def render_comprehensive_table(results_dict):
    st.markdown('### 📊 Complete Results Table')
    rows = []
    for name, item in results_dict.items():
        rpt = item['report']
        rows.append({'Sample':name,'Diagnosis':rpt['diagnosis'],'Blast %':rpt['blast_percentage'],'Blast Count':f"{rpt['blast_count']}/{rpt['counts']['WBC']}",'RBC':rpt['counts']['RBC'],'WBC':rpt['counts']['WBC'],'Platelets':rpt['counts']['Platelets'],'RBC Conf':f"{rpt['detection_confidence']['RBC']*100:.1f}%",'WBC Conf':f"{rpt['detection_confidence']['WBC']*100:.1f}%",'Platelet Conf':f"{rpt['detection_confidence']['Platelets']*100:.1f}%"})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=400)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Show Only Blast-like'):
            st.dataframe(df[df['Diagnosis']=='Blast-like'])
    with col2:
        if st.button('Show Only Normal'):
            st.dataframe(df[df['Diagnosis']=='Normal'])
    with col3:
        if st.button('Sort by Blast %'):
            st.dataframe(df.sort_values('Blast %', ascending=False))


def render_clinical_recommendations(results_dict):
    blast_count = sum(1 for item in results_dict.values() if item['report']['diagnosis']=='Blast-like')
    total = len(results_dict)
    abnormal_rate = (blast_count/total*100) if total>0 else 0
    if abnormal_rate >= 50:
        alert_level = '🔴 HIGH PRIORITY'
        alert_color = '#dc2626'
        recommendations = [
            'Immediate hematology consultation recommended',
            'Consider bone marrow biopsy for definitive diagnosis',
            'Complete blood count (CBC) with differential',
            'Flow cytometry for immunophenotyping',
            'Cytogenetic and molecular studies'
        ]
    elif abnormal_rate >= 25:
        alert_level = '🟡 MODERATE CONCERN'
        alert_color = '#f59e0b'
        recommendations = [
            'Repeat peripheral smear analysis',
            'Hematology review within 48-72 hours',
            'Complete blood count (CBC) correlation',
            'Consider additional morphological studies',
            'Monitor patient clinical status'
        ]
    else:
        alert_level = '🟢 LOW RISK'
        alert_color = '#059669'
        recommendations = [
            'Findings within acceptable range for screening',
            'Routine follow-up as clinically indicated',
            'Maintain standard monitoring protocols'
        ]

    st.markdown(f"""
    <div style="background: {alert_color}; color: white; padding: 1rem; border-radius: 12px;">
      <h3 style="margin:0">{alert_level}</h3>
      <p style="margin:0.2rem 0 0 0; font-size:1.05rem;">{abnormal_rate:.1f}% of samples show blast-like patterns</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('**Recommended Actions:**')
    for i, rec in enumerate(recommendations,1):
        st.markdown(f"{i}. {rec}")
    st.markdown('''---\n**Disclaimer:** This is an automated screening tool. All findings must be reviewed by a qualified hematopathologist.''')


def render_export_section(results_dict):
    st.markdown('### 💾 Export Options')
    rows = []
    for name, item in results_dict.items():
        rpt = item['report']
        rows.append({'Sample':name,'Diagnosis':rpt['diagnosis'],'Blast %':rpt['blast_percentage'],'Blast Count':rpt['blast_count']})
    df = pd.DataFrame(rows)
    c1, c2, c3 = st.columns(3)
    with c1:
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button('📄 Download Complete Analysis (CSV)', data=csv_bytes, file_name=f'batch_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', mime='text/csv')
    with c2:
        _, zip_bytes = export_batch_results(results_dict)
        st.download_button('🖼️ Download All Images (ZIP)', data=zip_bytes, file_name=f'batch_images_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip', mime='application/zip')
    with c3:
        st.download_button('📑 Download PDF Report (Not implemented)', data=b'', file_name='clinical_report.pdf', mime='application/pdf', disabled=True)


def render_integrated_results_dashboard(results_dict):
    if not results_dict:
        st.info('No results available. Upload and process images first.')
        return
    render_executive_summary(results_dict)
    st.divider()
    st.markdown('## 📊 Comparative Analysis')
    render_blast_comparison_chart(results_dict)
    render_cell_count_comparison(results_dict)
    col1, col2 = st.columns(2)
    with col1:
        render_diagnosis_distribution(results_dict)
    with col2:
        render_detailed_breakdown(results_dict)
    st.divider()
    st.markdown('## 🔥 Comprehensive Metrics Heatmap')
    render_metrics_heatmap(results_dict)
    st.divider()
    st.markdown('## 🔬 Integrated Findings Analysis')
    render_common_patterns(results_dict)
    render_sample_by_sample_view(results_dict)
    st.divider()
    render_comprehensive_table(results_dict)
    st.divider()
    render_clinical_recommendations(results_dict)
    st.divider()
    render_export_section(results_dict)


def render_final_aggregate_dashboard(results_dict):
    import pandas as pd
    import plotly.express as px

    if not results_dict:
        st.info("No data for aggregate analysis.")
        return

    # Flatten top-level items
    items = list(results_dict.values())
    reports_df = pd.json_normalize(items)

    # Sum total cell counts across dataset (from report.counts)
    rbc_col = None
    wbc_col = None
    plate_col = None
    for c in reports_df.columns:
        lc = c.lower()
        if "rbc" in lc and rbc_col is None:
            rbc_col = c
        if "wbc" in lc and wbc_col is None:
            wbc_col = c
        if "plate" in lc and plate_col is None:
            plate_col = c

    RBC_total = int(reports_df[rbc_col].sum()) if rbc_col and rbc_col in reports_df.columns else 0
    WBC_total = int(reports_df[wbc_col].sum()) if wbc_col and wbc_col in reports_df.columns else 0
    Platelets_total = int(reports_df[plate_col].sum()) if plate_col and plate_col in reports_df.columns else 0

    # Expand all wbc_predictions across dataset to aggregate classes and blast status
    try:
        wbc_preds = pd.json_normalize(items, record_path=["wbc_predictions"])  # may raise
    except Exception:
        wbc_preds = pd.DataFrame(columns=["wbc_class", "blast_status"])

    # WBC class distribution
    if not wbc_preds.empty and "wbc_class" in wbc_preds.columns:
        class_counts = wbc_preds["wbc_class"].value_counts().reindex(
            ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"],
            fill_value=0,
        )
    else:
        class_counts = pd.Series({"Basophil": 0, "Eosinophil": 0, "Lymphocyte": 0, "Monocyte": 0, "Neutrophil": 0})

    # Blast vs Normal counts
    if not wbc_preds.empty and "blast_status" in wbc_preds.columns:
        bs = wbc_preds["blast_status"].astype(str).str.lower()
        total_blast = int((bs == "blast-like").sum())
        total_normal = int((bs == "normal").sum())
    else:
        # Fallback to report-level counts if wbc_predictions missing
        blast_col = None
        for c in reports_df.columns:
            if "blast_count" in c.lower():
                blast_col = c
                break
        total_blast = int(reports_df[blast_col].sum()) if blast_col and blast_col in reports_df.columns else 0
        total_normal = max(0, WBC_total - total_blast)

    # Global blast percentage
    blast_pct = (total_blast / (total_blast + total_normal) * 100) if (total_blast + total_normal) > 0 else 0.0

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card_modern("Total WBCs", "🧬", f"{WBC_total}", 100, "accent")
    with c2:
        render_kpi_card_modern("Blast Cells", "⚠️", f"{total_blast}", 100, "accent")
    with c3:
        render_kpi_card_modern("Normal Cells", "✅", f"{total_normal}", 100, "accent")
    with c4:
        render_kpi_card_modern("Blast Burden", "📊", f"{blast_pct:.1f}%", 100, "accent")




    # 1) Total cell counts pie chart (RBC / WBC / Platelets)
    cells_df = pd.DataFrame({"Cell": ["RBC", "WBC", "Platelets"], "Count": [RBC_total, WBC_total, Platelets_total]})
    fig_cells = px.pie(cells_df, names="Cell", values="Count", title="Cell Counts Distribution — Dataset Total", hole=0.3, color_discrete_sequence=['#3B82F6', '#10B981', '#F59E0B'])
    fig_cells = apply_dark_theme(fig_cells)
    st.plotly_chart(fig_cells, use_container_width=True)

    # 2) WBC class distribution pie chart
    class_df = pd.DataFrame({"WBC Class": class_counts.index.tolist(), "Count": class_counts.values})
    fig_wbc_class = px.pie(class_df, names="WBC Class", values="Count", title="WBC Class Distribution — Dataset Total", hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_wbc_class = apply_dark_theme(fig_wbc_class)
    st.plotly_chart(fig_wbc_class, use_container_width=True)

    # 3) Blast vs Normal pie chart
    bn_df = pd.DataFrame({"Type": ["Blast-like", "Normal"], "Count": [total_blast, total_normal]})
    fig_bn = px.pie(bn_df, names="Type", values="Count", title="Blast-like vs Normal Cells — Dataset Total", hole=0.3, color_discrete_sequence=['#EF4444', '#3B82F6'])
    fig_bn = apply_dark_theme(fig_bn)
    st.plotly_chart(fig_bn, use_container_width=True)


def export_batch_results(results_dict):
    if not results_dict:
        st.info("No results to export.")
        return

    # CSV
    rows = []
    for name, it in results_dict.items():
        rpt = it["report"]
        row = {
            "filename": name,
            "total_wbc": rpt["counts"]["WBC"],
            "blast_percentage": rpt["blast_percentage"],
            "diagnosis": rpt["diagnosis"],
            "rbc_count": rpt["counts"]["RBC"],
            "platelet_count": rpt["counts"]["Platelets"],
        }
        rows.append(row)

    csv_buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    # ZIP of images
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, it in results_dict.items():
            # annotated
            ann = it.get("annotated_image")
            if ann is not None:
                bio = io.BytesIO()
                ann.save(bio, format="PNG")
                zf.writestr(f"annotated/{name}.png", bio.getvalue())

            # crops
            for i, c in enumerate(it.get("crops", [])):
                try:
                    bio = io.BytesIO()
                    c["crop"].save(bio, format="PNG")
                    zf.writestr(f"crops/{name}_crop_{i+1}.png", bio.getvalue())
                except Exception:
                    continue

    zip_buffer.seek(0)

    return csv_bytes, zip_buffer.getvalue()


def render_tabs_for_image(item):
    # item is one of results_dict[name]
    yolo_summary = item.get("yolo_summary")
    crops = item.get("crops", [])
    wbc_predictions = item.get("wbc_predictions", [])
    report = item.get("report")

    tab_detection, tab_wbc, tab_summary = st.tabs(["Detection", "WBC Analysis", "Summary Report"])

    with tab_detection:
        st.markdown(f"### Image: {item.get('image_name')}")
        left, right = st.columns([1, 1])
        with left:
            st.markdown("### Original Image")
            st.image(item.get("original_image"), use_container_width=True)
            st.markdown("### Key Metrics")
            render_kpi_card("RBC", str(report["counts"]["RBC"]), "#2563eb")
            render_kpi_card("WBC", str(report["counts"]["WBC"]), "#7c3aed")
            render_kpi_card("Platelets", str(report["counts"]["Platelets"]), "#ea580c")

        with right:
            st.markdown("### Annotated Detection")
            if yolo_summary and yolo_summary.get("annotated_image") is not None:
                st.image(yolo_summary.get("annotated_image"), use_container_width=True)
            render_detection_summary_box(report)

    with tab_wbc:
        st.markdown("### WBC Crop-Level Analysis")
        if not wbc_predictions:
            st.info("No WBC predictions available to display.")
        else:
            render_wbc_grid(wbc_predictions, crops)
            st.divider()
            st.markdown("### WBC Predictions Table")
            wbc_df = prepare_wbc_dataframe(wbc_predictions, crops)
            st.data_editor(
                wbc_df,
                use_container_width=True,
                hide_index=True,
                disabled=True,
                column_config={
                    "crop": st.column_config.ImageColumn("Cropped WBC"),
                    "blast_reasons": st.column_config.TextColumn("Blast Reasons"),
                },
            )

    with tab_summary:
        st.markdown("### Summary Report")
        diagnosis_banner(report["diagnosis"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Status", report["diagnosis"])
        m2.metric("Blast-like Cells", report["blast_count"])
        m3.metric("Blast Percentage", f"{report['blast_percentage']:.2f}%")

        st.markdown("### Reasoning")
        if report["diagnosis"] == "Blast-like":
            st.markdown("Blast-like finding based on the following reasons:")
            for reason in report["blast_reasons_summary"]:
                st.markdown(f"- **{reason}**")
        else:
            st.markdown("Overall result is **Normal** because blast-like patterns are below threshold.")
            if report["blast_reasons_summary"]:
                st.markdown("Observed blast-like indicators (limited):")
                for reason in report["blast_reasons_summary"]:
                    st.markdown(f"- {reason}")



def inject_css():
    css_code = """
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500;700&display=swap');
        
        :root {
          --bg-main: #0A0E1A;
          --bg-card: #111827;
          --bg-elevated: #1A2235;
          --border: #1E2D45;
          --accent: #2563EB;
          --success: #10B981;
          --warning: #F59E0B;
          --danger: #EF4444;
          --text-main: #F1F5F9;
          --text-secondary: #94A3B8;
          --text-muted: #475569;
        }

        html, body, [data-testid="stAppViewContainer"] {
          font-family: 'DM Sans', -apple-system, sans-serif !important;
          background-color: var(--bg-main) !important;
          color: var(--text-main) !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em !important;
            color: var(--text-main) !important;
        }

        [data-testid="stHeader"] {
          background-color: transparent !important;
        }

        [data-testid="stSidebar"] {
          background-color: #0D1526 !important;
          border-right: 1px solid var(--border) !important;
        }

        /* Top Nav */
        .top-nav {
          background: #0D1526;
          border-bottom: 1px solid var(--border);
          padding: 12px 24px;
          margin: -6rem -5rem 2rem -5rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          position: sticky;
          top: 0;
          z-index: 999;
        }

        .brand-container {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .brand-logo {
          font-size: 1.5rem;
        }

        .brand-text {
          display: flex;
          flex-direction: column;
        }

        .brand-title {
          font-family: 'DM Sans', sans-serif;
          font-weight: 700;
          font-size: 1.25rem;
          color: white;
          margin: 0;
          letter-spacing: 0.02em;
        }

        .brand-subtitle {
          font-family: 'DM Sans', sans-serif;
          font-size: 0.75rem;
          color: var(--text-secondary);
          text-transform: uppercase;
          margin: 0;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #0D1526 !important;
            border-bottom: 1px solid var(--border) !important;
            gap: 0;
            padding: 0;
            width: 100%;
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary) !important;
            padding: 14px 24px !important;
            border: none !important;
            background: transparent !important;
            font-weight: 500 !important;
            transition: all 200ms ease;
            border-radius: 0 !important;
            height: auto !important;
            white-space: pre-wrap;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #CBD5E1 !important;
            background: #111827 !important;
        }
        .stTabs [aria-selected="true"] {
            color: #F1F5F9 !important;
            background: #111827 !important;
            border-bottom: 2px solid var(--accent) !important;
            box-shadow: none !important;
        }

        /* File Uploader */
        [data-testid="stFileUploader"] > section {
            background: #111827 !important;
            border: 2px dashed var(--border) !important;
            border-radius: 12px !important;
            padding: 40px !important;
            transition: all 0.2s ease;
        }
        [data-testid="stFileUploader"] > section:hover {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.1) !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 28px !important;
            font: 600 14px 'DM Sans' !important;
            letter-spacing: 0.03em !important;
            box-shadow: 0 4px 15px rgba(37,99,235,0.35) !important;
            transition: all 200ms ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(37,99,235,0.45) !important;
            color: white !important;
        }
        
        [data-testid="stDownloadButton"] > button {
            background: transparent !important;
            border: 1.5px solid var(--accent) !important;
            color: var(--accent) !important;
            box-shadow: none !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
        }
        [data-testid="stDownloadButton"] > button:hover {
            background: rgba(37, 99, 235, 0.1) !important;
            transform: none !important;
            color: var(--accent) !important;
        }

        /* Premium Card / Containers */
        .premium-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 1.5rem;
            transition: all 0.2s ease;
        }
        
        /* KPI Cards */
        .kpi-card-v2 {
            background: #111827;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px 28px;
            position: relative;
            overflow: hidden;
            border-left: 4px solid var(--accent);
        }
        .kpi-label-v2 {
            font-family: 'DM Sans', sans-serif;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        .kpi-value-v2 {
            font-family: 'DM Mono', monospace;
            font-size: 42px;
            font-weight: 700;
            color: var(--accent);
            line-height: 1;
        }
        .kpi-trend {
            margin-top: 0.75rem;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--success);
        }

        /* Progress */
        [data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #2563EB, #7C3AED) !important;
            border-radius: 99px !important;
            height: 6px !important;
        }
        [data-testid="stProgress"] > div {
            background-color: #1A2235 !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: var(--border); }
        ::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }

        /* Expanders */
        [data-testid="stExpander"] {
            background: #0F1929 !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        [data-testid="stExpander"] summary svg {
            fill: var(--accent) !important;
        }

        /* Inputs */
        div[data-testid="stNumberInput"] input, 
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {
            background-color: #0A0E1A !important;
            color: white !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        div[data-testid="stNumberInput"] input:focus, 
        div[data-testid="stTextInput"] input:focus {
            border-color: var(--accent) !important;
        }

        /* Tables */
        [data-testid="stDataFrame"] {
            background: #111827 !important;
        }
        [data-testid="stDataFrame"] th {
            background-color: #1A2235 !important;
            color: var(--text-secondary) !important;
            text-transform: uppercase !important;
            font-size: 11px !important;
        }

        /* WBC Grid */
        .wbc-grid-v2 { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }
        .wbc-card-v2 { background: #111827; border-radius: 12px; padding: 16px; border: 1px solid var(--border); }
        .wbc-card-v2 img { border-radius: 8px; width: 100%; }
        .wbc-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .wbc-card-id { font-family: 'DM Mono', monospace; color: var(--text-secondary); font-weight: 600; }
        .wbc-badge { padding: 4px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
        .badge-blast-v2 { background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }
        .badge-normal-v2 { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }
        .wbc-class-title { font-family: 'DM Sans', sans-serif; font-size: 1.125rem; font-weight: 700; color: white; margin-top: 12px; margin-bottom: 4px; }
        .wbc-confidence { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 12px; }
        
        .attr-row { display: flex; justify-content: space-between; padding: 4px 0; font-family: 'DM Mono', monospace; font-size: 0.85rem; }
        .attr-label { color: var(--text-secondary); }
        .attr-value { color: var(--text-main); font-weight: 500; }

        .diag-banner-v2 {
            background: linear-gradient(135deg, #1A0A0A, #2D0F0F);
            border-left: 6px solid var(--danger);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 2rem;
            color: white;
        }
        .diag-title-v2 {
            font-family: 'DM Sans', sans-serif;
            font-size: 11px;
            font-weight: 600;
            color: var(--danger);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        }
        .diag-result-v2 {
            font-family: 'DM Sans', sans-serif;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    """
    st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)


def render_top_nav():
    st.markdown(
        """
        <div class="top-nav">
          <div class="brand-container">
            <div class="brand-logo">🔬</div>
            <div class="brand-text">
              <p class="brand-title">LeukoAnalyze Pro</p>
              <p class="brand-subtitle">Clinical Diagnostic Assistant</p>
            </div>
          </div>
          <div style="display: flex; gap: 24px; align-items: center;">
            <div style="text-align: right">
              <p style="margin: 0; font-size: 0.75rem; color: var(--text-muted); font-weight: 600;">STATUS</p>
              <p style="margin: 0; font-size: 0.875rem; color: var(--success); font-weight: 700;">● SYSTEM READY</p>
            </div>
            <div style="height: 32px; width: 1px; background: var(--border);"></div>
            <p style="margin: 0; font-weight: 700; color: var(--primary);">v2.4.0</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_kpi_card_modern(label: str, emoji: str, value: str, conf: float, color: str):
    st.markdown(
        f"""
        <div class="kpi-card-v2">
          <div class="kpi-label-v2">{label}</div>
          <div style="display: flex; justify-content: space-between; align-items: flex-end;">
            <div class="kpi-value-v2">{value}</div>
            <div style="font-size: 1.5rem;">{emoji}</div>
          </div>
          <div class="kpi-trend">
            <span style="color: var(--accent);">Confidence: {conf:.1f}%</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_upload_section(yolo_model, vit_bundle, rbc_class_id, wbc_class_id, platelet_class_id, conf_threshold):
    st.markdown("## 📤 Upload & Process Pipeline")
    st.markdown("<p style='color: #94A3B8; font-size: 1rem; margin-top: -10px; margin-bottom: 20px;'>Import peripheral blood smear images and configure batch processing parameters.</p>", unsafe_allow_html=True)
    
    if "processed_results" not in st.session_state:
        st.session_state.processed_results = {}

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("### 1. Select Images")
        uploaded_files = st.file_uploader(
            "Upload PNG, JPG, or JPEG files", 
            type=["png", "jpg", "jpeg"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            st.success(f"**{len(uploaded_files)}** files queued for processing.")

    with col2:
        st.markdown("### 2. Batch Settings")
        st.markdown("<div class='premium-card' style='padding: 20px; background: #1E293B;'>", unsafe_allow_html=True)
        max_images = int(st.number_input("Max Process Limit", min_value=1, max_value=200, value=50, help="Limit the number of images processed at once to prevent memory issues."))
        process_all = st.checkbox("Process all queued images", value=True)
        
        st.markdown("<hr style='border-color: #334155; margin: 15px 0;'>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All Results", use_container_width=True):
            st.session_state.processed_results = {}
            st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    uploaded_names = [getattr(f, "name", f"file_{i}") for i, f in enumerate(uploaded_files or [])]
    
    if not process_all and uploaded_files:
        files_to_process = [f for f in uploaded_files if getattr(f, "name", "") in st.multiselect("Select specific images to process", uploaded_names, default=uploaded_names)]
    else:
        files_to_process = uploaded_files

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 PROCESS BATCH", type="primary", use_container_width=True):
        if not files_to_process:
            st.warning("Please upload and select at least one image to process.")
        else:
            with st.spinner("Initializing AI pipeline..."):
                new_results = process_batch(files_to_process, yolo_model, vit_bundle, rbc_class_id, wbc_class_id, platelet_class_id, conf_threshold, max_images=max_images)
                st.session_state.processed_results.update(new_results)
            st.toast(f"Successfully processed {len(new_results)} images!", icon="🎉")

    # sidebar selector
    if st.session_state.processed_results:
        with st.sidebar:
            keys = list(st.session_state.processed_results.keys())
            sel = st.selectbox("Select processed image", keys, key="selected_image")
            st.markdown("---")
            if st.button("Download batch CSV"):
                csv_bytes, _ = export_batch_results(st.session_state.processed_results)
                st.download_button("Download CSV File", data=csv_bytes, file_name="batch_summary.csv", mime="text/csv")


def render_detection_dashboard(results):
    st.markdown("## 🔍 Detection Results")
    if not results:
        st.info("No processed results. Run processing in Upload & Process.")
        return

    sel = st.session_state.get("selected_image") or list(results.keys())[0]
    item = results.get(sel)
    if item is None:
        st.info("Selected image not found in results.")
        return

    report = item["report"]
    # KPI grid
    cols = st.columns(4)
    with cols[0]:
        render_kpi_card_modern("RED BLOOD CELLS", "🔴", str(report["counts"]["RBC"]), report["detection_confidence"]["RBC"] * 100.0, "#2563eb")
    with cols[1]:
        render_kpi_card_modern("WHITE BLOOD CELLS", "⚪", str(report["counts"]["WBC"]), report["detection_confidence"]["WBC"] * 100.0, "#7c3aed")
    with cols[2]:
        render_kpi_card_modern("PLATELETS", "🔶", str(report["counts"]["Platelets"]), report["detection_confidence"]["Platelets"] * 100.0, "#ea580c")
    with cols[3]:
        blast_pct = report["blast_percentage"]
        emoji = "✅" if blast_pct <= 20.0 else "⚠️"
        render_kpi_card_modern("BLAST BURDEN", emoji, f"{blast_pct:.1f}%", 100.0, "#059669" if blast_pct <= 20.0 else "#dc2626")

    st.markdown("---")
    # Side by side images
    left, right = st.columns([1, 1])
    with left:
        st.markdown("### 📸 Original Smear")
        st.image(item.get("original_image"), use_container_width=True)
    with right:
        st.markdown("### 🎯 Detection Overlay")
        ann = item.get("annotated_image")
        if ann is not None:
            st.image(ann, use_container_width=True)
        render_detection_summary_box(report)


def render_wbc_analysis_dashboard(results):
    st.markdown("## 🧬 WBC Analysis")
    if not results:
        st.info("No processed results.")
        return
    sel = st.session_state.get("selected_image") or list(results.keys())[0]
    item = results.get(sel)
    if item is None:
        st.info("Selected image not found.")
        return

    wbc_preds = item.get("wbc_predictions", [])
    crops = item.get("crops", [])
    if not wbc_preds:
        st.info("No WBC predictions for this image.")
        return

    st.markdown("<div class='wbc-grid-v2'>", unsafe_allow_html=True)
    for pred in wbc_preds:
        crop = next((c for c in crops if c["wbc_id"] == pred["wbc_id"]), None)
        status = pred["blast_status"]
        badge_class = "badge-blast-v2" if status == "blast-like" else "badge-normal-v2"
        icon = "⚠️" if status == "blast-like" else "✅"
        
        st.markdown(
            f"""
            <div class='wbc-card-v2'>
              <div class='wbc-card-header'>
                <span class='wbc-card-id'>ID: #{pred['wbc_id']}</span>
                <span class='wbc-badge {badge_class}'>{icon} {status.upper()}</span>
              </div>
              <div class='wbc-image-container'>
            """,
            unsafe_allow_html=True,
        )
        if crop is not None:
            st.image(crop["crop"], width=200)
        
        st.markdown(
            f"""
              </div>
              <div class='wbc-details'>
                <div class='wbc-class-title'>{pred['wbc_class']}</div>
                <div class='wbc-confidence'>Model Confidence: {pred['wbc_class_confidence']:.2%}</div>
                <div style='margin-top: 1rem;'>
            """,
            unsafe_allow_html=True,
        )
        
        with st.expander("🔬 View Morphological Profile"):
            for k, v in pred["attributes"].items():
                label = k.replace('_', ' ').title()
                st.markdown(f"""
                <div class='attr-row'>
                    <span class='attr-label'>{label}</span>
                    <span class='attr-value'>{v}</span>
                </div>
                """, unsafe_allow_html=True)
                
        if pred["blast_status"] == "blast-like":
            with st.expander("⚠️ Blast Indicators Found", expanded=True):
                for r in pred["blast_reasons"]:
                    st.markdown(f"<div style='font-size: 0.85rem; color: #b91c1c; font-weight: 500;'>• {format_blast_reason(r)}</div>", unsafe_allow_html=True)
                    
        st.markdown("</div></div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)



def render_diagnostic_report(results):
    st.markdown("## 📊 Diagnostic Report")
    if not results:
        st.info("No processed results.")
        return
    sel = st.session_state.get("selected_image") or list(results.keys())[0]
    item = results.get(sel)
    report = item.get("report")
    pct = report["blast_percentage"]
    is_blast = pct > 20.0
    color_start = "#059669" if not is_blast else "#dc2626"
    color_end = "#10b981" if not is_blast else "#ef4444"
    icon = "✅" if not is_blast else "⚠️"

    st.markdown(f"""
    <div class='diag-banner-v2' style='background: linear-gradient(135deg, {color_start} 0%, {color_end} 100%);'>
      <div class='diag-title-v2'>Automated Clinical Assessment</div>
      <div class='diag-result-v2'>{icon} {report['diagnosis'].upper()}</div>
      <div style='display: flex; gap: 2rem; align-items: center;'>
        <div style='background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 12px;'>
            <span style='opacity: 0.8; font-size: 0.8rem;'>BLAST BURDEN</span><br>
            <span style='font-size: 1.25rem; font-weight: 700;'>{pct:.2f}%</span>
        </div>
        <div style='background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 12px;'>
            <span style='opacity: 0.8; font-size: 0.8rem;'>CELL COUNT</span><br>
            <span style='font-size: 1.25rem; font-weight: 700;'>{report['blast_count']} Identified</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


    # Clinical interpretation
    st.markdown("<div class='card-wrap' style='border-left:4px solid var(--primary);'>", unsafe_allow_html=True)
    st.markdown("**Clinical Interpretation**")
    if is_blast:
        st.error("Blast-like pattern exceeds clinical threshold (>20%). Consider urgent hematology referral.")
    else:
        st.success("No actionable blast-like burden detected. Follow routine clinical workflow.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total WBCs Analyzed", report['counts']['WBC'])
    col2.metric("Blast-like Cells", report['blast_count'], f"{report['blast_percentage']:.2f}%")
    col3.metric("Threshold Status", "Above 20%" if is_blast else "Below 20%")

    if report['blast_reasons_summary']:
        st.markdown("**Morphological Indicators**")
        for r in report['blast_reasons_summary']:
            st.markdown(f"- {r}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_batch_analytics(results):
    st.markdown("## 📈 Batch Analytics")
    if not results:
        st.info("No processed results.")
        return
    render_final_aggregate_dashboard(results)
    st.markdown("---")
    # detailed table
    rows = []
    for name, it in results.items():
        rpt = it['report']
        rows.append({
            'filename': name,
            'total_wbc': rpt['counts']['WBC'],
            'blast_percentage': rpt['blast_percentage'],
            'diagnosis': rpt['diagnosis']
        })
    df = pd.DataFrame(rows)
    st.markdown("### Detailed Batch Results")
    st.dataframe(df)


def render_header():
    st.markdown(
        """
        <div class="premium-card" style="background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%); border-left: 5px solid var(--accent);">
            <h1 style="color: var(--primary); margin-bottom: 0.5rem; font-size: 2.25rem;">Advanced Hematology Analysis</h1>
            <p style="color: var(--text-muted); font-size: 1.1rem; max-width: 800px;">
                State-of-the-art AI pipeline for automated cell detection, morphological profiling, and diagnostic screening of peripheral blood smears.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()


def render_kpi_card(label: str, value: str, bg_color: str):
    st.markdown(
        f"""
        <div class="kpi-card" style="background:{bg_color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def diagnosis_banner(diagnosis: str):
    if diagnosis == "Blast-like":
        color = "#b42318"
        text = "Blast-like"
    else:
        color = "#067647"
        text = "Normal"
    st.markdown(
        f"<div class='diagnosis-banner' style='background:{color};'>Final Automated Assessment: {text}</div>",
        unsafe_allow_html=True,
    )


def pil_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def prepare_wbc_dataframe(predictions: List[Dict], crops: List[Dict]):
    crop_by_id = {item["wbc_id"]: item["crop"] for item in crops}
    rows = []
    for pred in predictions:
        crop = crop_by_id.get(pred["wbc_id"])
        row = {
            "wbc_id": pred["wbc_id"],
            "crop": pil_to_data_uri(crop) if crop is not None else "",
            "class": pred["wbc_class"],
            "blast": pred["blast_status"],
            "blast_reasons": ", ".join(format_blast_reason(r) for r in pred["blast_reasons"]) if pred["blast_status"] == "blast-like" else "",
        }
        for attr_name, attr_value in pred["attributes"].items():
            row[attr_name] = attr_value
        rows.append(row)
    return pd.DataFrame(rows)


def render_wbc_grid(predictions: List[Dict], crops: List[Dict]):
    crop_by_id = {item["wbc_id"]: item for item in crops}
    if not predictions:
        st.info("No WBC predictions available.")
        return

    for pred in predictions:
        crop_item = crop_by_id.get(pred["wbc_id"])
        st.markdown("<div class='crop-card'>", unsafe_allow_html=True)
        st.markdown(f"**WBC {pred['wbc_id']}**")
        if crop_item is not None:
            st.image(crop_item["crop"], width=220)
        st.markdown(f"Class: **{pred['wbc_class']}**")
        st.markdown("Attributes:")
        for attr_name, attr_value in pred["attributes"].items():
            label = attr_name.replace("_", " ").title()
            st.markdown(f"- {label}: **{attr_value}**")
        if pred["blast_status"] == "blast-like":
            reasons = [format_blast_reason(reason) for reason in pred["blast_reasons"]]
            st.markdown("<span class='badge badge-red'>Blast-like</span>", unsafe_allow_html=True)
            st.markdown("Blast reasons:")
            for reason in reasons:
                st.markdown(f"- **{reason}**")
        else:
            st.markdown("<span class='badge badge-green'>Normal</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_detection_summary_box(report: Dict):
    st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
    st.markdown("**Detection Summary**")
    st.markdown(f"- RBC: **{report['counts']['RBC']}** (avg conf: **{report['detection_confidence']['RBC']:.2f}**)")
    st.markdown(f"- WBC: **{report['counts']['WBC']}** (avg conf: **{report['detection_confidence']['WBC']:.2f}**)")
    st.markdown(f"- Platelets: **{report['counts']['Platelets']}** (avg conf: **{report['detection_confidence']['Platelets']:.2f}**)")
    st.markdown("</div>", unsafe_allow_html=True)


def render_top_nav():
    st.markdown(
        """
        <style>
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.7; }
            100% { transform: scale(1); opacity: 1; }
        }
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #10B981;
            border-radius: 50%;
            margin-right: 6px;
            animation: pulse 2s infinite ease-in-out;
        }
        .version-badge {
            background-color: #1A2235;
            color: #475569;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            font-family: 'DM Mono', monospace;
        }
        </style>
        <div class="top-nav">
          <div class="brand-container">
            <div class="brand-logo">🔬</div>
            <div class="brand-text">
              <p class="brand-title">LEUKOANALYZE PRO</p>
              <p class="brand-subtitle">Advanced Hematology Analysis</p>
            </div>
          </div>
          <div style="display: flex; gap: 24px; align-items: center;">
            <div style="text-align: right; display: flex; align-items: center; gap: 8px;">
              <span class="status-dot"></span>
              <p style="margin: 0; font-size: 0.875rem; color: #10B981; font-weight: 700; letter-spacing: 0.05em;">SYSTEM READY</p>
            </div>
            <div style="height: 32px; width: 1px; background: var(--border);"></div>
            <div class="version-badge">v2.4.0</div>
            <div style="height: 32px; width: 1px; background: var(--border);"></div>
            <button style="background: transparent; color: #94A3B8; border: 1px solid #1E2D45; border-radius: 6px; padding: 6px 12px; font-size: 0.8rem; cursor: pointer;">Deploy</button>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Leukemia Detection Dashboard", layout="wide", initial_sidebar_state="expanded")

    inject_css()
    render_top_nav()

    default_yolo = resolve_existing_path([
        DEFAULT_YOLO_PATH,
        "best_TXL-PBC_model.pt",
        "runs/detect/train/weights/best.pt",
    ])
    default_vit = resolve_existing_path([
        DEFAULT_VIT_PATH,
        "vit_best.pth",
        "wbc sample imag/vit_epoch_4.pth",
    ])
    default_csv = resolve_existing_path([
        "models/pbc_attr_v1_train.csv",
        "wbc sample imag/pbc_attr_v1_train.csv",
    ])

    st.sidebar.markdown("### ⚙️ Engine Settings")
    st.sidebar.markdown("<p style='color: #94A3B8; font-size: 0.85rem; margin-top:-10px; margin-bottom: 20px;'>Configure AI models and inference parameters.</p>", unsafe_allow_html=True)
    
    with st.sidebar.expander("🧩 Model Architectures", expanded=True):
        yolo_path = st.text_input("YOLO Model Path", value=default_yolo)
        vit_path = st.text_input("ViT Model Path", value=default_vit)
        csv_path = st.text_input("Attribute Definitions (CSV)", value=default_csv)

    with st.sidebar.expander("🎛️ Inference Parameters", expanded=False):
        conf_threshold = st.slider("WBC Confidence Threshold", min_value=0.3, max_value=0.9, value=DEFAULT_CONF_THRESHOLD, step=0.05)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    yolo_model = None
    vit_bundle = None

    try:
        yolo_model = load_yolo_model(yolo_path)
    except Exception as exc:
        st.sidebar.error(f"YOLO Failed: {exc}")

    try:
        vit_bundle = load_vit_model(vit_path, csv_path, device_str)
    except Exception as exc:
        st.sidebar.error(f"ViT Failed: {exc}")

    if yolo_model is None or vit_bundle is None:
        st.sidebar.warning("Please load valid model weights to initialize the pipeline.")
        st.stop()

    auto_rbc_id, auto_wbc_id, auto_platelet_id = infer_yolo_class_ids(yolo_model)
    with st.sidebar.expander("🏷️ Class Mappings", expanded=False):
        rbc_class_id = int(st.number_input("RBC ID", min_value=0, step=1, value=auto_rbc_id))
        wbc_class_id = int(st.number_input("WBC ID", min_value=0, step=1, value=auto_wbc_id))
        platelet_class_id = int(st.number_input("Platelet ID", min_value=0, step=1, value=auto_platelet_id))
        if isinstance(yolo_model.names, dict):
            st.caption(f"YOLO classes: {yolo_model.names}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
        <div style="font-size: 0.8rem; color: #64748B;">
            <div><strong>Compute Device:</strong> {device_str.upper()}</div>
            <div><strong>Hardware Accel:</strong> {"Enabled 🚀" if device_str == "cuda" else "Disabled 🐢"}</div>
            <div style="margin-top: 10px;">LeukoAnalyze Pro v2.4.0</div>
        </div>
    """, unsafe_allow_html=True)

    # ensure session state for results
    if "processed_results" not in st.session_state:
        st.session_state.processed_results = {}

    # Main horizontal tabs (multi-image views)
    tabs = st.tabs([
        "📤 Import Batch",
        "🎯 Cellular Detection (YOLO)",
        "🧬 Morphology Classification (ViT)",
        "📊 Cohort Analytics",
        "🧾 Official Clinical Report",
    ])

    results = st.session_state.processed_results

    with tabs[0]:
        # include upload controls at top of batch overview
        render_upload_section(yolo_model, vit_bundle, rbc_class_id, wbc_class_id, platelet_class_id, conf_threshold)

    with tabs[1]:
        render_detection_results_tab(results)

    with tabs[2]:
        render_wbc_analysis_tab(results)

    with tabs[3]:
        render_combined_diagnostic_tab(results)

    with tabs[4]:
        render_final_diagnostic_report(results)



if __name__ == "__main__":
    main()

#streamlit run "c:\Users\madhu\Desktop\major project\app.py"
