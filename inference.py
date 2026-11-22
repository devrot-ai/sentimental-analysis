"""
Inference entry point for the FCMF multimodal sentiment analysis model.

Accepts text input, one or more image paths, and a model checkpoint.
Outputs per-aspect sentiment predictions.

Original FCMF architecture by Nguyen et al. (2025).
Inference pipeline adapted for reproducibility by Kanishk Thakur.
"""

import os
import sys
import argparse
import json
import logging

import torch
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASPECT = ["Location", "Food", "Room", "Facilities", "Service", "Public_area"]
IMG_ASPECT = ["Food", "Room", "Facilities", "Service", "Public_area"]
POLARITY = ["None", "Negative", "Neutral", "Positive"]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_file(path, label="file"):
    """Check that a file exists and is readable."""
    p = Path(path)
    if not p.exists():
        logger.error(f"{label} not found: {path}")
        sys.exit(1)
    if not p.is_file():
        logger.error(f"{label} is not a file: {path}")
        sys.exit(1)
    return str(p)


def validate_checkpoint(path, label="checkpoint"):
    """Validate that a checkpoint file exists before attempting to load."""
    if path is None:
        logger.error(
            f"{label} path not specified. "
            f"Download pretrained weights from the original ViMACSA repository: "
            f"https://drive.google.com/file/d/1NHI24JxyjXfi1lU7nF99-TV7vBL-StZm/view?usp=sharing "
            f"and place them in the checkpoints/ directory."
        )
        sys.exit(1)
    return validate_file(path, label)


# ---------------------------------------------------------------------------
# Model loading (lazy imports to give clear errors)
# ---------------------------------------------------------------------------

def load_models(args, device):
    """Load all required models: YOLO, image classifier, RoI classifier, FCMF, ResNet.
    
    Returns a dict of loaded models ready for inference.
    """
    from fcmf_framework.image_process import (
        MyRoIModel, MyImgModel, load_model,
        image_processing, construct_visual_features,
        image_encoder, roi_encoder,
    )
    from fcmf_framework.fcmf_modeling import FCMF
    from torchvision.models import resnet152, ResNet152_Weights

    models = {}

    # --- YOLO (optional, for RoI detection at inference) ---
    yolo_path = validate_checkpoint(args.yolo_model_path, "YOLO model")
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(yolo_path)
        yolo_model.to(device)
        models["yolo"] = yolo_model
        logger.info(f"Loaded YOLO model from {yolo_path}")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    # --- RoI classifier ---
    roi_weight_path = validate_checkpoint(args.weight_roi_path, "RoI classifier weights")
    roi_model = MyRoIModel(len(IMG_ASPECT))
    roi_model.to(device)
    checkpoint = load_model(roi_weight_path)
    roi_model.load_state_dict(checkpoint["model_state_dict"])
    roi_model.eval()
    models["roi_model"] = roi_model
    logger.info(f"Loaded RoI classifier from {roi_weight_path}")

    # --- Image classifier ---
    img_weight_path = validate_checkpoint(args.weight_image_path, "Image classifier weights")
    image_model = MyImgModel(len(IMG_ASPECT))
    image_model.to(device)
    checkpoint = load_model(img_weight_path)
    image_model.load_state_dict(checkpoint["model_state_dict"])
    image_model.eval()
    models["image_model"] = image_model
    logger.info(f"Loaded image classifier from {img_weight_path}")

    # --- FCMF ---
    fcmf_path = validate_checkpoint(args.fcmf_checkpoint, "FCMF model weights")
    fcmf_model = FCMF(
        pretrained_path=args.pretrained_model,
        num_imgs=args.num_images,
        num_roi=args.num_rois,
    )
    fcmf_model.to(device)
    checkpoint = load_model(fcmf_path)
    fcmf_model.load_state_dict(checkpoint["model_state_dict"])
    fcmf_model.eval()
    models["fcmf"] = fcmf_model
    logger.info(f"Loaded FCMF model from {fcmf_path}")

    # --- Visual encoder (ResNet152) ---
    vis_path = validate_checkpoint(args.visual_model_checkpoint, "Visual encoder weights")
    visual_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    visual_model.to(device)
    checkpoint = load_model(vis_path)
    visual_model.load_state_dict(checkpoint["model_state_dict"])
    visual_model.eval()
    models["visual"] = visual_model
    logger.info(f"Loaded visual encoder from {vis_path}")

    return models


# ---------------------------------------------------------------------------
# Visual feature extraction
# ---------------------------------------------------------------------------

def extract_visual_features(models, list_image_path, num_imgs, num_rois, device):
    """Extract image and RoI visual features using ResNet152 and YOLO."""
    from fcmf_framework.image_process import (
        construct_visual_features, image_encoder, roi_encoder,
    )

    yolo_model = models["yolo"]
    visual_model = models["visual"]

    t_img_features, roi_img_features, roi_coors = construct_visual_features(
        yolo_model, list_image_path, 30, num_rois, num_imgs, device
    )
    t_img_features = t_img_features.unsqueeze(0).float().to(device)
    roi_img_features = roi_img_features.unsqueeze(0).float().to(device)
    roi_coors = roi_coors.unsqueeze(0).to(device)

    with torch.no_grad():
        encoded_img = []
        encoded_roi = []

        for img_idx in range(num_imgs):
            img_features = image_encoder(
                visual_model, t_img_features[:, img_idx, :]
            ).view(-1, 2048, 49).permute(0, 2, 1).squeeze(1)
            encoded_img.append(img_features)

            roi_f = []
            for roi_idx in range(num_rois):
                roi_features = roi_encoder(
                    visual_model, roi_img_features[:, img_idx, roi_idx, :]
                ).squeeze(1)
                roi_f.append(roi_features)

            roi_f = torch.stack(roi_f, dim=1)
            encoded_roi.append(roi_f)

        encoded_img = torch.stack(encoded_img, dim=1)
        encoded_roi = torch.stack(encoded_roi, dim=1)

    return roi_coors, encoded_img, encoded_roi


# ---------------------------------------------------------------------------
# Aspect-category prediction via image/RoI classifiers
# ---------------------------------------------------------------------------

def predict_image_aspects(models, list_image_path, device):
    """Use the image and RoI classifiers to predict aspect categories from images."""
    from fcmf_framework.image_process import image_processing

    list_image_aspect, list_roi_aspect = image_processing(
        models["image_model"],
        models["roi_model"],
        models["yolo"],
        list_image_path,
        30,
        IMG_ASPECT,
        device,
    )
    return list_image_aspect, list_roi_aspect


# ---------------------------------------------------------------------------
# FCMF prediction
# ---------------------------------------------------------------------------

def predict(tokenizer, text, models, list_image_path, num_imgs, num_rois, device):
    """Run the full FCMF inference pipeline.
    
    Returns:
        dict mapping each aspect to its predicted sentiment polarity.
    """
    from text_preprocess import TextNormalize, convert_unicode
    from underthesea import text_normalize

    # --- Text preprocessing ---
    normalize_class = TextNormalize()
    text = normalize_class.normalize(text_normalize(convert_unicode(text)))

    # --- Image aspect prediction ---
    if list_image_path:
        list_image_aspect, list_roi_aspect = predict_image_aspects(
            models, list_image_path, device
        )
    else:
        list_image_aspect = set()
        list_roi_aspect = set()

    if not list_image_aspect:
        list_image_aspect = {"empty"}
    if not list_roi_aspect:
        list_roi_aspect = {"empty"}

    # --- Auxiliary sentence ---
    joined_aspect = (
        f" {' , '.join(list_image_aspect)} </s></s>  {' , '.join(list_roi_aspect)}"
    )
    joined_aspect = joined_aspect.lower().replace("_", " ")

    # --- Visual features ---
    if list_image_path:
        roi_coors, encoded_img, encoded_roi = extract_visual_features(
            models, list_image_path, num_imgs, num_rois, device
        )
    else:
        # Zero visual features when no images provided
        roi_coors = torch.zeros(1, num_imgs, num_rois, 4).to(device)
        encoded_img = torch.zeros(1, num_imgs, 49, 2048).to(device)
        encoded_roi = torch.zeros(1, num_imgs, num_rois, 2048).to(device)

    # --- Per-aspect prediction ---
    fcmf_model = models["fcmf"]
    results = {}

    for id_asp, asp in enumerate(ASPECT):
        combine_text = f"{asp} </s></s> {text}"
        combine_text = combine_text.lower().replace("_", " ")

        tokens = tokenizer(
            combine_text,
            joined_aspect,
            max_length=170,
            truncation="only_first",
            padding="max_length",
            return_token_type_ids=True,
        )

        input_ids = torch.tensor(tokens["input_ids"]).unsqueeze(0).to(device)
        token_type_ids = torch.tensor(tokens["token_type_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(tokens["attention_mask"]).unsqueeze(0).to(device)
        added_input_mask = torch.tensor([1] * (170 + 49)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = fcmf_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                added_attention_mask=added_input_mask,
                visual_embeds_att=encoded_img,
                roi_embeds_att=encoded_roi,
                roi_coors=roi_coors,
            )

        pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)
        results[asp] = POLARITY[pred[0]]

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="FCMF Multimodal Aspect-Category Sentiment Analysis — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python inference.py \\\n"
            '    --text "Khách sạn này rất đẹp, nhân viên nhiệt tình" \\\n'
            "    --image_list image_0.png image_1.png \\\n"
            "    --num_rois 4\n"
        ),
    )

    # Required
    parser.add_argument("--text", type=str, required=True, help="Input text for analysis.")
    parser.add_argument(
        "--image_list", nargs="+", default=None,
        help="One or more image file paths. Optional — if omitted, text-only analysis is performed.",
    )

    # Model
    parser.add_argument(
        "--pretrained_model", type=str, default="xlm-roberta-base",
        help="HuggingFace pretrained model name (default: xlm-roberta-base).",
    )
    parser.add_argument("--num_images", type=int, default=7, help="Max images per sample (default: 7).")
    parser.add_argument("--num_rois", type=int, default=4, help="Number of RoIs per image (default: 4).")

    # Checkpoint paths
    parser.add_argument(
        "--fcmf_checkpoint", type=str, default="./checkpoints/fcmf_model.pth",
        help="Path to FCMF model weights.",
    )
    parser.add_argument(
        "--visual_model_checkpoint", type=str, default="./checkpoints/visual_model.pth",
        help="Path to visual encoder (ResNet) weights.",
    )
    parser.add_argument(
        "--yolo_model_path", type=str, default="./checkpoints/yolov8m.pt",
        help="Path to YOLOv8 model weights.",
    )
    parser.add_argument(
        "--weight_roi_path", type=str, default="./checkpoints/weight_roi_resnet152.pth",
        help="Path to RoI classifier weights.",
    )
    parser.add_argument(
        "--weight_image_path", type=str, default="./checkpoints/weight_image_resnet152.pth",
        help="Path to image classifier weights.",
    )

    # Configuration
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file. CLI arguments override config values.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device: 'cuda', 'cpu', or 'auto'.")
    parser.add_argument("--output_json", type=str, default=None, help="Save results to a JSON file.")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Configuration ---
    from configs import get_config

    config = get_config(config_path=args.config, cli_args=args)
    device = config["device"]
    logger.info(f"Device: {device}")

    # --- Validate images ---
    list_image_path = args.image_list or []
    for img_path in list_image_path:
        validate_file(img_path, f"Image '{img_path}'")

    if not list_image_path:
        logger.warning(
            "No images provided. Running text-only analysis. "
            "Visual features will be zeroed. For best results, provide associated images."
        )

    # --- Load tokenizer ---
    logger.info(f"Loading tokenizer: {args.pretrained_model}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # --- Load models ---
    logger.info("Loading models...")
    models = load_models(args, device)

    # --- Run prediction ---
    logger.info("Running inference...")
    results = predict(
        tokenizer=tokenizer,
        text=args.text,
        models=models,
        list_image_path=list_image_path,
        num_imgs=args.num_images,
        num_rois=args.num_rois,
        device=device,
    )

    # --- Display results ---
    print("\n" + "=" * 60)
    print("  FCMF Multimodal Aspect-Category Sentiment Analysis")
    print("=" * 60)
    print(f"\n  Input text: {args.text}")
    print(f"  Images:     {len(list_image_path)} provided")
    print(f"  Device:     {device}")
    print("\n  Predictions:")
    print("  " + "-" * 40)
    for aspect, sentiment in results.items():
        marker = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡", "None": "⚪"}.get(
            sentiment, "  "
        )
        print(f"  {marker}  {aspect:15s} → {sentiment}")
    print("  " + "-" * 40)
    print()

    # --- Save JSON output ---
    if args.output_json:
        output = {
            "text": args.text,
            "images": list_image_path,
            "predictions": results,
            "model": args.pretrained_model,
            "device": device,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output_json}")

    return results


if __name__ == "__main__":
    main()
