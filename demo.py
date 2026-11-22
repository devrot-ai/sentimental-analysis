"""
Optional Gradio demonstration interface for FCMF inference.

IMPORTANT: This demo requires pretrained model weights to produce real predictions.
Without checkpoints, it will show the interface but cannot run inference.

Download weights from the original ViMACSA repository:
https://drive.google.com/file/d/1NHI24JxyjXfi1lU7nF99-TV7vBL-StZm/view?usp=sharing

Original FCMF architecture by Nguyen et al. (2025).
Demo interface by Kanishk Thakur.
"""

import os
import sys
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if gradio is available
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not installed. Install with: pip install gradio")

# Constants
ASPECT = ["Location", "Food", "Room", "Facilities", "Service", "Public_area"]
POLARITY = ["None", "Negative", "Neutral", "Positive"]
SENTIMENT_COLORS = {
    "Positive": "#4caf50",
    "Negative": "#f44336",
    "Neutral": "#ff9800",
    "None": "#9e9e9e",
}

# --- Check if model weights are available ---
CHECKPOINTS_DIR = Path("./checkpoints")
WEIGHTS_AVAILABLE = all(
    (CHECKPOINTS_DIR / f).exists()
    for f in [
        "fcmf_model.pth",
        "visual_model.pth",
        "weight_roi_resnet152.pth",
        "weight_image_resnet152.pth",
        "yolov8m.pt",
    ]
)


def _check_weights_status():
    """Return a status message about checkpoint availability."""
    if WEIGHTS_AVAILABLE:
        return "✅ All model checkpoints found. Inference is available."
    
    missing = []
    expected = [
        "fcmf_model.pth",
        "visual_model.pth",
        "weight_roi_resnet152.pth",
        "weight_image_resnet152.pth",
        "yolov8m.pt",
    ]
    for f in expected:
        if not (CHECKPOINTS_DIR / f).exists():
            missing.append(f)
    
    return (
        f"⚠️ Missing checkpoints: {', '.join(missing)}\n\n"
        f"Download from: https://drive.google.com/file/d/1NHI24JxyjXfi1lU7nF99-TV7vBL-StZm/view\n"
        f"Place files in: {CHECKPOINTS_DIR.absolute()}\n\n"
        f"This interface is a preview only — predictions cannot be generated without weights."
    )


def predict_sentiment(text, image, pretrained_model="xlm-roberta-base", num_rois=4, num_images=7):
    """Run FCMF inference or return a placeholder if weights are unavailable."""
    
    if not text or not text.strip():
        return "Please enter some text.", _check_weights_status()
    
    if not WEIGHTS_AVAILABLE:
        placeholder = {asp: "⚠️ Weights required" for asp in ASPECT}
        result_text = "## Predictions (Weights Not Available)\n\n"
        result_text += "| Aspect | Sentiment |\n|--------|----------|\n"
        for asp in ASPECT:
            result_text += f"| {asp} | ⚠️ N/A |\n"
        result_text += f"\n*{_check_weights_status()}*"
        return result_text, _check_weights_status()
    
    # --- Actual inference ---
    try:
        import torch
        from transformers import AutoTokenizer
        from inference import load_models, predict
        import argparse

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Build args namespace
        args = argparse.Namespace(
            pretrained_model=pretrained_model,
            num_images=num_images,
            num_rois=num_rois,
            fcmf_checkpoint=str(CHECKPOINTS_DIR / "fcmf_model.pth"),
            visual_model_checkpoint=str(CHECKPOINTS_DIR / "visual_model.pth"),
            yolo_model_path=str(CHECKPOINTS_DIR / "yolov8m.pt"),
            weight_roi_path=str(CHECKPOINTS_DIR / "weight_roi_resnet152.pth"),
            weight_image_path=str(CHECKPOINTS_DIR / "weight_image_resnet152.pth"),
        )

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        models = load_models(args, device)

        # Handle image
        image_paths = []
        if image is not None:
            image_paths = [image]  # Gradio provides temp path

        results = predict(
            tokenizer=tokenizer,
            text=text,
            models=models,
            list_image_path=image_paths,
            num_imgs=num_images,
            num_rois=num_rois,
            device=device,
        )

        # Format results
        result_text = "## Predictions\n\n"
        result_text += "| Aspect | Sentiment |\n|--------|----------|\n"
        for asp, sentiment in results.items():
            emoji = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡", "None": "⚪"}.get(sentiment, "")
            result_text += f"| {asp} | {emoji} {sentiment} |\n"

        return result_text, "✅ Inference completed successfully."

    except Exception as e:
        error_msg = f"❌ Inference failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, error_msg


def build_demo():
    """Build and return the Gradio interface."""
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is not installed. Install with: pip install gradio")
    
    with gr.Blocks(title="FCMF — Multimodal Aspect-Category Sentiment Analysis") as demo:
        gr.Markdown(
            """
            # FCMF — Multimodal Aspect-Category Sentiment Analysis
            
            Fine-Grained Cross-Modal Fusion (FCMF) for predicting aspect-category sentiments 
            from text and images. Based on the [ViMACSA paper](https://arxiv.org/abs/2405.00543) 
            by Nguyen et al. (2025).
            
            **Aspects:** Location, Food, Room, Facilities, Service, Public Area  
            **Sentiments:** Positive, Negative, Neutral, None
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text for sentiment analysis (Vietnamese or multilingual)...",
                    lines=3,
                )
                image_input = gr.Image(label="Associated Image (optional)", type="filepath")
                
                with gr.Row():
                    pretrained_model = gr.Dropdown(
                        choices=[
                            "xlm-roberta-base",
                            "vinai/phobert-base",
                            "vinai/phobert-base-v2",
                            "uitnlp/visobert",
                        ],
                        value="xlm-roberta-base",
                        label="Pretrained Model",
                    )
                    num_rois = gr.Slider(
                        minimum=1, maximum=8, value=4, step=1, label="Number of RoIs"
                    )
                
                predict_btn = gr.Button("Analyze Sentiment", variant="primary")
            
            with gr.Column(scale=1):
                result_output = gr.Markdown(label="Predictions")
                status_output = gr.Textbox(label="Status", interactive=False)
        
        # Initialize with status
        demo.load(fn=lambda: _check_weights_status(), outputs=status_output)
        
        predict_btn.click(
            fn=predict_sentiment,
            inputs=[text_input, image_input, pretrained_model, num_rois],
            outputs=[result_output, status_output],
        )
        
        gr.Markdown(
            """
            ---
            **Model:** FCMF (Fine-Grained Cross-Modal Fusion) · 
            **Paper:** [arXiv:2405.00543](https://arxiv.org/abs/2405.00543) · 
            **Original Authors:** Nguyen, Quy Hoang; Nguyen, Minh-Van Truong; Van Nguyen, Kiet · 
            **Fork maintained by:** Kanishk Thakur
            """
        )
    
    return demo


if __name__ == "__main__":
    if not GRADIO_AVAILABLE:
        print("Error: Gradio is not installed. Install with: pip install gradio")
        sys.exit(1)
    
    demo = build_demo()
    print(f"\nModel weights status: {_check_weights_status()}\n")
    demo.launch(share=False)
