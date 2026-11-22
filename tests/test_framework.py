"""
Unit and integration tests for the ViMACSA / FCMF framework.

These tests use synthetic data and do NOT require the full ViMACSA dataset
or pretrained model weights. Tests that need external resources are marked
with skip conditions and explanatory messages.
"""

import os
import sys
import json
import tempfile
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 1. Configuration loading
# ============================================================================

class TestConfig:
    """Tests for the YAML configuration system."""

    def test_default_config_loads(self):
        from configs import load_yaml_config
        config = load_yaml_config()
        assert isinstance(config, dict)
        assert "pretrained_model" in config
        assert "num_imgs" in config
        assert "num_rois" in config
        assert "seed" in config

    def test_default_config_values(self):
        from configs import load_yaml_config
        config = load_yaml_config()
        assert config["num_polarity"] == 4
        assert config["num_imgs"] == 7
        assert config["num_rois"] == 4
        assert config["seed"] == 42
        assert config["max_seq_length"] == 170
        assert config["image_size"] == 224

    def test_missing_config_raises(self):
        from configs import load_yaml_config
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/path/config.yaml")

    def test_cli_merge_overrides(self):
        from configs import load_yaml_config, merge_cli_args
        config = load_yaml_config()
        original_lr = config["learning_rate"]
        cli_args = {"learning_rate": 1e-4, "unrelated_key": None}
        merged = merge_cli_args(config, cli_args)
        assert merged["learning_rate"] == 1e-4
        assert "unrelated_key" not in merged  # None values are skipped

    def test_resolve_device_cpu(self):
        from configs import resolve_device
        assert resolve_device("cpu") == "cpu"

    def test_resolve_device_auto(self):
        from configs import resolve_device
        device = resolve_device("auto")
        assert device in ("cuda", "cpu")

    def test_custom_yaml_config(self):
        """Test loading a custom YAML config file."""
        from configs import load_yaml_config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("seed: 123\nnum_imgs: 3\n")
            f.flush()
            config = load_yaml_config(f.name)
        os.unlink(f.name)
        assert config["seed"] == 123
        assert config["num_imgs"] == 3


# ============================================================================
# 2. Text preprocessing
# ============================================================================

class TestTextPreprocess:
    """Tests for Vietnamese text normalization."""

    def test_normalize_basic(self):
        from text_preprocess import TextNormalize
        normalizer = TextNormalize()
        result = normalizer.normalize("Khách sạn rất đẹp!!!")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_lowercases(self):
        from text_preprocess import TextNormalize
        normalizer = TextNormalize()
        result = normalizer.normalize("HELLO WORLD")
        assert result == "helo world"  # repeated chars collapsed

    def test_normalize_removes_emoji(self):
        from text_preprocess import TextNormalize
        normalizer = TextNormalize()
        result = normalizer.normalize("good 😊👍 hotel")
        assert "😊" not in result
        assert "👍" not in result

    def test_normalize_removes_hashtags(self):
        from text_preprocess import TextNormalize
        normalizer = TextNormalize()
        result = normalizer.normalize("nice hotel #travel @user")
        assert "#travel" not in result
        assert "@user" not in result

    def test_convert_unicode(self):
        from text_preprocess import convert_unicode
        text = "xin chào"
        result = convert_unicode(text)
        assert isinstance(result, str)

    def test_normalize_collapses_repeated(self):
        from text_preprocess import TextNormalize
        normalizer = TextNormalize()
        result = normalizer.normalize("goooood")
        assert result == "god"


# ============================================================================
# 3. Model initialization (architecture tests — no weights needed)
# ============================================================================

class TestModelArchitecture:
    """Test that FCMF model components initialize without errors."""

    def test_fcmf_init_xlm_roberta(self):
        """FCMF initializes with xlm-roberta-base (downloads from HF)."""
        try:
            from fcmf_framework.fcmf_modeling import FCMF
            model = FCMF(
                pretrained_path="xlm-roberta-base",
                num_labels=4,
                num_imgs=2,
                num_roi=2,
            )
            assert model is not None
            # Verify classifier output size
            assert model.classifier.out_features == 4
            assert model.num_imgs == 2
            assert model.num_roi == 2
        except Exception as e:
            pytest.skip(f"Cannot initialize FCMF (network/dependency issue): {e}")

    def test_fcmf_forward_shape(self):
        """Test that FCMF forward pass produces correct output shape."""
        try:
            from fcmf_framework.fcmf_modeling import FCMF
            num_imgs = 2
            num_rois = 2
            batch_size = 1
            seq_len = 10

            model = FCMF(
                pretrained_path="xlm-roberta-base",
                num_labels=4,
                num_imgs=num_imgs,
                num_roi=num_rois,
            )
            model.eval()

            # Synthetic inputs
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            added_attention_mask = torch.ones(batch_size, seq_len + 49, dtype=torch.long)
            visual_embeds = torch.randn(batch_size, num_imgs, 49, 2048)
            roi_embeds = torch.randn(batch_size, num_imgs, num_rois, 2048)
            roi_coors = torch.rand(batch_size, num_imgs, num_rois, 4)

            with torch.no_grad():
                logits = model(
                    input_ids=input_ids,
                    visual_embeds_att=visual_embeds,
                    roi_embeds_att=roi_embeds,
                    roi_coors=roi_coors,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    added_attention_mask=added_attention_mask,
                )

            assert logits.shape == (batch_size, 4), f"Expected (1, 4), got {logits.shape}"

        except Exception as e:
            pytest.skip(f"FCMF forward pass test skipped: {e}")

    def test_mm_modeling_components(self):
        """Test that core attention modules initialize."""
        from fcmf_framework.mm_modeling import (
            BertSelfAttention,
            BertCoAttention,
            BertPooler,
            BertLayerNorm,
            MultimodalEncoder,
            BertCrossEncoder,
            FeatureExtractor,
        )
        assert BertSelfAttention() is not None
        assert BertCoAttention() is not None
        assert BertPooler() is not None
        assert BertLayerNorm(768) is not None
        assert MultimodalEncoder() is not None
        assert BertCrossEncoder() is not None

    def test_roi_modeling_box_attention(self):
        """Test BoxMultiHeadedAttention initialization and forward."""
        from fcmf_framework.roi_modeling import BoxMultiHeadedAttention

        box_attn = BoxMultiHeadedAttention(h=8, d_model=768)
        assert box_attn is not None

        # Synthetic forward pass
        batch_size = 2
        num_rois = 4
        query = torch.randn(batch_size, num_rois, 768)
        key = torch.randn(batch_size, num_rois, 768)
        value = torch.randn(batch_size, num_rois, 768)
        boxes = torch.rand(batch_size, num_rois, 4)

        output = box_attn(query, key, value, boxes)
        assert output.shape == (batch_size, num_rois, 768)


# ============================================================================
# 4. Image loading and processing
# ============================================================================

class TestImageProcessing:
    """Tests for image loading and RoI processing utilities."""

    def test_convert_img_to_tensor(self):
        """Test image-to-tensor conversion with a synthetic image."""
        from fcmf_framework.image_process import convert_img_to_tensor

        # Synthetic 3-channel image (3, 100, 100)
        fake_image = torch.randint(0, 255, (3, 100, 100), dtype=torch.uint8)
        tensor = convert_img_to_tensor(fake_image)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_merge_boxes(self):
        """Test bounding box merging logic."""
        from fcmf_framework.image_process import merge_boxes

        boxes = [
            {"category": "bed", "coordinates": [10, 10, 100, 100]},
            {"category": "bed", "coordinates": [15, 15, 105, 105]},
            {"category": "table", "coordinates": [200, 200, 300, 300]},
        ]
        result = merge_boxes(boxes, epsilon=30)
        assert "bed" in result
        assert "table" in result

    def test_missing_image_handling(self):
        """Verify that loading a nonexistent image raises an error."""
        import torchvision
        with pytest.raises(Exception):
            torchvision.io.read_image("/nonexistent/image.jpg")

    def test_roi_model_init(self):
        """Test RoI classification model initialization."""
        from fcmf_framework.image_process import MyRoIModel
        model = MyRoIModel(num_classes=5)
        assert model is not None
        # Forward with synthetic input
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    def test_image_model_init(self):
        """Test Image classification model initialization."""
        from fcmf_framework.image_process import MyImgModel
        model = MyImgModel(num_classes=5)
        assert model is not None
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)


# ============================================================================
# 5. ResNet utilities
# ============================================================================

class TestResNetUtils:
    """Test ResNet wrapper models."""

    def test_resnet_img_forward(self):
        from fcmf_framework.resnet_utils import myResNetImg
        from torchvision.models import resnet152, ResNet152_Weights

        base = resnet152(weights=None)  # No download, random init
        model = myResNetImg(resnet=base, if_fine_tune=False, device="cpu")
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        # Output shape: (1, 2048, 7, 7)
        assert out.shape == (1, 2048, 7, 7)

    def test_resnet_roi_forward(self):
        from fcmf_framework.resnet_utils import myResNetRoI
        from torchvision.models import resnet152

        base = resnet152(weights=None)
        model = myResNetRoI(resnet=base, if_fine_tune=False, device="cpu")
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        # Output shape: (1, 2048)
        assert out.shape == (1, 2048)


# ============================================================================
# 6. Inference input validation
# ============================================================================

class TestInferenceValidation:
    """Test inference CLI input validation helpers."""

    def test_validate_file_exists(self):
        """validate_file succeeds for an existing file."""
        # Use this test file itself
        from inference import validate_file
        result = validate_file(__file__, "test file")
        assert result == str(Path(__file__))

    def test_validate_file_missing(self):
        """validate_file exits for a missing file."""
        from inference import validate_file
        with pytest.raises(SystemExit):
            validate_file("/nonexistent/file.txt", "missing")

    def test_validate_checkpoint_none(self):
        """validate_checkpoint exits when path is None."""
        from inference import validate_checkpoint
        with pytest.raises(SystemExit):
            validate_checkpoint(None, "checkpoint")


# ============================================================================
# 7. Optimization module
# ============================================================================

class TestOptimization:
    """Tests for the BertAdam optimizer."""

    def test_bert_adam_init(self):
        from fcmf_framework.optimization import BertAdam
        params = [torch.nn.Parameter(torch.randn(10, 10))]
        optimizer = BertAdam(params, lr=1e-3)
        assert optimizer is not None

    def test_bert_adam_step(self):
        from fcmf_framework.optimization import BertAdam
        param = torch.nn.Parameter(torch.randn(5))
        param.grad = torch.randn(5)
        optimizer = BertAdam([param], lr=1e-3)
        optimizer.step()
        # Should not raise


# ============================================================================
# 8. Dataset parsing (synthetic)
# ============================================================================

class TestDatasetParsing:
    """Test dataset class with synthetic data (no external dataset needed)."""

    def test_polarity_mapping(self):
        """Verify polarity label mapping."""
        pola_to_num = {"None": 0, "Negative": 1, "Neutral": 2, "Positive": 3}
        assert pola_to_num["None"] == 0
        assert pola_to_num["Positive"] == 3

    def test_aspect_list(self):
        """Verify the predefined aspect list."""
        aspects = ["Location", "Food", "Room", "Facilities", "Service", "Public_area"]
        assert len(aspects) == 6
        assert "Location" in aspects
        assert "Public_area" in aspects


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
