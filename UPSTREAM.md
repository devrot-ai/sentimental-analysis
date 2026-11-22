# Upstream Attribution

This project is a research fork of the **ViMACSA** (Vietnamese Multimodal Aspect-Category Sentiment Analysis) implementation.

## Original Repository

- **Repository:** [hoangquy18/Multimodal-Aspect-Category-Sentiment-Analysis](https://github.com/hoangquy18/Multimodal-Aspect-Category-Sentiment-Analysis)
- **Paper:** [New Benchmark Dataset and Fine-Grained Cross-Modal Fusion Framework for Vietnamese Multimodal Aspect-Category Sentiment Analysis](https://arxiv.org/abs/2405.00543)
- **Authors:** Nguyen, Quy Hoang; Nguyen, Minh-Van Truong; Van Nguyen, Kiet
- **Published in:** Multimedia Systems, Volume 31, Issue 1, pp. 1–28, 2025 (Springer)
- **Dataset:** ViMACSA — 4,876 text-image pairs with 14,618 fine-grained annotations (hotel domain, Vietnamese)

## Citation

```bibtex
@article{nguyen2025new,
  title={New benchmark dataset and fine-grained cross-modal fusion framework for Vietnamese multimodal aspect-category sentiment analysis},
  author={Nguyen, Quy Hoang and Nguyen, Minh-Van Truong and Van Nguyen, Kiet},
  journal={Multimedia Systems},
  volume={31},
  number={1},
  pages={1--28},
  year={2025},
  publisher={Springer}
}
```

## License Status

> **Important:** The original repository does not include an explicit license file. This fork preserves the upstream relationship via GitHub's fork mechanism and maintains full attribution. No standalone redistribution rights are assumed. All original code remains attributed to the original authors.

The file `fcmf_framework/optimization.py` contains code originally from Google AI Language Team / HuggingFace Inc., licensed under Apache License 2.0.

## What This Fork Modifies

This fork adapts the original ViMACSA implementation for research into **Multimodal Implicit Aspect-Based Sentiment Analysis**. The engineering modifications include:

- Centralized configuration system (YAML-based)
- Removal of hardcoded machine-specific paths
- Improved error handling and input validation
- Comprehensive test suite
- Rewritten documentation for reproducibility
- Optional Gradio demonstration interface

The original FCMF model architecture, training pipeline, dataset format, and core algorithms remain unchanged. All modifications are clearly identified in the commit history.

## Contact

- **Original authors:** nhq188@gmail.com
- **This fork maintained by:** Kanishk Thakur
