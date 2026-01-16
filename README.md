# Z-Image Turbo LoRA Training

Complete training system for Z-Image Turbo LoRAs with full W&B integration.

## Quick Start (Google Colab)

```python
# 1. Clone repo
!git clone https://github.com/YOUR_USERNAME/zimage-turbo-training.git
%cd zimage-turbo-training

# 2. Install
!pip install -e .

# 3. Use in your notebook
from wandb_monitor import TrainingMonitor
from dataset_processor import DatasetProcessor
# etc.
```

## Installation Options

### Option 1: Direct Import (Recommended for Colab)

```bash
!git clone https://github.com/YOUR_USERNAME/zimage-turbo-training.git /content/zimage-helpers
import sys
sys.path.insert(0, '/content/zimage-helpers')
```

### Option 2: Pip Install

```bash
!pip install git+https://github.com/YOUR_USERNAME/zimage-turbo-training.git
```

### Option 3: Download Individual Files

```python
BASE_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/zimage-turbo-training/main"

!wget {BASE_URL}/wandb_monitor.py
!wget {BASE_URL}/dataset_processor.py
!wget {BASE_URL}/xy_grid.py
!wget {BASE_URL}/upload_utils.py
```

## Features

- ✅ W&B integration with real-time monitoring
- ✅ Dataset preprocessing (flip, resize, validate)
- ✅ Multi-experiment ablation studies
- ✅ X/Y comparison grids
- ✅ Google Drive & HuggingFace uploads

## Usage

See the included notebook: `Z_Image_Turbo_Complete.ipynb`

## Documentation

Full documentation in [USAGE.md](USAGE.md)

## License

MIT License - see [LICENSE](LICENSE)
