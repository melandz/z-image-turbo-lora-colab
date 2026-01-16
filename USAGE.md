# Usage Guide

## GitHub Setup (One-Time)

1. **Create GitHub Repository**
   - Go to https://github.com/new
   - Name: `zimage-turbo-training`
   - Make it public (or private if you prefer)
   - Don't add README (we have one)

2. **Upload Files**
   ```bash
   # On your local machine
   git clone https://github.com/YOUR_USERNAME/zimage-turbo-training.git
   cd zimage-turbo-training
   
   # Copy all files you downloaded
   # Then push
   git add .
   git commit -m "Initial commit"
   git push
   ```

3. **Done!** Now you can use it in any Colab notebook

## Colab Usage

### Method 1: Simple Import (Recommended)

```python
# In your Colab notebook
!git clone https://github.com/YOUR_USERNAME/zimage-turbo-training.git /content/zimage-helpers

import sys
sys.path.insert(0, '/content/zimage-helpers')

from wandb_monitor import TrainingMonitor
from dataset_processor import DatasetProcessor
from xy_grid import XYGridGenerator
from upload_utils import UploadManager
```

### Method 2: Use the Complete Notebook

Just open `Z_Image_Turbo_Complete.ipynb` directly in Colab:
1. Go to: https://colab.research.google.com
2. File → Open Notebook → GitHub tab
3. Enter: `YOUR_USERNAME/zimage-turbo-training`
4. Select: `Z_Image_Turbo_Complete.ipynb`
5. Update `GITHUB_REPO` variable in cell 2
6. Run all cells!

## File Locations in Colab

When you run the notebook, files will be organized as:

```
/content/
├── zimage-helpers/           # Cloned GitHub repo
│   ├── wandb_monitor.py
│   ├── dataset_processor.py
│   ├── xy_grid.py
│   └── upload_utils.py
├── ai-toolkit/               # Training framework
├── raw_dataset/              # Your uploaded images
├── dataset/                  # Processed dataset
├── output/                   # Training outputs
│   ├── *.safetensors        # Model checkpoints
│   └── samples/             # Generated samples
└── *.png                     # Comparison grids
```

## Quick Reference

### Dataset Processing

```python
from dataset_processor import DatasetProcessor, create_dataset_config

processor = DatasetProcessor(output_dir='/content/dataset')

config = create_dataset_config(
    name='main',
    source_folder='/content/raw_dataset',
    flip_horizontal=True,
    repeats=2,
    resolutions=[512, 768, 1024]
)

stats = processor.process_multiple_datasets([config])
```

### W&B Monitoring

```python
from wandb_monitor import TrainingMonitor

monitor = TrainingMonitor(
    output_dir='/content/output',
    config={...},
    project='my-project',
    run_name='experiment_1'
)

monitor.start_monitoring()
# ... run training ...
monitor.finish()
```

### X/Y Grids

```python
from xy_grid import XYGridGenerator

gen = XYGridGenerator()

gen.compare_models(
    model_paths=['model1.safetensors', 'model2.safetensors'],
    prompts=['a cat', 'a dog'],
    output_path='comparison.png'
)
```

### Uploads

```python
from upload_utils import quick_backup

quick_backup(
    output_dir='/content/output',
    project_name='my_lora'
)
```

## Troubleshooting

### "Module not found"
→ Make sure `/content/zimage-helpers` is in `sys.path`

### "Repository not found"  
→ Check your GitHub repo is public or you're authenticated

### Import errors
→ Run `!pip install -r /content/zimage-helpers/requirements.txt`
