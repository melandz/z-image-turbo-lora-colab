"""
Upload Utilities for Google Drive and HuggingFace
"""

from pathlib import Path
import shutil
import tarfile
from huggingface_hub import HfApi, create_repo
from tqdm.auto import tqdm
import os

class UploadManager:
    """Manage uploads to Google Drive and HuggingFace"""
    
    def __init__(self, project_name="my_lora"):
        self.project_name = project_name
        self.hf_api = None
    
    def upload_to_drive(self, source_dir, drive_folder=None, compress=True):
        """
        Upload files to Google Drive
        
        Args:
            source_dir: Directory to upload
            drive_folder: Target folder in Drive (creates if not exists)
            compress: Whether to compress before uploading
        """
        try:
            from google.colab import drive
            
            # Ensure Drive is mounted
            if not Path('/content/drive/MyDrive').exists():
                drive.mount('/content/drive')
            
            # Set target folder
            if drive_folder is None:
                drive_folder = f'/content/drive/MyDrive/ZImage_Training/{self.project_name}'
            
            target_dir = Path(drive_folder)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            source = Path(source_dir)
            
            if compress:
                # Create tar.gz
                archive_name = f"{self.project_name}_{Path(source).name}.tar.gz"
                archive_path = Path('/content') / archive_name
                
                print(f"📦 Compressing {source}...")
                with tarfile.open(archive_path, 'w:gz') as tar:
                    for file in tqdm(list(source.rglob('*')), desc="Compressing"):
                        if file.is_file():
                            tar.add(file, arcname=file.relative_to(source.parent))
                
                # Copy to Drive
                print(f"☁️ Uploading to Google Drive...")
                target_file = target_dir / archive_name
                shutil.copy(archive_path, target_file)
                
                print(f"✓ Uploaded to: {target_file}")
                print(f"  Size: {target_file.stat().st_size / 1024 / 1024:.1f} MB")
                
                # Clean up
                archive_path.unlink()
                
            else:
                # Copy directory as-is
                print(f"☁️ Uploading to Google Drive...")
                target = target_dir / source.name
                
                if target.exists():
                    shutil.rmtree(target)
                
                shutil.copytree(source, target)
                print(f"✓ Uploaded to: {target}")
            
            return str(target_dir)
            
        except ImportError:
            print("❌ Not running in Google Colab, cannot upload to Drive")
            return None
        except Exception as e:
            print(f"❌ Error uploading to Drive: {e}")
            return None
    
    def upload_to_huggingface(self, model_path, repo_name=None, private=True, token=None):
        """
        Upload model to HuggingFace Hub
        
        Args:
            model_path: Path to .safetensors file
            repo_name: HF repo name (default: username/project_name)
            private: Whether to make repo private
            token: HF token (uses env var if not provided)
        """
        try:
            # Get token
            if token is None:
                token = os.environ.get('HF_TOKEN')
            
            if not token:
                print("❌ No HuggingFace token found")
                return None
            
            # Initialize API
            if self.hf_api is None:
                self.hf_api = HfApi(token=token)
            
            # Get username
            user_info = self.hf_api.whoami()
            username = user_info['name']
            
            # Create repo name
            if repo_name is None:
                repo_name = f"{username}/{self.project_name}"
            elif '/' not in repo_name:
                repo_name = f"{username}/{repo_name}"
            
            print(f"📤 Uploading to HuggingFace: {repo_name}")
            
            # Create repo if it doesn't exist
            try:
                create_repo(
                    repo_name,
                    private=private,
                    repo_type="model",
                    exist_ok=True,
                    token=token
                )
            except Exception as e:
                print(f"⚠ Repo creation warning: {e}")
            
            # Upload file
            model_path = Path(model_path)
            if not model_path.exists():
                print(f"❌ Model file not found: {model_path}")
                return None
            
            print(f"  Uploading: {model_path.name}")
            self.hf_api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=repo_name,
                repo_type="model",
                token=token
            )
            
            # Create README if not exists
            readme_content = self._generate_model_card(model_path.name)
            self.hf_api.upload_file(
                path_or_fileobj=readme_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=repo_name,
                repo_type="model",
                token=token
            )
            
            url = f"https://huggingface.co/{repo_name}"
            print(f"✓ Uploaded to: {url}")
            return url
            
        except Exception as e:
            print(f"❌ Error uploading to HuggingFace: {e}")
            return None
    
    def upload_dataset_to_huggingface(self, dataset_dir, repo_name=None, private=False, token=None):
        """
        Upload dataset to HuggingFace Hub
        
        Args:
            dataset_dir: Path to dataset folder
            repo_name: HF repo name
            private: Whether to make repo private
            token: HF token
        """
        try:
            if token is None:
                token = os.environ.get('HF_TOKEN')
            
            if not token:
                print("❌ No HuggingFace token found")
                return None
            
            if self.hf_api is None:
                self.hf_api = HfApi(token=token)
            
            user_info = self.hf_api.whoami()
            username = user_info['name']
            
            if repo_name is None:
                repo_name = f"{username}/{self.project_name}_dataset"
            elif '/' not in repo_name:
                repo_name = f"{username}/{repo_name}"
            
            print(f"📤 Uploading dataset to HuggingFace: {repo_name}")
            
            # Create repo
            try:
                create_repo(
                    repo_name,
                    private=private,
                    repo_type="dataset",
                    exist_ok=True,
                    token=token
                )
            except Exception as e:
                print(f"⚠ Repo creation warning: {e}")
            
            # Upload folder
            dataset_dir = Path(dataset_dir)
            self.hf_api.upload_folder(
                folder_path=str(dataset_dir),
                repo_id=repo_name,
                repo_type="dataset",
                token=token
            )
            
            url = f"https://huggingface.co/datasets/{repo_name}"
            print(f"✓ Dataset uploaded to: {url}")
            return url
            
        except Exception as e:
            print(f"❌ Error uploading dataset: {e}")
            return None
    
    def _generate_model_card(self, model_filename):
        """Generate README.md for HuggingFace"""
        return f"""---
license: apache-2.0
tags:
- text-to-image
- lora
- z-image-turbo
- diffusers
base_model: Tongyi-MAI/Z-Image-Turbo
---

# {self.project_name}

LoRA trained on Z-Image Turbo model.

## Usage

```python
from diffusers import ZImagePipeline
import torch

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16
).to("cuda")

# Load LoRA
pipe.load_lora_weights("{model_filename}")

# Generate
image = pipe(
    prompt="your prompt here",
    num_inference_steps=8,
    guidance_scale=0.0
).images[0]
```

## Training Details

- Base Model: Z-Image Turbo
- Training Framework: Ostris AI-Toolkit
- Model File: {model_filename}

## Recommended Settings

- Inference Steps: 8
- Guidance Scale: 0.0
- Resolution: 1024x1024
"""


def quick_backup(output_dir, project_name="lora_backup"):
    """
    Quick helper to backup outputs to both Drive and HF
    
    Args:
        output_dir: Directory with training outputs
        project_name: Name for the backup
    """
    manager = UploadManager(project_name)
    
    print("=" * 60)
    print("BACKING UP TRAINING OUTPUTS")
    print("=" * 60)
    
    # Upload to Drive
    print("\n1. Google Drive Upload")
    drive_path = manager.upload_to_drive(output_dir, compress=True)
    
    # Upload models to HF
    print("\n2. HuggingFace Upload")
    output_path = Path(output_dir)
    model_files = list(output_path.glob("*.safetensors"))
    
    if model_files:
        for model_file in model_files:
            if "final" in model_file.name or model_file == model_files[-1]:
                manager.upload_to_huggingface(model_file)
    else:
        print("⚠ No .safetensors files found")
    
    print("\n" + "=" * 60)
    print("BACKUP COMPLETE")
    print("=" * 60)
