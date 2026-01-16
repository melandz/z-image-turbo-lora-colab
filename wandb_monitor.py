"""
W&B Monitoring System for AI-Toolkit Training
Monitors output folder and logs metrics/images to W&B in real-time
"""

import wandb
import time
import re
import os
from pathlib import Path
from PIL import Image
from datetime import datetime
from threading import Thread, Event
import json

class TrainingMonitor:
    """Monitors training folder and logs to W&B"""
    
    def __init__(self, output_dir, config, project="zimage-turbo", entity=None, run_name=None, group=None):
        self.output_dir = Path(output_dir)
        self.config = config
        self.run = None
        self.stop_event = Event()
        self.monitor_thread = None
        self.last_step = -1
        self.last_log_time = time.time()
        
        # Initialize W&B
        if run_name is None:
            run_name = f"{config.get('project_name', 'lora')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            job_type='training',
            config=self._prepare_config(config),
            tags=self._get_tags(config),
        )
        
        print(f"✓ W&B Run: {self.run.name}")
        print(f"   URL: {self.run.url}")
    
    def _prepare_config(self, config):
        """Prepare config for W&B"""
        return {
            # Model
            'model': config.get('model_path', 'Tongyi-MAI/Z-Image-Turbo'),
            'adapter_version': config.get('adapter_version', 'v1'),
            'quantization': config.get('quantization', 'qfloat8'),
            'low_vram': config.get('low_vram', True),
            
            # Network
            'network_type': config.get('network_type', 'lora'),
            'lora_rank': config.get('lora_rank', 16),
            'lora_alpha': config.get('lora_alpha', 16),
            
            # Training
            'learning_rate': config.get('learning_rate', 1e-4),
            'total_steps': config.get('total_steps', 2000),
            'batch_size': config.get('batch_size', 1),
            'optimizer': config.get('optimizer', 'adamw8bit'),
            'gradient_accumulation': config.get('gradient_accumulation', 1),
            'gradient_checkpointing': config.get('gradient_checkpointing', True),
            
            # Dataset
            'num_training_images': config.get('num_images', 0),
            'num_captions': config.get('num_captions', 0),
            'resolutions': config.get('resolutions', [512, 768, 1024]),
            
            # Sampling
            'sample_every': config.get('sample_every', 200),
            'sample_steps': config.get('sample_steps', 8),
            
            # Other
            'seed': config.get('seed', 42),
        }
    
    def _get_tags(self, config):
        """Generate tags for the run"""
        tags = ['z-image-turbo', config.get('network_type', 'lora')]
        if config.get('experiment_name'):
            tags.append(config['experiment_name'])
        return tags
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("✓ W&B monitoring started")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        log_file = self.output_dir / "training.log"
        samples_dir = self.output_dir / "samples"
        
        while not self.stop_event.is_set():
            try:
                # Parse logs for metrics
                if log_file.exists():
                    self._parse_and_log_metrics(log_file)
                
                # Check for new sample images
                if samples_dir.exists():
                    self._log_new_samples(samples_dir)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"⚠ Monitoring error: {e}")
                time.sleep(10)
    
    def _parse_and_log_metrics(self, log_file):
        """Parse training log and extract metrics"""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            for line in reversed(lines[-50:]):  # Check last 50 lines
                # Look for step info: "Step 100/2000"
                step_match = re.search(r'Step\s+(\d+)/(\d+)', line)
                if step_match:
                    current_step = int(step_match.group(1))
                    
                    if current_step > self.last_step:
                        self.last_step = current_step
                        
                        # Extract loss if present
                        loss_match = re.search(r'loss[:\s=]+([0-9.]+)', line, re.IGNORECASE)
                        if loss_match:
                            loss = float(loss_match.group(1))
                            
                            # Calculate epoch (approximate)
                            total_steps = int(step_match.group(2))
                            epoch = (current_step / total_steps) * self.config.get('max_epochs', 10)
                            
                            # Log to W&B
                            self.run.log({
                                'loss': loss,
                                'step': current_step,
                                'epoch': epoch,
                                'progress': current_step / total_steps,
                            }, step=current_step)
                        
                        # Extract learning rate if present
                        lr_match = re.search(r'lr[:\s=]+([0-9.e-]+)', line, re.IGNORECASE)
                        if lr_match:
                            lr = float(lr_match.group(1))
                            self.run.log({'learning_rate': lr}, step=current_step)
                        
        except Exception as e:
            print(f"⚠ Error parsing logs: {e}")
    
    def _log_new_samples(self, samples_dir):
        """Log newly generated sample images"""
        try:
            # Find all sample images
            sample_files = sorted(samples_dir.glob("**/*.png"))
            
            for img_path in sample_files:
                # Check if we've already logged this
                img_id = str(img_path.relative_to(self.output_dir))
                
                # Extract step from filename (e.g., "sample_000200_0.png")
                step_match = re.search(r'(\d+)', img_path.stem)
                if step_match:
                    step = int(step_match.group(1))
                    
                    # Only log if this step is new
                    if step > self.last_step - 50:  # Within recent steps
                        img = Image.open(img_path)
                        
                        # Resize for W&B (256px max)
                        img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                        
                        # Extract prompt from filename or metadata
                        prompt = self._extract_prompt(img_path)
                        
                        self.run.log({
                            f"samples/{img_path.parent.name}": wandb.Image(
                                img,
                                caption=f"Step {step}: {prompt}"
                            )
                        }, step=step)
        
        except Exception as e:
            print(f"⚠ Error logging samples: {e}")
    
    def _extract_prompt(self, img_path):
        """Try to extract prompt from filename or metadata"""
        # Try to find associated txt file
        txt_file = img_path.with_suffix('.txt')
        if txt_file.exists():
            return txt_file.read_text(encoding='utf-8').strip()[:100]
        return "sample"
    
    def log_dataset_info(self, dataset_path):
        """Log dataset as W&B artifact"""
        try:
            dataset_path = Path(dataset_path)
            artifact = wandb.Artifact(
                name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type="dataset",
                description="Training dataset"
            )
            
            # Add files
            for img_file in dataset_path.glob("*.png"):
                artifact.add_file(str(img_file))
            for txt_file in dataset_path.glob("*.txt"):
                artifact.add_file(str(txt_file))
            
            self.run.log_artifact(artifact)
            print(f"✓ Dataset logged to W&B ({len(list(dataset_path.glob('*.png')))} images)")
            
        except Exception as e:
            print(f"⚠ Error logging dataset: {e}")
    
    def log_final_model(self, model_path, metadata=None):
        """Log final trained model as artifact"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                print(f"⚠ Model not found: {model_path}")
                return
            
            artifact = wandb.Artifact(
                name=f"lora_{self.run.name}",
                type="model",
                description="Trained LoRA weights",
                metadata=metadata or {}
            )
            
            artifact.add_file(str(model_path))
            self.run.log_artifact(artifact)
            
            print(f"✓ Model logged to W&B: {model_path.name}")
            
        except Exception as e:
            print(f"⚠ Error logging model: {e}")
    
    def log_comparison_table(self, images_data):
        """Log comparison table with multiple images"""
        try:
            columns = ["step", "prompt", "image", "seed"]
            data = []
            
            for img_data in images_data:
                img = Image.open(img_data['path']) if isinstance(img_data['path'], (str, Path)) else img_data['path']
                img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                
                data.append([
                    img_data.get('step', 0),
                    img_data.get('prompt', ''),
                    wandb.Image(img),
                    img_data.get('seed', 42)
                ])
            
            table = wandb.Table(data=data, columns=columns)
            self.run.log({"sample_comparison": table})
            
            print(f"✓ Logged comparison table with {len(data)} images")
            
        except Exception as e:
            print(f"⚠ Error logging table: {e}")
    
    def finish(self, final_metrics=None):
        """Stop monitoring and finish W&B run"""
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Log final summary metrics
        if final_metrics:
            for key, value in final_metrics.items():
                self.run.summary[key] = value
        
        self.run.finish()
        print("✓ W&B run finished")


# Utility function to wrap training
def train_with_wandb(train_func, config, **wandb_kwargs):
    """
    Wrapper to run training with W&B monitoring
    
    Args:
        train_func: Function that runs training (takes output_dir as arg)
        config: Training configuration dict
        **wandb_kwargs: Additional W&B init kwargs
    """
    monitor = TrainingMonitor(
        output_dir=config.get('output_dir', '/content/output'),
        config=config,
        **wandb_kwargs
    )
    
    # Log dataset
    if 'dataset_path' in config:
        monitor.log_dataset_info(config['dataset_path'])
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Run training
        result = train_func()
        
        # Log final model
        output_dir = Path(config.get('output_dir', '/content/output'))
        for model_file in output_dir.glob("*.safetensors"):
            monitor.log_final_model(model_file)
        
        monitor.finish(final_metrics={'training_complete': 1})
        return result
        
    except Exception as e:
        monitor.finish(final_metrics={'training_failed': 1, 'error': str(e)})
        raise
