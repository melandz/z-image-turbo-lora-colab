"""
X/Y Grid Comparison Generator
Creates comparison grids for evaluating different models, prompts, seeds
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import torch
from diffusers import ZImagePipeline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm.auto import tqdm

class XYGridGenerator:
    """Generate X/Y comparison grids"""
    
    def __init__(self, base_model="Tongyi-MAI/Z-Image-Turbo", device="cuda"):
        self.base_model = base_model
        self.device = device
        self.pipe = None
    
    def load_pipeline(self):
        """Load the base pipeline"""
        if self.pipe is None:
            print("Loading Z-Image Turbo pipeline...")
            self.pipe = ZImagePipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            print("✓ Pipeline loaded")
    
    def generate_grid(self, x_axis, y_axis, output_path, **gen_kwargs):
        """
        Generate X/Y grid
        
        Args:
            x_axis: Dict with 'type' and 'values'
                - type: 'prompt', 'seed', 'model', 'steps'
                - values: List of values for X axis
            y_axis: Dict with 'type' and 'values'
                - type: 'prompt', 'seed', 'model', 'steps'
                - values: List of values for Y axis
            output_path: Where to save the grid
            **gen_kwargs: Additional generation kwargs (width, height, guidance_scale, etc.)
        """
        self.load_pipeline()
        
        x_type = x_axis['type']
        y_type = y_axis['type']
        x_values = x_axis['values']
        y_values = y_axis['values']
        
        print(f"Generating {len(y_values)}x{len(x_values)} grid ({y_type} vs {x_type})")
        
        # Generate all images
        images = []
        total = len(x_values) * len(y_values)
        pbar = tqdm(total=total, desc="Generating")
        
        for y_val in y_values:
            row_images = []
            for x_val in x_values:
                # Prepare generation params
                params = self._prepare_params(x_type, x_val, y_type, y_val, gen_kwargs)
                
                # Generate image
                img = self._generate_image(params)
                row_images.append(img)
                pbar.update(1)
            
            images.append(row_images)
        
        pbar.close()
        
        # Create grid
        grid = self._create_grid(
            images,
            x_labels=[self._format_label(x_type, v) for v in x_values],
            y_labels=[self._format_label(y_type, v) for v in y_values],
            x_title=x_type.capitalize(),
            y_title=y_type.capitalize()
        )
        
        # Save
        grid.save(output_path, 'PNG')
        print(f"✓ Grid saved: {output_path}")
        
        return grid
    
    def _prepare_params(self, x_type, x_val, y_type, y_val, base_kwargs):
        """Prepare generation parameters from axis values"""
        params = base_kwargs.copy()
        
        # Apply X axis value
        if x_type == 'prompt':
            params['prompt'] = x_val
        elif x_type == 'seed':
            params['generator'] = torch.Generator(self.device).manual_seed(x_val)
        elif x_type == 'steps':
            params['num_inference_steps'] = x_val
        elif x_type == 'model':
            params['lora_path'] = x_val
        
        # Apply Y axis value
        if y_type == 'prompt':
            params['prompt'] = y_val
        elif y_type == 'seed':
            params['generator'] = torch.Generator(self.device).manual_seed(y_val)
        elif y_type == 'steps':
            params['num_inference_steps'] = y_val
        elif y_type == 'model':
            params['lora_path'] = y_val
        
        # Ensure we have required params
        if 'prompt' not in params:
            params['prompt'] = "a beautiful landscape"
        if 'num_inference_steps' not in params:
            params['num_inference_steps'] = 8
        if 'guidance_scale' not in params:
            params['guidance_scale'] = 0
        if 'width' not in params:
            params['width'] = 512
        if 'height' not in params:
            params['height'] = 512
        
        return params
    
    def _generate_image(self, params):
        """Generate a single image"""
        # Handle LoRA loading if needed
        lora_path = params.pop('lora_path', None)
        if lora_path:
            self.pipe.load_lora_weights(lora_path)
        
        # Generate
        result = self.pipe(**params)
        image = result.images[0]
        
        # Unload LoRA if we loaded one
        if lora_path:
            self.pipe.unload_lora_weights()
        
        return image
    
    def _create_grid(self, images, x_labels, y_labels, x_title, y_title):
        """Create the final grid image"""
        rows = len(images)
        cols = len(images[0])
        
        # Get image size from first image
        img_width, img_height = images[0][0].size
        
        # Label dimensions
        label_height = 40
        label_width = 120
        margin = 10
        
        # Calculate grid size
        grid_width = label_width + (img_width + margin) * cols + margin
        grid_height = label_height + (img_height + margin) * rows + margin
        
        # Create canvas
        grid = Image.new('RGB', (grid_width, grid_height), 'white')
        draw = ImageDraw.Draw(grid)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
            title_font = font
        
        # Draw X axis title and labels
        draw.text((grid_width // 2, 5), x_title, fill='black', font=title_font, anchor='mm')
        for i, label in enumerate(x_labels):
            x = label_width + (img_width + margin) * i + img_width // 2
            draw.text((x, label_height - 5), label, fill='black', font=font, anchor='mm')
        
        # Draw Y axis title
        # (Vertical text is complex, so we'll skip for now)
        draw.text((5, grid_height // 2), y_title, fill='black', font=title_font, anchor='lm')
        
        # Draw Y labels and images
        for row_idx, row_images in enumerate(images):
            y_pos = label_height + (img_height + margin) * row_idx + margin
            
            # Y label
            label = y_labels[row_idx]
            draw.text((label_width - 5, y_pos + img_height // 2), label, fill='black', font=font, anchor='rm')
            
            # Images
            for col_idx, img in enumerate(row_images):
                x_pos = label_width + (img_width + margin) * col_idx + margin
                grid.paste(img, (x_pos, y_pos))
        
        return grid
    
    def _format_label(self, label_type, value):
        """Format label for display"""
        if label_type == 'prompt':
            return value[:20] + "..." if len(value) > 20 else value
        elif label_type == 'seed':
            return f"seed:{value}"
        elif label_type == 'steps':
            return f"{value}s"
        elif label_type == 'model':
            return Path(value).stem[:15]
        return str(value)
    
    def compare_models(self, model_paths, prompts, output_path, **kwargs):
        """
        Quick helper: Compare multiple models with same prompts
        
        Args:
            model_paths: List of LoRA paths
            prompts: List of prompts to test
            output_path: Output path for grid
        """
        return self.generate_grid(
            x_axis={'type': 'model', 'values': model_paths},
            y_axis={'type': 'prompt', 'values': prompts},
            output_path=output_path,
            **kwargs
        )
    
    def compare_seeds(self, prompt, seeds, output_path, **kwargs):
        """
        Quick helper: Compare different seeds for same prompt
        
        Args:
            prompt: Prompt to use
            seeds: List of seeds
            output_path: Output path
        """
        return self.generate_grid(
            x_axis={'type': 'seed', 'values': seeds},
            y_axis={'type': 'prompt', 'values': [prompt]},
            output_path=output_path,
            **kwargs
        )
    
    def cleanup(self):
        """Free GPU memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            print("✓ Pipeline unloaded")


def create_comparison_grid_simple(image_paths, labels, output_path, grid_size=(3, 3)):
    """
    Simple grid from existing images
    
    Args:
        image_paths: List of image paths
        labels: List of labels (same length as image_paths)
        output_path: Where to save
        grid_size: (rows, cols)
    """
    rows, cols = grid_size
    images = [Image.open(p).convert('RGB') for p in image_paths[:rows*cols]]
    
    # Resize all to same size
    target_size = (512, 512)
    images = [img.resize(target_size, Image.Resampling.LANCZOS) for img in images]
    
    # Create grid
    margin = 10
    label_height = 30
    
    grid_width = cols * (target_size[0] + margin) + margin
    grid_height = rows * (target_size[1] + label_height + margin) + margin
    
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        
        x = margin + col * (target_size[0] + margin)
        y = margin + row * (target_size[1] + label_height + margin)
        
        # Paste image
        grid.paste(img, (x, y))
        
        # Draw label
        label_y = y + target_size[1] + 5
        draw.text((x + target_size[0] // 2, label_y), label, fill='black', font=font, anchor='mt')
    
    grid.save(output_path, 'PNG')
    print(f"✓ Grid saved: {output_path}")
    return grid
