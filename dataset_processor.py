"""
Dataset Preprocessing for Z-Image Turbo Training
Handles image flipping, resizing, caption management
"""

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import shutil

class DatasetProcessor:
    """Process and prepare datasets for training"""
    
    def __init__(self, output_dir='/content/dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            'source_images': 0,
            'processed_images': 0,
            'with_captions': 0,
            'without_captions': 0,
        }
    
    def process_dataset(self, source_folder, config=None):
        """
        Process a dataset folder
        
        Args:
            source_folder: Path to source images
            config: Dict with processing options:
                - flip_horizontal: bool
                - flip_vertical: bool
                - repeats: int
                - resolutions: list (not used for processing, just metadata)
        """
        config = config or {}
        source = Path(source_folder)
        
        if not source.exists():
            print(f"⚠ Source folder not found: {source}")
            return self.stats
        
        # Get all images
        image_files = self._find_images(source)
        
        if not image_files:
            print(f"⚠ No images found in {source}")
            return self.stats
        
        print(f"📦 Processing {len(image_files)} images...")
        self.stats['source_images'] = len(image_files)
        
        # Process each image
        for img_path in tqdm(image_files, desc="Processing"):
            self._process_image(img_path, config)
        
        return self.stats
    
    def _find_images(self, folder):
        """Find all image files"""
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.PNG', '*.JPG', '*.JPEG']:
            images.extend(folder.glob(ext))
        return images
    
    def _process_image(self, img_path, config):
        """Process a single image with all variants"""
        try:
            base_name = img_path.stem
            img = Image.open(img_path).convert('RGB')
            
            # Original
            variants = [(img, f"{base_name}.png")]
            
            # Horizontal flip
            if config.get('flip_horizontal', False):
                h_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
                variants.append((h_flip, f"{base_name}_hflip.png"))
            
            # Vertical flip
            if config.get('flip_vertical', False):
                v_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
                variants.append((v_flip, f"{base_name}_vflip.png"))
            
            # Both flips
            if config.get('flip_horizontal') and config.get('flip_vertical'):
                hv_flip = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
                variants.append((hv_flip, f"{base_name}_hvflip.png"))
            
            # Save all variants
            variant_names = []
            for img_variant, filename in variants:
                output_path = self.output_dir / filename
                img_variant.save(output_path, 'PNG')
                variant_names.append(filename)
                self.stats['processed_images'] += 1
            
            # Handle captions
            caption_file = img_path.with_suffix('.txt')
            if caption_file.exists():
                caption = caption_file.read_text(encoding='utf-8').strip()
                for variant_name in variant_names:
                    txt_path = self.output_dir / variant_name.replace('.png', '.txt')
                    txt_path.write_text(caption, encoding='utf-8')
                    self.stats['with_captions'] += 1
            else:
                self.stats['without_captions'] += len(variant_names)
            
            # Repeats (if needed)
            repeats = config.get('repeats', 1)
            if repeats > 1:
                for i in range(1, repeats):
                    for img_variant, filename in variants:
                        repeat_name = filename.replace('.png', f'_repeat{i}.png')
                        output_path = self.output_dir / repeat_name
                        img_variant.save(output_path, 'PNG')
                        
                        # Copy caption for repeat
                        if caption_file.exists():
                            txt_path = self.output_dir / repeat_name.replace('.png', '.txt')
                            txt_path.write_text(caption, encoding='utf-8')
                        
                        self.stats['processed_images'] += 1
        
        except Exception as e:
            print(f"⚠ Error processing {img_path.name}: {e}")
    
    def process_multiple_datasets(self, dataset_configs):
        """
        Process multiple dataset folders
        
        Args:
            dataset_configs: List of dicts with:
                - name: Dataset name
                - source_folder: Path to images
                - flip_horizontal: bool
                - flip_vertical: bool
                - repeats: int
                - resolutions: list
        """
        # Clear output directory
        if self.output_dir.exists():
            for f in self.output_dir.glob('*'):
                f.unlink()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        total_stats = {
            'source_images': 0,
            'processed_images': 0,
            'with_captions': 0,
            'without_captions': 0,
        }
        
        for ds_config in dataset_configs:
            print(f"\n📦 Processing: {ds_config.get('name', 'unnamed')}")
            stats = self.process_dataset(
                source_folder=ds_config['source_folder'],
                config=ds_config
            )
            
            # Aggregate stats
            for key in total_stats:
                total_stats[key] += stats[key]
        
        # Print summary
        self._print_summary(total_stats)
        return total_stats
    
    def _print_summary(self, stats):
        """Print processing summary"""
        print("\n" + "=" * 60)
        print("DATASET PROCESSING COMPLETE")
        print("=" * 60)
        print(f"  Source images: {stats['source_images']}")
        print(f"  Processed images: {stats['processed_images']}")
        print(f"  With captions: {stats['with_captions']}")
        print(f"  Without captions: {stats['without_captions']}")
        print(f"  Output: {self.output_dir}")
        print("=" * 60)
    
    def get_stats(self):
        """Get current processing stats"""
        return self.stats.copy()
    
    def validate_dataset(self):
        """Validate the processed dataset"""
        images = list(self.output_dir.glob('*.png'))
        captions = list(self.output_dir.glob('*.txt'))
        
        print("\n📊 Dataset Validation:")
        print(f"  Total images: {len(images)}")
        print(f"  Total captions: {len(captions)}")
        
        # Check for orphaned files
        image_stems = {img.stem for img in images}
        caption_stems = {txt.stem for txt in captions}
        
        images_without_captions = image_stems - caption_stems
        captions_without_images = caption_stems - image_stems
        
        if images_without_captions:
            print(f"  ⚠ Images without captions: {len(images_without_captions)}")
        if captions_without_images:
            print(f"  ⚠ Captions without images: {len(captions_without_images)}")
        
        if not images_without_captions and not captions_without_images:
            print("  ✓ All images have matching captions!")
        
        return {
            'total_images': len(images),
            'total_captions': len(captions),
            'images_without_captions': len(images_without_captions),
            'captions_without_images': len(captions_without_images),
        }


def create_dataset_config(name, source_folder, **kwargs):
    """
    Helper to create dataset config dict
    
    Args:
        name: Dataset name
        source_folder: Path to images
        **kwargs: Additional config (flip_horizontal, flip_vertical, repeats, resolutions)
    """
    config = {
        'name': name,
        'source_folder': source_folder,
        'flip_horizontal': kwargs.get('flip_horizontal', False),
        'flip_vertical': kwargs.get('flip_vertical', False),
        'repeats': kwargs.get('repeats', 1),
        'resolutions': kwargs.get('resolutions', [512, 768, 1024]),
    }
    return config
