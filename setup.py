from setuptools import setup, find_packages

setup(
    name="zimage-turbo-training",
    version="1.0.0",
    description="Complete training system for Z-Image Turbo LoRAs with W&B integration",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "wandb>=0.16.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "huggingface-hub>=0.20.0",
        "torch>=2.0.0",
        "diffusers>=0.25.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
    ],
    python_requires=">=3.8",
)
