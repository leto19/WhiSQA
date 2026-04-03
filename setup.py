from pathlib import Path

from setuptools import find_packages, setup


setup(
    name="whisqa",
    version="0.1.0",
    description="Whisper-based speech quality assessment.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["whisqa", "whisqa.*"]),
    package_data={"whisqa": ["checkpoints/*.pt", "data/*.npz"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy==1.24.4",
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "torchinfo==1.8.0",
        "transformers==4.35.0",
        "soundfile",
    ],
    entry_points={"console_scripts": ["whisqa=whisqa.cli:main"]},
)
