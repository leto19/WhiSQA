"""
Core functionality for WhiSQA speech quality assessment.
"""

import os
import torch
import torchaudio
from typing import List, Union
from pathlib import Path

from .models import (
    whisperMetricPredictorEncoderLayersTransformerSmall,
    whisperMetricPredictorEncoderLayersTransformerSmalldim
)


def _get_device():
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # for M1 Macs
    else:
        return torch.device("cpu")  # May be slow!


def _load_model(model_type: str, device: torch.device, checkpoint_dir: str = None):
    """Load the appropriate model with checkpoints."""
    if checkpoint_dir is None:
        # Try to find checkpoints relative to package
        pkg_dir = Path(__file__).parent
        checkpoint_dir = pkg_dir / "checkpoints"
        
        # Fallback to current working directory
        if not checkpoint_dir.exists():
            checkpoint_dir = Path.cwd() / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_dir)
    
    if model_type == "single":
        model = whisperMetricPredictorEncoderLayersTransformerSmall()
        checkpoint_path = checkpoint_dir / "single_head_model.pt"
    elif model_type == "multi":
        model = whisperMetricPredictorEncoderLayersTransformerSmalldim()
        checkpoint_path = checkpoint_dir / "multi_head_model.pt"
    else:
        raise ValueError(f"Model type '{model_type}' not supported. Use 'single' or 'multi'.")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    
    return model


def _preprocess_audio(audio_file: str, device: torch.device):
    """Preprocess audio file to the required format."""
    waveform, sample_rate = torchaudio.load(audio_file)

    # Convert to mono if needed
    if waveform.shape[0] != 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    return waveform.to(device)


def get_score(audio_file: str, model_type: str = "single", checkpoint_dir: str = None) -> torch.Tensor:
    """
    Get a quality score for a given audio file.

    Args:
        audio_file (str): Path to the audio file. Will be resampled to 16kHz and converted to mono if needed.
        model_type (str): "single" for MOS score or "multi" for multidimensional 
                         [MOS, Noisiness, Coloration, Discontinuity, Loudness].
        checkpoint_dir (str, optional): Directory containing model checkpoints. 
                                      If None, will search in package directory or current directory.

    Returns:
        torch.Tensor: Either MOS score (single) or 5-dimensional quality scores (multi).
    """
    device = _get_device()
    model = _load_model(model_type, device, checkpoint_dir)
    waveform = _preprocess_audio(audio_file, device)
    
    with torch.no_grad():
        score = model(waveform)
        if model_type == "multi":
            score = score.squeeze(0)
    
    return score


def get_score_multi(audio_files: List[str], model_type: str = "single", 
                   batch_size: int = 16, checkpoint_dir: str = None) -> torch.Tensor:
    """
    Get quality scores for multiple audio files in batches.

    Args:
        audio_files (List[str]): List of paths to audio files. They will be resampled to 16kHz 
                               and converted to mono if needed.
        model_type (str): "single" for MOS scores or "multi" for multidimensional scores.
        batch_size (int): Number of files to process in each batch.
        checkpoint_dir (str, optional): Directory containing model checkpoints.

    Returns:
        torch.Tensor: Quality scores for all files. Shape depends on model_type.
    """
    device = _get_device()
    model = _load_model(model_type, device, checkpoint_dir)
    
    # Preprocess all audio files
    waveforms = []
    max_len = 0
    print(f"Processing {len(audio_files)} files...")
    
    for audio_file in audio_files:
        waveform = _preprocess_audio(audio_file, "cpu")  # Keep on CPU for now
        waveforms.append(waveform)
        if waveform.shape[1] > max_len:
            max_len = waveform.shape[1]

    # Pad waveforms to the same length
    padded_waveforms = []
    for waveform in waveforms:
        pad_len = max_len - waveform.shape[1]
        if pad_len > 0:
            padded_waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            padded_waveform = waveform
        padded_waveforms.append(padded_waveform)
    
    # Stack all waveforms
    padded_waveforms = torch.cat(padded_waveforms, dim=0).to(device)

    # Process in batches
    scores = []
    with torch.no_grad():
        for i in range(0, len(audio_files), batch_size):
            batch_waveforms = padded_waveforms[i:i+batch_size]
            batch_scores = model(batch_waveforms) * 5  # Scale to 0-5 range
            if model_type == "multi":
                batch_scores = batch_scores.squeeze(1)
            scores.append(batch_scores.cpu())

    return torch.cat(scores, dim=0)