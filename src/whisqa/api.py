import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, Optional

from .models.whisper_ni_predictors import (
    whisperMetricPredictorEncoderLayersTransformerSmall,
    whisperMetricPredictorEncoderLayersTransformerSmalldim,
)
from .resources import package_file


SUPPORTED_MODEL_TYPES = {"single", "multi"}
TARGET_SAMPLE_RATE = 16000


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_type: str, device: Optional[torch.device] = None) -> torch.nn.Module:
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Model type not supported: {model_type}")

    device = device or get_device()

    if model_type == "single":
        model = whisperMetricPredictorEncoderLayersTransformerSmall()
        checkpoint_name = "single_head_model.pt"
    else:
        model = whisperMetricPredictorEncoderLayersTransformerSmalldim()
        checkpoint_name = "multi_head_model.pt"

    with package_file("checkpoints", checkpoint_name) as checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def _load_waveform(audio_file: str) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_file)

    if waveform.shape[0] != 1:
        raise ValueError("Number of input channels must be 1")
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )

    return waveform.squeeze(0)


def get_scores(audio_files: Iterable[str], model_type: str = "single") -> torch.Tensor:
    audio_files = list(audio_files)
    if not audio_files:
        raise ValueError("At least one audio file is required")

    device = get_device()
    model = load_model(model_type=model_type, device=device)
    waveforms = [_load_waveform(audio_file) for audio_file in audio_files]
    batch = pad_sequence(waveforms, batch_first=True).to(device)

    with torch.inference_mode():
        score = model(batch)

    if model_type == "multi":
        return score.squeeze(-1)
    return score.squeeze(-1)


def get_score(audio_file: str, model_type: str = "single") -> torch.Tensor:
    score = get_scores([audio_file], model_type=model_type)
    if model_type == "multi":
        return score.squeeze(0)
    return score.squeeze(0)


def batch_scores_to_dicts(scores: torch.Tensor, model_type: str) -> list[dict[str, float]]:
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    return [scores_to_dict(score, model_type) for score in scores]


def _normalize_single_score(score: torch.Tensor) -> torch.Tensor:
    if score.ndim == 0:
        return score.unsqueeze(0)
    return score


def scores_to_dict(score: torch.Tensor, model_type: str) -> dict[str, float]:
    score = _normalize_single_score(score)

    if model_type == "single":
        return {"MOS": score.item() * 5}

    return {
        "MOS": score[0].item() * 5,
        "Noisiness": score[1].item() * 5,
        "Coloration": score[2].item() * 5,
        "Discontinuity": score[3].item() * 5,
        "Loudness": score[4].item() * 5,
    }
