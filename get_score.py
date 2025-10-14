from models.whisper_ni_predictors import whisperMetricPredictorEncoderLayersTransformerSmall, whisperMetricPredictorEncoderLayersTransformerSmalldim
import sys
import torchaudio
import argparse
import torch

def get_score(audio_file: str, model_type: str) -> torch.Tensor:
    """
    Get a score for a given audio file and print it. 

    Args:
        audio_file (str): Path to the audio file, must be 16K sample rate and mono. 
        model_type (str): Single MOS (more accurate) or multidimensional [MOS, Noisiness, Coloration, Discontinuity and Loudness].

    Returns:
        score (torch.Tensor): either MOS score or MOS + speech dimensions 
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
         device = torch.device("mps") #for M1 Macs
    else:
        device = torch.device("cpu") #May be slow ! 

    if model_type == "single":
        model = whisperMetricPredictorEncoderLayersTransformerSmall()
        model.load_state_dict(torch.load("checkpoints/single_head_model.pt",map_location=device))
    elif model_type == "multi":
        model = whisperMetricPredictorEncoderLayersTransformerSmalldim()
        model.load_state_dict(torch.load("checkpoints/multi_head_model.pt",map_location=device))
    else:
        raise ValueError("Model type not supported")

    model.eval()
    model.to(device)
    waveform, sample_rate = torchaudio.load(audio_file)

    #check channels
    if waveform.shape[0] != 1:
        #convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Check sample rate
    if sample_rate != 16000:
        #resample
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    waveform = waveform.to(device)
    score = model(waveform)
    if model_type == "multi":
        score = score.squeeze(0)
    return score 


def get_score_multi(audio_files: list[str], model_type: str,batch_size=16) -> torch.Tensor:
    """
    Get scores for a list of audio files in a batch.

    Args:
        audio_files (list[str]): List of paths to audio files. They will be resampled to 16k and converted to mono if needed.
        model_type (str): Single MOS (more accurate) or multidimensional [MOS, Noisiness, Coloration, Discontinuity and Loudness].

    Returns:
        scores (torch.Tensor): either MOS scores or MOS + speech dimensions for all files.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # for M1 Macs
    else:
        device = torch.device("cpu")  # May be slow !

    if model_type == "single":
        model = whisperMetricPredictorEncoderLayersTransformerSmall()
        model.load_state_dict(torch.load("checkpoints/single_head_model.pt", map_location=device))
    elif model_type == "multi":
        model = whisperMetricPredictorEncoderLayersTransformerSmalldim()
        model.load_state_dict(torch.load("checkpoints/multi_head_model.pt", map_location=device))
    else:
        raise ValueError("Model type not supported")

    model.eval()
    model.to(device)

    waveforms = []
    max_len = 0
    for audio_file in audio_files:
        waveform, sample_rate = torchaudio.load(audio_file)

        # Check channels
        if waveform.shape[0] != 1:
            # Convert to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Check sample rate
        if sample_rate != 16000:
            # Resample
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
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
    padded_waveforms = torch.cat(padded_waveforms, dim=0).to(device)

    scores = []
    with torch.no_grad():
        for i in range(0, len(audio_files), batch_size):
            batch_waveforms = padded_waveforms[i:i+batch_size]
            batch_scores = model(batch_waveforms)*5
            if model_type == "multi":
                batch_scores = batch_scores.squeeze(1)
            scores.append(batch_scores.cpu())

    return torch.cat(scores, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a score for a given audio file")
    parser.add_argument("--audio_file", type=str, help="Path to the audio file")
    parser.add_argument("--audio_files", type=str, nargs='+', help="List of paths to audio files for batch processing")
    parser.add_argument("--model_type", type=str, help="Single headed MOS or multidimension [MOS,Noisiness, Coloration,Discontinuity and Loudness]", default="single")
    args = parser.parse_args()

    if args.audio_file:
        score = get_score(args.audio_file, args.model_type)
        print(f"Score for {args.audio_file}: {score}")

    if args.audio_files:
        scores = get_score_multi(args.audio_files, args.model_type)
        for audio_file, score in zip(args.audio_files, scores):
            print(f"Score for {audio_file}: {score}")
            