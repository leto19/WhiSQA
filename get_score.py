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
        raise ValueError("Number of input channels must be 1")

    # Check sample rate
    if sample_rate != 16000:
        raise ValueError("Sample rate must be 16000")

    waveform = waveform.to(device)
    score = model(waveform)
    if model_type == "multi":
        score = score.squeeze(0)
    return score 

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a score for a given audio file")
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    parser.add_argument("--model_type", type=str, help="Single headed MOS or multidimension [MOS,Noisiness, Coloration,Discontinuity and Loudness]", default="single")
    args = parser.parse_args()


    score = get_score(args.audio_file, args.model_type)
    print(score.shape)
    if args.model_type == "single":
        print("MOS", score.item() * 5)
    else:
        mos = score[0].item() * 5
        noisiness = score[1].item() * 5
        coloration = score[2].item() * 5
        discontinuity = score[3].item() * 5
        loudness = score[4].item() * 5
        print("MOS", mos)
        print("Noisiness", noisiness)
        print("Coloration", coloration)
        print("Discontinuity", discontinuity)
        print("Loudness", loudness)

    sys.exit(0)
