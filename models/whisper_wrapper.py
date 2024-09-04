from torch import Tensor, nn
import torch
import torch.nn.functional as F
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperForConditionalGeneration
from functools import lru_cache
from typing import Optional, Union
import numpy as np


SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
#print("N_SAMPLES: ",N_SAMPLES)
N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = N_MELS,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed("mel_filters.npz",mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80))
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load("models/mel_filters.npz",allow_pickle=True) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)



def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array







class WhisperWrapper_full(nn.Module):
    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None, num_layers = 12, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # using layer = -1 returns all layers in form (1, time, feat_dim, layers)
        # otherwise single layer in form (1, time, feat_dim)

        self.num_layers = num_layers
        self.use_feat_extractor = use_feat_extractor
        if layer is None:
            self.layer = 12
        else:
            self.layer = layer

        # if use_feat_extractor:
        #     self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        if pretrained_model is None:
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(pretrained_model)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
    def forward(self, data):

        if self.use_feat_extractor:
            #print(data.shape)
            #print(type(data))
            data = log_mel_spectrogram(data,padding=N_SAMPLES)
            data = pad_or_trim(data, length=3000).to(self.device)
            #print("feature shape: ",data.shape)
            #print("requires grad: ",data.requires_grad)

        outputs = self.model.generate(
            input_features = data,
            output_hidden_states = True,
            return_dict_in_generate = True
        )
        #print(outputs.sequences)
        #print(outputs.decoder_hidden_states[0][0].shape)
        if self.layer == -1:
            decoder_hidden = []
            for layer in range(self.num_layers):
                hidden = torch.stack([outputs.decoder_hidden_states[word][layer][:][:] for word in range(len(outputs.decoder_hidden_states))])
                #hidden has dim ('word', batch, layer,feat_dim)
                hidden = hidden.permute(1,0,3,2)
                #hidden has dim (batch, 'word', feat_dim, layer)
                #print(layer,hidden.shape)
                decoder_hidden.append(hidden)
            decoder_hidden = torch.stack(decoder_hidden, dim = -1).squeeze(3)
            #print("decoder_hidden size: ",decoder_hidden.size())
        elif self.layer == None:
            decoder_hidden = torch.stack([outputs.decoder_hidden_states[word][self.num_layers-1][0][0] for word in range(len(outputs.decoder_hidden_states))])
            decoder_hidden = decoder_hidden.unsqueeze(0)
        else:
            decoder_hidden = torch.stack([outputs.decoder_hidden_states[word][self.layer][0][0] for word in range(len(outputs.decoder_hidden_states))])
            decoder_hidden = decoder_hidden.unsqueeze(0)
        #print(f"decoder_hidden size: {decoder_hidden.size()}")
        # print(decoder_hidden.size())
        #input(">>>")
        return decoder_hidden


class WhisperWrapper_encoder(nn.Module):
    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_feat_extractor = use_feat_extractor
        self.layer = layer

        if not use_feat_extractor:
              self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        if pretrained_model is None:
            model = WhisperModel.from_pretrained("openai/whisper-small")
        else:
            model = WhisperModel.from_pretrained(pretrained_model)
        self.model = model.encoder
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, data):

        if self.use_feat_extractor:
            #print(data.shape)
            data_padded = pad_or_trim(data, length=N_SAMPLES).to(self.device)
            #print("data padded shape: ",data_padded.shape)
            data_feats = log_mel_spectrogram(data_padded)

            #print("feature shape after log mel: ",data_feats.shape)
        else:
            #print("data shape: ",data.shape)
            d_list = []
            for d in data:
                d_list.append(d.to('cpu').tolist())
            data = self.feature_extractor(d_list, sampling_rate = 16000, return_tensors = 'pt')
            #print(data)
            data_feats = data.input_features.to(self.device)
            #print("data shape after",data_feats.shape)
        if self.layer is None:
            data = self.model(
                input_features = data_feats, 
                return_dict = True
            )
            #print(data)
            data = data[0]
            #print(data.shape)
        elif self.layer == -1:
            data = self.model(
                    input_features = data_feats, 
                    return_dict = True,
                    output_hidden_states = True
                )
            #print(data.hidden_states[0].shape)
            layers = []
            for layer in range(len(data.hidden_states)):
                
                layers.append(data.hidden_states[layer])
            data = torch.stack(layers, dim = -1)
            #print(data.shape)
        else:
            data = self.model(
                input_features = data_feats, 
                return_dict = True,
                output_hidden_states = True
            )
            data = data.hidden_states[self.layer]

        return data
    
class WhisperWrapper_encoder_debug(nn.Module):
    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_feat_extractor = use_feat_extractor
        self.layer = layer

        if not use_feat_extractor:
              self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        if pretrained_model is None:
            model = WhisperModel.from_pretrained("openai/whisper-small")
        else:
            model = WhisperModel.from_pretrained(pretrained_model)

        self.model = model.encoder
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, data):

        if self.use_feat_extractor:
            print(data.shape)
            data_padded = pad_or_trim(data, length=N_SAMPLES).to(self.device)
            print("data padded shape: ",data_padded.shape)
            data_feats = log_mel_spectrogram(data_padded)

            print("feature shape after log mel: ",data_feats.shape)
        else:
            print("data shape: ",data.shape)
            d_list = []
            for d in data:
                d_list.append(d.to('cpu').tolist())
            data = self.feature_extractor(d_list, sampling_rate = 16000, return_tensors = 'pt')
            #print(data)
            data_feats = data.input_features.to(self.device)
            print("data shape after",data_feats.shape)
        if self.layer is None:
            data = self.model(
                input_features = data_feats, 
                return_dict = True
            )
            #print(data)
            data = data[0]
            print(data.shape)
        elif self.layer == -1:
            data = self.model(
                    input_features = data_feats, 
                    return_dict = True,
                    output_hidden_states = True
                )
            #print(data.hidden_states[0].shape)
            layers = []
            for layer in range(len(data.hidden_states)):
                
                layers.append(data.hidden_states[layer])
            data = torch.stack(layers, dim = -1)
            #print(data.shape)
        else:
            data = self.model(
                input_features = data_feats, 
                return_dict = True,
                output_hidden_states = True
            )
            data = data.hidden_states[self.layer]

        return data, data_feats

