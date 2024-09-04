import math
from torch import Tensor, nn
import torch

try:
    from transformer_config import Config
except:
    from models.transformer_config import Config

class TransformerWrapper(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        # Normalization.
        self.norm = nn.BatchNorm1d(config.dim_input)

        # Position encoding.
        # if config.xlsr_name == "hubert_encoder" or config.xlsr_name == "hubert_full" or config.xlsr_name == "whisper_full":
        #     self.position_encoding = PositionalEncodingVariable(config)
        # else:
        #     self.position_encoding = PositionalEncoding(config)
        self.position_encoding = PositionalEncoding(config)

        # Down-projection to transformer dim.
        self.linear_proj = nn.Linear(
            in_features=config.dim_input,
            out_features=config.dim_transformer,
        )
        self.linear_proj_drop = nn.Dropout(config.dropout)

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_transformer,
            dim_feedforward=config.dim_transformer*2,
            nhead=config.nhead_transformer,
            batch_first=True,
            dropout=config.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.nlayers_transformer,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:

        # Normalization.
        # Transform from (N, L, C) to (N, C, L) and back.
        x = self.norm(x.transpose(-1,-2)).transpose(-1,-2)

        # Linear projection down to transformer dim.
        x = self.linear_proj(x)
        x = self.linear_proj_drop(x)

        # Position encoding + transformer.
        x = self.position_encoding(x)
        x = self.transformer_encoder(x, mask)

        x = self.dropout(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        d_model: int = config.dim_transformer
        seq_len: int = config.feat_seq_len
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(2*seq_len) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe.expand(x.shape)
        return x

class PositionalEncodingVariable(nn.Module):
    """
    Positional encoding module for variable-length sequences.

    Args:
        config (Config): Configuration object containing model parameters.

    Attributes:
        pe (Tensor): Positional encoding tensor.

    """

    def __init__(self, config: Config):
        super().__init__()

        d_model: int = config.dim_transformer
        seq_len: int = config.feat_seq_len
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(2*seq_len) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        #self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape [seq_len, batch_size, embedding_dim]

        Returns:
            Tensor: Output tensor with positional encoding applied.

        """
        d_model: int = 256
        seq_len: int = x.shape[1]
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(2*seq_len) / d_model))
        pe = torch.zeros(1, seq_len, d_model).cuda()
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.pe.to(x.device)

        x = x + self.pe.expand(x.shape)
        return x


if __name__ == "__main__":
    import torchinfo 
    from transformer_config import WHISPER_ENCODER_CONFIG, WHISPER_ENCODER_CONFIG_SMALL
    from whisper_wrapper import WhisperWrapper_encoder
    import matplotlib.pyplot as plt
    import torchaudio
    trans = TransformerWrapper(WHISPER_ENCODER_CONFIG_SMALL).cuda()
    whisper_enc = WhisperWrapper_encoder(use_feat_extractor=True).cuda().eval()
    print(torchinfo.summary(trans))
    #input = torch.randn(2, 16000*5).cuda()
    input = torchaudio.load("/mnt/parscratch/users/acp20glc/VoiceBank/clean_testset_wav_16k/p232_001.wav")[0].cuda()
    whisper_rep = whisper_enc(input)
    print(whisper_rep.shape)
    trans_rep = trans(whisper_rep)
    print(trans_rep.shape)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(input[0].cpu().numpy())
    axs[0].set_title('Input')
    axs[1].imshow(whisper_rep[0].detach().cpu().numpy().T)
    axs[1].set_title('Whisper Representation')
    axs[2].imshow(trans_rep[0].detach().cpu().numpy().T)
    axs[2].set_title('Transformer Representation')
    plt.savefig("transformer_rep.png")
    
