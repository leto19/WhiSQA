from enum import Enum
import torch
from torch.nn.functional import pad

class Input(Enum):
    MFCC = 0
    XLSR = 1

class CenterCrop(torch.nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor):
        # Center crop.
        unsqueezed = False
        if x.dim() == 2:
            unsqueezed = True
            x = x.unsqueeze(0)
        assert x.dim() == 3 # N, L, C

        if x.size(1) > self.seq_len:
            center_start_idx = int(x.size(1) / 2 - self.seq_len / 2)
            start_idx = center_start_idx
            end_idx = start_idx + self.seq_len
            x = x[:, start_idx:end_idx, :]
        if x.size(1) < self.seq_len:
            to_pad = self.seq_len - x.size(1)
            # Pad the end of sequence dimension.
            x = pad(x, (0,0,0,to_pad,0,0), mode="constant", value=0.0)

        if unsqueezed:
            x = x.squeeze(0)

        return x
    
class Config:

    name: str = None
    input: Input = None
    feat_seq_len: int = None
    dim_input: int = None
    dim_transformer: int = None
    dim_head_in: int = None
    dim_head_out: int = None

    def __init__(
        self,
        name: str,
        input: Input,
        feat_seq_len: int,
        dim_transformer: int = None,
        xlsr_name: str = None,
        nhead_transformer: int = 4,
        nlayers_transformer: int = 2,
    ):
        if input == Input.MFCC:
            xlsr_name = None

        # Check valid parameters.
        assert feat_seq_len > 0, "feat_seq_len must be positive."

        # Save parameters.
        self.name = name
        self.input = input
        self.feat_seq_len = feat_seq_len
        self.dim_transformer = dim_transformer
        self.xlsr_name = xlsr_name
        self.nhead_transformer = nhead_transformer
        self.nlayers_transformer = nlayers_transformer
        if xlsr_name is not None:
            # From XLS-R paper Table 2: Model architectures.
            if xlsr_name == "wav2vec2-xls-r-300m":
                _b = 24
                _h = 1024
            elif xlsr_name == "wav2vec2-xls-r-1b":
                _b = 48
                _h = 1280
            elif xlsr_name == "wav2vec2-xls-r-2b":
                _b = 48
                _h = 1920
            elif xlsr_name == "hubert_encoder":
                _b = -1
                _h = 512
            elif xlsr_name == "hubert_encoder_t":
                _b = -1
                _h = 384
            elif xlsr_name == "hubert_full":
                _b = -1
                _h = 768
            elif xlsr_name == "hubert_full_t":
                _b = -1
                _h = 384
            elif xlsr_name == "whisper_encoder":
                _b = -1
                _h = 768
            elif xlsr_name == "whisper_encoder_ref":
                _b = -1
                _h = 768*2
            elif xlsr_name == "whisper_encoder_t":
                _b = -1
                _h = 1500
            elif xlsr_name == "whisper_full":
                _b = -1
                _h = 768
            elif xlsr_name == "whisper_full_t":
                _b = -1
                _h = 384
            self.xlsr_layers = _b + 1  # +1 for CNN activation "layer0"
            self.dim_input = _h
        else:
            if self.feat_seq_len == 80: #handle transposed mfcc
                self.xlsr_layers = None
                self.dim_input = 3000
            else:
                self.xlsr_layers = None
                self.dim_input = 80  # MFCC

        self.dim_head_in = self.dim_transformer  # * self.feat_seq_len
        self.dim_head_out = 1

        self.dropout = 0.1  # TODO


# Length of feature frame window.
FEAT_SEQ_LEN = 256


####################### TRANSFORMER_32DEEP_CONFIG ####################
MFCC_TRANSFORMER_32DEEP_CONFIG = Config(
    "MFCC_TRANSFORMER_32DEEP_CONFIG",
    Input.MFCC,
    feat_seq_len=FEAT_SEQ_LEN,
    dim_transformer=256,
    xlsr_name=None,
    nhead_transformer=4,
    nlayers_transformer=4,
)

HUBERT_ENCODER_CONFIG= Config(
    "HUBERT_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=256,
    dim_transformer=256,
    xlsr_name="hubert_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)



HUBERT_ENCODER_CONFIG_T = Config(
    "HUBERT_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=512,
    dim_transformer=256,
    xlsr_name="hubert_encoder_t",
    nhead_transformer=4,
    nlayers_transformer=4,
)





HUBERT_FULL_CONFIG = Config(
    "HUBERT_FULL_CONFIG",
    Input.XLSR,
    feat_seq_len=768,
    dim_transformer=256,
    xlsr_name="hubert_full",
    nhead_transformer=4,
    nlayers_transformer=4,
)


WHISPER_ENCODER_CONFIG = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=1500,
    dim_transformer=768,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)

WHISPER_ENCODER_CONFIG = Config(
    "WHISPER_ENCODER_CONFIG_REF",
    Input.XLSR,
    feat_seq_len=1500,
    dim_transformer=768,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)


WHISPER_ENCODER_CONFIG_MEDIUM = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=1500,
    dim_transformer=512,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)



WHISPER_ENCODER_CONFIG_SMALL = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=1500,
    dim_transformer=256,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)
WHISPER_ENCODER_CONFIG_SMALL_T = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=768,
    dim_transformer=256,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)

WHISPER_ENCODER_CONFIG_MEL = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.MFCC,
    feat_seq_len=3000,
    dim_transformer=256,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)



WHISPER_ENCODER_CONFIG_SMALLER = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=1500,
    dim_transformer=128,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)

WHISPER_ENCODER_CONFIG_SMALLER_T = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=768,
    dim_transformer=128,
    xlsr_name="whisper_encoder_t",
    nhead_transformer=4,
    nlayers_transformer=4,
)

WHISPER_FULL_CONFIG_SMALL= Config(
    "WHISPER_FULL_CONFIG",
    Input.XLSR,
    feat_seq_len=768,
    dim_transformer=256,
    xlsr_name="whisper_full",
    nhead_transformer=4,
    nlayers_transformer=4,
)


XLSR_300M_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_32DEEP_CONFIG",
    Input.XLSR,
    feat_seq_len=FEAT_SEQ_LEN,
    dim_transformer=32,
    xlsr_name="wav2vec2-xls-r-300m",
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_1B_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_1B_TRANSFORMER_32DEEP_CONFIG",
    Input.XLSR,
    feat_seq_len=FEAT_SEQ_LEN,
    dim_transformer=32,
    xlsr_name="wav2vec2-xls-r-1b",
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_2B_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_2B_TRANSFORMER_32DEEP_CONFIG",
    Input.XLSR,
    feat_seq_len=FEAT_SEQ_LEN,
    dim_transformer=32,
    xlsr_name="wav2vec2-xls-r-2b",
    nhead_transformer=4,
    nlayers_transformer=4,
)