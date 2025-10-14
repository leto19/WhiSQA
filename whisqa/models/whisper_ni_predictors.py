import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .whisper_wrapper import WhisperWrapper_encoder
from .transformer_wrapper import TransformerWrapper
from .transformer_config import Config, Input    

class AttentionPool(nn.Module):
    """Attention-based pooling module with feed-forward network."""
    
    def __init__(self, dim_head_in):
        super().__init__()
        
        self.linear1 = nn.Linear(dim_head_in, 2*dim_head_in)
        self.linear2 = nn.Linear(2*dim_head_in, 1)
        
        self.linear3 = nn.Linear(dim_head_in, 1)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: Tensor):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        att = F.softmax(att, dim=2)
      
        x = torch.bmm(att, x) 
      
        x = x.squeeze(1)
        
        x = self.linear3(x)
        
        return x  



    
class whisperMetricPredictorEncoderLayersTransformerSmall(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)

        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG",
        Input.XLSR,
        feat_seq_len=feat_seq,
        dim_transformer=256,
        xlsr_name="whisper_encoder",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool = AttentionPool(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        out_feats = self.feat_extract(x) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 13 tensors
        #print(self.layer_weights)
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out
    
    
class whisperMetricPredictorEncoderLayersTransformerSmalldim(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)

        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG",
        Input.XLSR,
        feat_seq_len=feat_seq,
        dim_transformer=256,
        xlsr_name="whisper_encoder",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool1 = AttentionPool(self.config.dim_transformer)
        self.attenPool2 = AttentionPool(self.config.dim_transformer)
        self.attenPool3 = AttentionPool(self.config.dim_transformer)
        self.attenPool4 = AttentionPool(self.config.dim_transformer)
        self.attenPool5 = AttentionPool(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        out_feats = self.feat_extract(x) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 13 tensors
        #print(self.layer_weights)
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)

        out1 = self.attenPool1(out) #attenPool returns (B, 1)
        out1 = self.sigmoid(out1) #sigmoid returns (B, 1)
        out2 = self.attenPool2(out) #attenPool returns (B, 1)
        out2 = self.sigmoid(out2)

        out3 = self.attenPool3(out) #attenPool returns (B, 1)
        out3 = self.sigmoid(out3)

        out4 = self.attenPool4(out) #attenPool returns (B, 1)
        out4 = self.sigmoid(out4)

        out5 = self.attenPool5(out) #attenPool returns (B, 1)
        out5 = self.sigmoid(out5)

        #return all 5 outputs ona new dimension
        out = torch.stack([out1,out2,out3,out4,out5],dim=1)
        
        return out


class WhisperMetricPredictor(nn.Module):
    """Unified transformer-based metric estimator using Whisper encoder features.
    
    Args:
        feat_seq: Feature sequence length (default: 1500)
        num_outputs: Number of output dimensions (1 for single MOS, 5 for multi-dimensional)
    """
    
    def __init__(self, feat_seq=1500, num_outputs=1):
        super().__init__()
        
        # Feature extraction
        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        
        # Layer weighting for combining encoder layers
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)
        
        # Input normalization
        self.norm_input = nn.BatchNorm1d(768)

        # Transformer configuration
        self.config = Config(
            "WHISPER_ENCODER_CONFIG",
            Input.XLSR,
            feat_seq_len=feat_seq,
            dim_transformer=256,
            xlsr_name="whisper_encoder",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        
        # Transformer and pooling
        self.transformer = TransformerWrapper(self.config)
        
        # Multiple attention pooling heads for multi-dimensional output
        self.attention_pools = nn.ModuleList([
            AttentionPool(self.config.dim_transformer) for _ in range(num_outputs)
        ])
        
        self.sigmoid = nn.Sigmoid()
        self.num_outputs = num_outputs

    def forward(self, x):
        # Extract and combine encoder layer features
        out_feats = self.feat_extract(x)  # List of 13 tensors (B, 1500, 512)
        out_feats = out_feats @ self.softmax(self.layer_weights)  # Weighted combination
        
        # Normalize features
        out_feats = self.norm_input(out_feats.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Apply transformer
        out = self.transformer(out_feats)  # (B, 1500, 256)
        
        # Apply attention pooling and sigmoid activation
        if self.num_outputs == 1:
            out = self.attention_pools[0](out)
            out = self.sigmoid(out)
        else:
            outputs = []
            for pool in self.attention_pools:
                pool_out = self.sigmoid(pool(out))
                outputs.append(pool_out)
            out = torch.stack(outputs, dim=1)
        
        return out


# Legacy alias
PoolAttFF = AttentionPool
