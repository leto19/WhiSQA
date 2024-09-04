import torch
import torch.nn.functional as F
from torch import Tensor, nn
try:
    from whisper_wrapper import WhisperWrapper_full,WhisperWrapper_encoder,pad_or_trim, log_mel_spectrogram
    from transformer_wrapper import TransformerWrapper
    from transformer_config import CenterCrop,Config,Input
except:
    from models.whisper_wrapper import WhisperWrapper_full,WhisperWrapper_encoder, pad_or_trim, log_mel_spectrogram
    from models.transformer_wrapper import TransformerWrapper
    from models.transformer_config import CenterCrop,Config,Input    

class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
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



    
class whisperMetricPredictorEncoderTransformerSmall(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)

        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)

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

        
        
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        out_feats = self.feat_extract(x) #whisper encoder returns (B, 1500, 512)
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out


class whisperMetricPredictorEncoderTransformerSmallT(nn.Module):
    """Transformer based varient on metric estimator

    based on
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG",
        Input.XLSR,
        feat_seq_len=768,
        dim_transformer=256,
        xlsr_name="whisper_encoder_t",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        out_feats = self.feat_extract(x) #whisper encoder returns (B, 1500, 512)
        out_feats = out_feats.permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)

        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out
    
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

        
        
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
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

        
        
        self.attenPool1 = PoolAttFF(self.config.dim_transformer)
        self.attenPool2 = PoolAttFF(self.config.dim_transformer)
        self.attenPool3 = PoolAttFF(self.config.dim_transformer)
        self.attenPool4 = PoolAttFF(self.config.dim_transformer)
        self.attenPool5 = PoolAttFF(self.config.dim_transformer)
        
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
class whisperMetricPredictorEncoderLayersTransformerSmallRef(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)
        self.norm_input_ref = nn.BatchNorm1d(768)
        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.layer_weights_ref = nn.Parameter(torch.ones(13))
        self.softmax_ref = nn.Softmax(dim=0)
        self.softmax = nn.Softmax(dim=0)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG_REF",
        Input.XLSR,
        feat_seq_len=feat_seq,
        dim_transformer=256,
        xlsr_name="whisper_encoder_ref",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool1 = PoolAttFF(self.config.dim_transformer)
      
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
    
        out_feats = self.feat_extract(x) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 13 tensors
        print(self.layer_weights)
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        
        out_feats_ref = self.feat_extract(y) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        out_feats_ref = out_feats_ref @ self.softmax_ref(self.layer_weights_ref) #weighted sum of the 13 tensors
        print(self.layer_weights_ref)
        out_feats_ref = self.norm_input_ref(out_feats_ref.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        
        
        #concatenate the two inputs
        #print("out_feats",out_feats.shape)
        #print("out_feats_ref",out_feats_ref.shape)
        out_feats = torch.cat([out_feats,out_feats_ref],dim=2)
        #print(out_feats.shape)


        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)


        out1 = self.attenPool1(out) #attenPool returns (B, 1)
        out1 = self.sigmoid(out1) #sigmoid returns (B, 1)
        
        
        return out1#,out_feats
class whisperMetricPredictorEncoderLayersTransformerSmallDimRef(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)
        self.norm_input_ref = nn.BatchNorm1d(768)
        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.layer_weights_ref = nn.Parameter(torch.ones(13))
        self.softmax_ref = nn.Softmax(dim=0)
        self.softmax = nn.Softmax(dim=0)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG_REF",
        Input.XLSR,
        feat_seq_len=feat_seq,
        dim_transformer=256,
        xlsr_name="whisper_encoder_ref",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool1 = PoolAttFF(self.config.dim_transformer)
        self.attenPool2 = PoolAttFF(self.config.dim_transformer)
        self.attenPool3 = PoolAttFF(self.config.dim_transformer)
        self.attenPool4 = PoolAttFF(self.config.dim_transformer)
        self.attenPool5 = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
    
        out_feats = self.feat_extract(x) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 13 tensors
        print(self.layer_weights)
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        
        out_feats_ref = self.feat_extract(y) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        out_feats_ref = out_feats_ref @ self.softmax_ref(self.layer_weights_ref) #weighted sum of the 13 tensors
        print(self.layer_weights_ref)
        out_feats_ref = self.norm_input_ref(out_feats_ref.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        
        
        #concatenate the two inputs
        #print("out_feats",out_feats.shape)
        #print("out_feats_ref",out_feats_ref.shape)
        out_feats = torch.cat([out_feats,out_feats_ref],dim=2)
        #print(out_feats.shape)


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
        
        return out#,out_feats



class whisperMetricPredictorEncoderLayersTransformerSmallT(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG",
        Input.XLSR,
        feat_seq_len=768,
        dim_transformer=256,
        xlsr_name="whisper_encoder_t",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        out_feats = self.feat_extract(x) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 13 tensors
        print(self.layer_weights)

        out_feats = out_feats.permute(0,2,1) #swap axes to (B, 512, 1500)

        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out


class whisperMetricPredictorMelTransformerSmall(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(self, feat_seq=3000):
        super().__init__()


        self.config = Config(
        "MFCC_TRANSFORMER_32DEEP_CONFIG",
        Input.MFCC,
        feat_seq_len=feat_seq,
        dim_transformer=256,
        xlsr_name=None,
        nhead_transformer=4,
        nlayers_transformer=4,
    )
        self.norm_input = nn.BatchNorm1d(80)

        self.transformer = TransformerWrapper(self.config)

        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N_SAMPLES = 16000*30
        data_padded = pad_or_trim(x, length=N_SAMPLES) #pad or trim to 30 seconds, returns (B, 480000)
        data_feats = log_mel_spectrogram(data_padded).swapaxes(1,2) #returns (B, 3000, 80)
    
        data_feats = self.norm_input(data_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 3000, 80)
        out_trans = self.transformer(data_feats) # transformer returns (B, 3000, 256)
        out = self.attenPool(out_trans) #attenPool returns (B, 1)
        out = self.sigmoid(out)

        return out


class whisperMetricPredictorMelTransformerSmallT (nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(self, feat_seq=3000):
        super().__init__()


        self.config = Config(
        "MFCC_TRANSFORMER_32DEEP_CONFIG",
        Input.MFCC,
        feat_seq_len=80,
        dim_transformer=256,
        xlsr_name="mel_T",
        nhead_transformer=4,
        nlayers_transformer=4,
    )
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.transformer = TransformerWrapper(self.config)

        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N_SAMPLES = 16000*30
        data_padded = pad_or_trim(x, length=N_SAMPLES) #pad or trim to 30 seconds, returns (B, 480000)
        data_feats = log_mel_spectrogram(data_padded) #returns (B, 80, 3000)
    
        data_feats = self.norm_input(data_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 80, 3000)
        out_trans = self.transformer(data_feats) # transformer returns (B, 3000, 256)
        out = self.attenPool(out_trans) #attenPool returns (B, 1)
        out = self.sigmoid(out)

        return out



class whisperMetricPredictorFullTransformerSmall(nn.Module):
    def __init__(self, feat_seq=768//2):
        super().__init__()



        self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)
        self.config = Config(
            "WHISPER_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=feat_seq,
            dim_transformer=256,
            xlsr_name="whisper_full",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.cc = CenterCrop(feat_seq)
        self.norm_input = nn.BatchNorm1d(768)
        self.transformer = TransformerWrapper(self.config)
        self.norm_input = nn.BatchNorm1d(768)
        self.attenPool = PoolAttFF(self.config.dim_transformer)


        self.sigmoid = nn.Sigmoid()
    def forward(self, x):        
        out_feats = self.feat_extract(x)[:,:,:,-1] #whisper encoder returns (B, 1500, 768)
        out_feats = self.cc (out_feats) #center crop to 384
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 384, 768)
        out = self.transformer(out_feats) # transformer returns (B, 384, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out

class whisperMetricPredictorFullTransformerSmallT(nn.Module):
    def __init__(self, feat_seq=384):
        super().__init__()



        self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)
        self.config = Config(
            "WHISPER_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=768,
            dim_transformer=256,
            xlsr_name="whisper_full_t",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.cc = CenterCrop(feat_seq)
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.transformer = TransformerWrapper(self.config)
        self.attenPool = PoolAttFF(self.config.dim_transformer)


        self.sigmoid = nn.Sigmoid()
    def forward(self, x):        
        out_feats = self.feat_extract(x)[:,:,:,-1] #whisper encoder returns (B, W, 768)
        out_feats = self.cc (out_feats) #center crop to 384
        
        out_feats= out_feats.permute(0,2,1) #swap axes to (B, 768, 384)

        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 768, 384)
        
        out = self.transformer(out_feats) # transformer returns (B, 768, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out



class whisperMetricPredictorFullLayersTransformerSmall(nn.Module):
    def __init__(self, feat_seq=768//2):
        super().__init__()



        self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)
        self.config = Config(
            "WHISPER_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=feat_seq,
            dim_transformer=256,
            xlsr_name="whisper_full",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.cc = CenterCrop(feat_seq)
        self.norm_input = nn.BatchNorm1d(768)
        self.transformer = TransformerWrapper(self.config)
        self.norm_input = nn.BatchNorm1d(768)
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        self.layer_weights = nn.Parameter(torch.ones(12))
        self.softmax = nn.Softmax(dim=0)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):        
        out_feats = self.feat_extract(x) #whisper encoder returns list (B, 1500, 768,12)
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 12 tensors (B, 1500, 768) 
        print(self.layer_weights)
        out_feats = self.cc (out_feats) #center crop to 384
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 384, 768)
        out = self.transformer(out_feats) # transformer returns (B, 384, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out


class whisperMetricPredictorFullLayersTransformerSmallT(nn.Module):
    def __init__(self, feat_seq=384):
        super().__init__()



        self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)
        self.config = Config(
            "WHISPER_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=768,
            dim_transformer=256,
            xlsr_name="whisper_full_t",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.cc = CenterCrop(feat_seq)
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.transformer = TransformerWrapper(self.config)
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        self.layer_weights = nn.Parameter(torch.ones(12))
        self.softmax = nn.Softmax(dim=0)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):        
        out_feats = self.feat_extract(x) #whisper encoder returns list (B, 1500, 768,12)
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 12 tensors (B, 1500, 768) 
        print(self.layer_weights)
        out_feats = self.cc (out_feats) #center crop to 384
        
        out_feats= out_feats.permute(0,2,1) #swap axes to (B, 768, 384)

        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 768, 384)
        
        out = self.transformer(out_feats) # transformer returns (B, 768, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out

if __name__ == "__main__":
    import torchinfo
    import torchaudio
    import matplotlib.pyplot as plt
    import numpy as np



    aud_path = "/mnt/parscratch/users/acp20glc/VoiceBank/clean_testset_wav_16k/p232_001.wav"
    clean,_ = torchaudio.load(aud_path)
    clean = clean.cuda()
    
    noisy,_ = torchaudio.load(aud_path.replace("clean_testset_wav_16k","noisy_testset_wav_16k"))
    noisy = noisy.cuda()
    model = whisperMetricPredictorEncoderLayersTransformerSmallDimRef()
    torchinfo.summary(model, input_size=[(16,16000),(16,16000)])
    output,out_feat= model(clean,noisy)
    print(output.shape)
    #cast out_feat to numpy
    out_feat = out_feat.cpu().detach().numpy()
    print(out_feat.shape)
    #apply a sigmoid to the output
    out_feat = 1/(1+np.exp(-out_feat))

    print(out_feat.shape)
    plt.imshow(out_feat[0].T)
    plt.xlabel("Time")
    plt.ylabel("Feature")
    #chunk y axis into blocks of 768
    plt.yticks(np.arange(0, 768*2, 768//2))
    #add a horrizontal line halfway through the vertical axis
    plt.axhline(y=768, color='r', linestyle='--')
    plt.savefig("out_feat.png")




    # model = whisperMetricPredictorEncoderTransformerSmall()
    # torchinfo.summary(model, input_size=(16,16000))
    # print(input.shape)
    # output= model(input)
    # print(output.shape)

    # del model
    # model = whisperMetricPredictorEncoderLayersTransformerSmall()
    # torchinfo.summary(model, input_size=(16,16000))
    # print(input.shape)
    # output= model(input)
    # print(output.shape)

    # del model
    # model = whisperMetricPredictorEncoderTransformerSmallT()
    # torchinfo.summary(model, input_size=(16,16000))
    # print(input.shape)
    # output= model(input)
    # print(output.shape)

    # del model
    # model = whisperMetricPredictorEncoderLayersTransformerSmallT()
    # torchinfo.summary(model, input_size=(16,16000))
    # print(input.shape)
    # output= model(input)
    # print(output.shape)

    # del model
    # model = whisperMetricPredictorMelTransformerSmall()
    # torchinfo.summary(model, input_size=(16,16000))
    # print(input.shape)
    # output= model(input)
    # print(output.shape)

    # del model
    # model = whisperMetricPredictorMelTransformerSmallT()
    # torchinfo.summary(model, input_size=(16,16000))
    # print(input.shape)
    # output= model(input)
    # print(output.shape)

    # del model
    # model = whisperMetricPredictorFullTransformerSmall()
    # torchinfo.summary(model, input_size=(16,16000))
    # print(input.shape)
    # output= model(input)
    # print(output.shape)

    # del model
    # model = whisperMetricPredictorFullLayersTransformerSmall()
    # torchinfo.summary(model, input_size=(16,16000))
    # print(input.shape)
    # output= model(input)
    # print(output.shape)

    # del model
    # model = whisperMetricPredictorFullTransformerSmallT()
    # torchinfo.summary(model, input_size=(16,16000))
    # print(input.shape)
    # output= model(input)
    # print(output.shape)

    # del model
    # model = whisperMetricPredictorFullLayersTransformerSmallT()
    # torchinfo.summary(model, input_size=(16,16000))
    # print(input.shape)
    # output= model(input)
    # print(output.shape)

    # print("done :)")
    