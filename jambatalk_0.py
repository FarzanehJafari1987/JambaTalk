import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from wav2vec import Wav2Vec2Model, Wav2Vec2ForCTC, linear_interpolation

from mamba.mamba_ssm.modules.mamba_simple import Mamba 
from moe_mamba import MambaMoELayer
from transformer import TransformerDecoderLayerGQA_RoPE

# Input Representation Adjustment, brrowed from https://github.com/galib360/FaceXHuBERT
def inputRepresentationAdjustment(audio_embedding_matrix, vertex_matrix, ifps, ofps):
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embedding_matrix.shape[1] % 2 != 0:
            audio_embedding_matrix = audio_embedding_matrix[:, :audio_embedding_matrix.shape[1] - 1]

        if audio_embedding_matrix.shape[1] > vertex_matrix.shape[1] * 2:
            audio_embedding_matrix = audio_embedding_matrix[:, :vertex_matrix.shape[1] * 2]

        elif audio_embedding_matrix.shape[1] < vertex_matrix.shape[1] * 2: 
            vertex_matrix = vertex_matrix[:, :audio_embedding_matrix.shape[1] // 2]
    else:
        factor = -1 * (-ifps // ofps)
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True,
                                               mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

    frame_num = vertex_matrix.shape[1]
    audio_embedding_matrix = torch.reshape(audio_embedding_matrix, (
    1, audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor))
    return audio_embedding_matrix, vertex_matrix, frame_num


class JambaTalk(nn.Module):
    def __init__(self, args):
        super(JambaTalk, self).__init__()
        self.dataset = args.dataset
        self.audio_encoder = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.text_encoder = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.audio_encoder.feature_extractor._freeze_parameters()

        if args.dataset == "vocaset":
            pkl_path = "./vocaset/FLAME_masks.pkl"
            with open(pkl_path, 'rb') as f:
                self.lip_mask = pickle.load(f, encoding='latin1')["lips"]
                self.lip_map = nn.Linear(254 * 3, 1024)

        elif args.dataset == "BIWI":
            with open('./BIWI/BIWI_lip.pkl', 'rb') as f:
                self.lip_mask = pickle.load(f, encoding='latin1')
                self.lip_map = nn.Linear(758 * 3, 1024)

        # mamba layer
        self.mamba = Mamba(d_model = args.feature_dim)

        # mamba_moe layer
        self.mamba_moe = MambaMoELayer(dim=args.feature_dim, d_state=8, d_conv=8, num_experts=2, num_experts_per_token=2)

        if args.dataset == "vocaset":  
            self.transformer_decoder = TransformerDecoderLayerGQA_RoPE(
                d_model=args.feature_dim,
                n_query_heads=8,
                n_kv_heads=8,
            )

        elif args.dataset == "BIWI":
            self.transformer_decoder = TransformerDecoderLayerGQA_RoPE(
                d_model=args.feature_dim,
                n_query_heads=4,
                n_kv_heads=4,
            )           
        
        if args.dataset == "vocaset":
            self.audio_feature_map = nn.Linear(1024, args.feature_dim)
            self.transformer = nn.Transformer(d_model=1024, batch_first=True)
        elif args.dataset == "BIWI":
            self.audio_feature_map = nn.Linear(2048, args.feature_dim)
            self.transformer = nn.Transformer(d_model=1024, batch_first=True)
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        self.device = args.device
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.lm_head = nn.Linear(1024, 33)

        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, audio, template, vertice):

        template = template.unsqueeze(1)
        frame_num = vertice.shape[1]
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state

        if self.dataset == "BIWI":
            hidden_states, vertice, frame_num = inputRepresentationAdjustment(hidden_states, vertice, 50, 25)
            hidden_states = hidden_states[:, :frame_num]
        elif self.dataset == "vocaset":
            pass
        vertice_input = self.audio_feature_map(hidden_states)

        if self.dataset == "BIWI":
            vertice_out = self.mamba(vertice_input) 
            vertice_out = self.mamba_moe(vertice_out)
            vertice_out = self.mamba(vertice_out) 
            vertice_out = self.mamba_moe(vertice_out) 
            vertice_out = self.transformer_decoder(vertice_out, vertice_out) 
            vertice_out = self.mamba_moe(vertice_out)
            vertice_out = self.mamba(vertice_out)
            vertice_out = self.mamba_moe(vertice_out) 
            vertice_out = self.mamba(vertice_out)
            vertice_out = self.mamba_moe(vertice_out)

        elif self.dataset == "vocaset":
            vertice_out = self.mamba(vertice_input) 
            vertice_out = self.mamba_moe(vertice_out) 
            vertice_out = self.transformer_decoder(vertice_out, vertice_out) 
            vertice_out = self.mamba_moe(vertice_out) 
            vertice_out = self.mamba(vertice_out)
            vertice_out = self.mamba_moe(vertice_out)

        vertice_out = self.vertice_map_r(vertice_out)
        audio_model = self.text_encoder(audio)
        text_hidden_states = audio_model.hidden_states
        text_logits = audio_model.logits
        frame_num = text_hidden_states.shape[1]
        lip_out = vertice_out.reshape(vertice_out.shape[0], vertice_out.shape[1], -1, 3)[:, :, self.lip_mask,
                  :].reshape(vertice_out.shape[0], vertice_out.shape[1], -1)
        lip_gt = vertice.reshape(vertice.shape[0], vertice.shape[1], -1, 3)[:, :, self.lip_mask, :].reshape(
            vertice.shape[0], vertice.shape[1], -1)
        lip_offset = self.lip_map(lip_out)

        if self.dataset == "vocaset":
            lip_offset = linear_interpolation(lip_offset, 30, 50, output_len=frame_num)
        elif self.dataset == "BIWI":
            text_hidden_states = text_hidden_states[:, :vertice_out.shape[1] * 2]
            text_logits = text_logits[:, :vertice_out.shape[1] * 2]
            frame_num = text_hidden_states.shape[1]
            lip_offset = linear_interpolation(lip_offset, 25, 50, output_len=frame_num)
        lip_features = self.transformer(lip_offset, lip_offset)
        logits = self.lm_head(self.dropout(lip_features))
        vertice_out = vertice_out + template

        return vertice_out, vertice, lip_features, text_hidden_states, logits, text_logits

    def predict(self, audio, template):
        template = template.unsqueeze(1)
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        if self.dataset == "BIWI":
            if hidden_states.shape[1] % 2 != 0:
                hidden_states = hidden_states[:, :hidden_states.shape[1] - 1]
            hidden_states = torch.reshape(hidden_states, (1, hidden_states.shape[1] // 2, hidden_states.shape[2] * 2))
        elif self.dataset == "vocaset":
            pass
        vertice_input = self.audio_feature_map(hidden_states)

        if self.dataset == "BIWI":
            vertice_out = self.mamba(vertice_input) 
            vertice_out = self.mamba_moe(vertice_out)
            vertice_out = self.mamba(vertice_out) 
            vertice_out = self.mamba_moe(vertice_out) 
            vertice_out = self.transformer_decoder(vertice_out, vertice_out) 
            vertice_out = self.mamba_moe(vertice_out)
            vertice_out = self.mamba(vertice_out)
            vertice_out = self.mamba_moe(vertice_out) 
            vertice_out = self.mamba(vertice_out)
            vertice_out = self.mamba_moe(vertice_out)

        elif self.dataset == "vocaset":
            vertice_out = self.mamba(vertice_input) 
            vertice_out = self.mamba_moe(vertice_out) 
            vertice_out = self.transformer_decoder(vertice_out, vertice_out) 
            vertice_out = self.mamba_moe(vertice_out) 
            vertice_out = self.mamba(vertice_out)
            vertice_out = self.mamba_moe(vertice_out)

        vertice_out = self.vertice_map_r(vertice_out)
        lip_offset = vertice_out.reshape(vertice_out.shape[0], vertice_out.shape[1], -1, 3)[:, :, self.lip_mask,
                     :].reshape(vertice_out.shape[0], vertice_out.shape[1], -1)
        lip_offset = self.lip_map(lip_offset)

        if self.dataset == "vocaset":
            lip_offset = linear_interpolation(lip_offset, 30, 50, output_len=None)
        elif self.dataset == "BIWI":
            lip_offset = linear_interpolation(lip_offset, 25, 50, output_len=None)
        lip_features = self.transformer(lip_offset, lip_offset)
        if self.dataset == "vocaset":
            vertice_out = vertice_out + template
        elif self.dataset == "BIWI":
            vertice_out = vertice_out + template
        logits = self.lm_head(self.dropout(lip_features))

        return vertice_out, lip_features, logits
