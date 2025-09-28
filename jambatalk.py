import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from wav2vec import Wav2Vec2Model, Wav2Vec2ForCTC, linear_interpolation
from mamba_ssm.modules.mamba_simple import Mamba 
from moe_mamba import MambaMoELayer
from transformer import TransformerDecoderLayerGQA_RoPE

# Borrowed/adapted from FaceXHuBERT
def inputRepresentationAdjustment(audio_embedding_matrix, vertex_matrix, ifps, ofps):
    """Align audio features to vertex frames by FPS.
    Args:
        audio_embedding_matrix: (B, T_a, C)
        vertex_matrix: (B, T_v, V*3)  or  (B, T_v, V, 3)
        ifps: input fps for audio embeddings (e.g., 50 for BIWI)
        ofps: output fps for vertices (e.g., 25 for BIWI)
    Returns:
        audio_embedding_matrix_aligned, vertex_matrix_aligned, frame_num (T_v)
    """

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
    """
    Key changes:
      - Factorized repeated Mamba/MoE/Decoder stacks into helper methods
      - Unified dataset-dependent settings (FPS, heads) in __init__
      - Removed duplicated code between forward() and predict() by using a shared backbone
      - Kept output signatures identical to original for training compatibility
    """

    def __init__(self, args):
        super(JambaTalk, self).__init__()
        self.dataset = args.dataset
        self.device = args.device
        self.feature_dim = args.feature_dim
        self.vertice_dim = args.vertice_dim

        # Encoders
        self.audio_encoder = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.text_encoder = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.audio_encoder.feature_extractor._freeze_parameters()

        # Dataset-specific configs
        if self.dataset == "vocaset":
            pkl_path = "./vocaset/FLAME_masks.pkl"
            with open(pkl_path, "rb") as f:
                self.lip_mask = pickle.load(f, encoding="latin1")["lips"]
            self.lip_map = nn.Linear(254 * 3, 1024)
            n_q, n_kv = 8, 8
            # (source fps -> target fps for lip features)
            self.src_fps, self.tgt_fps = 30, 50
            audio_in_size = 1024
        elif self.dataset == "BIWI":
            with open("./BIWI/BIWI_lip.pkl", "rb") as f:
                self.lip_mask = pickle.load(f, encoding="latin1")
            self.lip_map = nn.Linear(758 * 3, 1024)
            n_q, n_kv = 4, 4
            self.src_fps, self.tgt_fps = 25, 50
            audio_in_size = 2048
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        # Sequence backbone
        self.mamba = Mamba(d_model=self.feature_dim)
        self.mamba_moe = MambaMoELayer(dim=self.feature_dim, d_state=8, d_conv=8, num_experts=2, num_experts_per_token=2)
        self.transformer_decoder = TransformerDecoderLayerGQA_RoPE(
            d_model=self.feature_dim, n_query_heads=n_q, n_kv_heads=n_kv
        )

        # Projections
        self.audio_feature_map = nn.Linear(audio_in_size, self.feature_dim)
        self.vertice_map_r = nn.Linear(self.feature_dim, self.vertice_dim)
        nn.init.constant_(self.vertice_map_r.weight, 0.0)
        nn.init.constant_(self.vertice_map_r.bias, 0.0)

        # Lip-text pathway
        # lip_map -> 1024, transformer(1024), lm_head(33)
        self.transformer = nn.Transformer(d_model=1024, batch_first=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.lm_head = nn.Linear(1024, 33) 

    # ---------- helper stacks ----------
    def _stack_mamba_moe_head(self, x, depth: int):
        for _ in range(depth):
            x = self.mamba(x)
            x = self.mamba_moe(x)
        return x

    def _transformer_block(self, x):
        return self.transformer_decoder(x, x)
    
    def _stack_mamba_moe_tail(self, x, depth: int):
        for _ in range(depth):
            x = self.mamba_moe(x)
            x = self.mamba(x)
        return x

    def _backbone(self, vertice_input, use_decoder_blocks: int, extra_mamba_blocks: int):
        # Head stack
        x = self._stack_mamba_moe_head(vertice_input, depth=use_decoder_blocks)
        # Transformer
        x = self._transformer_block(x)
        # Tail stack
        x = self._stack_mamba_moe_tail(x, depth=extra_mamba_blocks)
        # Mamba_moe
        x = self.mamba_moe(x)
        return x

    def _run_sequence_backbone(self, vertice_input):
        """Dataset-specific depth schedule to mirror original behavior exactly."""
        if self.dataset == "BIWI":
            out = self._backbone(vertice_input, use_decoder_blocks=2, extra_mamba_blocks=2)
        else:  # vocaset
            out = self._backbone(vertice_input, use_decoder_blocks=1, extra_mamba_blocks=1)
        return out
        
    # ---------- core forward utilities ----------
    def _prepare_hidden_states(self, audio, frame_num=None):
        """Run audio encoder and align for dataset specifics.
        Returns (hidden_states_aligned, frame_num_for_vertices).
        """
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        if self.dataset == "BIWI":
            if frame_num is None:
                raise ValueError("frame_num must be provided for BIWI alignment in training forward")
            hidden_states, _, frame_num = inputRepresentationAdjustment(hidden_states, torch.zeros(1, frame_num, 1, device=hidden_states.device), 50, 25)
            hidden_states = hidden_states[:, :frame_num]
        return hidden_states, frame_num

    def _lip_text_path(self, vertice_out, text_hidden_states=None, text_logits=None, out_len_override=None):
        # Extract predicted lip region from vertices
        B, T, _ = vertice_out.shape
        lip_pred = (
            vertice_out.view(B, T, -1, 3)[:, :, self.lip_mask, :].reshape(B, T, -1)
        )
        lip_offset = self.lip_map(lip_pred)

        # Interpolate to target 50Hz timeline
        out_len = out_len_override
        lip_offset = linear_interpolation(lip_offset, self.src_fps, 50, output_len=out_len)

        # Small transformer to get lip features
        lip_features = self.transformer(lip_offset, lip_offset)
        logits = self.lm_head(self.dropout(lip_features))
        return lip_features, logits

    # ---------- model API ----------
    def forward(self, audio, template, vertice):
        """Training forward pass.
        Returns: vertice_out, vertice(gt passthrough), lip_features, text_hidden_states, logits, text_logits
        """
        template = template.unsqueeze(1)
        frame_num = vertice.shape[1]

        # Audio to features (aligned per dataset)
        hidden_states, _ = self._prepare_hidden_states(audio, frame_num=frame_num)
        vertice_input = self.audio_feature_map(hidden_states)

        # Sequence backbone (Mamba/MoE + Decoder per dataset schedule)
        vertice_out = self._run_sequence_backbone(vertice_input)

        # Map to vertices & add template
        vertice_out = self.vertice_map_r(vertice_out)
        vertice_out = vertice_out + template

        # Text branch (for lip-text alignment & CTC)
        audio_model = self.text_encoder(audio)
        text_hidden_states = audio_model.hidden_states
        text_logits = audio_model.logits

        # Dataset-specific trim/interp for text branch
        if self.dataset == "BIWI":
            # Make text streams length match 2x vertice_out
            max_len = vertice_out.shape[1] * 2
            text_hidden_states = text_hidden_states[:, :max_len]
            text_logits = text_logits[:, :max_len]
            out_len = text_hidden_states.shape[1]
        else:
            out_len = text_hidden_states.shape[1]

        lip_features, logits = self._lip_text_path(
            vertice_out, text_hidden_states=text_hidden_states, text_logits=text_logits, out_len_override=out_len
        )

        return vertice_out, vertice, lip_features, text_hidden_states, logits, text_logits

    @torch.no_grad()
    def predict(self, audio, template):
        """Inference path (no text encoder).
        Returns: vertice_out, lip_features, logits
        """
        template = template.unsqueeze(1)

        # Original predict reshaped BIWI embeddings from 50Hz -> 25Hz by factor 2
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        if self.dataset == "BIWI":
            # ensure even length for folding
            if hidden_states.shape[1] % 2 != 0:
                hidden_states = hidden_states[:, : hidden_states.shape[1] - 1]
            hidden_states = torch.reshape(hidden_states, (1, hidden_states.shape[1] // 2, hidden_states.shape[2] * 2))

        vertice_input = self.audio_feature_map(hidden_states)
        vertice_out = self._run_sequence_backbone(vertice_input)

        vertice_out = self.vertice_map_r(vertice_out)
        vertice_out = vertice_out + template

        # Lip-text small head (no text encoder during inference)
        lip_features, logits = self._lip_text_path(vertice_out, out_len_override=None)
        return vertice_out, lip_features, logits
