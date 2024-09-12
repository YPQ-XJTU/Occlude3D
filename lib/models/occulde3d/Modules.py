import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.roi_align as roi_align

def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask, pos):
        output = src

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.self_attn = LinearAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_key_padding_mask, pos):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class CenterFusionModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CenterFusionModule, self).__init__()

        self.input_channels = input_channels

        self.fpos = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.W_favg = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # MLP for heatmap prediction
        self.mlp_heatmap = nn.Sequential(
            nn.Linear(input_channels, input_channels),
            nn.ReLU(),
            nn.Linear(input_channels, output_channels)
        )

        # MLP for offset prediction
        self.mlp_offset = nn.Sequential(
            nn.Linear(input_channels, input_channels),
            nn.ReLU(),
            nn.Linear(input_channels, output_channels)
        )

        # MLP for size prediction
        self.mlp_size = nn.Sequential(
            nn.Linear(input_channels, input_channels),
            nn.ReLU(),
            nn.Linear(input_channels, output_channels)
        )

    def _position_based_fusion(self, heatmap_pred, offset_pred, size_pred, roi_mask):
        # Concatenate the predictions with roi_mask
        fused_input = torch.cat((heatmap_pred, offset_pred, size_pred, roi_mask), dim=1)

        # Position-based fusion
        F_e_V = self.fpos(fused_input)

        avg_pool = F.adaptive_avg_pool2d(F_e_V, output_size=1)

        W_output = self.W_favg(avg_pool)

        # Sigmoid activation
        weights = self.sigmoid(W_output)

        # Element-wise multiplication
        fused_feature = weights * F_e_V

        return fused_feature

    def _convolution_layers(self, fused_feature):
        # Apply convolutional layers
        x = self.conv1(fused_feature)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x

    def get_roi_feat_by_mask(self, feat, inds, mask, calibs):
        BATCH_SIZE, _, HEIGHT, WIDTH = feat.size()
        device_id = feat.device
        num_masked_bin = mask.sum()
        res = {}

        if num_masked_bin != 0:
            scale_box2d_masked = extract_input_from_tensor(feat, inds, mask)
            roi_feature_masked = roi_align(feat, scale_box2d_masked, [7, 7])
            roi_calibs = calibs[scale_box2d_masked[:, 0].long()]
            coords_in_camera_coord = torch.cat([self.project2rect(roi_calibs, torch.cat(
                [scale_box2d_masked[:, 1:3], torch.ones([num_masked_bin, 1]).to(device_id)], -1))[:, :2],
                                                self.project2rect(roi_calibs, torch.cat([scale_box2d_masked[:, 3:5],
                                                                                         torch.ones(
                                                                                             [num_masked_bin, 1]).to(
                                                                                             device_id)], -1))[:, :2]],
                                               -1)
            coords_in_camera_coord = torch.cat([scale_box2d_masked[:, 0:1], coords_in_camera_coord], -1)
            coord_maps = torch.cat([torch.cat([coords_in_camera_coord[:, 1:2] + i * (
                        coords_in_camera_coord[:, 3:4] - coords_in_camera_coord[:, 1:2]) / (7 - 1) for i in range(7)],
                                              -1).unsqueeze(1).repeat([1, 7, 1]).unsqueeze(1),
                                    torch.cat([coords_in_camera_coord[:, 2:3] + i * (
                                                coords_in_camera_coord[:, 4:5] - coords_in_camera_coord[:, 2:3]) / (
                                                           7 - 1) for i in range(7)], -1).unsqueeze(2).repeat(
                                        [1, 1, 7]).unsqueeze(1)], 1)

            cls_hots = torch.zeros(num_masked_bin, self.cls_num).to(device_id)
            cls_hots[torch.arange(num_masked_bin).to(device_id)] = 1.0

            roi_feature_masked = torch.cat(
                [roi_feature_masked, coord_maps, cls_hots.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, 7, 7])], 1)
        return roi_feature_masked

    def get_roi_feat(self, feat, inds, mask, calibs):
        BATCH_SIZE, _, HEIGHT, WIDTH = feat.size()
        device_id = feat.device
        coord_map = torch.cat([torch.arange(WIDTH).unsqueeze(0).repeat([HEIGHT, 1]).unsqueeze(0),
                               torch.arange(HEIGHT).unsqueeze(-1).repeat([1, WIDTH]).unsqueeze(0)], 0).unsqueeze(
            0).repeat([BATCH_SIZE, 1, 1, 1]).type(torch.float).to(device_id)
        box2d_centre = coord_map + self.offset_2d
        box2d_maps = torch.cat([box2d_centre - self.size_2d / 2, box2d_centre + self.size_2d / 2], 1)
        box2d_maps = torch.cat([torch.arange(BATCH_SIZE).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
            [1, 1, HEIGHT, WIDTH]).type(torch.float).to(device_id), box2d_maps], 1)
        res = self.get_roi_feat_by_mask(feat, box2d_maps, inds, mask, calibs)
        return res

    def get_roi_head_predictions(self, roi_mask):
        # Extract necessary information from roi_mask
        heatmap = roi_mask['heatmap']
        offset_2d = roi_mask['offset_2d']
        size_2d = roi_mask['size_2d']

        # Predictions using MLP models
        heatmap_pred = self.mlp_heatmap(heatmap)
        offset_pred = self.mlp_offset(offset_2d)
        size_pred = self.mlp_size(size_2d)

        return heatmap_pred, offset_pred, size_pred

    def forward(self, feat, inds, mask, calibs):
        roi_mask = self.get_roi_feat(feat, inds, mask, calibs)

        heatmap_pred, offset_pred, size_pred = self.get_roi_head_predictions(roi_mask)

        fused_feature = self._position_based_fusion(heatmap_pred, offset_pred, size_pred)

        x = self._convolution_layers(fused_feature)

        return x

class DepthEncoder(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        depth_num_bins = int(model_cfg["num_depth_bins"])
        depth_min = float(model_cfg["depth_min"])
        depth_max = float(model_cfg["depth_max"])
        self.depth_max = depth_max

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        # Create modules
        d_model = model_cfg["hidden_dim"]
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))

        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())

        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))

        depth_encoder_layer = TransformerEncoderLayer(
            d_model, nhead=8, dim_feedforward=256, dropout=0.1)

        self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)

        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)

    def forward(self, vis_depth, att_depth, mask, pos):
        assert len(vis_depth) == 4  # Assuming vis_depth has shape [B, C, H, W]
        assert len(att_depth) == 4  # Assuming att_depth has shape [B, C, H, W]

        # Perform encoding for visual depth
        vis_depth_embed = self.depth_encoder(vis_depth, mask, pos)

        # Perform encoding for attention depth
        att_depth_embed = self.depth_encoder(att_depth, mask, pos)

        return vis_depth_embed, att_depth_embed

