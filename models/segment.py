import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers import EsmModel, EsmTokenizer
import torch.nn.functional as F
from models.loss_func import WeightedRMSELoss


class MultiScaleSamplingLayer(nn.Module):
    def __init__(self, scales, embedding_dim):
        super().__init__()
        self.samplingLayers = nn.ModuleList([nn.Conv1d(embedding_dim, embedding_dim, 2, 2) for _ in range(scales)])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        multi_scale_inputs = [x]
        for layer in self.samplingLayers:
            x = layer(x)
            multi_scale_inputs.append(x)
        multi_scale_inputs = [_.permute(0, 2, 1) for _ in multi_scale_inputs]
        return multi_scale_inputs


class SequenceSegmentation(nn.Module):
    def __init__(self, segment_lengths):
        super().__init__()
        self.segment_lengths = segment_lengths

    def forward(self, x):
        for idx, length in enumerate(self.segment_lengths):
            x[idx] = x[idx].reshape(x[idx].shape[0], -1, length, x[idx].shape[-1])
        return x


class MultiScaleSegmentConv2d(nn.Module):
    def __init__(self, kernel_size, segment_lengths, embedding_dim):
        super().__init__()
        self.segment_lengths = segment_lengths
        self.conv_layers = nn.ModuleList([nn.Conv2d(embedding_dim, embedding_dim, (kernel_size, _), stride=(1, 1), padding=(2, 0)) for _ in segment_lengths])

    def forward(self, x):
        segments = [layer(_.permute(0, 3, 1, 2)) for _, layer in zip(x, self.conv_layers)]
        segments = [torch.squeeze(_.permute(0, 2, 3, 1), dim=2) for _ in segments]
        return segments


class DualGroupedSegmentAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, width):
        super().__init__()
        self.col_attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True), num_layers=num_layers)
        self.row_attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True), num_layers=num_layers)
        self.width = width

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, self.width, x.shape[-1])
        x_S = x.reshape(-1, x.shape[2], x.shape[-1])
        x_L = x.reshape(-1, x.shape[1], x.shape[-1])
        x_S = self.row_attention(x_S)
        x_L = self.col_attention(x_L)
        x = x_L.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[-1]) + x_S.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[-1])
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x


class MultiScaleSegmentAttention(nn.Module):
    def __init__(self, scales, embedding_dim, num_heads, num_layers, attention_range):
        super().__init__()
        self.segment_attentions = nn.ModuleList([DualGroupedSegmentAttention(embedding_dim, num_heads, num_layers, attention_range) for _ in range(scales + 1)])

    def forward(self, x):
        # input x is a list of tensors with shape (batch_size, num_segments, segment_dim), the segment_dim euqal to embedding_dim
        for idx, layer in enumerate(self.segment_attentions):
            x[idx] = layer(x[idx])
        return x


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_layer1 = nn.Linear(input_dim, input_dim)
        self.attention_layer2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_scores = self.attention_layer1(x)
        attn_scores = torch.tanh(attn_scores)
        attn_scores = self.attention_layer2(attn_scores).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled_output = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return pooled_output, attn_weights


class MultiScalePooling(nn.Module):
    def __init__(self, scales, embedding_dim):
        super().__init__()
        self.poolingLayers = nn.ModuleList([AttentionPooling(embedding_dim) for _ in range(scales+1)])

    def forward(self, x):
        pooled_outputs = []
        for idx, layer in enumerate(self.poolingLayers):
            pooled_output, attn_weights = layer(x[idx])
            pooled_outputs.append((pooled_output, attn_weights))
        return pooled_outputs


class MergeRegressionHead(nn.Module):
    def __init__(self, scales, input_dim):
        super().__init__()
        self.regression_layers = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(scales+1)])

    def forward(self, x):
        multi_scale_pred = [layer(_) for _, layer in zip(x, self.regression_layers)]
        multi_scale_pred = torch.cat(multi_scale_pred, dim=1)
        pred_max = torch.max(multi_scale_pred, 1)[0]
        pred_min = torch.min(multi_scale_pred, 1)[0]
        pred = torch.mean(multi_scale_pred, 1)
        return pred, pred_min, pred_max


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pretrain_model = EsmModel.from_pretrained(config['pretrain_model'])
        self.sample_layer = MultiScaleSamplingLayer(config['scales'], config['embed_dim'])
        self.segmentation = SequenceSegmentation(config['segment_lengths'])
        self.segment_conv2d = MultiScaleSegmentConv2d(config['kernel_size'], config['segment_lengths'], config['embed_dim'])
        self.segment_attention = MultiScaleSegmentAttention(config['scales'], config['embed_dim'], config['num_heads'], config['num_layers'], config['attention_range'])
        self.segment_pooling = MultiScalePooling(config['scales'], config['embed_dim'])
        self.regression = MergeRegressionHead(config['scales'], config['embed_dim'])
        self._keys_to_ignore_on_save = []
        self.loss_fct = WeightedRMSELoss()
        self.inference = False

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            outputs = self.pretrain_model(input_ids, attention_mask)

        hidden_state = outputs.last_hidden_state

        hidden_state = self.sample_layer(hidden_state)
        hidden_state = self.segmentation(hidden_state)
        hidden_state = self.segment_conv2d(hidden_state)
        hidden_state_conv = torch.mean(sum(hidden_state), dim=1)
        hidden_state = self.segment_attention(hidden_state)
        hidden_state_attention = torch.mean(sum(hidden_state), dim=1)
        hidden_state = self.segment_pooling(hidden_state)
        hidden_state, attn_weights = zip(*hidden_state)
        hidden_state_pooling = sum(hidden_state)
        attn_weights = torch.mean(torch.stack(attn_weights), 0)
        pred, pred_min, pred_max = self.regression(hidden_state)

        if self.inference:
            return ModelOutput(pred=pred,
                               pred_min=pred_min,
                               pred_max=pred_max,
                               attn_weights=attn_weights,
                               hidden_state_conv=hidden_state_conv,
                               hidden_state_attention=hidden_state_attention,
                               hidden_state_pooling=hidden_state_pooling)

        loss = None
        if labels is not None:
            loss = self.loss_fct(pred, labels)
        return ModelOutput(loss=loss, pred=pred)


class Collator:
    def __init__(self, pretrain_model):
        self.tokenizer = EsmTokenizer.from_pretrained(pretrain_model)

    def __call__(self, batch):
        seqs = [_.sequence for _ in batch]
        labels = [_.label for _ in batch]
        labels = torch.tensor(labels).float()
        inputs = self.tokenizer(list(seqs), return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
        inputs['labels'] = labels
        return inputs

