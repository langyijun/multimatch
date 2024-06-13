import torch
import torch.nn as nn
from torch.nn import functional as F

def metric_loss(feats_x_ulb_w, feats_x_ulb_s):
    similarity_matrix_same_w_s = F.cosine_similarity(feats_x_ulb_w, feats_x_ulb_s, dim=1)
    similarity_matrix_diff_w_w = F.cosine_similarity(feats_x_ulb_w.unsqueeze(1), feats_x_ulb_w.unsqueeze(0), dim=2)
    similarity_matrix_diff_w_s = F.cosine_similarity(feats_x_ulb_w.unsqueeze(1), feats_x_ulb_s.unsqueeze(0), dim=2)

    similarity_matrix_diff_w_w = similarity_matrix_diff_w_w - torch.diag_embed(
        torch.diag(similarity_matrix_diff_w_w))
    similarity_matrix_diff_w_s = similarity_matrix_diff_w_s - torch.diag_embed(
        torch.diag(similarity_matrix_diff_w_s))
    loss = torch.mean(-similarity_matrix_same_w_s + torch.logsumexp(torch.mul(similarity_matrix_diff_w_w, 1000000), dim=1) / 1000000)
    return loss

class MetricLoss(nn.Module):
    def forward(self, feats_x_ulb_w, feats_x_ulb_s):
        return metric_loss(feats_x_ulb_w, feats_x_ulb_s)