import torch
import torch.nn as nn
from torch.nn import functional as F

def metric_loss(logits_x_ulb, logits_x_ulb_w, logits_x_ulb_s, feats_x_ulb, feats_x_ulb_w, feats_x_ulb_s, delta, m_model):



    # feats_matrix_x_w = F.cosine_similarity(feats_x_ulb, feats_x_ulb_w, dim=1)
    # feats_matrix_x_s = F.cosine_similarity(feats_x_ulb, feats_x_ulb_s, dim=1)
    # log_pred_x = F.log_softmax(logits_x_ulb, dim=-1)
    # log_pred_w = F.log_softmax(logits_x_ulb_w, dim=-1)
    # log_pred_s = F.log_softmax(logits_x_ulb_s, dim=-1)
    # ce_x_w = torch.sum(-log_pred_x * log_pred_w, dim=1)
    # ce_x_s = torch.sum(-log_pred_x * log_pred_s, dim=1)
    # loss = torch.norm(F.normalize(feats_matrix_x_w / feats_matrix_x_s, p=2, dim=0) - F.normalize(ce_x_w / ce_x_s, p=2, dim=0), p=1)
    # return loss



    # feats_matrix_x_x = F.cosine_similarity(feats_x_ulb.unsqueeze(1), feats_x_ulb.unsqueeze(0), dim=2)
    # feats_matrix_x_x = feats_matrix_x_x - torch.diag_embed(torch.diag(feats_matrix_x_x))
    # feats_matrix_x_w = F.cosine_similarity(feats_x_ulb, feats_x_ulb_w, dim=1)
    # feats_matrix_x_s = F.cosine_similarity(feats_x_ulb, feats_x_ulb_s, dim=1)
    # loss_w = torch.mean(
    #     torch.clamp(-feats_matrix_x_w + torch.logsumexp(torch.mul(feats_matrix_x_x, 1000000), dim=1) / 1000000 + delta, min=0))
    # loss_s = torch.mean(
    #     torch.clamp(-feats_matrix_x_s + torch.logsumexp(torch.mul(feats_matrix_x_x, 1000000), dim=1) / 1000000 + delta, min=0))
    # loss = loss_w + loss_s
    # return loss

    similarity_matrix_same_x_w = F.cosine_similarity(feats_x_ulb, feats_x_ulb_w, dim=1)
    similarity_matrix_same_x_s = F.cosine_similarity(feats_x_ulb, feats_x_ulb_s, dim=1)
    similarity_matrix_diff_x_w = F.cosine_similarity(feats_x_ulb.unsqueeze(1), feats_x_ulb_w.unsqueeze(0), dim=2)
    similarity_matrix_diff_x_s = F.cosine_similarity(feats_x_ulb.unsqueeze(1), feats_x_ulb_s.unsqueeze(0), dim=2)

    similarity_matrix_same_w_s = F.cosine_similarity(feats_x_ulb_w, feats_x_ulb_s, dim=1)
    similarity_matrix_diff_w_w = F.cosine_similarity(feats_x_ulb_w.unsqueeze(1), feats_x_ulb_w.unsqueeze(0), dim=2)
    similarity_matrix_diff_w_s = F.cosine_similarity(feats_x_ulb_w.unsqueeze(1), feats_x_ulb_s.unsqueeze(0), dim=2)
    similarity_matrix_diff_s_s = F.cosine_similarity(feats_x_ulb_s.unsqueeze(1), feats_x_ulb_s.unsqueeze(0), dim=2)
    similarity_matrix_diff_s_w = F.cosine_similarity(feats_x_ulb_s.unsqueeze(1), feats_x_ulb_w.unsqueeze(0), dim=2)

    similarity_matrix_diff_x_w = similarity_matrix_diff_x_w - torch.diag_embed(
        torch.diag(similarity_matrix_diff_x_w))
    similarity_matrix_diff_x_s = similarity_matrix_diff_x_s - torch.diag_embed(
        torch.diag(similarity_matrix_diff_x_s))

    similarity_matrix_diff_w_w = similarity_matrix_diff_w_w - torch.diag_embed(
        torch.diag(similarity_matrix_diff_w_w))
    similarity_matrix_diff_s_s = similarity_matrix_diff_s_s - torch.diag_embed(
        torch.diag(similarity_matrix_diff_s_s))
    similarity_matrix_diff_w_s = similarity_matrix_diff_w_s - torch.diag_embed(
        torch.diag(similarity_matrix_diff_w_s))
    similarity_matrix_diff_s_w = similarity_matrix_diff_s_w - torch.diag_embed(
        torch.diag(similarity_matrix_diff_s_w))

    loss_w_w_s_s = torch.sum(torch.clamp(-similarity_matrix_same_w_s + torch.logsumexp(torch.mul(torch.cat((similarity_matrix_diff_w_w, similarity_matrix_diff_s_s), axis=1), 1000000), dim=1) / 1000000 + delta, min=0))
    loss_s = torch.sum(torch.clamp(-similarity_matrix_same_w_s + torch.logsumexp(torch.mul(similarity_matrix_diff_s_s, 1000000), dim=1) / 1000000 + delta, min=0))
    loss_w = torch.sum(torch.clamp(-similarity_matrix_same_w_s + torch.logsumexp(torch.mul(similarity_matrix_diff_w_w, 1000000), dim=1) / 1000000 + delta, min=0))
    loss_w_all = torch.sum(torch.clamp(-similarity_matrix_same_w_s + torch.logsumexp(
        torch.mul(torch.cat((similarity_matrix_diff_w_w, similarity_matrix_diff_w_s), axis=1), 1000000),
        dim=1) / 1000000 + delta, min=0))
    loss_s_all = torch.sum(torch.clamp(-similarity_matrix_same_w_s + torch.logsumexp(
        torch.mul(torch.cat((similarity_matrix_diff_s_s, similarity_matrix_diff_s_w), axis=1), 1000000),
        dim=1) / 1000000 + delta, min=0))

    loss_x_w = torch.sum(torch.clamp(-similarity_matrix_same_x_w + torch.logsumexp(torch.mul(similarity_matrix_diff_x_w, 1000000), dim=1) / 1000000 + delta, min=0))
    loss_x_s = torch.sum(torch.clamp(-similarity_matrix_same_x_s + torch.logsumexp(torch.mul(similarity_matrix_diff_x_s, 1000000), dim=1) / 1000000 + delta, min=0))

    if m_model == 1:
        return loss_w_w_s_s
    elif m_model == 2:
        return (loss_w_all + loss_s_all) / 2
    elif m_model == 3:
        return (loss_w + loss_s) / 2
    elif m_model == 4:
        return loss_x_s
    else:
        return 0
    # euclidean_distances_same_w_s = torch.norm(feats_x_ulb_w - feats_x_ulb_s, dim=1)
    # euclidean_distances_diff_w_w = torch.norm(feats_x_ulb_w[:, None, :] - feats_x_ulb_w[None, :, :], dim=2)
    # euclidean_distances_diff_w_s = torch.norm(feats_x_ulb_w[:, None, :] - feats_x_ulb_s[None, :, :], dim=2)
    # modified_euclidean_distances_diff_w_w = euclidean_distances_diff_w_w.clone()
    # modified_euclidean_distances_diff_w_w.fill_diagonal_(float('inf'))
    # loss = torch.mean(torch.clamp(euclidean_distances_same_w_s + torch.logsumexp(torch.mul(-modified_euclidean_distances_diff_w_w, 1000000), dim=1) / 1000000 + delta, min=0))
    # return loss

class MetricLoss(nn.Module):
    def forward(self, logits_x_ulb, logits_x_ulb_w, logits_x_ulb_s, feats_x_ulb, feats_x_ulb_w, feats_x_ulb_s, delta, m_model):
        return metric_loss(logits_x_ulb, logits_x_ulb_w, logits_x_ulb_s, feats_x_ulb, feats_x_ulb_w, feats_x_ulb_s, delta, m_model)