# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import torch.nn as nn
from torch.nn import functional as F

from .cross_entropy import ce_loss


def consistency_loss(logits, targets, name='ce', mask=None, use_non_zero=False):
    """
    consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagation, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask  # 论文公式(8)

    if use_non_zero:
        if torch.sum(loss) == 0:
            return loss.mean()

        else:
            loss_1index = torch.nonzero(loss)
            loss_non0 = loss[loss_1index]

            return loss_non0.mean()
    else:
        return loss.mean()


class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """

    def forward(self, logits, targets, name='ce', mask=None, use_non_zero=False):
        return consistency_loss(logits, targets, name, mask, use_non_zero)


class RankingLoss(torch.nn.Module):
    def __init__(self, embedding_size=128, margin_latent=0.5, margin_embedding=0.5, N=2, lambda_1=1, lambda_2=1, r=1.0):
        torch.nn.Module.__init__(self)
        self.margin_latent = margin_latent  # latent space
        self.margin_embedding = margin_embedding  # embedding margin
        self.N = N  # num of generation samples
        self.r = r  # noise rate
        self.embedding_size = embedding_size  # latent space embedding size
        self.margin_latent = margin_latent
        self.margin_embedding = margin_embedding
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.directions = torch.nn.Parameter(torch.randn(self.N, self.embedding_size).cuda())
        torch.nn.init.kaiming_normal_(self.directions, mode='fan_out')
        self.rate = (torch.tensor([i for i in range(1, self.N + 1)]) * self.r).reshape(-1, 1).cuda()

    def get_loss_in_latent(self, X_w, X_s1, X_s2):
        cos_sim_w_s1 = F.cosine_similarity(X_w, X_s1, dim=1, eps=1e-8)
        cos_sim_w_s2 = F.cosine_similarity(X_w, X_s2, dim=1, eps=1e-8)
        loss = torch.log(1 + torch.exp(cos_sim_w_s1 - cos_sim_w_s2 + self.margin_latent))

        # satisfy_size = X_w.size(0)
        # loss = list()
        # for i in range(satisfy_size):
        #     weak = X_w[i]
        #     strong1 = X_s1[i]
        #     strong2 = X_s2[i]
        #     cos_w_s1 = F.linear(l2_norm(weak), l2_norm(strong1))
        #     cos_w_s2 = F.linear(l2_norm(weak), l2_norm(strong2))
        #     loss1 = torch.log(1 + torch.exp(cos_w_s2 - cos_w_s1 + self.margin_latent))
        #     loss.append(loss1)
        # loss = sum(loss) / satisfy_size
        return loss.mean()

    def get_loss_after_latent(self, X_w, target_X_w, model, mask):
        sementic_changes = F.normalize(self.directions, p=2, dim=1)
        sementic_changes = F.normalize(sementic_changes * self.rate, p=2, dim=1)

        # 调整 X_w 的形状使其变为 batch_size*N*embedding_size
        gen_samples_ = X_w.unsqueeze(1)
        gen_samples_ = gen_samples_ + sementic_changes
        gen_samples_ = F.normalize(gen_samples_, dim=2, p=2)

        feat_X_w_ = model.classifier(X_w)
        feat_gen_ = model.classifier(gen_samples_)
        cosine_sim = F.cosine_similarity(feat_X_w_.unsqueeze(1), feat_gen_, dim=2, eps=1e-8)

        loss_gen_ranking_ = torch.log(
            1 + torch.exp(cosine_sim[:, :-1] - cosine_sim[:, 1:] + self.margin_embedding)).mean()
        gen_samples_ce_loss = 0
        for i in range(self.N):
            gen_samples_ce_loss += (F.cross_entropy(gen_samples_[:, i, :], target_X_w, reduction='none') * mask).mean()
        loss = loss_gen_ranking_ + gen_samples_ce_loss

        # satisfy_size = X_w.size(0)
        # sementic_changes = self.directions
        # sementic_changes = l2_norm(sementic_changes)
        # # get random directions
        # for i in range(self.N):
        #     sementic_changes[i] = sementic_changes[i] * (i + 1) * self.r
        # sementic_changes = l2_norm(sementic_changes)
        # loss_ranking = list()
        # loss_anchor = list()
        # for i in range(satisfy_size):
        #     label = target_X_w[i]
        #     if len(label.shape) == 0:  # 方式出现训练最后一个step时，出现v是一维的情况
        #         label = torch.unsqueeze(label, 0)
        #     gen_samples = torch.empty(self.N, X_w.size(1)).cuda()
        #     for j in range(self.N):
        #         gen_samples[j] = X_w[i] + sementic_changes[j]
        #     gen_samples = l2_norm(gen_samples)
        #     feat_X_w = model.fc(X_w[i])
        #     feat_gen = model.fc(gen_samples)
        #     cos_gx = F.linear(feat_gen, feat_X_w)
        #     loss_gen_ranking = torch.log(1 + torch.exp(cos_gx[1] - cos_gx[0] + self.margin_embedding))
        #     loss_ranking.append(loss_gen_ranking)
        #     # loss_gen_anchor = 0.5 * torch.log(1 + torch.exp(self.margin_anchor - cos_gx[0])) +
        #     #                  0.5 * torch.log(1 + torch.exp(self.margin_anchor - cos_gx[1]))
        #     feat_gen_0 = torch.unsqueeze(feat_gen[0], 0)
        #     feat_gen_1 = torch.unsqueeze(feat_gen[1], 0)
        #     loss_gen_anchor = 0.5 * (
        #                 F.cross_entropy(feat_gen_0, label, reduction='none') + F.cross_entropy(feat_gen_1, label,
        #                                                                                        reduction='none'))
        #     loss_anchor.append(loss_gen_anchor)
        # loss = (sum(loss_ranking) + sum(loss_anchor)) / satisfy_size

        return loss

    def forward(self, X_w_latent, X_s1_latent, X_s2_latent, target_X_w, model, mask):
        loss1 = self.get_loss_in_latent(X_w_latent, X_s1_latent, X_s2_latent)
        loss2 = self.get_loss_after_latent(X_w_latent, target_X_w, model, mask)
        return self.lambda_1 * loss1 + self.lambda_2 * loss2
