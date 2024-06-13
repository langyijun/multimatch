import torch
import torch.nn.functional as F

from .utils import FreeMatchThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.comatch.mcc import MinimumClassConfusionLoss
import torch.nn as nn
import numpy as np

# TODO: move these to .utils or algorithms.utils.loss
def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    # mask:经过阈值过滤后的掩码
    # logits_s:无标签强增强样本的预测分布
    # prob_model:局部自适应阈值
    # label_hist:标签直方图
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()  # 论文公式(9)下半部分, prob_s.mean()为公式(9)上半部分

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)  # 论文公式(11)的前半部分

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)  #交叉熵
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()

def ce_loss(logits, targets, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)

@ALGORITHMS.register('freematch')
class FreeMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, use_quantile=args.use_quantile, clip_thresh=args.clip_thresh)
        self.lambda_e = args.ent_loss_ratio

        self.rot_classifier = nn.Linear(128, 4, bias=False).to(self.gpu)
        nn.init.xavier_normal_(self.rot_classifier.weight.data)

        self.lambda_mcc = args.lambda_mcc
        self.use_mcc = args.use_mcc

        self.lambda_rot = args.lambda_rot
        self.use_rot = args.use_rot

        self.use_non_zero = args.use_non_zero

        self.delta = args.delta
        self.m_model = args.m_model
        self.lambda_m = args.lambda_m

    def init(self, T, hard_label=True, ema_p=0.999, use_quantile=True, clip_thresh=False):
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh


    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FreeMatchThresholdingHook(num_classes=self.num_classes, momentum=self.args.ema_p), "MaskingHook")

        # if self.args.pseudo_label_model == 2:
        #     lb_class_dist = [0 for _ in range(self.num_classes)]
        #     for c in self.dataset_dict['train_lb'].targets:
        #         lb_class_dist[c] += 1
        #     lb_class_dist = np.array(lb_class_dist)
        #     lb_class_dist = lb_class_dist / lb_class_dist.sum()
        #     self.register_hook(
        #         DistAlignEMAHook(num_classes=self.num_classes, p_target_type='gt', p_target=lb_class_dist),
        #         "DistAlignHook")
        #
        # elif self.args.pseudo_label_model == 3:
        #     self.register_hook(
        #         DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p,
        #                          p_target_type='uniform' if self.args.dist_uniform else 'model'),
        #         "DistAlignHook")


        super().set_hooks()


    def train_step(self, x_lb, y_lb, x_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb = outputs['logits'][num_lb:2*num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][2*num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb = outputs['feat'][num_lb:2 * num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][2*num_lb:].chunk(2)

                # k次增强
                # x_ulb_w = x_ulb_w.view(-1, 3, 32, 32)
                # x_ulb_s = x_ulb_s.view(-1, 3, 32, 32)
                # inputs = torch.cat((x_lb, x_ulb, x_ulb_w, x_ulb_s))
                # outputs = self.model(inputs)
                # logits_x_lb = outputs['logits'][:num_lb]
                # feats_x_lb = outputs['feat'][:num_lb]
                # logits_x_ulb = outputs['logits'][num_lb:2 * num_lb]
                # feats_x_ulb = outputs['feat'][num_lb:2 * num_lb]
                # logits_x_ulb_w_list, logits_x_ulb_s_list = outputs['logits'][2 * num_lb:].chunk(2)
                # feats_x_ulb_w_list, feats_x_ulb_s_list = outputs['feat'][2 * num_lb:].chunk(2)

            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']

            # feats_x_ulb_w = torch.mean(torch.stack(feats_x_ulb_w_list.chunk(3), dim=0), dim=0)
            # feats_x_ulb_s = torch.mean(torch.stack(feats_x_ulb_s_list.chunk(3), dim=0), dim=0)
            # logits_x_ulb_w = torch.mean(torch.stack(logits_x_ulb_w_list.chunk(3), dim=0), dim=0)
            # logits_x_ulb_s = torch.mean(torch.stack(logits_x_ulb_s_list.chunk(3), dim=0), dim=0)
            # feats_x_ulb_w, _, _ = feats_x_ulb_w_list.chunk(3)
            # feats_x_ulb_s, _, _ = feats_x_ulb_s_list.chunk(3)
            # logits_x_ulb_w, _, _ = logits_x_ulb_w_list.chunk(3)
            # logits_x_ulb_s, _, _ = logits_x_ulb_s_list.chunk(3)

            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}


            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')



            # calculate mask



            # generate unlabeled targets using pseudo label hook
            # if self.args.pseudo_label_model == 1:
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)

            # 5.27对称
            pseudo_label_2 = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=logits_x_ulb_s,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)
            mask_2 = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_s)



            # elif self.args.pseudo_label_model == 2:
            #     prob_x_ulb_w = self.call_hook("dist_align", "DistAlignHook",
            #                                   probs_x_ulb=self.compute_prob(logits_x_ulb_w))
            #     pseudo_label = prob_x_ulb_w ** (1 / self.T)
            #     pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True)).detach()
            #     mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)
            #
            # elif self.args.pseudo_label_model == 3:
            #     probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            #     probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
            #     probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w,
            #                                    probs_x_lb=probs_x_lb)
            #     # calculate weight
            #     mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            #     pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
            #                                   # make sure this is logits, not dist aligned probs
            #                                   # uniform alignment in softmatch do not use aligned probs for generating pseudo labels
            #                                   logits=logits_x_ulb_w,
            #                                   use_hard_label=self.use_hard_label,
            #                                   T=self.T)

            # calculate unlabeled loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask,
                                          use_non_zero=self.use_non_zero)

            unsup_loss_2 = self.consistency_loss(logits_x_ulb_w,
                                               pseudo_label_2,
                                               'ce',
                                               mask=mask_2,
                                               use_non_zero=self.use_non_zero)
            
            # calculate entropy loss
            if mask.sum() > 0:
               ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
            else:
               ent_loss = 0.0

            if mask_2.sum() > 0:
               ent_loss_2, _ = entropy_loss(mask_2, logits_x_ulb_w, self.p_model, self.label_hist)
            else:
               ent_loss_2 = 0.0

            if self.use_mcc:
                mcc_loss_fn = MinimumClassConfusionLoss()   # 最小化类内混淆
                mcc_loss = mcc_loss_fn(logits_x_ulb_w)
            else:
                mcc_loss = 0.0

            if self.use_rot:
                x_ulb_r = torch.cat(
                    [torch.rot90(x_ulb_w[:num_lb], i, [2, 3]) for i in range(4)], dim=0).to(self.gpu)
                y_ulb_r = torch.cat(
                    [torch.empty(x_ulb_w[:num_lb].size(0)).fill_(i).long() for i in range(4)], dim=0).to(self.gpu)
                self.bn_controller.freeze_bn(self.model)
                logits_rot = self.rot_classifier(self.model(x_ulb_r)['feat'])
                self.bn_controller.unfreeze_bn(self.model)
                rot_loss = ce_loss(logits_rot, y_ulb_r, reduction='mean')
            else:
                rot_loss = 0.0

            metric_loss = self.metric_loss(logits_x_ulb, logits_x_ulb_w, logits_x_ulb_s, feats_x_ulb, feats_x_ulb_w, feats_x_ulb_s, self.delta, self.m_model)

            total_loss = sup_loss + self.lambda_u * (unsup_loss + unsup_loss_2) / 2 + self.lambda_e * (ent_loss + ent_loss_2) / 2 + self.lambda_m * metric_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(),
                                         metric_loss=metric_loss.item(),
                                         # mcc_loss=mcc_loss.item(),
                                         # rot_loss=rot_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['MaskingHook'].p_model.cpu()
        save_dict['time_p'] = self.hooks_dict['MaskingHook'].time_p.cpu()
        save_dict['label_hist'] = self.hooks_dict['MaskingHook'].label_hist.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--ent_loss_ratio', float, 0.01),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
        ]