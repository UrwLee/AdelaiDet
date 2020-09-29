import torch
from torch.nn import functional as F
from torch import nn

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import cv2
import random

from adet.utils.comm import compute_locations, aligned_bilinear


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        x_concat = None
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )

            if i < n_layers - 1:
                x = F.relu(x)
            if i == n_layers-2:
                x_concat = x

        return x,x_concat

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        locations = compute_locations(
            mask_feats['mask_feats_body'].size(2), mask_feats['mask_feats_body'].size(3),
            stride=mask_feat_stride, device=mask_feats['mask_feats_body'].device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats['mask_feats_body'].size()
        # make sure self.in_channels == mask_feat.channel

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats['mask_feats_body'].dtype)

            mask_head_inputs_body = torch.cat([
                relative_coords, mask_feats['mask_feats_body'][im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
            mask_head_inputs_edge = torch.cat([
                relative_coords, mask_feats['mask_feats_edge'][im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs_body = mask_feats['mask_feats_body'][im_inds].reshape(n_inst, self.in_channels, H * W)
            mask_head_inputs_edge = mask_feats['mask_feats_edge'][im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs_body = mask_head_inputs_body.reshape(1, -1, H, W)
        mask_head_inputs_edge = mask_head_inputs_edge.reshape(1, -1, H, W)

        weights_body, biases_body = parse_dynamic_params(
            mask_head_params[:,:self.num_gen_params], self.channels,
            self.weight_nums, self.bias_nums
        )
        weights_edge, biases_edge = parse_dynamic_params(
            mask_head_params[:,self.num_gen_params:-81], self.channels,
            self.weight_nums, self.bias_nums
        )
        weights_final, biases_final = parse_dynamic_params(
            mask_head_params[:, -81:], self.channels,
            self.weight_nums[1:], self.bias_nums[1:]
        )

        mask_logits_body,mask_logits_body_concat = self.mask_heads_forward(mask_head_inputs_body, weights_body, biases_body, n_inst)
        mask_logits_edge,mask_logits_edge_concat = self.mask_heads_forward(mask_head_inputs_edge, weights_edge, biases_edge, n_inst)

        mask_logits_final,_ = self.mask_heads_forward(mask_logits_body_concat+mask_logits_edge_concat,weights_final,biases_final,n_inst)

        mask_logits_body = mask_logits_body.reshape(-1, 1, H, W)
        mask_logits_edge = mask_logits_edge.reshape(-1, 1, H, W)
        mask_logits_final = mask_logits_final.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits_body = aligned_bilinear(mask_logits_body, int(mask_feat_stride / self.mask_out_stride))
        mask_logits_edge = aligned_bilinear(mask_logits_edge, int(mask_feat_stride / self.mask_out_stride))
        mask_logits_final = aligned_bilinear(mask_logits_final, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits_body.sigmoid(),mask_logits_edge.sigmoid(),mask_logits_final.sigmoid()

    def generate_body_and_edge(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        unknown = np.array(np.not_equal(mask, 0).astype(np.float32))
        dilation = cv2.dilate(unknown, kernel, iterations=2)
        corrosion = cv2.erode(unknown, kernel, iterations=1)
        edge = dilation - corrosion
        # body = np.where(mask!=0,1,0)
        body = np.where(edge == 1, 0, mask/255)
        return edge.astype(np.uint8), body.astype(np.uint8)

    def onehot_to_binary_edges(self,mask, radius):
        """
        Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

        """
        import time
        t1 = time.time()
        if radius < 0:
            return mask

        # We need to pad the borders for boundary conditions
        mask_pad = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        edgemap = np.zeros(mask.shape)

        dist = distance_transform_edt(mask_pad) + distance_transform_edt(1.0 - mask_pad)
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
        edgemap = (edgemap > 0).astype(np.uint8)
        print(time.time()-t1)
        return edgemap

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None,image_tensor = None):
        if self.training:
            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats['mask_feats_body'].dtype)
            device = gt_bitmasks.device
            edges = []
            bodys = []

            for i in range(gt_bitmasks.shape[0]):
                mask = gt_bitmasks[i, 0, :, :].cpu().numpy() * 255
                edge, body = self.generate_body_and_edge(mask)
                # edge = self.onehot_to_binary_edges(mask,2)
                # edge_small = self.onehot_to_binary_edges(mask,1)
                # body = (mask*edge_small).astype('uint8')
                edges.append(edge[np.newaxis, :, :])
                bodys.append(body[np.newaxis, :, :])
            gt_bitmasks_edge = torch.from_numpy(np.concatenate(edges)).unsqueeze(dim=1).to(device)
            gt_bitmasks_body = torch.from_numpy(np.concatenate(bodys)).unsqueeze(dim=1).to(device)
            ##########################################################################################

            ############################## debug ##########################################################################
            debug = False
            if debug:
                image_tensor = image_tensor.cpu()
                gt_im_id = []
                for per_im in gt_instances:
                    gt_im_id += per_im.im_id
                gt_im_id = (torch.tensor(gt_im_id))[gt_inds]
                gt_bitmasks_full = torch.cat(
                    [per_im.gt_bitmasks_full for per_im in gt_instances])  # torch.Size([num_gt, 200, 312])
                gt_bitmasks_full = gt_bitmasks_full[gt_inds].unsqueeze(dim=1).to(
                    dtype=mask_feats['mask_feats_body'].dtype)  # torch.Size([500, 1, 200, 312])
                edges_full = []
                bodys_full = []
                for i in range(gt_bitmasks_full.shape[0]):
                    mask = gt_bitmasks_full[i, 0, :, :].cpu().numpy() * 255
                    edge, body = self.generate_body_and_edge(mask)
                    # edge = self.onehot_to_binary_edges(mask, 2)
                    # edge_small = self.onehot_to_binary_edges(mask, 1)
                    # body = (mask * edge_small).astype('uint8')
                    edges_full.append(edge[np.newaxis, :, :])
                    bodys_full.append(body[np.newaxis, :, :])
                gt_bitmasks_edge_full = torch.from_numpy(np.concatenate(edges_full)).unsqueeze(dim=1)
                gt_bitmasks_body_full = torch.from_numpy(np.concatenate(bodys_full)).unsqueeze(dim=1)
                for i in range(len(gt_im_id)):
                    image = image_tensor[gt_im_id[i], :, :, :].permute(1, 2, 0).numpy()
                    image = image * [1.0, 1.0, 1.0] + [103.530, 116.280, 123.675]
                    image = image.astype('float32')
                    mask = gt_bitmasks_full[i, 0, :, :].cpu().numpy() * 255
                    edge = (gt_bitmasks_edge_full[i, 0, :, :].cpu().numpy() * 255).astype('float32')
                    body = (gt_bitmasks_body_full[i, 0, :, :].cpu().numpy() * 255).astype('float32')
                    diff = (mask - body).astype('float32')
                    image_mask = cv2.addWeighted(image, 0.5, np.repeat(mask[:, :, np.newaxis], 3, 2), 0.5, 0.0)
                    image_edge = cv2.addWeighted(image, 0.5, np.repeat(edge[:, :, np.newaxis], 3, 2), 0.5, 0.0)
                    image_body = cv2.addWeighted(image, 0.5, np.repeat(body[:, :, np.newaxis], 3, 2), 0.5, 0.0)
                    show = np.concatenate([image_mask, image_edge, image_body, np.repeat(diff[:, :, np.newaxis], 3, 2)], 1)
                    cv2.imwrite('/opt/tiger/toutiao/labcv/dmx_loop/data/visiual/condinst/train_test_mask/{}.jpg'.format(
                        str(random.randint(0, 10000))), show)
            ################################################################################################################
            if len(pred_instances) == 0:
                loss_mask = mask_feats['mask_feats_edge'].sum() * 0+mask_feats['mask_feats_body'].sum() * 0 + pred_instances.mask_head_params.sum() * 0
            else:
                mask_scores_body,mask_scores_edge,mask_scores_final = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances)
                mask_losses_body = dice_coefficient(mask_scores_body, gt_bitmasks_body)
                mask_losses_edge = dice_coefficient(mask_scores_edge, gt_bitmasks_edge)
                mask_losses_final = dice_coefficient(mask_scores_final, gt_bitmasks)
                loss_mask =0.5* mask_losses_body.mean()+0.5*mask_losses_edge.mean()+mask_losses_final.mean()
                
            return loss_mask.float()
        else:
            if len(pred_instances) > 0:

                mask_scores_body,mask_scores_edge,mask_scores_final = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_scores_final.float()
                pred_instances.pred_body_masks = mask_scores_body.float()
                pred_instances.pred_edge_masks = mask_scores_edge.float()

            return pred_instances

def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    # layer = getattr(cfg.MODEL, 'BNFUNC')
    normalization_layer =  torch.nn.BatchNorm2d(in_channels)
    return normalization_layer
