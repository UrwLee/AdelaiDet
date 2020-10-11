# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.structures import ImageList
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures.instances import Instances
from detectron2.structures.masks import polygons_to_bitmask

from .dynamic_mask_head import build_dynamic_mask_head
from .mask_branch import build_mask_branch

from adet.utils.comm import aligned_bilinear

__all__ = ["CondInst"]


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class CondInst(nn.Module):
    """
    Main class for CondInst architectures (see https://arxiv.org/abs/2003.05664).
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.mask_head = build_dynamic_mask_head(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.max_proposals = cfg.MODEL.CONDINST.MAX_PROPOSALS

        # build top module
        in_channels = self.proposal_generator.in_channels_to_top_module

        self.controller_body = nn.Conv2d(
            in_channels, self.mask_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )
        self.controller_edge = nn.Conv2d(
            in_channels, self.mask_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )
        self.controller_final = nn.Conv2d(
            in_channels, 81,
            kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.controller_body.weight, std=0.01)
        torch.nn.init.constant_(self.controller_body.bias, 0)
        torch.nn.init.normal_(self.controller_edge.weight, std=0.01)
        torch.nn.init.constant_(self.controller_edge.bias, 0)
        torch.nn.init.normal_(self.controller_final.weight, std=0.01)
        torch.nn.init.constant_(self.controller_final.bias, 0)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1))
        else:
            gt_instances = None

        proposals, proposal_losses,cls_fea_fusion = self.proposal_generator(
            images, features, gt_instances, self.controller_body,self.controller_edge,self.controller_final
        )

        mask_feats, sem_losses = self.mask_branch(features, gt_instances,cls_fea_fusion = cls_fea_fusion)

        if self.training:
            loss_mask = self._forward_mask_heads_train(proposals, mask_feats, gt_instances,images.tensor)

            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            losses.update({"loss_mask": loss_mask})
            return losses
        else:
            pred_instances_w_masks = self._forward_mask_heads_test(proposals, mask_feats)

            padded_im_h, padded_im_w = images.tensor.size()[-2:]
            processed_results = []
            for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
                instances_per_im = self.postprocess(
                    instances_per_im, height, width,
                    padded_im_h, padded_im_w
                )

                processed_results.append({
                    "instances": instances_per_im
                })

            return processed_results

    def _forward_mask_heads_train(self, proposals, mask_feats, gt_instances,images):
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]

        if 0 <= self.max_proposals < len(pred_instances):
            inds = torch.randperm(len(pred_instances), device=mask_feats['mask_feats_body'].device).long()
            logger.info("clipping proposals from {} to {}".format(
                len(pred_instances), self.max_proposals
            ))
            pred_instances = pred_instances[inds[:self.max_proposals]]

        pred_instances.mask_head_params_body = pred_instances.top_feats_body
        pred_instances.mask_head_params_edge = pred_instances.top_feats_edge
        pred_instances.mask_head_params_final = pred_instances.top_feats_final

        loss_mask = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            pred_instances, gt_instances,images
        )

        return loss_mask

    def _forward_mask_heads_test(self, proposals, mask_feats):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params_body = pred_instances.top_feats_body
        pred_instances.mask_head_params_edge = pred_instances.top_feats_edge
        pred_instances.mask_head_params_final = pred_instances.top_feats_final

        pred_instances_w_masks = self.mask_head(
            mask_feats, self.mask_branch.out_stride, pred_instances
        )

        return pred_instances_w_masks

    def add_bitmasks(self, instances, im_h, im_w):
        for i in range(len(instances)):
            per_im_gt_inst = instances[i]
            if not per_im_gt_inst.has("gt_masks"):
                continue
            polygons = per_im_gt_inst.get("gt_masks").polygons
            per_im_bitmasks = []
            per_im_bitmasks_full = []
            im_id = []
            for per_polygons in polygons:
                bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                bitmask = torch.from_numpy(bitmask).to(self.device).float()
                start = int(self.mask_out_stride // 2)
                bitmask_full = bitmask.clone()
                bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                assert bitmask.size(0) * self.mask_out_stride == im_h
                assert bitmask.size(1) * self.mask_out_stride == im_w

                per_im_bitmasks.append(bitmask)
                per_im_bitmasks_full.append(bitmask_full)
                im_id.append(i)

            per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
            per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            per_im_gt_inst.im_id = im_id

    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks, factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()
        # edge and body

        if results.has("pred_edge_masks"):
            # edge
            pred_edge_masks = aligned_bilinear(
                results.pred_edge_masks, factor)
            pred_edge_masks = pred_edge_masks[:, :, :resized_im_h, :resized_im_w]
            pred_edge_masks = F.interpolate(
                pred_edge_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False)
            pred_edge_masks = pred_edge_masks[:, 0, :, :]
            results.pred_edge = (pred_edge_masks > mask_threshold).float()
            # body
            pred_body_masks = aligned_bilinear(
                results.pred_body_masks, factor)
            pred_body_masks = pred_body_masks[:, :, :resized_im_h, :resized_im_w]
            pred_body_masks = F.interpolate(
                pred_body_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False)
            pred_body_masks = pred_body_masks[:, 0, :, :]
            results.pred_body = (pred_body_masks > mask_threshold).float()

        return results
