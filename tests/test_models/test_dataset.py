import unittest
import torch
import sys
#sys.path.append('../../')
sys.path.append('/root/autodl-fs/projects/DINO')
from models.dino.dn_components import dn_post_process,prepare_for_cdn
import torch.nn as nn
from test_dino import printDict

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
class TestDnComponents(unittest.TestCase):
    def setUp(self):
        self.args = {
            "config_file": "config/DINO/DINO_4scale.py",
            "options": {
                "dn_scalar": 100,
                "embed_init_tgt": True,
                "dn_label_coef": 1.0,
                "dn_bbox_coef": 1.0,
                "use_ema": False,
                "dn_box_noise_scale": 1.0
            },
            "dataset_file": "exdark",
            "coco_path": "/root/autodl-tmp/COCO/",
            "coco_panoptic_path": None,
            "remove_difficult": False,
            "fix_size": False,
            "output_dir": "logs/DINO/R50-MS4",
            "note": "",
            "device": "cuda",
            "seed": 42,
            "resume": "",
            "pretrain_model_path": "./pretrained/checkpoint0033_4scale.pth",
            "finetune_ignore": ["label_enc.weight", "class_embed"],
            "start_epoch": 0,
            "eval": False,
            "num_workers": 10,
            "test": False,
            "debug": False,
            "find_unused_params": False,
            "save_results": False,
            "save_log": False,
            "pre_encoder": False,
            "world_size": 1,
            "dist_url": "env://",
            "rank": 0,
            "local_rank": 0,
            "amp": False,
            "distributed": False,


            "data_aug_scales": [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
            "data_aug_max_size": 1333,
            "data_aug_scales2_resize": [400, 500, 600],
            "data_aug_scales2_crop": [384, 600],
            "data_aug_scale_overlap": None,
            "num_classes": 13,
            "lr": 0.0001,
            "param_dict_type": "default",
            "lr_backbone": 1e-05,
            "lr_backbone_names": ["backbone.0"],
            "lr_linear_proj_names": ["reference_points", "sampling_offsets"],
            "lr_linear_proj_mult": 0.1,
            "ddetr_lr_param": False,
            "batch_size": 1,
            "weight_decay": 0.0001,
            "epochs": 12,
            "lr_drop": 11,
            "save_checkpoint_interval": 1,
            "clip_max_norm": 0.1,
            "onecyclelr": False,
            "multi_step_lr": False,
            "lr_drop_list": [33, 45],
            "modelname": "dino",
            "frozen_weights": None,
            "backbone": "resnet50",
            "use_checkpoint": False,
            "dilation": False,
            "position_embedding": "sine",
            "pe_temperatureH": 20,
            "pe_temperatureW": 20,
            "return_interm_indices": [1, 2, 3],
            "backbone_freeze_keywords": None,
            "enc_layers": 6,
            "dec_layers": 6,
            "unic_layers": 0,
            "pre_norm": False,
            "dim_feedforward": 2048,
            "hidden_dim": 256,
            "dropout": 0.0,
            "nheads": 8,
            "num_queries": 900,
            "query_dim": 4,
            "num_patterns": 0,
            "pdetr3_bbox_embed_diff_each_layer": False,
            "pdetr3_refHW": -1,
            "random_refpoints_xy": False,
            "fix_refpoints_hw": -1,
            "dabdetr_yolo_like_anchor_update": False,
            "dabdetr_deformable_encoder": False,
            "dabdetr_deformable_decoder": False,
            "use_deformable_box_attn": False,
            "box_attn_type": "roi_align",
            "dec_layer_number": None,
            "num_feature_levels": 4,
            "enc_n_points": 4,
            "dec_n_points": 4,
            "decoder_layer_noise": False,
            "dln_xy_noise": 0.2,
            "dln_hw_noise": 0.2,
            "add_channel_attention": False,
            "add_pos_value": False,
            "two_stage_type": "standard",
            "two_stage_pat_embed": 0,
            "two_stage_add_query_num": 0,
            "two_stage_bbox_embed_share": False,
            "two_stage_class_embed_share": False,
            "two_stage_learn_wh": False,
            "two_stage_default_hw": 0.05,
            "two_stage_keep_all_tokens": False,
            "num_select": 300,
            "transformer_activation": "relu",
            "batch_norm_type": "FrozenBatchNorm2d",
            "masks": False,
            "aux_loss": True,
            "set_cost_class": 2.0,
            "set_cost_bbox": 5.0,
            "set_cost_giou": 2.0,
            "cls_loss_coef": 1.0,
            "mask_loss_coef": 1.0,
            "dice_loss_coef": 1.0,
            "bbox_loss_coef": 5.0,
            "giou_loss_coef": 2.0,
            "enc_loss_coef": 1.0,
            "interm_loss_coef": 1.0,
            "no_interm_box_loss": False,
            "focal_alpha": 0.25,
            "decoder_sa_type": "sa",
            "matcher_type": "HungarianMatcher",
            "decoder_module_seq": ["sa", "ca", "ffn"],
            "nms_iou_threshold": -1,
            "dec_pred_bbox_embed_share": True,
            "dec_pred_class_embed_share": True,
            "use_dn": True,
            "dn_number": 100,
            "dn_box_noise_scale": 1.0,
            "dn_label_noise_ratio": 0.5,
            "embed_init_tgt": True,
            "dn_labelbook_size": 91,
            "match_unstable_error": True,
            "use_ema": False,
            "ema_decay": 0.9997,
            "ema_epoch": 0,
            "use_detached_boxes_dec_out": False,


            "dn_scalar": 100,
            "dn_label_coef": 1.0,
            "dn_bbox_coef": 1.0
        }
        # with open('/root/autodl-tmp/projects/DINO/args.json', 'r') as file:
        #     json_str = file.read()
        #self.args = json.loads(json_str)
        #print(self.args.keys())
        class DotDict(dict):
            def __getattr__(self, attr):
                if attr in self:
                    return self[attr]
                else:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__
        self.args= DotDict(self.args)
        #print("self.args.ddataset_file",self.args.dataset_file)
        self.dataset_val = build_dataset(image_set='val', args=self.args)
    
    def test_anns(self):
        from util.box_ops import box_cxcywh_to_xyxy
        coco = self.dataset_val.coco
        anns = coco.loadAnns(coco.getAnnIds(self.dataset_val.ids))
        print(anns[:10])
        img, target = self.dataset_val[0]
        print("img",img,"target",target)
        # boxes = target['boxes']
        # s = target['orig_size']
        # w,h = s[0],s[1]
        # for box in boxes:
        #     box = box * torch.tensor([w,h,w,h])
        #     box = box_cxcywh_to_xyxy(box)
        #     print("box",box)
        

if __name__ == "__main__":
    unittest.main()
