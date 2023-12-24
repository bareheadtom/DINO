import unittest
import torch
import sys
sys.path.append('../../')
sys.path.append('/root/autodl-fs/projects/DINO')
from models.dino.dn_components import dn_post_process,prepare_for_cdn
from models.dino.dino import DINO,build_dino
from models.dino.position_encoding import *
from models.dino.backbone import *
import json
def printDict(map):
    #print("printDict")
    for k,v in map.items():
        if isinstance(v, dict):
            printDict(v)
        elif isinstance(v, list):
            for it in v:
                printDict(it)
        elif type(v) == int:
            map[k] = v
        else:
            map[k] = v.shape
import copy
def getrecur(map):
    print("getrecur")
    if isinstance(map, dict):
        for k,v in map.items():
            getrecur(v)
    elif isinstance(v, list):
        for it in v:
            getrecur(it)
    elif hasattr(map, 'shape'):
        map = map.shape

def getShapeDict(map):
    mapt = copy.deepcopy(map)
    getShapeDict(mapt)
    return mapt

class Tensor1:
    def __init__(self) -> None:
        self.a = 12
    def __str__(self):
        return f"Tensor shape: {self.a}"

class TestDINO(unittest.TestCase):

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
        print("self.args.ddataset_file",self.args.dataset_file)
        self.model, self.criterion, self.postprocessors = build_dino(self.args)

    
    def test_model(self):
        print("\n************test_model")
        targets = [
            {"labels": torch.tensor([1, 2, 3]).to('cuda'), "boxes": torch.tensor([[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8],[0.1,0.2,0.3,0.4]]).to('cuda')},
            {"labels": torch.tensor([2, 3]).to('cuda'), "boxes": torch.tensor([[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8]]).to('cuda')}
        ]
        samples = NestedTensor(tensors=torch.randn(2,3,300,300).to('cuda'),mask=torch.randn(2,300,300).to('cuda'))
        self.model = self.model.to('cuda')
        output = self.model.forward(samples, targets)
        printDict(output)
        #print("output",output)
        {
            'pred_logits': torch.Size([2, 900, 13]),
            'pred_boxes': torch.Size([2, 900, 4]),
            'aux_outputs': [{
                'pred_logits': torch.Size([2, 900, 13]),
                'pred_boxes': torch.Size([2, 900, 4])
            }, {
                'pred_logits': torch.Size([2, 900, 13]),
                'pred_boxes': torch.Size([2, 900, 4])
            }, {
                'pred_logits': torch.Size([2, 900, 13]),
                'pred_boxes': torch.Size([2, 900, 4])
            }, {
                'pred_logits': torch.Size([2, 900, 13]),
                'pred_boxes': torch.Size([2, 900, 4])
            }, {
                'pred_logits': torch.Size([2, 900, 13]),
                'pred_boxes': torch.Size([2, 900, 4])
            }],
            'interm_outputs': {
                'pred_logits': torch.Size([2, 900, 13]),
                'pred_boxes': torch.Size([2, 900, 4])
            },
            'interm_outputs_for_matching_pre': {
                'pred_logits': torch.Size([2, 900, 13]),
                'pred_boxes': torch.Size([2, 900, 4])
            },
            'dn_meta': {
                'pad_size': 198,
                'num_dn_group': 33,
                'output_known_lbs_bboxes': {
                    'pred_logits': torch.Size([2, 198, 13]),
                    'pred_boxes': torch.Size([2, 198, 4]),
                    'aux_outputs': [{
                        'pred_logits': torch.Size([2, 198, 13]),
                        'pred_boxes': torch.Size([2, 198, 4])
                    }, {
                        'pred_logits': torch.Size([2, 198, 13]),
                        'pred_boxes': torch.Size([2, 198, 4])
                    }, {
                        'pred_logits': torch.Size([2, 198, 13]),
                        'pred_boxes': torch.Size([2, 198, 4])
                    }, {
                        'pred_logits': torch.Size([2, 198, 13]),
                        'pred_boxes': torch.Size([2, 198, 4])
                    }, {
                        'pred_logits': torch.Size([2, 198, 13]),
                        'pred_boxes': torch.Size([2, 198, 4])
                    }]
                }
            }
        }
        
    def t1est_getShapeDict(self):
        a = {
            'a': [{'a1':torch.randn(2,3,12,12),'a1':torch.randn(6,3,12,12),'a1':torch.randn(2,4,12,12)}],
            'b':{
                'b1':"aaa",
                'b2':torch.randn(2,3,2,12)
            },
            'c':5
        }
        
        print(a)
    
    def t1est__prepare_for_cdn(self):
        print("\n*********************t1est__prepare_for_cdn")
    
    
    
    def t1est_forward(self):
        print("test_forward")

    def t1est_criterion(self):
        print("\n************test_criterion")
    
    def t1est_postprocessors(self):
        print("\n************test_postprocessors")
        

if __name__ == "__main__":
    unittest.main()
