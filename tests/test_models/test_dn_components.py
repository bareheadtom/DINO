import unittest
import torch
import sys
#sys.path.append('../../')
sys.path.append('/root/autodl-fs/projects/DINO')
from models.dino.dn_components import dn_post_process,prepare_for_cdn
import torch.nn as nn
from test_dino import printDict

class TestDnComponents(unittest.TestCase):
    def t1est_prepare_for_cdn(self):
        print("\n************test_prepare_for_cdn")
        targets = [
            {"labels": torch.tensor([1, 2, 3]).to('cuda'), "boxes": torch.tensor([[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8],[0.1,0.2,0.3,0.4]]).to('cuda')},
            {"labels": torch.tensor([2, 3]).to('cuda'), "boxes": torch.tensor([[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8]]).to('cuda')}
        ]
        dn_number, dn_label_noise_ratio, dn_box_noise_scale = 100, 0.5, 1.0
        dn_labelbook_size = 91
        out = prepare_for_cdn(dn_args=(targets, dn_number, dn_label_noise_ratio, dn_box_noise_scale),
                        training=True, num_queries=900, num_classes=13,
                        hidden_dim=256, label_enc=nn.Embedding(91 + 1, 256).to('cuda')
                        )
        input_query_label, input_query_bbox, attn_mask, dn_meta = out
        print("input_query_label",input_query_label.shape)
        print("input_query_bbox",input_query_bbox.shape)
        print("attn_mask",attn_mask.shape)
        print("dn_meta",dn_meta)
        # input_query_label torch.Size([2, 198, 256])
        # input_query_bbox torch.Size([2, 198, 4])
        # attn_mask torch.Size([1098, 1098]) 900 + 198
        # dn_meta {'pad_size': 198, 'num_dn_group': 33}

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    def test_dn_post_process(self):
        print("\n************test_dn_post_process")
        outputs_class, outputs_coord, dn_meta = torch.randn(6,2,1098,13), torch.randn(6,2,1098,4),{'pad_size':198,'num_dn_group':33}
        outputs_class, outputs_coord = dn_post_process(outputs_class, outputs_coord, dn_meta,True,self._set_aux_loss)
        print("result outputs_class, outputs_coord",outputs_class, outputs_coord)
        #Tensor shape: torch.Size([6, 2, 900, 13]) Tensor shape: torch.Size([6, 2, 900, 4])
        print("dn_meta",dn_meta)
        # {
        #     'pad_size': 198,
        #     'num_dn_group': 33,
        #     'output_known_lbs_bboxes': {
        #         'pred_logits': Tensor shape: torch.Size([2, 198, 13]),
        #         'pred_boxes': Tensor shape: torch.Size([2, 198, 4]),
        #         'aux_outputs': [{
        #             'pred_logits': Tensor shape: torch.Size([2, 198, 13]),
        #             'pred_boxes': Tensor shape: torch.Size([2, 198, 4])
        #         }, {
        #             'pred_logits': Tensor shape: torch.Size([2, 198, 13]),
        #             'pred_boxes': Tensor shape: torch.Size([2, 198, 4])
        #         }, {
        #             'pred_logits': Tensor shape: torch.Size([2, 198, 13]),
        #             'pred_boxes': Tensor shape: torch.Size([2, 198, 4])
        #         }, {
        #             'pred_logits': Tensor shape: torch.Size([2, 198, 13]),
        #             'pred_boxes': Tensor shape: torch.Size([2, 198, 4])
        #         }, {
        #             'pred_logits': Tensor shape: torch.Size([2, 198, 13]),
        #             'pred_boxes': Tensor shape: torch.Size([2, 198, 4])
        #         }]
        #     }
        # }
        

if __name__ == "__main__":
    unittest.main()
