import unittest
import torch
import sys
sys.path.append('../')
sys.path.append('/root/autodl-fs/projects/DINO')
from models.dino.dn_components import dn_post_process,prepare_for_cdn
from models.dino.deformable_transformer import DeformableTransformer,DeformableTransformerEncoderLayer,TransformerEncoder,DeformableTransformerDecoderLayer,TransformerDecoder
import torch.nn as nn
from models.dino.utils import sigmoid_focal_loss, MLP
import copy
class TestDeformableTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer= DeformableTransformer(
            d_model=256,
            dropout=0.0,
            nhead=8,
            num_queries=900,
            dim_feedforward=2048,
            num_encoder_layers=6,
            num_unicoder_layers=0,
            num_decoder_layers=6,
            normalize_before=False,
            return_intermediate_dec=True,
            query_dim=4,
            activation='relu',
            num_patterns=0,
            modulate_hw_attn=True,
            # for deformable encoder
            deformable_encoder=True,
            deformable_decoder=True,
            num_feature_levels=4,
            enc_n_points=4,
            dec_n_points=4,
            use_deformable_box_attn=False,
            box_attn_type='roi_align',
            # init query
            learnable_tgt_init=True,
            decoder_query_perturber=None,

            add_channel_attention=False,
            add_pos_value=False,
            random_refpoints_xy=False,

            # two stage
            two_stage_type='standard',  # ['no', 'standard', 'early']
            two_stage_pat_embed=0,
            two_stage_add_query_num=0,
            two_stage_learn_wh=False,
            two_stage_keep_all_tokens=False,
            # evo of #anchors
            dec_layer_number=None,
            rm_self_attn_layers=None,
            key_aware_type=None,
            # layer share
            layer_share_type=None,
            # for detach
            rm_detach=None,
            decoder_sa_type='sa',
            module_seq=['sa', 'ca', 'ffn'],
            # for dn
            embed_init_tgt=True,
            use_detached_boxes_dec_out=False
        )

        _class_embed = nn.Linear(256, 13)
        _bbox_embed = MLP(256, 256, 4, 3)
        self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
        self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
    def test_DeformableTransformer(self):
        print("\n************test_DeformableTransformer")
        # srcs, masks, input_query_bbox, poss,input_query_label,attn_mask
        # [Tensor shape: torch.Size([2, 256, 38, 38]), Tensor shape: torch.Size([2, 256, 19, 19]), Tensor shape: torch.Size([2, 256, 10, 10]), Tensor shape: torch.Size([2, 256, 5, 5])] 
        # [Tensor shape: torch.Size([2, 38, 38]), Tensor shape: torch.Size([2, 19, 19]), Tensor shape: torch.Size([2, 10, 10]), Tensor shape: torch.Size([2, 5, 5])] 
        #Tensor shape: torch.Size([2, 198, 4]) 
        #[Tensor shape: torch.Size([2, 256, 38, 38]), Tensor shape: torch.Size([2, 256, 19, 19]), Tensor shape: torch.Size([2, 256, 10, 10]), Tensor shape: torch.Size([2, 256, 5, 5])] 
        #Tensor shape: torch.Size([2, 198, 256]) Tensor shape: torch.Size([1098, 1098])
        srcs = [torch.randn(2, 256, 38, 38).to('cuda'),torch.randn(2, 256, 19, 19).to('cuda'),torch.randn(2, 256, 10, 10).to('cuda'),torch.randn(2, 256, 5, 5).to('cuda')]
        masks = [torch.ones(2,38,38, dtype=torch.bool).to('cuda'),torch.ones(2,19, 19, dtype=torch.bool).to('cuda'),torch.ones(2,10, 10, dtype=torch.bool).to('cuda'),torch.ones(2,5, 5, dtype=torch.bool).to('cuda')]
        poss = [torch.randn(2, 256, 38, 38).to('cuda'),torch.randn(2, 256, 19, 19).to('cuda'),torch.randn(2, 256, 10, 10).to('cuda'),torch.randn(2, 256, 5, 5).to('cuda')]
        input_query_bbox = torch.randn(2, 198, 4).to('cuda')
        input_query_label = torch.randn(2,198,256).to('cuda')
        attn_mask = torch.ones(1098, 1098, dtype=torch.bool).to('cuda')
        self.transformer.to('cuda')
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer.forward(
            srcs =srcs,
            masks = masks,
            pos_embeds = poss,
            refpoint_embed = input_query_bbox,
            tgt = input_query_label,
            attn_mask = attn_mask
        )
        print("out")
        #print("out",hs, reference, hs_enc, ref_enc, init_box_proposal)
        # [torch.Size([2, 1098, 256]), torch.Size([2, 1098, 256]), torch.Size([2, 1098, 256]), torch.Size([2, 1098, 256]), torch.Size([2, 1098, 256]), torch.Size([2, 1098, 256])]
        # [torch.Size([2, 1098, 4])] 
        # torch.Size([1, 2, 900, 256]) 
        # torch.Size([1, 2, 900, 4]) 
        # torch.Size([2, 900, 4])
    def t1est_TransformerEncoder(self):
        print("\n*********t1est_TransformerEncoder")
        deformableTransformerEncoderLayer = DeformableTransformerEncoderLayer(
            d_model=256,
            d_ffn=2048,
            dropout=0.0,
            activation="relu",
            n_levels=4,
            n_heads=8,
            n_points=4,
            add_channel_attention=False,
            use_deformable_box_attn=False,
            box_attn_type='roi_align'
        )
        transformerEncoder = TransformerEncoder(
            encoder_layer=deformableTransformerEncoderLayer,
            num_layers=6,
            norm=None,
            d_model=256,
            num_queries=900,
            deformable_encoder=True,
            enc_layer_share=False,
            two_stage_type='standard'
        )
        transformerEncoder = transformerEncoder.to('cuda')
        src_flatten = torch.randn(2,1930,256).to('cuda')
        lvl_pos_embed_flatten = torch.randn(2,1930,256).to('cuda')
        spatial_shapes = torch.tensor([[38,38],[19,19],[10,10],[5,5]]).to('cuda')
        level_start_index = torch.tensor([0,1444,1805,1905]).to('cuda')
        valid_ratios = torch.tensor([[[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]]]).to('cuda')
        mask_flatten = torch.ones(2,1930,dtype=bool).to('cuda')
        memory, enc_intermediate_output, enc_intermediate_refpoints = transformerEncoder(
           src = src_flatten,
           pos = lvl_pos_embed_flatten,
           level_start_index=level_start_index,
           spatial_shapes=spatial_shapes,
           valid_ratios=valid_ratios,
           key_padding_mask=mask_flatten 
        )
        out = memory, enc_intermediate_output, enc_intermediate_refpoints
        for o in out:
            if not (o ==None):
                print(o.shape)
        #torch.Size([2, 1930, 256])

    def t1est_TransformerDecoder(self):
        deformableTransformerDecoderLayer = DeformableTransformerDecoderLayer(
            d_model=256,
            d_ffn=2048,
            dropout=0.0,
            activation='relu',
            n_levels=4,
            n_heads=8,
            n_points=4,
            use_deformable_box_attn=False,
            box_attn_type='roi_align',
            key_aware_type=None,
            decoder_sa_type='sa',
            module_seq=['sa','ca','ffn']
        )
        transformerDecoder = TransformerDecoder(
            decoder_layer=deformableTransformerDecoderLayer,
            num_layers=6,
            norm=nn.LayerNorm(256),
            return_intermediate=True,
            d_model=256,
            query_dim=4,
            modulate_hw_attn=True,
            num_feature_levels=4,
            deformable_decoder=True,
            decoder_query_perturber=None,
            dec_layer_number=None,
            rm_dec_query_scale=True,
            dec_layer_share=False,
            use_detached_boxes_dec_out=False
        )
        transformerDecoder = transformerDecoder.to('cuda')
        tgt_tranpose = torch.randn(1098,2,256).to('cuda')
        memory_transpose = torch.randn(1930,2,256).to('cuda')
        attn_mask = torch.ones(1098,1098,dtype=bool).to('cuda')


        mask_flatten = torch.ones(2,1930,dtype=bool).to('cuda')
        lvl_pos_embed_flatten_transppose = torch.randn(1930,2,256).to('cuda')
        refpoint_embed_transpose = torch.randn(1098,2,4).to('cuda')
        hs, references = transformerDecoder.forward(
                    tgt=tgt_tranpose,
                    memory=memory_transpose,
                    tgt_mask=attn_mask,
                    memory_mask=None,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=mask_flatten,
                    pos=lvl_pos_embed_flatten_transppose,
                    refpoints_unsigmoid=refpoint_embed_transpose,
                    level_start_index=torch.tensor([   0, 1444, 1805, 1905]).to('cuda'),
                    spatial_shapes=torch.tensor([[38, 38],
                                                [19, 19],
                                                [10, 10],
                                                [ 5,  5]]).to('cuda'),
                    valid_ratios=torch.tensor([[[0., 0.],
                                                [0., 0.],
                                                [0., 0.],
                                                [0., 0.]],

                                                [[0., 0.],
                                                [0., 0.],
                                                [0., 0.],
                                                [0., 0.]]]).to('cuda')
                )
        print("hs")
        for h in hs:
            print(h.shape)
        print("references")
        for reference in references:
            print(reference.shape)
        # hs
        # torch.Size([2, 1098, 256])
        # torch.Size([2, 1098, 256])
        # torch.Size([2, 1098, 256])
        # torch.Size([2, 1098, 256])
        # torch.Size([2, 1098, 256])
        # torch.Size([2, 1098, 256])
        # references
        # torch.Size([2, 1098, 4])

    def t1est_DeformableTransformerEncoderLayer(self):
        print("\n*********t1est_DeformableTransformerEncoderLayer")
        deformableTransformerEncoderLayer = DeformableTransformerEncoderLayer(
            d_model=256,
            d_ffn=2048,
            dropout=0.0,
            activation="relu",
            n_levels=4,
            n_heads=8,
            n_points=4,
            add_channel_attention=False,
            use_deformable_box_attn=False,
            box_attn_type='roi_align'
        )
        deformableTransformerEncoderLayer = deformableTransformerEncoderLayer.to('cuda')
        output = torch.randn(2,1930,256).to('cuda')
        pos = torch.randn(2,1930,256).to('cuda')
        reference_points = torch.randn(2,1930,4,2).to('cuda')
        spatial_shapes=torch.tensor([[38,38],[19,19],[10,10],[5,5]]).to('cuda')
        level_start_index=torch.tensor([0,1444,1805,1905]).to('cuda')
        key_padding_mask = torch.ones(2,1930,dtype=bool).to('cuda')
        output = deformableTransformerEncoderLayer.forward(
            src = output,
            pos = pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask
        )
        print(output.shape)
        #torch.Size([2, 1930, 256])

    def test_DeformableTransformerDecoderLayer(self):
        print("\n*********t1est_DeformableTransformerDecoderLayer")
        deformableTransformerDecoderLayer = DeformableTransformerDecoderLayer(
            d_model=256,
            d_ffn=2048,
            dropout=0.0,
            activation='relu',
            n_levels=4,
            n_heads=8,
            n_points=4,
            use_deformable_box_attn=False,
            box_attn_type='roi_align',
            key_aware_type=None,
            decoder_sa_type='sa',
            module_seq=['sa','ca','ffn']
        )
        deformableTransformerDecoderLayer = deformableTransformerDecoderLayer.to('cuda')
        output = torch.randn(1098,2,256).to('cuda')
        query_pos = torch.randn(1098,2,256).to('cuda')
        query_sine_embed = torch.randn(1098,2,512).to('cuda')
        tgt_key_padding_mask = None
        reference_points_input = torch.randn(1098,2,4,4).to('cuda')

        memory = torch.randn(1930,2,256).to('cuda')
        memory_key_padding_mask = torch.ones(2,1930,dtype=bool).to('cuda')
        level_start_index = torch.tensor([0, 1444, 1805, 1905]).to('cuda')
        spatial_shapes = torch.tensor([[38, 38],
        [19, 19],
        [10, 10],
        [ 5,  5]]).to('cuda')
        pos = torch.randn(1930,2,256).to('cuda')

        tgt_mask = torch.ones(1098,1098,dtype=bool).to('cuda')
        memory_mask = None
        outputde = deformableTransformerDecoderLayer.forward(
            tgt=output,
            tgt_query_pos=query_pos,
            tgt_query_sine_embed=query_sine_embed,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_reference_points=reference_points_input,

            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_level_start_index=level_start_index,
            memory_spatial_shapes=spatial_shapes,
            memory_pos=pos,

            self_attn_mask=tgt_mask,
            cross_attn_mask=memory_mask
        )
        print("outputde",outputde.shape)
        #outputde torch.Size([1098, 2, 256])



if __name__ == "__main__":
    unittest.main()


