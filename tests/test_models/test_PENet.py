import unittest
import torch
import sys
#sys.path.append('../../')
sys.path.append('/root/autodl-fs/projects/DINO')
from models.dino.dn_components import dn_post_process,prepare_for_cdn
import torch.nn as nn
from test_dino import printDict
import numpy as np
from evaluation import eval_map
from models.enhance.penet import PENet
class TestPenet(unittest.TestCase): 
    def test_Penet(self):
        penet = PENet()
        penet = penet.to('cuda')
        x = torch.randn(2,3,223,223).to('cuda')
        out = penet(x)
        print("out",out.shape)
            



if __name__ == "__main__":
    unittest.main()
