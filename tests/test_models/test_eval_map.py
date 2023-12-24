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
def bbox2result(bboxes, labels, num_classes):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor | np.ndarray): shape (n, 5)
            labels (torch.Tensor | np.ndarray): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
        else:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            return [bboxes[labels == i, :] for i in range(num_classes)]
class TestDnComponents(unittest.TestCase): 
    def t1est_eval_map(self):
            # batch head
            
            resultshead = [(torch.randn(3,5),torch.tensor([1,4,5])),(torch.randn(1,5),torch.tensor([5])),(torch.randn(4,5),torch.tensor([1,4,7,5]))]
            # detector
            bbox_results = [
                bbox2result(det_bboxes, det_labels, 12)
                for det_bboxes, det_labels in resultshead
            ]
            #print("bbox_results",bbox_results)
            results = bbox_results
            annotations = [{
                'bboxes': np.array([[ 84., 186., 243., 322.],
                                [494., 239., 638., 316.],
                                [373., 201., 419., 229.],
                                [404., 210., 451., 237.],
                                [433., 220., 491., 245.],
                                [454., 236., 524., 267.],
                                [549., 209., 618., 246.]], dtype=np.float32), 
                'labels': np.array([1, 1, 1, 1, 1, 1, 4]), 
                # 'bboxes_ignore': np.array([], shape=(0, 4), dtype=np.float32), 
                # 'labels_ignore': np.array([], dtype=np.int64)
            },{
                'bboxes': np.array([[ 84., 186., 243., 322.],
                                [494., 239., 638., 316.],
                                [373., 201., 419., 229.],
                                [404., 210., 451., 237.],
                                [433., 220., 491., 245.],
                                [454., 236., 524., 267.],
                                [549., 209., 618., 246.]], dtype=np.float32), 
                'labels': np.array([1, 1, 1, 1, 1, 1, 4]), 
                # 'bboxes_ignore': np.array([], shape=(0, 4), dtype=np.float32), 
                # 'labels_ignore': np.array([], dtype=np.int64)
            },{
                'bboxes': np.array([[ 84., 186., 243., 322.],
                                [494., 239., 638., 316.],
                                [373., 201., 419., 229.],
                                [404., 210., 451., 237.],
                                [433., 220., 491., 245.],
                                [454., 236., 524., 267.],
                                [549., 209., 618., 246.]], dtype=np.float32), 
                'labels': np.array([1, 1, 1, 1, 1, 1, 4]), 
                # 'bboxes_ignore': np.array([], shape=(0, 4), dtype=np.float32), 
                # 'labels_ignore': np.array([], dtype=np.int64)
            }
            ]
            mean_ap, _ = eval_map(results,annotations)
    
    def test_dataTranslate(self):
        results = [{'scores':torch.ones(10),'labels':torch.randn(10),'boxes':torch.randn(10,4)},{'scores':torch.ones(10),'labels':torch.randn(10),'boxes':torch.randn(10,4)}]
        targets = [{'boxes':torch.randn(4,4),'labels':torch.randn(4)},{'boxes':torch.randn(4,4),'labels':torch.randn(4)}]

        resultshead = []
        for i in range(len(results)):
             result = results[i]
             scores = result['scores']
             boxes = result['boxes']
             labels = result['labels']
             scores = scores.unsqueeze(1)
             print("\nboxes",boxes)
             boxes = torch.cat((boxes,scores),dim=1)
             print("boxes",boxes)
             resultshead.append((boxes, labels))

        bbox_results = [
            bbox2result(det_bboxes, det_labels, 12)
            for det_bboxes, det_labels in resultshead
        ]
        #print("bbox_results",bbox_results)

        annotations = [{'bboxes':target['boxes'], 'labels': target['labels']}for target in targets]

        mean_ap, _ = eval_map(bbox_results,annotations,dataset=('Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair',
               'Cup', 'Dog', 'Motorbike', 'People', 'Table'))



if __name__ == "__main__":
    unittest.main()
