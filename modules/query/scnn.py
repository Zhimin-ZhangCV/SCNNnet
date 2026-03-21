import math
from turtle import pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry
from modules.utils import _l2norm
from .similarity import Similarity
from modules.SCNNnet import SCNNnet
from modules.CBAM import CBAM

from modules.C3N import C3N

import pdb



@registry.Query.register("scnn")
class SCNN(nn.Module):
    
    def __init__(self, in_channels, cfg):

        super().__init__()
        """
        @inproceedings{li2019DN4,
            title={Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning},
            author={Li, Wenbin and Wang, Lei and Xu, Jinglin and Huo, Jing and Gao Yang and Luo, Jiebo},
            booktitle={CVPR},
            year={2019}
        }
        https://github.com/WenbinLee/DN4
        """

        self.cfg = cfg
        self.neighbor_k = 1

        self.inner_simi = Similarity(cfg, metric='cosine')
        # self.inner_simi = Similarity(cfg, metric='innerproduct')
        # self.inner_simi = Similarity(cfg, metric='euclidean')
        # self.inner_simi = Similarity(cfg, metric='neg_ed')
        
        self.criterion = nn.CrossEntropyLoss()
        self.k_shot_average = cfg.model.dn4.larger_shot == "average"
        if cfg.model.encoder == "R12":
            self.channel = 640
        else:
            self.channel = 64
        self.h = 5
        self.w = 5
        self.acmix = SCNNnet(in_planes=self.channel, out_planes=self.channel, cfg=self.cfg)
        

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
   
        s = support_xf.shape[1]
      
        support_xf = support_xf.squeeze(0)#[10, 640, 5, 5]
        query_xf = query_xf.squeeze(0)#[150, 640, 5, 5]

  
 
        
        feature_map = self.acmix(support_xf,query_xf)

       
        support_xf = feature_map[:s].view(1, -1, self.channel, self.h, self.w)
        query_xf = feature_map[s:].view(1, -1, self.channel, self.h, self.w) #[1, 150, 640, 5, 5]
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1] #50
        if self.k_shot_average:
            support_xf = support_xf.view(b, n_way, k_shot, c, h, w).mean(2)
            support_xf = support_xf.view(b, n_way, c, h * w) 


        support_xf = support_xf.view(b, n_way, c, h * w)
     
        


        innerproduct_matrix = self.inner_simi(support_xf, query_xf) 
        innerproduct_matrix = F.normalize(innerproduct_matrix, dim=1)

        topk_value, _ = torch.topk(innerproduct_matrix, self.neighbor_k, -1) # [b, q, N, M_q, neighbor_k]
     
        similarity_matrix = topk_value.mean(-1).view(b, q, n_way, -1).sum(-1)
        similarity_matrix = similarity_matrix.view(b * q, n_way)
        

        query_y = query_y.view(b * q)

        if self.training:
            loss = self.criterion(similarity_matrix, query_y)
            return {"dn4_loss": loss}
        else:
            _, predict_labels = torch.max(similarity_matrix, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards


if __name__ == "__main__":
    pass