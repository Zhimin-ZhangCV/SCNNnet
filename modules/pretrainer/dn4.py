# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import modules.registry as registry

# from modules.utils import _l2norm
# from modules.similarity import Similarity
# from modules.INSTA import INSTA
# from modules.SpatialFormer import SFTA
# from modules.SpatialFormer import SFSA
# from modules.neck import tSF

# @registry.Pretrainer.register("DN4")
# class PretrainDN4(BasePretrainer):
#     def __init__(self, cfg):
#         super().__init__(cfg)

#         self.cfg = cfg
#         self.neighbor_k = 1

#         self.inner_simi = Similarity(cfg, metric='cosine')
#         self.criterion = nn.CrossEntropyLoss()
#         self.k_shot_average = cfg.model.dn4.larger_shot == "average"
#         if cfg.model.encoder == "R12":
#             self.channel = 640
#         else:
#             self.channel = 64
#         self.insta = INSTA(self.channel, 10, 0.2, 3)
#         self.query_embed = nn.Embedding(100, self.channel)
#         self.SFTA=SFTA(feature_dim=self.channel, num_heads=4, FFN_method='MLP', mode='SFTA', softmax_score=False)  
#         self.SFSA=SFSA(feature_dim=self.channel, num_heads=4, FFN_method='MLP', mode='SFSA')          
#         self.neck_tSF = tSF(feature_dim=self.channel,num_queries=100,num_heads=8, FFN_method='MLP_one')
#         self.n_way = cfg.train.n_way
#         self.k_shot = cfg.train.k_shot

#     def get_recon_dist(self,query,support,alpha,beta,Woodbury=True):
#         # query: way*query_shot*resolution, d
#         # support: way, shot*resolution , d
#         # Woodbury: whether to use the Woodbury Identity as the implementation or not

#         # correspond to kr/d in the paper
#         reg = support.size(1)/support.size(2)
        
#         # correspond to lambda in the paper
#         lam = reg*alpha.exp()+1e-6

#         # correspond to gamma in the paper
#         rho = beta.exp()

#         st = support.permute(0,2,1) # way, d, shot*resolution

#         if Woodbury:
#             # correspond to Equation 10 in the paper
            
#             sts = st.matmul(support) # way, d, d
#             m_inv = (sts+torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse() # way, d, d
#             hat = m_inv.matmul(sts) # way, d, d
        
#         else:
#             # correspond to Equation 8 in the paper
            
#             sst = support.matmul(st) # way, shot*resolution, shot*resolution
#             m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() # way, shot*resolution, shot*resolutionsf 
#             hat = st.matmul(m_inv).matmul(support) # way, d, d

#         Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d

#         dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way
#         return dist

#     def get_neg_l2_dist(self, support_xf, query_xf):
#         b, q, c, h, w = query_xf.shape

#         support_xf = support_xf / math.sqrt(self.encoder.out_channels)
#         query_xf = query_xf / math.sqrt(self.encoder.out_channels)

#         support_xf = support_xf.view(b, self.n_way, self.k_shot, c, -1).permute(0, 1, 2, 4, 3).contiguous()
#         support_xf = support_xf.view(b, self.n_way, -1, c)
#         query_xf = query_xf.view(b, q, c, -1).permute(0, 1, 3, 2).contiguous()
#         query_xf = query_xf.view(b, q * h * w, c)

#         alpha = self.r[0]
#         beta = self.r[1]

#         recon_dist = torch.zeros(b, q * h * w, self.n_way).to(query_xf.device)
#         for i in range(b):
#             recon_dist[i] = self.get_recon_dist(query_xf[i], support_xf[i], alpha, beta)
#         neg_l2_dist = recon_dist.neg().view(-1, h * w, self.n_way).mean(1)
#         return neg_l2_dist

#     def forward_train(self, x, y):
#         enc = self.encoder(x)        
#         # *******************************************************************
#         #     tSF
#         # *******************************************************************
#         feature = enc
#         feature,_ = self.neck_tSF(feature.view(*feature.shape[1:]))
#         feature = feature.unsqueeze(0)
#         support_xf_tsf = feature[:, :support_xf.shape[1], :, :, :]
#         query_xf_tsf = feature[:, support_xf.shape[1]:, :, :, :]
#         support_xf = support_xf_tsf
#         query_xf = query_xf_tsf
#         # *******************************************************************
      
#         b, q, c, h, w = query_xf.shape
#         s = support_xf.shape[1]
#         if self.k_shot_average:
#             support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).mean(2)
#             support_xf = support_xf.view(b, self.n_way, c, h * w)

#         support_xf = support_xf.view(b, self.n_way, c, h * w)
#         # print('support_xf.shape',support_xf.shape)

#         innerproduct_matrix = self.inner_simi(support_xf, query_xf)
#         topk_value, _ = torch.topk(innerproduct_matrix, self.neighbor_k, -1) # [b, q, N, M_q, neighbor_k]
#         similarity_matrix = topk_value.mean(-1).view(b, q, self.n_way, -1).sum(-1)
#         similarity_matrix = similarity_matrix.view(b * q, self.n_way)

#         query_y = y.view(b * q)
#         loss = self.criterion(similarity_matrix, query_y)
#         return {"pretrain_dn4_loss": loss}

#     def forward_test(self, support_x, support_y, query_x, query_y):
#         # *******************************************************************
#         #     tSF
#         # *******************************************************************
#         feature = torch.cat([support_xf, query_xf], 1)
#         feature,_ = self.neck_tSF(feature.view(*feature.shape[1:]))
#         feature = feature.unsqueeze(0)
#         support_xf_tsf = feature[:, :support_xf.shape[1], :, :, :]
#         query_xf_tsf = feature[:, support_xf.shape[1]:, :, :, :]
#         support_xf = support_xf_tsf
#         query_xf = query_xf_tsf
#         # *******************************************************************
      
#         b, q, c, h, w = query_xf.shape
#         s = support_xf.shape[1]
#         if self.k_shot_average:
#             support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).mean(2)
#             support_xf = support_xf.view(b, self.n_way, c, h * w)

#         support_xf = support_xf.view(b, self.n_way, c, h * w)
#         # print('support_xf.shape',support_xf.shape)

#         innerproduct_matrix = self.inner_simi(support_xf, query_xf)
#         topk_value, _ = torch.topk(innerproduct_matrix, self.neighbor_k, -1) # [b, q, N, M_q, neighbor_k]
#         similarity_matrix = topk_value.mean(-1).view(b, q, self.n_way, -1).sum(-1)
#         similarity_matrix = similarity_matrix.view(b * q, self.n_way)

#         query_y = query_y.view(b * q)
#         _, predict_labels = torch.max(similarity_matrix, 1)
#         rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
#         return rewards