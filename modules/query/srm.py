import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry
import numpy as np

def auxrank(support):
    way = support.size(0)
    shot = support.size(1)
    support = support/support.norm(2).unsqueeze(-1)
    L1 = torch.zeros((way**2-way)//2).long().cuda()
    L2 = torch.zeros((way**2-way)//2).long().cuda()
    counter = 0
    for i in range(way):
        for j in range(i):
            L1[counter] = i
            L2[counter] = j
            counter += 1
    s1 = support.index_select(0, L1) # (s^2-s)/2, s, d
    s2 = support.index_select(0, L2) # (s^2-s)/2, s, d
    dists = s1.matmul(s2.permute(0,2,1)) # (s^2-s)/2, s, s
    assert dists.size(-1)==shot
    frobs = dists.pow(2).sum(-1).sum(-1)
    return frobs.sum().mul(.03)



@registry.Query.register("SRM")
class SRM(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()
        """
        @InProceedings{Wertheimer_2021_CVPR,
            author    = {Wertheimer, Davis and Tang, Luming and Hariharan, Bharath},
            title     = {Few-Shot Classification With Feature Map Reconstruction Networks},
            booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
            month     = {June},
            year      = {2021},
            pages     = {8012-8021}
        }

        https://github.com/Tsingularity/FRN
        """

        self.cfg = cfg

        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.r = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.d = in_channels
        self.resolution = 25

        self.r = nn.Parameter(torch.zeros(2),requires_grad=True)
        self.r_Q = nn.Parameter(torch.zeros(2), requires_grad=True)

        self.criterion = nn.CrossEntropyLoss()

    def get_recon_dist(self,query,support,alpha,beta,Woodbury=True):
        # query: way*query_shot*resolution, d
        # support: way, shot*resolution , d
        # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        reg = support.size(1)/support.size(2)

        # correspond to lambda in the paper
        lam = reg*alpha.exp()

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0,2,1) # way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper

            sts = st.matmul(support) # way, d, d
            m_inv = (sts+torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse() # way, d, d
            hat = m_inv.matmul(sts) # way, d, d

        else:
            # correspond to Equation 8 in the paper

            sst = support.matmul(st) # way, shot*resolution, shot*resolution
            m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() # way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support) # way, d, d

        Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d

        Q = query.view(-1, self.resolution, self.d)
        st_Q = Q.permute(0, 2, 1)

        reg_Q = Q.size(1) / Q.size(2)
        lam_Q = reg_Q * self.r_Q[0].exp() + 1e-6
        rho_Q = self.r_Q[1].exp()

        sts_Q = st_Q.matmul(Q)  # way, d, d  [150,640,640]
        m_inv_Q = (sts_Q + torch.eye(sts_Q.size(-1)).to(sts_Q.device).unsqueeze(0).mul(lam_Q)).inverse()
        # m_inv_Q = (sts_Q + torch.eye(sts_Q.size(-1)).to(sts_Q.device).unsqueeze(0).mul(lam)).inverse()
        hat_Q = m_inv_Q.matmul(sts_Q)

        Q_recon = query.view(-1, self.resolution, self.d).matmul(hat_Q).mul(rho_Q).view(-1, self.d)

        dist1 = (Q_bar - Q_recon.unsqueeze(0)).pow(2).sum(2).permute(1, 0)
        dist2 = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)

        dist = dist1 + dist2
        return dist

    def get_neg_l2_dist(self, support_xf, query_xf):
        b, q, c, h, w = query_xf.shape

        support_xf = support_xf / math.sqrt(self.d)
        query_xf = query_xf / math.sqrt(self.d)

        support_xf = support_xf.view(b, self.n_way, self.k_shot, c, -1).permute(0, 1, 2, 4, 3).contiguous()
        support_xf = support_xf.view(b, self.n_way, -1, c)
        query_xf = query_xf.view(b, q, c, -1).permute(0, 1, 3, 2).contiguous()
        query_xf = query_xf.view(b, q * h * w, c)

        alpha = self.r[0]
        beta = self.r[1]

        recon_dist = torch.zeros(b, q * h * w, self.n_way).to(query_xf.device)
        for i in range(b):
            recon_dist[i] = self.get_recon_dist(query_xf[i], support_xf[i], alpha, beta)
        neg_l2_dist = recon_dist.neg().view(-1, h * w, self.n_way).mean(1)
        return neg_l2_dist, support_xf

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        # print('n_way', n_way)  # 10
        # print('query_y.shape', query_y.shape)  # [1, 150]
        self.n_way = n_way
        self.k_shot = k_shot
        query_shot = query_xf.size(1) / n_way

        neg_l2_dist, s = self.get_neg_l2_dist(support_xf, query_xf)  # 计算负欧几里得距离

        y = torch.from_numpy(np.repeat(range(n_way), query_shot ))
        # print('y.shape', y.shape) # [75]
        y_oh = 1 - torch.zeros((len(y), n_way)).scatter_(1, y.unsqueeze(1), 1)
        # print('y_oh.shape', y_oh.shape) # [75, 5]
        y_oh = y_oh.cuda()

        # print('s.shape', s.shape) # [1, 10, 25, 640]
        
        s = s.view(s.size(1), -1, s.size(3))


        aux_loss = auxrank(s)

        logits = neg_l2_dist * self.scale  # 将负欧几里得距离乘以一个比例因子,得到分类器输出logits


        frn_loss_dual = -1 * (y_oh * logits).sum(-1).mean()


        query_y = query_y.view(-1)  # 将query_y展开为一维
        if self.training:  # 如果是训练阶段
            # print('logits.shape', logits.shape)
            # print('query_y.shape', query_y.shape)
            loss = self.criterion(logits, query_y) + aux_loss + 0.001 * frn_loss_dual # 计算损失函数
            return {"FRN_loss": loss}  # 返回损失函数
        else:  # 如果是测试阶段
            _, predict_labels = torch.max(neg_l2_dist, 1)  # 将logits张量再第二维度上取最大值,得到预测标签
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]  # 将预测标签与查询集标签进行比较,得到奖励,返回奖励列表
            return rewards
