import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple
import pdb



class FFN_MLP(nn.Module):
    def __init__(self, feature_dim):
        super(FFN_MLP, self).__init__()
        self.linear1 = nn.Linear(feature_dim, feature_dim*2)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(feature_dim*2, feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)

    def forward(self, src):
        src2 = self.linear2((self.relu(self.linear1(src))))
        src = src + src2
        src = self.norm3(src)
        return src
    


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]



class SCNNnet(nn.Module):
    def __init__(self, cfg, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(SCNNnet, self).__init__()
        self.cfg = cfg
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.head_dim = self.out_planes // self.head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.r1 = torch.nn.Parameter(torch.tensor([1.], requires_grad=True))
        self.r2 = torch.nn.Parameter(torch.tensor([1.], requires_grad=True))
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1) 
        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2 #3
        self.pad_att = nn.ReflectionPad2d(self.padding_att) #3
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.FFN = FFN_MLP(in_planes)      


    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x
    

    def get_4d_correlation_map(self, spt, qry):
        way = spt.shape[0] 
        num_qry = qry.shape[0]

        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)
        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)
        return similarity_map_einsum


    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)
    
      
        
    def forward(self, spt, qry):
        if self.training:
            self.n_way = self.cfg.train.n_way
        else:
            self.n_way = self.cfg.val.n_way

        spt = spt.squeeze(0)
        qry = qry.squeeze(0)

        # print(spt.shape)
        # print(qry.shape)
        # pdb.set_trace()
        x = torch.cat((spt,qry),dim=0)

        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        

        b, c, h, w = q.shape
        scaling = float(self.head_dim) ** -0.5  

        h_out, w_out = h//self.stride, w//self.stride

        pe = self.conv_p(position(h, w, x.is_cuda)) #生成位置编码 PE .shape==[1, 160, 5, 5]


        q_att = q.view(b*self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b*self.head, self.head_dim, h, w)
        v_att = v.view(b*self.head, self.head_dim, h, w)
        
        # print("\n{{{{{{{{{{")
        # print(self.stride) == 1
        # print("}}}}}}}}}}}")
        # pdb.set_trace()

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b*self.head, self.head_dim, self.kernel_att,self.kernel_att, h_out, w_out) # b*head, head_dim, k_att^2, h_out, w_out
        k_enhance = unfold_k * k_att.unsqueeze(2).unsqueeze(2)
        unfold_k = k_enhance.view(b*self.head,self.head_dim,-1,h_out,w_out)
        
        # unfold_k = k_att.view(b*self.head,self.head_dim,-1,h_out,w_out)
        

        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att,self.kernel_att, h_out, w_out) # 1, head_dim, k_att^2, h_out, w_out
        rpe_enhance = unfold_rpe * pe.unsqueeze(2).unsqueeze(2)
        unfold_rpe = rpe_enhance.view(1,self.head_dim,-1,h_out,w_out)
        ##
        # unfold_rpe = pe.view(1,self.head_dim,-1,h_out,w_out)
        ##

       
        att = (q_att.unsqueeze(2)*(unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1) # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        unfold_v = self.unfold(self.pad_att(v_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out)

        out_sae = (att.unsqueeze(1) * unfold_v).sum(2).view(b, self.out_planes, h_out, w_out) 

# ----------------------------------------------------------------------------------------------------------------------------------------------------
        
        IC_spt = spt.squeeze(0)

        IC_spt = self.normalize_feature(IC_spt)
        IC_qry = self.normalize_feature(qry)

        IC_cor = self.get_4d_correlation_map(IC_spt, IC_qry) 
        num_qry, way, H_s, W_s, H_q, W_q = IC_cor.size()

        cor_spt = IC_cor.view(num_qry, way, H_s * W_s, H_q, W_q)
        cor_qry = IC_cor.view(num_qry, way, H_s, W_s, H_q * W_q)
        

        cor_spt = self.gaussian_normalize(cor_spt, dim=2)
        cor_qry = self.gaussian_normalize(cor_qry, dim=4)

        cor_spt = F.softmax(cor_spt , dim=2)
        cor_spt = cor_spt.view(num_qry, way, H_s, W_s, H_q, W_q)
   
        cor_qry = F.softmax(cor_qry , dim=4)
        cor_qry = cor_qry.view(num_qry, way, H_s, W_s, H_q, W_q)


        att_s = cor_spt.sum(dim=[4, 5])
        att_q = cor_qry.sum(dim=[2, 3]) 
        # print(att_s.shape) #[150, 50, 5, 5])
        # print(att_q.shape) [150, 50, 5, 5]


        att_s = F.normalize(att_s, p=2, dim=1, eps=1e-8)
        att_q = F.normalize(att_q, p=2, dim=1, eps=1e-8)

        # print(IC_qry.shape) torch.Size([150, 640, 5, 5])  
        # print(IC_spt.shape) #torch.Size([50, 640, 5, 5])

        out_att_s = att_s.unsqueeze(2) * IC_spt.unsqueeze(0) #torch.Size([150, 50, 640, 5, 5] support num = 10way*5shot
        out_att_q = att_q.unsqueeze(2) * IC_qry.unsqueeze(1) #torch.Size([150, 50, 640, 5, 5]) query num=15query*10way

    

        out_att_s = out_att_s.mean(0)
        out_att_q = out_att_q.mean(1)
        out_ic = torch.cat((out_att_s,out_att_q),dim=0)



        
        out = self.r1 * out_sae + self.r2 * out_ic
        # out = out_ic
        # out = out_sae
        # out = self.FFN(out.view(b, c, h * w).transpose(1, 2)).transpose(1, 2).contiguous().view(b, c, h, w)
        return out

