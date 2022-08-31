import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from pcflow.utils.model_utils.flownet3d_utils.layers import PointNetSetAbstraction,PointNetFeaturePropogation,FlowEmbedding,PointNetSetUpConv



class FlowNet3D(BaseModel):
    def __init__(self,
                 cfg=None,
                 extra_input_channel=0,
                 sa1_npoint=1024,
                 sa1_radius=0.5,
                 sa1_nsample=16,
                 fe_radius=10.0,
                 fe_nsample=64,
                 su1_nsample=8,
                 su1_redius=2.4):
        super().__init__(cfg)

        self.extra_input_channel = extra_input_channel
        self.sa1 = PointNetSetAbstraction(npoint=sa1_npoint, radius=sa1_radius, nsample=sa1_nsample, in_channel=extra_input_channel, mlp=[32,32,64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=sa1_npoint//4, radius=sa1_radius*2, nsample=sa1_nsample, in_channel=64, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=sa1_npoint//16, radius=sa1_radius*4, nsample=sa1_nsample//2, in_channel=128, mlp=[128, 128, 256], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=sa1_npoint//64, radius=sa1_radius*8, nsample=sa1_nsample//2, in_channel=256, mlp=[256,256,512], group_all=False)
        
        self.fe_layer = FlowEmbedding(radius=fe_radius, nsample=fe_nsample, in_channel = 128, mlp=[128, 128, 128], pooling='max', corr_func='concat')
        
        self.su1 = PointNetSetUpConv(nsample=su1_nsample, radius=su1_redius, f1_channel = 256, f2_channel = 512, mlp=[], mlp2=[256, 256])
        self.su2 = PointNetSetUpConv(nsample=su1_nsample, radius=su1_redius/2, f1_channel = 128+128, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        self.su3 = PointNetSetUpConv(nsample=su1_nsample, radius=su1_redius/4, f1_channel = 64, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        self.fp = PointNetFeaturePropogation(in_channel = 256+3, mlp = [256, 256])
        
        self.conv1 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2=nn.Conv1d(128, 3, kernel_size=1, bias=True)
        self.init_weights()
        
    def forward(self, pc1, pc2, feature1=None, feature2=None, **kwargs):
        l1_pc1, l1_feature1 = self.sa1(pc1, feature1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1)
        
        l1_pc2, l1_feature2 = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)
        
        _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)

        l3_pc1, l3_feature1 = self.sa3(l2_pc1, l2_feature1_new)
        l4_pc1, l4_feature1 = self.sa4(l3_pc1, l3_feature1)
        
        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)
        
        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        sf = self.conv2(x)
        return sf
        
if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048)).cuda()
    label = torch.randn(8,16).cuda()
    model = FlowNet3D().cuda()
    output = model(input,input,input,input)
    print(output.size())
