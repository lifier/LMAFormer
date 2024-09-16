import torch.nn as nn
import torch
from MFIRSTD.models.MAFEM_block.submodules_sc import DeformableAttnBlock
from torch.nn.init import xavier_uniform_, constant_
def make_model(args):
    return SC(in_channels=args.n_colors,
                        n_sequence=args.n_sequence,
                        out_channels=args.n_colors,
                        n_resblock=args.n_resblock,
                        n_feat=args.n_feat)

class SC(nn.Module):
    def __init__(self, in_channels=192, n_sequence=3, out_channels=2):
        super(SC, self).__init__()
        self.MMA = DeformableAttnBlock(n_heads=4,d_model=in_channels,n_levels=5,n_points=12,n_sequence=n_sequence)
        # self.pos_em  = PositionalEncodingPermute3D(3)
        self.motion_branch = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=2*in_channels, out_channels=96//2, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.LeakyReLU(0.1,inplace=True),
                    torch.nn.Conv2d(in_channels=96//2, out_channels=64//2, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.LeakyReLU(0.1,inplace=True),
                    torch.nn.Conv2d(in_channels=64//2, out_channels=32//2, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.LeakyReLU(0.1,inplace=True),
        )
        self.motion_out = torch.nn.Conv2d(in_channels=32//2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        constant_(self.motion_out.weight.data, 0.)
        constant_(self.motion_out.bias.data, 0.)
        pad = int((n_sequence-1)/2)
        self.collect_eng = torch.nn.Conv3d(in_channels=in_channels,out_channels=384, kernel_size=(1,1,n_sequence), padding=(0,0,pad), dilation=1)
        self.bn_last = torch.nn.BatchNorm3d(num_features=384)
        self.sigmoid = torch.nn.Sigmoid()
    def compute_flow(self, frames):
        n, t, c, h, w = frames.size()
        frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
        frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_forward = self.estimate_flow(frames_1, frames_2).view(n, t-1, 2, h, w)
        flows_backward = self.estimate_flow(frames_2,frames_1).view(n, t-1, 2, h, w)

        return flows_forward,flows_backward
    def estimate_flow(self,frames_1, frames_2):
        return self.motion_out(self.motion_branch(torch.cat([frames_1, frames_2],1)))
        
    def forward(self, features):
        flow_forward,flow_backward = self.compute_flow(features)
        frame,srcframe = self.MMA(features,features,flow_forward,flow_backward)
        frame = self.collect_eng(frame.permute(0,2,1,3,4))
        frame = self.bn_last(frame).permute(0,2,1,3,4)
        frame = self.sigmoid(frame)
        mid_loss = None

        return frame, flow_forward,flow_backward
