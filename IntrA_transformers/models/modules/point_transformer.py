import torch
import torch.nn as nn
from external_libs.pointops.functions import pointops
import einops
import re

torch.autograd.set_detect_anomaly(True)


class LayerNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            super()
            .forward(input.transpose(1, 2).contiguous())
            .transpose(1, 2)
            .contiguous()
        )




class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            self.relu,
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid_planes),
            self.relu,
            nn.Linear(mid_planes, out_planes // share_planes),
            LayerNorm1d(out_planes // share_planes),
            self.relu,
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        p_r = self.linear_p(p_r)
        r_qk = (
            x_k
            - x_q.unsqueeze(1)
            + einops.reduce(
                p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes
            )
        )
        w = self.linear_w(r_qk)  # (n, nsample, c)
        w = self.softmax(w)
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes),
            w,
        )
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.linear_identity = nn.Linear(in_planes, out_planes, bias=False)
            self.pool = nn.AvgPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            identity = x[idx.long(), :]  # (m, c)
            x = pointops.queryandgroup(
                self.nsample,
                p.contiguous(),
                n_p.contiguous(),
                x.contiguous(),
                None,
                o,
                n_o,
                use_xyz=True,
            )
            x = self.relu(
                self.bn(self.linear(x).transpose(1, 2).contiguous())
            )  # (m, c, nsample)

            x = self.pool(x).squeeze(-1) + self.relu(
                self.bn(self.linear_identity(identity))
            )  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]



class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                # concat each with per-cloud mean => finally x = mlp[ x, mlp[mean(x)] ]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            # upsample pxo2 to pxo1
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformer(nn.Module):
    def __init__(
        self,
        block_num,
        blocks,
        in_channels,
        out_channels,
        stride,
        nsample,
        planes,
        block=Bottleneck,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_planes = in_channels
        share_planes = 8
        self.block_num = block_num
        self.planes = planes

        for i_block in range(block_num):
            self.add_module(
                "enc{}".format(i_block + 1),
                self._make_enc(
                    block,
                    planes[i_block],
                    blocks[i_block],
                    share_planes,
                    stride=stride[i_block],
                    nsample=nsample[i_block],
                ),
            )
        for i_block in range(block_num,0,-1):
            self.add_module(
                "dec{}".format(i_block),
                self._make_dec(
                    block,
                    planes[i_block - 1],
                    blocks[i_block - 1],
                    share_planes,
                    nsample=nsample[i_block - 1],
                    is_head=False if (i_block)!=block_num else True
                ),
            )


        self.up_mlp = []
        for i in range(block_num):
            self.up_mlp.append(nn.Sequential(nn.Linear(self.planes[i], self.planes[i]), nn.BatchNorm1d(self.planes[i]), nn.ReLU(inplace=True), nn.Linear(self.planes[i], self.planes[0])).cuda())

        self.cls = nn.Sequential(nn.Linear((self.block_num + 1)*planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], self.out_channels))



    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)


    def forward(self, inputs):

        B = len(inputs)
        N, _ = inputs[0].shape
        o0 = [N]
        for i in range(1, len(inputs)):
            N, _ = inputs[i].shape
            o0.append(N + o0[-1])

        x0 = torch.cat(inputs, dim=0).cuda()
        p0 = x0[:, :3].contiguous()
        o0 = torch.Tensor(o0).int().cuda()

        p, x, o = p0, x0, o0
        encoders_output = []
        encoders_output.append(self._modules["enc{}".format(1)]([p, x, o]))
        for i in range(1, self.block_num):
            encoders_output.append(self._modules["enc{}".format(i + 1)](encoders_output[-1]))
        
        concat_feats = []
        for index, enc_out in enumerate(encoders_output):
            p, x, o = enc_out
            x = pointops.interpolation(p, p0, x, o, o0, k=1)
            concat_feats.append(self.up_mlp[index](x))
            
        decoders_output = []

        decoders_output.append(self._modules["dec{}".format(self.block_num)][1:]([encoders_output[self.block_num-1][0],self._modules["dec{}".format(self.block_num)][0](encoders_output[self.block_num-1]),encoders_output[self.block_num-1][2]]))
        for i in range(self.block_num-1, 0, -1):
            decoders_output.append(

                self._modules["dec{}".format(i)][1:]([encoders_output[i-1][0],self._modules["dec{}".format(i)][0](encoders_output[i-1],decoders_output[-1]),encoders_output[i-1][2]])

            )
        concat_feats.append(decoders_output[-1][1])
        
        x = self.cls(torch.cat(concat_feats, dim=1))

        output = [x[0:o0[0],:]]
        for i in range(1, B):
            output.append(x[o0[i-1]:o0[i],:])

        return output
