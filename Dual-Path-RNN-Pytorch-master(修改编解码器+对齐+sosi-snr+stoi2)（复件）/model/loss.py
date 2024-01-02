import torch
import numpy
from model.stoiq import stoi_loss
from itertools import permutations

def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    global s_hat, s_s
    x = x.view(32000)
    s = s.view(32000)

    max = -100

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    s1 = torch.cat((s, s), dim=-1)
    # print(s1)
    for i in range(0, 101):
        s2 = s1[0 + i:len(x) + i]
        if x.shape != s2.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s2.shape))
        x_zm = x - torch.mean(x, dim=-1, keepdim=True, dtype=torch.float32)
        s_zm = s2 - torch.mean(s2, dim=-1, keepdim=True, dtype=torch.float32)
        # t = torch.sum(
        #     x_zm * s_zm, dim=-1,
        #     keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True) ** 2 + eps)
        t = (l2norm(x_zm, keepdim=True) ** 2 + eps)* s_zm / torch.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True)
        k = eps + l2norm(t) / (l2norm(x_zm - t) + eps)
        theta = torch.arcsin(torch.sqrt(1/k))
        an = 20 * torch.log10(1.0/(torch.sin(theta/2.0)*torch.sin(theta/2.0)+eps))
        if an > max:
            max = an
            s_hat=x_zm
            s_hat=s_hat.unsqueeze(0)
            s_s=s_zm
            s_s=s_s.unsqueeze(0)
    st_oi = stoi_loss(s_hat, s_s, torch.tensor([32000]), reduction='mean')
        # print(an) 打印SI-SNR
    return -st_oi+max


def Loss(ests, egs):
    # spks x n x S
    refs = egs
    num_spks = len(refs)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    # P x N
    N = egs[0].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N
