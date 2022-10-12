from config import *
from scipy.linalg import hadamard


class DHDLoss(nn.Module):
    def __init__(self, temp):
        super(DHDLoss, self).__init__()
        self.temp = temp

    def forward(self, X, P, L):
        X = F.normalize(X, p = 2, dim = -1)
        P = F.normalize(P, p = 2, dim = -1)
        
        D = F.linear(X, P) / self.temp

        L /= torch.sum(L, dim=1, keepdim=True).expand_as(L)

        xent_loss = torch.mean(torch.sum(-L * F.log_softmax(D, -1), -1))
        return xent_loss


class CSQLoss(nn.Module):
    def __init__(self, bit, NB_CLS, is_single_label, device):
        super(CSQLoss, self).__init__()
        self.is_single_label = is_single_label
        self.hash_targets = self.get_hash_targets(NB_CLS, bit).to(device)
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(device)
        self.criterion = nn.BCELoss().to(device)

    def forward(self, X, L):
        X = X.tanh()
        hash_center = self.label2center(L)
        center_loss = self.criterion(0.5 * (X + 1), 0.5 * (hash_center + 1))
        return center_loss

    def label2center(self, L):
        if self.is_single_label:
            hash_center = self.hash_targets[L.argmax(axis=1)]
        else:
            center_sum = L @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets


class DCHLoss(nn.Module):
    def __init__(self, N_bits, batch_size, device, gamma=20.0, lambda1=0.1):
        super(DCHLoss, self).__init__()
        self.gamma = gamma
        self.lambda1 = lambda1
        self.N_bits = N_bits
        self.one = torch.ones((batch_size, N_bits)).to(device)

    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        return (1 - cos.clamp(max=0.99)) * self.N_bits / 2

    def forward(self, X, L):
        s = (L @ L.t() > 0).float()

        if (1 - s).sum() != 0 and s.sum() != 0:
            positive_w = s * s.numel() / s.sum()
            negative_w = (1 - s) * s.numel() / (1 - s).sum()
            w = positive_w + negative_w
        else:
            w = 1

        d_hi_hj = self.d(X, X)
        cauchy_loss = w * (s * torch.log(d_hi_hj / self.gamma) + torch.log(1 + self.gamma / d_hi_hj))
        loss = cauchy_loss.mean()
        return loss


def Quantization_1_loss(X):
    Qloss = torch.mean(torch.abs(torch.abs(X)-1.0))
    return Qloss
