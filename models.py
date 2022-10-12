from config import *


class Augmentation(nn.Module):
    def __init__(self, org_size, Aw=1.0):
        super(Augmentation, self).__init__()
        self.gk = int(org_size*0.1)
        if self.gk%2==0:
            self.gk += 1
        self.Aug = nn.Sequential(
        Kg.RandomResizedCrop(size=(org_size, org_size), p=1.0*Aw),
        Kg.RandomHorizontalFlip(p=0.5*Aw),
        Kg.ColorJitter(brightness=0.4, contrast=0.8, saturation=0.8, hue=0.2, p=0.8*Aw),
        Kg.RandomGrayscale(p=0.2*Aw),
        Kg.RandomGaussianBlur((self.gk, self.gk), (0.1, 2.0), p=0.5*Aw))

    def forward(self, x):
        return self.Aug(x)


class ResNet(nn.Module):
    def __init__(self, pretrained):
        super(ResNet, self).__init__()
        # Download the random initialized / supervised pretrained model from torchvision
        self.pretrained = models.resnet50(pretrained=pretrained)
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == 'avgpool':
                break

        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None
        
    def forward(self,x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return x


class DINO(nn.Module):
    def __init__(self):
        super(DINO, self).__init__()
        # Download the self-supervised pretrained model from "https://github.com/facebookresearch/dino"
        self.pm = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')        
    def forward(self,x):
        x = self.pm(x)
        return x


class Hash_func(nn.Module):
    def __init__(self, fc_dim, N_bits, NB_CLS):
        super(Hash_func, self).__init__()
        self.Hash = nn.Sequential(nn.Linear(fc_dim, N_bits, bias=False))
        self.P = nn.Parameter(torch.FloatTensor(NB_CLS, N_bits), requires_grad=True)            # For DHD loss
        nn.init.xavier_uniform_(self.P, gain=nn.init.calculate_gain('tanh'))

    def forward(self, X):
        X = self.Hash(X)
        return torch.tanh(X)

