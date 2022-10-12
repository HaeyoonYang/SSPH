from models import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss import *
from Dataloader import Loader
from Retrieval import DoRetrieval
from tensorboardX import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('SSPH', add_help=False)

    parser.add_argument('--gpu_id', default="0", type=str, help="""Define GPU id.""")
    parser.add_argument('--data_dir', default="../data", type=str, help="""Path to dataset.""")
    parser.add_argument('--dataset', default="nuswide", type=str, help="""Dataset name: nuswide, coco.""")
    
    parser.add_argument('--batch_size', default=128, type=int, help="""Training mini-batch size.""")
    parser.add_argument('--num_workers', default=12, type=int, help="""Number of data loading workers per GPU.""")
    parser.add_argument('--encoder', default="self-supervised", type=str, help="""Encoder network: none, supervised, self-supervised.""")
    parser.add_argument('--N_bits', default=64, type=int, help="""Number of bits to retrieval.""")
    parser.add_argument('--init_lr', default=3e-4, type=float, help="""Initial learning rate.""")
    parser.add_argument('--baseline_lr', default=0.1, type=float, help="""Hash function learning rate.""")
    parser.add_argument('--warm_up', default=10, type=int, help="""Learning rate warm-up end.""")
    parser.add_argument('--lambda1', default=0.1, type=float, help="""Balancing hyper-paramter on Quantization loss.""")
    parser.add_argument('--std', default=0.5, type=float, help="""Gaussian estimator standrad deviation.""")
    parser.add_argument('--temp', default=0.2, type=float, help="""Temperature scaling parameter on hash proxy loss.""")
    parser.add_argument('--transformation_scale', default=0.5, type=float, help="""Transformation scaling for self teacher: AlexNet=0.2, else=0.5.""")

    parser.add_argument('--max_epoch', default=100, type=int, help="""Number of epochs to train.""")
    parser.add_argument('--eval_epoch', default=1, type=int, help="""Compute mAP for Every N-th epoch.""")
    parser.add_argument('--eval_init', default=1, type=int, help="""Compute mAP after N-th epoch.""")
    parser.add_argument('--loss_type', default="DHD", type=str, help="""type of loss function: DHD, CSQ, DCH""")
    parser.add_argument('--output_dir', default="trained", type=str, help="""Path to save logs and checkpoints.""")

    return parser


def train(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = args.data_dir
    dname = args.dataset

    N_bits = args.N_bits
    init_lr = args.init_lr
    baseline_lr = args.baseline_lr
    batch_size = args.batch_size

    summary = SummaryWriter()

    # Set dataset
    if dname=='nuswide':
        NB_CLS=21
        Top_N=5000
        args.temp = 0.6
        is_single_label = False
    elif dname=='coco':
        NB_CLS=80
        Top_N=5000
        args.temp = 0.4
        is_single_label = False
    else:
        print("Wrong dataset name.")
        return 

    # Set encoder
    if args.encoder=='none':
        Baseline = ResNet(pretrained=False)
        fc_dim = 2048
    elif args.encoder=='supervised':
        Baseline = ResNet(pretrained=True)
        fc_dim = 2048
    elif args.encoder=='self-supervised':
        Baseline = DINO()
        fc_dim = 2048
    else:
        print("Wrong encoder name.")
        return

    # Set directory & save arguments
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    f = open(f'{args.output_dir}/logs/{args.encoder}_{dname}_{args.N_bits}_{args.transformation_scale}_{args.init_lr}_{args.baseline_lr}.txt', 'w')
    f.write(f'Dataset: {dname},\tclasses: {str(NB_CLS)},\tTop_N: {str(Top_N)},\tTemperature: {str(args.temp)}\n')
    f.write(f'Baseline: {args.encoder},\tfc_dim: {str(fc_dim)},\tBits: {args.N_bits}\tTransformation_scale: {str(args.transformation_scale)}\n')
    f.write(f'Initial lr: {args.init_lr},\tBaseline lr: x {args.baseline_lr}\n')
    f.write('\n\n')
    f.write('===================================================\n')
    print('Dataset: ', dname, '\tclasses: ', str(NB_CLS), '\tTop_N: ', str(Top_N), '\tTemperature: ', str(args.temp))
    print('Baseline: ', args.encoder, '\tfc_dim: ', str(fc_dim), '\tBits: ', str(args.N_bits), '\ttransformation scale: ', str(args.transformation_scale))
    print('Initial lr: ', str(args.init_lr), '\tBaseline lr: x ', str(args.baseline_lr))

    # Load dataset
    Img_dir = os.path.join(path, dname, dname+'256')
    Train_dir = os.path.join(path, dname, dname+'_Train.txt')
    Gallery_dir = os.path.join(path, dname, dname+'_DB.txt')
    Query_dir = os.path.join(path, dname, dname+'_Query.txt')
    org_size = 256
    input_size = 224

    trainset = Loader(Img_dir, Train_dir, NB_CLS)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True,
                                            shuffle=True, num_workers=args.num_workers)
    
    # Define augmentation
    AugT = Augmentation(org_size, args.transformation_scale)
    Crop = nn.Sequential(Kg.CenterCrop(input_size))
    Norm = nn.Sequential(Kg.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))

    # Load network
    H = Hash_func(fc_dim, N_bits, NB_CLS)
    net = nn.Sequential(Baseline, H)
    net = net.to(device)

    # Set supervised training objectives
    DHD_criterion = DHDLoss(args.temp)
    CSQ_criterion = CSQLoss(N_bits, NB_CLS, is_single_label, device)
    DCH_criterion = DCHLoss(N_bits, batch_size, device)

    params = [{'params': Baseline.parameters(), 'lr': baseline_lr * init_lr},
            {'params': H.parameters()}]

    optimizer = torch.optim.Adam(params, lr=init_lr, weight_decay=10e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=0, last_epoch=-1)
    
    MAX_mAP = 0.0
    mAP = 0.0

    # Start training
    for epoch in range(args.max_epoch):
        f.write(f'\nEpoch: {epoch},\tLR: {optimizer.param_groups[0]["lr"]}, {optimizer.param_groups[1]["lr"]}\n\n')
        S_loss = 0.0            # supervised loss (DHD or CSQ or DCH)
        Q_loss = 0.0            # Quantization loss

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            It = Norm(Crop(AugT(inputs)))
            Xt = net(It)
            
            # Supervised & Quantization loss
            l1 = torch.tensor(0., device=device)
            l2 = torch.tensor(0., device=device)

            if args.loss_type == 'DHD':
                l1 = DHD_criterion(Xt, H.P, labels)
            elif args.loss_type == 'CSQ':
                l1 = CSQ_criterion(Xt, labels)
            elif args.loss_type == 'DCH':
                l1 = DCH_criterion(Xt, labels)

            l2 = Quantization_1_loss(Xt) * args.lambda1
            loss = l1 + l2

            loss.backward()
            optimizer.step()
            
            # print statistics
            S_loss += l1.item()
            Q_loss += l2.item()

            if (i+1) % 10 == 0:
                f.write('[%3d] CE: %.4f, \tQuant: %.4f, \tmAP: %.4f, \tMAX mAP: %.4f\n' %
                    (i+1, S_loss / 10, Q_loss / 10, mAP, MAX_mAP))
                print('[%3d] CE: %.4f, Quant: %.4f, mAP: %.4f, MAX mAP: %.4f' %
                    (i+1, S_loss / 10, Q_loss / 10, mAP, MAX_mAP))
                S_loss = 0.0
                Q_loss = 0.0

            if epoch >= args.warm_up:
                scheduler.step()

        # Calculate mAP
        if (epoch+1) % args.eval_epoch == 0 and (epoch+1) >= args.eval_init:
            mAP = DoRetrieval(device, net.eval(), Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args)
            summary.add_scalar('mAP', mAP, epoch)

            if mAP > MAX_mAP:
                MAX_mAP = mAP
                print('[%3d] Reached MAX mAP: %.4f\n' % (epoch+1, MAX_mAP))
                best_dict = {'epoch': epoch,
                            'net': net.state_dict(),
                            'lr': [optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']]}
                torch.save(best_dict, f'{args.output_dir}/models/{args.encoder}_{dname}_{args.N_bits}_{args.transformation_scale}_{args.baseline_lr}.pth')
                del best_dict
            net.train()

        summary.add_scalar('H Loss', S_loss, epoch)
        summary.add_scalar('Q Loss', Q_loss, epoch)
    f.close()

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSPH', parents=[get_args_parser()])
    args = parser.parse_args()
    train(args)
