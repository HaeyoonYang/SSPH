from models import *
from loss import *
from config import *
from Dataloader import *
from Retrieval import DoRetrieval
import shutil


# Visualize top-10 retrieved images for each query image
def Visualize(gallery_c, query_c, g_txt_dir, q_txt_dir, img_dir, enc):
    with open(g_txt_dir, 'r') as f:
        lists_gallery = f.readlines()
    with open(q_txt_dir, 'r') as f:
        lists_query = f.readlines()
    lists_gallery_org = np.asarray(lists_gallery)
    lists_query = np.asarray(lists_query)
    
    os.makedirs('./Vis/'+enc, exist_ok=True)
    save_f = open('./Vis/'+enc+'.txt', 'w')

    # rnd_i = [1,2,10,100,1000,2090]                                          # manually select queries to visualize
    rnd_i = sorted(np.random.randint(query_c.shape[0], size=10))            # randomly select queries to visualize         
    print("\nQuery:")
    print(rnd_i)
    for k in range(len(rnd_i)):
        dist = (gallery_c.shape[1] - query_c[rnd_i[k]] @ gallery_c.T)
        rank = np.argsort(dist)
        rnd_query_dir = lists_query[rnd_i[k]]
        lists_gallery = lists_gallery_org[rank]
        item = np.array2string(rnd_query_dir).split(" ")
        rnd_query_name = img_dir + '/' + item[0][1:]
        rnd_query_label = ''.join(item[1:]).rstrip("\\n'")
        print(rnd_query_name)
        save_f.write(f"QUERY:\n{rnd_query_name}\n{rnd_query_label}\n")
        rnd_query_res_path ='./Vis/' + enc + '/%d_query.png'%(rnd_i[k])
        shutil.copy(rnd_query_name, rnd_query_res_path)

        # log top-10 retrieved images from Gallery
        save_f.write("\n======================================================\n")
        save_f.write("GALLERY")
        save_f.write("\n======================================================\n")
        for i in range(1, 11):
            item = np.array2string(lists_gallery[i-1]).split(" ")
            gallery_name = img_dir + '/' + item[0][1:]
            gallery_label = ''.join(item[1:]).rstrip("\\n'")
            save_f.write(f"{gallery_name}\n{gallery_label}\n\n")
            gallery_res_path = './Vis/' + enc + '/%d_gallery%d.png' % (rnd_i[k], i)
            shutil.copy(gallery_name, gallery_res_path)
        save_f.write("\n\n")
    save_f.close()


def get_args_parser():
    parser = argparse.ArgumentParser('SSPH', add_help=False)

    parser.add_argument('--gpu_id', default="0", type=str, help="""Define GPU id.""")
    parser.add_argument('--data_dir', default="../Fast-Image-Retrieval/data/", type=str, help="""Path to dataset.""")
    parser.add_argument('--model_path', default="./SSPH_trained_models/selfsupervised_nuswide_64_0.5_0.1.pth", type=str, help="""Path to pretrained weights.""")
    parser.add_argument('--dataset', default="nuswide", type=str, help="""Dataset name: nuswide, coco.""")
    parser.add_argument('--encoder', default="self-supervised", type=str, help="""Encoder network: none, supervised, self-supervised.""")
    parser.add_argument('--N_bits', default=64, type=int, help="""Number of bits to retrieval.""")

    parser.add_argument('--batch_size', default=128, type=int, help="""Training mini-batch size.""")
    parser.add_argument('--num_workers', default=12, type=int, help="""Number of data loading workers per GPU.""")

    return parser


def test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = args.data_dir
    dname = args.dataset
    N_bits = args.N_bits

    if dname=='nuswide':
        NB_CLS=21
        Top_N=5000
    elif dname=='coco':
        NB_CLS=80
        Top_N=5000
    else:
        print("Wrong dataset name.")
        return

    # Load Gallery (DataBase) & Query
    Img_dir = os.path.join(path, dname, dname+'256')
    Gallery_dir = os.path.join(path, dname, dname+'_DB.txt')
    Query_dir = os.path.join(path, dname, dname+'_Query.txt')
    
    Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS)
    Gallery_loader = torch.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers)
    Query_set = Loader(Img_dir, Query_dir, NB_CLS)
    Query_loader = torch.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load network
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

    H = Hash_func(fc_dim, N_bits, NB_CLS)
    net = nn.Sequential(Baseline, H)
    net = net.to(device)
    net.load_state_dict(torch.load(args.model_path)['net'])             # Load trained model
    net.eval()

    Crop_Normalize = nn.Sequential(
        Kg.CenterCrop(224),
        Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225]))
    )

    # Generate hash codes for Gallery & Query
    with torch.no_grad():
        for i, data in enumerate(Gallery_loader, 0):
            gallery_x_batch, gallery_y_batch = data[0].to(device), data[1].to(device)
            gallery_x_batch = Crop_Normalize(gallery_x_batch)

            outputs = net(gallery_x_batch)

            if i == 0:
                gallery_c = outputs
                gallery_y = gallery_y_batch
            else:
                gallery_c = torch.cat([gallery_c, outputs], 0)
                gallery_y = torch.cat([gallery_y, gallery_y_batch], 0)
            # print(i, 'Gallery')

        for i, data in enumerate(Query_loader, 0):
            query_x_batch, query_y_batch = data[0].to(device), data[1].to(device)
            query_x_batch = Crop_Normalize(query_x_batch)

            outputs = net(query_x_batch)

            if i == 0:
                query_c = outputs
                query_y = query_y_batch
            else:
                query_c = torch.cat([query_c, outputs], 0)
                query_y = torch.cat([query_y, query_y_batch], 0)
            # print(i, 'Query')

    print("Database:", gallery_c.size(), ",\tQuery:", query_c.size())
    
    mAP = DoRetrieval(device, net.eval(), Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args)
    gallery_c = gallery_c.sign().cpu().detach().numpy()
    query_c = query_c.sign().cpu().detach().numpy()
    Visualize(gallery_c, query_c, Gallery_dir, Query_dir, Img_dir, os.path.basename(args.model_path)[:-4])
    print("\n ==> mAP score: %.4f\n" % (mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSPH', parents=[get_args_parser()])
    args = parser.parse_args()
    test(args)
