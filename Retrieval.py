from Dataloader import *
from tqdm import tqdm

def Evaluate_mAP(device, gallery_codes, query_codes, gallery_labels, query_labels, Top_N=None):
    num_query = query_labels.shape[0]
    mean_AP = 0.0
    with tqdm(total=num_query, desc="Evaluate mAP", bar_format='{desc:<15}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
        for i in range(num_query):
            # Retrieve images from database
            retrieval = (query_labels[i, :] @ gallery_labels.t() > 0).float()

            # Calculate hamming distance
            hamming_dist = (gallery_codes.shape[1] - query_codes[i, :] @ gallery_codes.t())

            # Arrange position according to hamming distance
            retrieval = retrieval[torch.argsort(hamming_dist)][:Top_N]

            # Retrieval count
            retrieval_cnt = retrieval.sum().int().item()

            # Can not retrieve images
            if retrieval_cnt == 0:
                continue

            # Generate score for every position
            score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

            # Acquire index
            index = (torch.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float().to(device)

            mean_AP += (score / index).mean()
            pbar.update(1)

        mean_AP = mean_AP / num_query
    return mean_AP


def DoRetrieval(device, net, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    # Load DataBase
    Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS)
    Gallery_loader = torch.utils.data.DataLoader(Gallery_set, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load Query
    Query_set = Loader(Img_dir, Query_dir, NB_CLS)
    Query_loader = torch.utils.data.DataLoader(Query_set, batch_size=args.batch_size, num_workers=args.num_workers)

    Crop_Normalize = nn.Sequential(
        Kg.CenterCrop(224),
        Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225]))
    )

    with torch.no_grad():
        # Build Gallery; generate real-valued hash codes for DataBase
        with tqdm(total=len(Gallery_loader), desc="Build Gallery", bar_format='{desc:<15}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, data in enumerate(Gallery_loader, 0):
                gallery_x_batch, gallery_y_batch = data[0].to(device), data[1].to(device)
                gallery_x_batch = Crop_Normalize(gallery_x_batch)
                if gallery_y_batch.dim() == 1:
                    gallery_y_batch = torch.eye(NB_CLS, device=device)[gallery_y_batch]

                outputs = net(gallery_x_batch)

                if i == 0:
                    gallery_c = outputs
                    gallery_y = gallery_y_batch
                else:
                    gallery_c = torch.cat([gallery_c, outputs], 0)
                    gallery_y = torch.cat([gallery_y, gallery_y_batch], 0)
                pbar.update(1)
        
        # Build Query; generate real-valued hash codes for Query
        with tqdm(total=len(Query_loader), desc="Compute Query", bar_format='{desc:<15}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, data in enumerate(Query_loader, 0):
                query_x_batch, query_y_batch = data[0].to(device), data[1].to(device)
                query_x_batch = Crop_Normalize(query_x_batch)
                if query_y_batch.dim() == 1:
                    query_y_batch = torch.eye(NB_CLS, device=device)[query_y_batch]

                outputs = net(query_x_batch)

                if i == 0:
                    query_c = outputs
                    query_y = query_y_batch
                else:
                    query_c = torch.cat([query_c, outputs], 0)
                    query_y = torch.cat([query_y, query_y_batch], 0)
                pbar.update(1)

    # generate binary hash codes for DataBase and Query
    gallery_c = torch.sign(gallery_c)
    query_c = torch.sign(query_c)

    # compute mAP 
    mAP = Evaluate_mAP(device, gallery_c, query_c, gallery_y, query_y, Top_N)
    return mAP

