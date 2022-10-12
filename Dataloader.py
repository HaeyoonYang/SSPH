from config import *


class Loader(Dataset):
    def __init__(self, img_dir, txt_dir, NB_CLS=None):
        self.img_dir = img_dir
        self.file_list = np.loadtxt(txt_dir, dtype='str')
        self.NB_CLS = NB_CLS

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                self.file_list[idx][0])
        image = Image.open(img_name)

        if self.NB_CLS != None:
            # one-hot label
            if len(self.file_list[idx])>2:
                label = [int(self.file_list[idx][i]) for i in range(1,self.NB_CLS+1)]
                label = torch.FloatTensor(label)
            # label in 1-digit number
            else:
                label = int(self.file_list[idx][1])
            return transforms.ToTensor()(image), label
        else:
            return transforms.ToTensor()(image)


class tSNE_Loader(Dataset):
    def __init__(self, img_dir, txt_dir, NB_CLS=None):
        self.img_dir = img_dir
        self.file_list = np.loadtxt(txt_dir, dtype='str')
        self.NB_CLS = NB_CLS

        self.train_data = []
        self.train_labels = []
        for file in self.file_list:
            data_name = file[0]

            if self.NB_CLS != None:
                # one-hot label
                if len(file)>2:
                    label = [int(file[i]) for i in range(1,self.NB_CLS+1)]
                    label = torch.FloatTensor(label)

                    if np.count_nonzero(label==1) > 1:
                        nonzero_index = np.nonzero(np.array(label, dtype=np.int))[0]
                        for c in nonzero_index:
                            self.train_data.append(data_name)
                            label_tmp = [1 if i == c else 0 for i in range(len(label))]
                            label_tmp = torch.FloatTensor(label_tmp)
                            self.train_labels.append(label_tmp)
                    else:
                        self.train_data.append(data_name)
                        self.train_labels.append(label)
                # label in 1-digit number
                else:
                    label = int(file[1])
                    label = torch.FloatTensor(label)
                    self.train_data.append(data_name)
                    self.train_labels.append(label)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                self.train_data[idx])
        image = Image.open(img_name)
        label = self.train_labels[idx]

        return transforms.ToTensor()(image), label