import argparse
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import torch.backends.cudnn as cudnn
import sys 
from PIL import Image
sys.path.append("..")
sys.path.append("../../../src/models")
from make_swin import *
from tqdm import tqdm
from resnet import ResNet50
from chouzhen_one import getVideoPng
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
class Dataset_f(torch.utils.data.Dataset):
    def __init__(self, dataset,default_size, transform=None):
        super(Dataset_f, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.default_size = default_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname = self.dataset[index]
        fpath = self.dataset[index]
#         try:
        if os.path.isfile(fpath):
            img = Image.open(fpath).convert('RGB')
        else:
            img = Image.new('RGB', self.default_size)
            print('No such images: {:s}'.format(fpath))
        if self.transform is not None:
            img = self.transform(img)
        return img, fname
#         except:
#             print(fname)
#             fname = self.dataset[0]
#             fpath = self.dataset[0]
#             if os.path.isfile(fpath):
#                 img = Image.open(fpath).convert('RGB')
#             else:
#                 img = Image.new('RGB', self.default_size)
#                 print('No such images: {:s}'.format(fpath))
#             if self.transform is not None:
#                 img = self.transform(img)
#             return img, fname
            
    
def get_data(imglist_file, batch_size):
    
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    data_transformer = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        normalizer,
    ])
    
    img_lists = []    # image nums
    for line in tqdm(open(imglist_file).readlines()):
        img_lists.append(line.strip())
    #img_lists=img_lists[3586016:]
    dataset = Dataset_f(img_lists, 224,data_transformer)
    # data transforms


    # data loaders
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size, num_workers=8,
        shuffle=False, pin_memory=True)

    return dataset, data_loader

def write_feature(features, fname, args):
    for i in range(len(features)):
        video=fname[i].split("/")[-2]
        jpgname=fname[i].split("/")[-1].split(".")[0]
        outpath=args.out_path_root+"/"+video
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        
        np.save(os.path.join(outpath, "shot_{}.npy".format(jpgname)), features[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Place feature using resnet with ImageNet pretrain")
    parser.add_argument('--imagelist_file', type=str,\
                        default="/hetu_group/wuxiangyu/cx/algo/script/train_list1.txt")
    parser.add_argument('--out_path_root', type=str, default= \
                        "/hetu_group/wuxiangyu/cx/algo/SceneSeg/sceneseg_predata/swin_l_scene128_384")
    parser.add_argument('--resume', type=str, default= \
                        "swin_base_patch4_window12_384_22k.pth")
    parser.add_argument('-b', '--batch-size', type=int, default=42)
    args = parser.parse_args()
    
    
#    model = ResNet50(pretrained=True)
    
    model = swin_base_patch4_window12_384_in22k()
    state = torch.load(args.resume, map_location='cpu')
    filtered_dict = {k: v for k, v in state['model'].items() if k != "head.weight" and k != "head.bias"}
    model.load_state_dict(filtered_dict, strict=False)
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    print("model ready")
    
    if not os.path.exists(args.out_path_root):
        os.mkdir(args.out_path_root)

#    videos=os.listdir(args.image_root)
#    for video in tqdm(videos):
#        print("start {}".format(video))
    #img_path=os.path.join(args.image_root, video)
#        getVideoPng(video, args.video_root, img_path)
    
    dataset, data_loader = get_data(imglist_file=args.imagelist_file, batch_size=args.batch_size)

    #out_path=os.path.join(args.out_path_root, video)
    for img, fname in tqdm(data_loader):
        output, features = model(img.cuda())
        features = features.cpu().data
        write_feature(features=features, fname=fname, args=args) 
        
