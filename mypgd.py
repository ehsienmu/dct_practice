
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os

import torch_dct as dct

from tqdm import tqdm

class mydct(object):
    def __call__(self, img):
        """
        :param img: (PIL): Image 

        :return: ycbr color space image (PIL)
        """
        # img = dct.dct_2d(img)  
        # # img = cv2.cvtColor(img, cv2.COLOR_BGR2ycbcr)
        # t = torch.from_numpy(img)

        return dct.dct_2d(img)  #Image.fromarray(t)

    def __repr__(self):
        return self.__class__.__name__+'()'


train_tfm_dct = transforms.Compose([
    # transforms.RandomResizedCrop((128, 128)),
    # transforms.RandomHorizontalFlip(),
    # transforms.AutoAugment(),
    # transforms.ColorJitter(0.2, 0.2),
    # transforms.RandomAffine(0, None, (0.7, 1.3)),
    transforms.ToTensor(),
    mydct(),
])
test_tfm_dct = transforms.Compose([
    # transforms.RandomResizedCrop((128, 128)),
    # transforms.RandomHorizontalFlip(),
    # transforms.AutoAugment(),
    # # transforms.ColorJitter(0.2, 0.2),
    # transforms.RandomAffine(0, None, (0.7, 1.3)),
    transforms.ToTensor(),
    mydct(),
])

train_tfm = transforms.Compose([
    # transforms.RandomResizedCrop((128, 128)),
    # transforms.RandomHorizontalFlip(),
    # transforms.AutoAugment(),
    # transforms.ColorJitter(0.2, 0.2),
    # transforms.RandomAffine(0, None, (0.7, 1.3)),
    transforms.ToTensor(),
])
test_tfm = transforms.Compose([
    # transforms.RandomResizedCrop((128, 128)),
    # transforms.RandomHorizontalFlip(),
    # transforms.AutoAugment(),
    # # transforms.ColorJitter(0.2, 0.2),
    # transforms.RandomAffine(0, None, (0.7, 1.3)),
    transforms.ToTensor(),
])

def prepare_loaders(BATCH_SIZE=128, dataset='mnist', do_dct=False):
    if dataset == 'mnist':
        DSET = datasets.MNIST
    elif dataset == 'cifar10':
        DSET = datasets.CIFAR10
    else:
        print('not found')

    if(do_dct):
        train_dataset = DSET(root='~/DATA', train=True, transform=train_tfm_dct, download=True)
        test_dataset = DSET(root='~/DATA', train=False, transform=test_tfm_dct, download=True)
    else:
        train_dataset = DSET(root='~/DATA', train=True, transform=train_tfm, download=True)
        test_dataset = DSET(root='~/DATA', train=False, transform=test_tfm, download=True)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

def train(args, model, train_loader, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    best = 0
    model.to(args.device)
    ASR = Tacc = Aacc = 0
    for epoch in range(10):
        pbar = tqdm(train_loader, ncols=88, desc='train'+str(epoch))
        for imgs, labels in pbar:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            advnoise = args.eps * torch.randn_like(imgs)
            for adv_steps in tqdm(range(args.stack_iter), ncols=88, leave=False):
                model.train()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()

        ASR, Tacc, Aacc = test(args, model, test_loader)
        print("Epoch: {}, Attack Success Rate: {}, Test Accuracy: {}, Attacked Accuracy: {}".format(epoch, ASR, Tacc, Aacc))
        if Aacc > best:
            best = Aacc
            torch.save({'state_dict':model.state_dict(), 'ASR':ASR, 'Tacc':Tacc, 'Aacc':Aacc,}, args.save_dir/'best.pth')

        torch.save({'state_dict':model.state_dict(), 'ASR':ASR, 'Tacc':Tacc, 'Aacc':Aacc,}, args.save_dir/'epoch{}.pth'.format(epoch))


def test(args, model, test_loader, fast=False):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct = attacked = total = 0
    bbar = test_loader if fast else tqdm(test_loader, ncols=88)
    for imgs, targets in bbar:
        imgs, targets = imgs.to(args.device), targets.to(args.device)
        correct_mask = model(imgs).max(1)[1] == targets
        imgs = imgs[correct_mask]
        targets = targets[correct_mask]

        adv_imgs = imgs.clone()         # the adversarial image
        adv_imgs += args.eps * torch.randn_like(adv_imgs) # init_noise
        pbar = range(args.attack_iter)
        pbar = pbar if fast else tqdm(pbar, desc='attack', leave=False, ncols=88)
        
        for i in pbar:
            img_var = adv_imgs.clone()
            img_var.requires_grad = True
            # print(img_var)
            output = model(img_var)
            loss = criterion(output, targets)
            model.zero_grad()
            loss.backward()
            gradient = img_var.grad.data.sign()
            adv_imgs += args.alpha * gradient
            # RGB images + noise
            adv_imgs = torch.where(adv_imgs < imgs-args.eps, imgs-args.eps, adv_imgs)
            adv_imgs = torch.where(adv_imgs > imgs+args.eps, imgs+args.eps, adv_imgs)
            adv_imgs = adv_imgs.clamp(0,1) # check valid range

        adv_good = model(adv_imgs).max(1)[1] == targets
        correct += float(adv_good.sum())
        attacked += float(len(adv_good))
        total += float(len(correct_mask))

        ASR = 1 - correct/attacked
        Tacc = attacked/total
        Aacc = correct/total
        if fast:
            return ASR, Tacc, Aacc

    return ASR, Tacc, Aacc

def make_sure_dir(directory):
    print("current dir is: %s" % (os.getcwd()))
    # edf
    # if os.path.isdir(path):
    #     print(path, "Exists")
    # else:
    #     print(path, "Doesn't exists")
    #     os.mkdir(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--model-type', default='resnet18', type=str)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--datatype', default='mnist', type=str)
    parser.add_argument('--ckptdir', default='ckpt/runs', type=str)
    parser.add_argument('--name', default='test', type=str)

    parser.add_argument('--attack-iter', default=10, type=int)
    parser.add_argument('--stack-iter', default=10, type=int)
    parser.add_argument('--eps', default=8/255, type=float)
    parser.add_argument('--alpha', default=2/255, type=float)
    parser.add_argument('--dct', default=False, type=bool)
    
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.save_dir = Path(args.ckptdir) / args.name
    make_sure_dir(args.save_dir)

    # folder = "abc"
    # os.chdir(".")


    print('>> saving checkpoints to', args.save_dir)
    if args.model_type == 'WideResNet':
        from wideresnet import WideResNet
        model = WideResNet(depth=28, widen_factor=10, dropRate=0.0)
    elif args.model_type == 'resnet18':
        model = models.resnet18()
        if args.datatype == 'mnist':
            model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        model.fc = torch.nn.Linear(512, 10)
    
    if args.resume:
        state_dict = torch.load(args.resume)['state_dict']
        if args.model_type == 'WideResNet':
            submodel = WideResNet(28, 10, widen_factor=10, dropRate=0.0, sub=True)
            submodel = torch.nn.DataParallel(submodel, device_ids = [args.device])
            submodel.load_state_dict(state_dict)
            # submodel = submodel.module
            model = submodel.module
            # model = WideResNet(28, 10, widen_factor=10, dropRate=0.0)
            # model.load_state_dict({k: v for k, v in model.state_dict().items() if k in model.state_dict()})
        else:
            model.load_state_dict(state_dict)

    train_loader, test_loader = prepare_loaders(args.batch_size, args.datatype, args.dct)
    model.to(args.device)

    if args.test:
        assert args.resume is not None
        ASR, Tacc, Aacc = test(args, model, test_loader)
        print("Attack Success Rate: {}, Test Accuracy: {}, Attacked Accuracy: {}".format(ASR, Tacc, Aacc))
    else:
        train(args, model, train_loader, test_loader)
