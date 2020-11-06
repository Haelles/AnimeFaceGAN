# train, generate ; dataset
from Config import opt
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
import models
import tqdm
import time
import os
import PIL
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# from Visualize.Visualizer import Visualizer


def train(**kwargs):
    opt._parse(kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # vis = Visualizer(opt.env)

    print("begin to load data\n")
    transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    true_dataset = datasets.ImageFolder(opt.load_data_path, transform)
    true_data_loader = DataLoader(true_dataset, opt.batch_size, num_workers=opt.num_workers, drop_last=True)
    true_labels = torch.ones(opt.batch_size).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)
    noises = torch.randn(opt.batch_size, opt.noise_dimension, 1, 1).to(device)
    fix_noises = torch.randn(opt.batch_size, opt.noise_dimension, 1, 1).to(device)  # 固定值，用于save/generate

    print("load data success\nbegin to construct net\n")
    discriminator, generator = get_net(device)

    optimizer_d = optim.Adam(discriminator.parameters(), lr=opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=opt.lr2, betas=(opt.beta1, 0.999))

    cal_d = nn.BCELoss().to(device)
    cal_g = nn.BCELoss().to(device)

    discriminator.train()
    generator.train()

    print("construct net success\nbegin to train\n")

    pre_loss_g = 1e4
    pre_loss_d = 1e4
    for i in range(opt.epoch):
        meter_d = 0.0
        meter_g = 0.0

        for j, (data, _) in tqdm.tqdm(enumerate(true_data_loader)):  # ？
            # 训练判别器
            data = data.to(device)
            optimizer_d.zero_grad()
            res = discriminator(data)
            loss_d1 = cal_d(res, true_labels)
            loss_d1.backward()
            # optimizer_d.step()

            # optimizer_d.zero_grad()
            noises.copy_(torch.randn(opt.batch_size, opt.noise_dimension, 1, 1))
            fake_images = generator(noises).detach()
            res = discriminator(fake_images)
            loss_d2 = cal_d(res, fake_labels)
            loss_d2.backward()

            optimizer_d.step()
            # 真图片和假图片各反向一次，但计算loss时合起来算一次
            temp_loss_d = float(0.5 * loss_d1 + 0.5 * loss_d2)
            meter_d += float(temp_loss_d)

            # 每训练5次判别器，训练1次生成器 (result: bad loss)
            # if (j + 1) % 5 == 0:
            optimizer_g.zero_grad()
            # noises.copy_(torch.randn(opt.batch_size, opt.noise_dimension, 1, 1))
            fake_images = generator(noises)
            pred = discriminator(fake_images)
            loss_g = cal_g(pred, true_labels)
            loss_g.backward()
            optimizer_g.step()
            meter_g += float(loss_g)

            if (j + 1) % 70 == 0:
                with torch.no_grad():
                    fake_images = generator(fix_noises)
                    pred = discriminator(fake_images)
                    top_k = pred.topk(opt.generate_num)[1]
                    result = []
                    for num in top_k:
                        result.append(fake_images[num])
                    torchvision.utils.save_image(result, "images/epoch-" + str(i + 1) + "-loss_g-" + str(
                        round(loss_g.item(), 4)) + time.strftime("-%H:%M:%S") + ".png", normalize=True, range=(-1, 1))
                    print("epoch:%d | index:%d | loss_d: %.4f | loss_g: %.4f\n" % (
                    i + 1, j + 1, meter_d / (j + 1), meter_g / (j + 1)))

        meter_g /= (j + 1)
        meter_d /= (j + 1)

        if (i + 1) % 10 == 0:
            # save
            discriminator.save()
            generator.save()

        if meter_g > pre_loss_g:
            lr = opt.lr2 * opt.lr_decay
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = lr
        pre_loss_g = meter_g

        if meter_d > pre_loss_d:
            lr = opt.lr1 * 0.99  # a smaller decay
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = lr
        pre_loss_d = meter_d


def generate(**kwargs):
    opt._parse(kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noises = torch.randn(opt.batch_size, opt.noise_dimension, 1, 1).to(device)
    discriminator, generator = get_net(device)
    discriminator.eval()
    generator.eval()

    with torch.no_grad():
        fake_images = generator(noises)
        pred = discriminator(fake_images)
        top_k = pred.topk(opt.generate_num)[1]
        result = []

        true_labels = torch.ones(opt.batch_size).to(device)
        cal_loss = nn.BCELoss().to(device)
        loss = cal_loss(pred, true_labels)

        for i in top_k:
            result.append(fake_images[i])

        torchvision.utils.save_image(result, "images/test-" + str(round(loss.item(), 4)) + time.strftime(
            "-%m_%d_%H:%M:%S") + ".png",
                                     normalize=True, range=(-1, 1))


def get_net(device):
    discriminator = getattr(models, 'Discriminator')(opt).to(device)
    generator = getattr(models, 'Generator')(opt).to(device)
    if opt.load_discriminator is not None:
        discriminator.load(opt.load_discriminator)
    if opt.load_generator is not None:
        generator.load(opt.load_generator)
    return discriminator, generator


def get_broken():
    root = "data/cropped/"
    files = os.listdir(root)
    file_list = [
        path for path in files
    ]
    for path in file_list:
        try:
            img = PIL.Image.open(root + path, "r")
        except PIL.UnidentifiedImageError:
            os.remove(root + path)
            print("remove" + root + path + "\n")
        except IOError:
            os.remove(root + path)
            print("remove" + root + path + "\n")


def remove_broken():
    original = os.getcwd()
    file = open("remove", "r")
    os.chdir(os.getcwd() + "/data/cropped")
    for file in file.readlines():
        file_path = file.split("\n")[0]
        # print(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(file_path)
    os.chdir(original)


if __name__ == '__main__':
    import fire

    fire.Fire()


