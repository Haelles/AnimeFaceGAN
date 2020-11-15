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
from tensorboardX import SummaryWriter
import i2v
from scipy.linalg import sqrtm
import numpy as np
import os
from models.model_new import weights_init

# ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(**kwargs):
    opt._parse(kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter('runs/week07_01/')
    iter_count = opt.iter_count

    print("begin to load data\n")
    transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    true_dataset = datasets.ImageFolder(opt.load_data_path, transform)
    true_data_loader = DataLoader(true_dataset, opt.batch_size, num_workers=opt.num_workers, drop_last=True,
                                  shuffle=True)
    true_labels = torch.ones(opt.batch_size).to(device)
    # one_sided_label_smoothing = torch.ones(opt.batch_size).fill_(0.9).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)
    noises = torch.randn(opt.batch_size, opt.noise_dimension, 1, 1).to(device)
    # fix_noises = torch.randn(opt.batch_size, opt.noise_dimension, 1, 1).to(device)  # 固定值，用于save/generate

    print("load data success\nbegin to construct net\n")
    discriminator, generator = get_net(device)

    lr1 = opt.lr1
    lr2 = opt.lr2
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr1, betas=(opt.beta1, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=lr2, betas=(opt.beta1, 0.999))

    cal = nn.BCELoss().to(device)

    discriminator.train()
    generator.train()

    print("construct net success\nbegin to train\n")

    for i in range(opt.epoch):
        meter_d = 0.0
        meter_g = 0.0
        cnt = 0

        for j, (true_images, _) in tqdm.tqdm(enumerate(true_data_loader)):
            true_images = true_images.to(device)
            optimizer_d.zero_grad()
            res = discriminator(true_images)
            loss_d1 = cal(res, true_labels)
            loss_d1.backward()

            noises.copy_(torch.randn(opt.batch_size, opt.noise_dimension, 1, 1))
            fake_images = generator(noises.detach())
            res = discriminator(fake_images)
            loss_d2 = cal(res, fake_labels)
            loss_d2.backward()

            temp_loss_d = loss_d1 + loss_d2
            meter_d += float(temp_loss_d)
            optimizer_d.step()

            optimizer_g.zero_grad()
            fake_images = generator(noises)
            pred = discriminator(fake_images)
            loss_g = cal(pred, true_labels)
            loss_g.backward()
            optimizer_g.step()
            meter_g += float(loss_g)
            if (j + 1) % 30 == 0:
                writer.add_scalar('mini_loss_g_1', loss_g.item(), iter_count + 1)  # mini batch loss
                writer.add_scalar('mini_loss_d_1', temp_loss_d, iter_count + 1)

            cnt += 1  # 1 -> 248
            iter_count += 1  # 1 -> 248iters * 60 epochs

        # record epoch loss
        meter_g /= cnt
        meter_d /= cnt
        with torch.no_grad():
            discriminator.eval()
            generator.eval()
            noises.copy_(torch.randn(opt.batch_size, opt.noise_dimension, 1, 1))
            fake_images = generator(noises)
            pred = discriminator(fake_images)
            loss_g = cal(pred, fake_labels)
            discriminator.train()
            generator.train()

            # top_k = pred.topk(opt.generate_num, dim=0)[1]
            # print(top_k)
            # result = []
            # for num in top_k:
            #    result.append(fake_images[num])
            # print(fake_images[num].shape)

            # accuracy = 0.0
            # for p in pred.tolist():
            #     if p > 0.5:
            #         accuracy += 1.0
            # accuracy = round(accuracy / opt.batch_size, 2)

            writer.add_scalar('epoch_loss_g_1', meter_g, i + opt.epoch_count + 1)
            writer.add_scalar('epoch_loss_d_1', meter_d, i + opt.epoch_count + 1)
            # writer.add_scalar('accuracy', accuracy, i + opt.epoch_count + 1)

            # writer.add_scalar('epoch_fid', cal_fid(illust2vec), i + opt.epoch_count + 1)

            torchvision.utils.save_image(fake_images[0:64, :, :, :],
                                         "images/epoch-" + str(i + opt.epoch_count + 1) + "-loss_g-" + str(
                                             round(loss_g.item(), 4)) + time.strftime("-%H:%M:%S") + ".jpg",
                                         normalize=True, range=(-1, 1))
            # print("epoch:%d | avg_loss_d: %.4f | avg_loss_g: %.4f | accuracy: %.2f\n"
            # % (i + 1, meter_d, meter_g, accuracy))

        if (i + 1) % 10 == 0:
            discriminator.save_with_label("method01")
            generator.save_with_label("method01")


def generate(**kwargs):
    opt._parse(kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noises = torch.randn(opt.batch_size, opt.noise_dimension, 1, 1).to(device)
    discriminator, generator = get_net(device)
    discriminator.eval()
    generator.eval()

    with torch.no_grad():
        fake_images = generator(noises)
        root = "resize/week07_01/"
        for i in range(4):
            for k in range(opt.batch_size):
                torchvision.utils.save_image(fake_images[k], root + str(k + 1 + 64 * i) + ".jpg", normalize=True, range=(-1, 1))


def get_net(device):
    discriminator = getattr(models, 'net_D1')(opt).to(device)
    generator = getattr(models, 'net_G1')(opt).to(device)
    discriminator.apply(weights_init)  # 尝试
    generator.apply(weights_init)
    if opt.load_discriminator is not None:
        discriminator.load(opt.load_discriminator)
    if opt.load_generator is not None:
        generator.load(opt.load_generator)
    return discriminator, generator


if __name__ == '__main__':
    import fire
    fire.Fire()


