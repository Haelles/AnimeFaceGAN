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


# ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(**kwargs):
    opt._parse(kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter('runs/week6_q5')
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
    one_sided_label_smoothing = torch.ones(opt.batch_size).fill_(0.9).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)
    noises = torch.randn(opt.batch_size, opt.noise_dimension, 1, 1).to(device)
    fix_noises = torch.randn(opt.batch_size, opt.noise_dimension, 1, 1).to(device)  # 固定值，用于save/generate

    print("load data success\nbegin to construct net\n")
    discriminator, generator = get_net(device)

    lr1 = opt.lr1
    lr2 = opt.lr2
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr1, betas=(opt.beta1, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=lr2, betas=(opt.beta1, 0.999))

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
        cnt = 0

        for j, (data, _) in tqdm.tqdm(enumerate(true_data_loader)):
            data = data.to(device)
            optimizer_d.zero_grad()
            res = discriminator(data)
            loss_d1 = cal_d(res, true_labels)
            loss_d1.backward()

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
            fake_images = generator(noises)
            pred = discriminator(fake_images)
            loss_g = cal_g(pred, true_labels)
            loss_g.backward()
            optimizer_g.step()
            meter_g += float(loss_g)

            iter_count += 1
            if j % 30 == 0:
                writer.add_scalar('mini_loss_g', loss_g.item(), iter_count)  # mini batch loss
                writer.add_scalar('mini_loss_d', float(0.5 * loss_d1 + 0.5 * loss_d2), iter_count)
            cnt += 1

        # record epoch loss
        meter_g /= cnt
        meter_d /= cnt
        with torch.no_grad():
            discriminator.eval()
            generator.eval()
            noises.copy_(torch.randn(opt.batch_size, opt.noise_dimension, 1, 1))
            fake_images = generator(noises)
            pred = discriminator(fake_images)
            loss_g = cal_g(pred, true_labels)
            discriminator.train()
            generator.train()

            top_k = pred.topk(opt.generate_num)[1]
            result = []
            for num in top_k:
                result.append(fake_images[num])

            accuracy = 0.0
            for p in pred.tolist():
                if p > 0.5:
                    accuracy += 1.0
            accuracy = round(accuracy / opt.batch_size, 2)

            writer.add_scalar('epoch_loss_g', meter_g, i + opt.epoch_count + 1)
            writer.add_scalar('epoch_loss_d', meter_d, i + opt.epoch_count + 1)
            writer.add_scalar('accuracy', accuracy, i + opt.epoch_count + 1)

            root = "resize/fake/"
            for k in range(opt.batch_size):
                torchvision.utils.save_image(fake_images[k], root + str(i + 1) + ".png", normalize=True, range=(-1, 1))
            writer.add_scalar('iter_fid', cal_fid(), i + opt.epoch_count + 1)

            torchvision.utils.save_image(result, "images/epoch-" + str(i + opt.epoch_count + 1) + "-loss_g-" + str(
                round(loss_g.item(), 4)) + "-accuracy-" + str(accuracy) + time.strftime("-%H:%M:%S") + ".png",
                                         normalize=True, range=(-1, 1))
            # print("epoch:%d | avg_loss_d: %.4f | avg_loss_g: %.4f | accuracy: %.2f\n"
            # % (i + 1, meter_d, meter_g, accuracy))

        if (i + 1) % 10 == 0:
            discriminator.save()
            generator.save()

        if meter_d > pre_loss_d:
            lr1 *= opt.lr_decay
            print("epoch:%d | new lr1: %.15f\n" % (i + opt.epoch_count + 1, lr1))
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = lr1
        pre_loss_d = meter_d

        if meter_g > pre_loss_g:
            lr2 *= opt.lr_decay
            print("epoch:%d | new lr2: %.15f\n" % (i + opt.epoch_count + 1, lr2))
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = lr2
        pre_loss_g = meter_g


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

        accuracy = 0.0
        for p in pred.tolist():
            if p > 0.5:
                accuracy += 1.0

        for i in top_k:
            result.append(fake_images[i])

        torchvision.utils.save_image(result, "images/test-" + str(round(loss.item(), 4)) + "-accuracy-" + str(
            round(accuracy / opt.batch_size, 2)) + time.strftime("-%m_%d_%H:%M:%S") + ".png", normalize=True,
                                     range=(-1, 1))


def get_net(device):
    discriminator = getattr(models, 'Discriminator')(opt).to(device)
    generator = getattr(models, 'Generator')(opt).to(device)
    if opt.load_discriminator is not None:
        discriminator.load(opt.load_discriminator)
    if opt.load_generator is not None:
        generator.load(opt.load_generator)
    return discriminator, generator


def cal_fid():
    """
    与256张真实图片对比，每张图片通过Illustation2vec feature extractor得到一个1*4096向量
    然后计算FID
    :return:fid
    """
    from scipy.linalg import sqrtm
    import i2v
    import numpy as np
    import os

    illust2vec = i2v.make_i2v_with_chainer("illust2vec_ver200.caffemodel")

    # 真实图片的特征向量
    root = "resize/true/"
    paths = os.listdir(root)
    images_list = [Image.open(root + path) for path in paths]
    true_vec = illust2vec.extract_feature(images_list)

    # 计算生成图片的特征向量
    root = "resize/fake/"
    paths = os.listdir(root)
    images_list = [Image.open(root + path) for path in paths]
    fake_vec = illust2vec.extract_feature(images_list)
    # 清空文件夹，下次继续使用
    for path in paths:
        os.remove(root + path)

    # 计算FID
    mu1 = true_vec.mean(axis=0)
    sigma1 = np.cov(true_vec, rowvar=False)
    mu2 = fake_vec.mean(axis=0)
    sigma2 = np.cov(fake_vec, rowvar=False)
    # calculate sum squared difference between means
	sum_squared_diff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
	return sum_squared_diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


if __name__ == '__main__':
    import fire
    fire.Fire()



