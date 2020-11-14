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

# ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(**kwargs):
    f = open('lr.txt', 'a')
    opt._parse(kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter('runs/week06_q7_3')
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
    LAMBDA = opt.LAMBDA
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr1, betas=(opt.beta1, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=lr2, betas=(opt.beta1, 0.999))

    # cal_d = nn.BCELoss().to(device)
    # cal_g = nn.BCELoss().to(device)

    discriminator.train()
    generator.train()

    print("construct net success\nbegin to train\n")

    pre_loss_g = 1e4
    pre_loss_d = 1e4

    for i in range(opt.epoch):
        meter_d = 0.0
        meter_g = 0.0
        cnt1 = 0
        cnt2 = 0

        for j, (true_images, _) in tqdm.tqdm(enumerate(true_data_loader)):
            true_images = true_images.to(device)
            optimizer_d.zero_grad()
            res = discriminator(true_images)
            loss_d1 = -1 * res.mean()
            loss_d1.backward()

            noises.copy_(torch.randn(opt.batch_size, opt.noise_dimension, 1, 1))
            fake_images = generator(noises).detach()
            res = discriminator(fake_images)
            loss_d2 = res.mean()
            loss_d2.backward()

            loss_d3 = cal_gradient_penalty(discriminator, device, true_images, fake_images, LAMBDA)
            loss_d3.backward()
            optimizer_d.step()

            temp_loss_d = loss_d1 + loss_d2 + loss_d3
            meter_d += float(temp_loss_d)

            # 每训练5次判别器，训练1次生成器
            if (j + 1) % 5 == 0:
                optimizer_g.zero_grad()
                fake_images = generator(noises)
                pred = discriminator(fake_images)
                loss_g = -1 * pred.mean()
                loss_g.backward()
                optimizer_g.step()
                meter_g += float(loss_g)
                cnt2 += 1
                if (j + 1) % 30 == 0:
                    writer.add_scalar('mini_loss_g', loss_g.item(), iter_count + 1)  # mini batch loss
                    writer.add_scalar('mini_loss_d', temp_loss_d, iter_count + 1)

            cnt1 += 1  # 1 -> 248
            iter_count += 1  # 1 -> 248iters * 60 epochs

        # record epoch loss
        meter_g /= cnt2
        meter_d /= cnt1
        with torch.no_grad():
            discriminator.eval()
            generator.eval()
            noises.copy_(torch.randn(opt.batch_size, opt.noise_dimension, 1, 1))
            fake_images = generator(noises)
            pred = discriminator(fake_images)
            loss_g = -1 * pred.mean()
            discriminator.train()
            generator.train()

            # 事先用os.mkdir建好文件夹，后续用CPU并行计算FID
            root = "resize/fake3/" + str(i + opt.epoch_count + 1) + "/"
            for k in range(opt.batch_size):
                torchvision.utils.save_image(fake_images[k], root + str(k + 1) + ".jpg", normalize=True, range=(-1, 1))

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

            writer.add_scalar('epoch_loss_g', meter_g, i + opt.epoch_count + 1)
            writer.add_scalar('epoch_loss_d', meter_d, i + opt.epoch_count + 1)
            # writer.add_scalar('accuracy', accuracy, i + opt.epoch_count + 1)

            # writer.add_scalar('epoch_fid', cal_fid(illust2vec), i + opt.epoch_count + 1)

            torchvision.utils.save_image(fake_images[0:64, :, :, :],
                                         "images/epoch-" + str(i + opt.epoch_count + 1) + "-loss_g-" + str(
                                             round(loss_g.item(), 4)) + time.strftime("-%H:%M:%S") + ".jpg",
                                         normalize=True, range=(-1, 1))
            # print("epoch:%d | avg_loss_d: %.4f | avg_loss_g: %.4f | accuracy: %.2f\n"
            # % (i + 1, meter_d, meter_g, accuracy))

        if (i + 1) % 10 == 0:
            discriminator.save()
            generator.save()

        if abs(meter_d) > abs(pre_loss_d):
            lr1 *= opt.lr_decay
            print("epoch:%d | new lr1: %.15f\n" % (i + opt.epoch_count + 1, lr1))
            f.write("epoch:%d | new lr1: %.15f\n" % (i + opt.epoch_count + 1, lr1))
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = lr1
        pre_loss_d = meter_d

        if abs(meter_g) > abs(pre_loss_g):
            lr2 *= opt.lr_decay
            print("epoch:%d | new lr2: %.15f\n" % (i + opt.epoch_count + 1, lr2))
            f.write("epoch:%d | new lr2: %.15f\n" % (i + opt.epoch_count + 1, lr2))
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = lr2
        pre_loss_g = meter_g

    f.close()


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

        root = "resize/modify2/"
        for i in range(4):
            for k in range(opt.batch_size):
                torchvision.utils.save_image(fake_images[k], root + str(k + 1 + 64 * i) + ".jpg", normalize=True, range=(-1, 1))


def get_net(device):
    discriminator = getattr(models, 'Discriminator')(opt).to(device)
    generator = getattr(models, 'Generator')(opt).to(device)
    if opt.load_discriminator is not None:
        discriminator.load(opt.load_discriminator)
    if opt.load_generator is not None:
        generator.load(opt.load_generator)
    return discriminator, generator


def cal_gradient_penalty(discriminator, device, true_images, fake_images, LAMBDA):
    """

    :param discriminator:  net D
    :param device: GPU0
    :param true_images: real samples
    :param fake_images: fake samples
    :return: gradient_penalty
    """
    alpha = torch.randn(true_images.size()[0], 1, 1, 1).to(device)
    interpolates = (alpha * true_images + (1 - alpha) * fake_images).requires_grad_(True).to(device)
    d_interpolates = discriminator(interpolates)
    grad_outputs = torch.ones(d_interpolates.size()).to(device)
    gradients = torch.autograd.grad(d_interpolates, interpolates,
                                    grad_outputs=grad_outputs, create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size()[0], -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA


def cal_fid():
    """
    与真实图片对比，每张图片通过Illustation2vec feature extractor得到一个1*4096向量
    然后计算FID
    :return:fid
    """

    illust2vec = i2v.make_i2v_with_chainer("illust2vec_ver200.caffemodel")
    # 真实图片的特征向量
    root = "resize/true/"
    paths = os.listdir(root)
    true_list = []
    for path in paths:
        if "jpg" in path:
            true_list.append(Image.open(root + path))
    res_true = []
    for i in range(opt.batch_size):
        result_real = illust2vec.extract_feature([true_list[i]])
        res_true.append(result_real)
        # print(str(i))
    true_vec = np.concatenate(tuple(res_true), axis=0)
    mu1 = true_vec.mean(axis=0)
    sigma1 = np.cov(true_vec, rowvar=False)
    print("true_vec_done")

    fid1 = temp_cal("resize/fake/", 256, mu1, sigma1)
    print("fid1:" + str(fid1))
    fid2 = temp_cal("resize/fake2/40/", 64, mu1, sigma1)
    print("fid2:" + str(fid2))
    fid3 = temp_cal("resize/fake3/40/", 64, mu1, sigma1)
    print("fid2:" + str(fid3))


def temp_cal(root, illust2vec, size, mu1, sigma1):
    fake_list = []
    paths = os.listdir(root)
    for path in paths:
        if "jpg" in path:
            fake_list.append(Image.open(root + path))
    res_fake = []
    for j in range(size):
        result_fake = illust2vec.extract_feature([fake_list[j]])
        res_fake.append(result_fake)
        # print(str(j))
    fake_vec = np.concatenate(tuple(res_fake), axis=0)
    mu2 = fake_vec.mean(axis=0)
    sigma2 = np.cov(fake_vec, rowvar=False)
    # calculate sum squared difference between means
    sum_squared_diff = np.sum((mu1 - mu2) ** 2.0)
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

