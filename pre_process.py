import os
import cv2
import PIL
import random
from PIL import Image
from shutil import copyfile


def get_true_for_cal_fid():
    source = "../data/cropped/"
    target = "./resize/true/"
    files = os.listdir(source)
    seed = random.randint(1, 100)
    for i in range(256):
        copyfile(source + files[seed * (i + 1)], target + str(i + 1) + ".jpg")


def get_broken():
    root = "../data/cropped/"
    files = os.listdir(root)
    file_list = [
        path for path in files
    ]
    for path in file_list:
        try:
            img = cv2.imread(root + path, 1)
        except PIL.UnidentifiedImageError:
            os.remove(root + path)
            print("remove" + root + path + "\n")
        except FileNotFoundError:
            os.remove(root + path)
            print("remove" + root + path + "\n")
        except IOError:
            os.remove(root + path)
            print("remove" + root + path + "\n")


def resize_original():
    """
    用于改变原始图片尺寸
    :return:无
    """
    root = "data/cropped/"
    files = os.listdir(root)
    file_list = [
        path for path in files
    ]
    cnt = 0
    for file in file_list:
        img = cv2.resize(cv2.imread(root + file), (96, 96), interpolation=cv2.INTER_AREA)
        cv2.imwrite("resize/true/" + str(cnt) + ".jpg", img)
        cnt += 1


def get_96_size():
    import os
    import shutil

    source_path = os.path.abspath(r'../动漫头像/多类头像资源')
    target_path = os.path.abspath(r'../动漫头像/96-96')

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if os.path.exists(source_path):
        for root, dirs, files in os.walk(source_path):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, target_path)
                # print(src_file)
    print('copy files finished!')


def make_fake_dir():
    root = "./resize/fake"
    for i in range(256):
        os.mkdir(root + str(i + 1))


if __name__ == '__main__':
    import fire
    fire.Fire()

