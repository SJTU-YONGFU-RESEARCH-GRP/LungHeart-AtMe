import os
import sys
import json
import pickle
import random
import logging

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
from Args import args


error_file = ['102_1b1_Ar_sc_Meditron5.png']


def read_split_data(train_root: str, val_root: str, train_val_set_plot_path: str, class_js_path: str, data_path_lists: str):
    '''
    generate path lists and label lists
    :param train_root: root path of train set
    :param val_root: root path of test set
    :param train_val_set_plot_path: path to save train set and test set data distribution histogram
    :param class_js_path: path to save classes.json
    :param data_path_lists: path to save data lists
    :return: 4 lists: train_images_path, train_images_label, val_images_path, val_images_label
    '''
    random.seed(0)
    assert os.path.exists(train_root), "dataset root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "dataset root: {} does not exist.".format(val_root)

    # A folder corresponds to a category
    spectrogram_class = [cla for cla in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, cla))]
    spectrogram_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(spectrogram_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open(class_js_path, 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_train_num = []
    every_class_val_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    for cla in spectrogram_class:
        train_cla_path = os.path.join(train_root, cla)
        val_cla_path = os.path.join(val_root, cla)
        train_images = [os.path.join(train_cla_path, i) for i in os.listdir(train_cla_path)
                        if (os.path.splitext(i)[-1] in supported) and i not in error_file]
        val_images = [os.path.join(val_cla_path, i) for i in os.listdir(val_cla_path)
                      if (os.path.splitext(i)[-1] in supported) and i not in error_file]
        image_class = class_indices[cla]
        every_class_train_num.append(len(train_images))
        every_class_val_num.append(len(val_images))

        for img_path in train_images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)
        for img_path in val_images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_train_num) + sum(every_class_val_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = True
    if plot_image:
        # draw histogram for every class
        plt.bar(range(len(spectrogram_class)), every_class_train_num, align='center', label='train')
        plt.bar(range(len(spectrogram_class)), every_class_val_num, align='center',
                bottom=every_class_train_num, label='valid', tick_label=spectrogram_class, fc='y')
        plt.xticks(range(len(spectrogram_class)), spectrogram_class)
        for i, v in enumerate(every_class_train_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
            plt.text(x=i, y=every_class_val_num[i] + v + 5, s=str(every_class_val_num[i]), ha='center')
        plt.xlabel('image classes')
        plt.ylabel('number of train-valid set images')
        plt.title('spectrogram-classes distribution')
        plt.legend()
        plt.savefig(train_val_set_plot_path)
        plt.clf()
        # plt.show()
    
    data_lists = [train_images_path, train_images_label, val_images_path, val_images_label]
    write_pickle(data_lists, data_path_lists)

    return data_lists


def plot_data_loader_image(class_js_path: str, data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = class_js_path
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    '''
    save list_info to file as python object
    '''
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    '''
    read information from file as python object
    '''
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
    return info_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r", encoding='utf-8') as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r", encoding='utf-8') as package_f:
            logger.info(package_f.read())

    return logger


def cal_scores(Confusion_matrix):
    Sp_total = (Confusion_matrix[3][3] + Confusion_matrix[4][4]) / (sum(Confusion_matrix[3]) + sum(Confusion_matrix[4]))
    Se_total = (Confusion_matrix[0][0] + Confusion_matrix[1][1] + Confusion_matrix[2][2] + Confusion_matrix[5][5]) / (
            sum(Confusion_matrix[0]) + sum(Confusion_matrix[1]) + sum(Confusion_matrix[2]) + sum(Confusion_matrix[5]))

    acc_lung = (Confusion_matrix[0][0] + Confusion_matrix[1][1] + Confusion_matrix[4][4] + Confusion_matrix[5][5]) / (
            sum(Confusion_matrix[0]) + sum(Confusion_matrix[1]) + sum(Confusion_matrix[4]) + sum(Confusion_matrix[5]))
    acc_heart = (Confusion_matrix[2][2] + Confusion_matrix[3][3]) / (
                sum(Confusion_matrix[2]) + sum(Confusion_matrix[3]))

    Sp_lung = Confusion_matrix[4][4] / (sum(Confusion_matrix[4]))
    Sp_heart = Confusion_matrix[3][3] / (sum(Confusion_matrix[3]))

    Se_lung = (Confusion_matrix[0][0] + Confusion_matrix[1][1] + Confusion_matrix[5][5]) / (
           sum(Confusion_matrix[0]) + sum(Confusion_matrix[1]) + sum(Confusion_matrix[5]))
    Se_heart = Confusion_matrix[2][2] / (sum(Confusion_matrix[2]))

    results = [[Se_total, Sp_total, (Se_total + Sp_total) / 2],
               [acc_lung, Se_lung, Sp_lung, (Se_lung + Sp_lung) / 2],
               [acc_heart, Se_heart, Sp_heart, (Se_heart + Sp_heart) / 2]]

    return results


def loss_function(device):
    pos_weight = torch.tensor(args.pos_weight, dtype=torch.float).to(device)
    # pos_weight = None
    criterion = torch.nn.CrossEntropyLoss(weight=pos_weight)
    # weighted_BCE = criterion()
    return criterion


class Focal_Loss():
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps = 1e-7
        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)

        target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)

        ce = -1 * torch.log(y_pred + eps) * target
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = torch.zeros(1).to(device)  # Loss
    total_accu = torch.zeros(1).to(device)   # Number of samples with correct predictions
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)  # configure progress bars

    pred_list = np.array([])
    targ_list = np.array([])
    for step, data in enumerate(data_loader):
        images, labels = data

        sample_num += images.shape[0]

        in_img = images.to(device)
        # Tensor: (32, 6)
        pred = model(in_img)
        # print("Outside: input size", in_img.size(), "output_size", pred.size())

        # Tensor: (32,)
        pred_classes = torch.max(pred, dim=1)[1]
        total_accu += torch.eq(pred_classes, labels.to(device)).sum()

        # label: Tensor: (32,)
        loss = loss_function(device)(pred, labels.to(device))
        loss.backward()
        total_loss += loss.detach()

        pred_np = pred_classes.cpu().detach().numpy()
        # print("pred_np: ", pred_np)
        if not (len(pred_list.shape) == len(pred_np.shape)):
            pred_list = pred_np
        else:
            pred_list = np.concatenate([pred_list, pred_np], axis=0)
        # print("pred_list: ", pred_list)
        targ_np = labels.numpy()
        # print("targ_np: ", targ_np)
        if not (len(targ_list.shape) == len(targ_np.shape)):
            targ_list = targ_np
        else:
            targ_list = np.concatenate([targ_list, targ_np], axis=0)
        # print("targ_list: ", targ_list)

        data_loader.desc = "[train epoch {}] loss: {:.3f}, total_acc: {:.3f}".format(
            epoch, total_loss.item() / (step + 1), total_accu.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return total_loss.item() / (step + 1), total_accu.item() / sample_num, [targ_list, pred_list]


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    # loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    total_accu = torch.zeros(1).to(device)   # Number of samples with correct predictions
    total_loss = torch.zeros(1).to(device)  # Loss

    pred_list = np.array([])
    targ_list = np.array([])
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        # print("step: ", step)
        # print("label: ", labels)
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        total_accu += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(device)(pred, labels.to(device))
        # loss = loss_function(pred, labels.to(device))
        total_loss += loss

        pred_np = pred_classes.cpu().detach().numpy()
        if not (len(pred_list.shape) == len(pred_np.shape)):
            pred_list = pred_np
        else:
            pred_list = np.concatenate([pred_list, pred_np], axis=0)
        targ_np = labels.numpy()
        if not (len(targ_list.shape) == len(targ_np.shape)):
            targ_list = targ_np
        else:
            targ_list = np.concatenate([targ_list, targ_np], axis=0)

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               total_loss.item() / (step + 1),
                                                                               total_accu.item() / sample_num)

    return total_loss.item() / (step + 1), total_accu.item() / sample_num, [targ_list, pred_list]


def plot_results(csv_path, save_dir):
    result = pd.read_csv(csv_path)
    epoch = result.loc[:, 'epoch']
    train_acc_total = result.loc[:, 'train_acc_total']
    train_acc_heart = result.loc[:, 'train_acc_heart']
    train_acc_lung = result.loc[:, 'train_acc_lung']
    train_loss_total = result.loc[:, 'train_loss_total']
    train_se_total = result.loc[:, 'train_se_total']
    train_se_heart = result.loc[:, 'train_se_heart']
    train_se_lung = result.loc[:, 'train_se_lung']
    train_sp_total = result.loc[:, 'train_sp_total']
    train_sp_heart = result.loc[:, 'train_sp_heart']
    train_sp_lung = result.loc[:, 'train_sp_lung']
    train_score_total = result.loc[:, 'train_score_total']
    train_score_heart = result.loc[:, 'train_score_heart']
    train_score_lung = result.loc[:, 'train_score_lung']

    val_acc_total = result.loc[:, 'val_acc_total']
    val_acc_heart = result.loc[:, 'val_acc_heart']
    val_acc_lung = result.loc[:, 'val_acc_lung']
    val_loss_total = result.loc[:, 'val_loss_total']
    val_se_total = result.loc[:, 'val_se_total']
    val_se_heart = result.loc[:, 'val_se_heart']
    val_se_lung = result.loc[:, 'val_se_lung']
    val_sp_total = result.loc[:, 'val_sp_total']
    val_sp_heart = result.loc[:, 'val_sp_heart']
    val_sp_lung = result.loc[:, 'val_sp_lung']
    val_score_total = result.loc[:, 'val_score_total']
    val_score_heart = result.loc[:, 'val_score_heart']
    val_score_lung = result.loc[:, 'val_score_lung']

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(epoch, train_acc_total, label="train_acc_total")
    plt.plot(epoch, train_loss_total, label="train_loss_total")
    plt.plot(epoch, val_acc_total, label="val_acc_total")
    plt.plot(epoch, val_loss_total, label="val_loss_total")
    plt.title("Total Accuracy and Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy/Loss")
    plt.legend(loc="upper left")
    my_x_ticks = np.arange(0, args.epochs, 10)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    pic_path = os.path.join(save_dir, 'Acc_Loss_Total.png')
    plt.savefig(pic_path)
    # plt.show()
    plt.close()

    plt.plot(epoch, train_acc_lung, label="train_acc_lung")
    plt.plot(epoch, train_se_lung, label="train_se_lung")
    plt.plot(epoch, train_sp_lung, label="train_sp_lung")
    plt.plot(epoch, train_score_lung, label="train_score_lung")
    plt.plot(epoch, val_acc_lung, label="val_acc_lung")
    plt.plot(epoch, val_se_lung, label="val_se_lung")
    plt.plot(epoch, val_sp_lung, label="val_sp_lung")
    plt.plot(epoch, val_score_lung, label="val_score_lung")
    plt.title("Lung Metrics")
    plt.xlabel("Epoch #")
    plt.ylabel("Metrics")
    plt.legend(loc="upper left")
    my_x_ticks = np.arange(0, args.epochs, 10)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    pic_path = os.path.join(save_dir, 'Metrics_Lung.png')
    plt.savefig(pic_path)
    # plt.show()
    plt.close()

    plt.plot(epoch, train_acc_heart, label="train_acc_heart")
    plt.plot(epoch, train_se_heart, label="train_se_heart")
    plt.plot(epoch, train_sp_heart, label="train_sp_heart")
    plt.plot(epoch, train_score_heart, label="train_score_heart")
    plt.plot(epoch, val_acc_heart, label="val_acc_heart")
    plt.plot(epoch, val_se_heart, label="val_se_heart")
    plt.plot(epoch, val_sp_heart, label="val_sp_heart")
    plt.plot(epoch, val_score_heart, label="val_score_heart")
    plt.title("Heart Metrics")
    plt.xlabel("Epoch #")
    plt.ylabel("Metrics")
    plt.legend(loc="upper left")
    my_x_ticks = np.arange(0, args.epochs, 10)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    pic_path = os.path.join(save_dir, 'Metrics_Heart.png')
    plt.savefig(pic_path)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    read_split_data(args.data_path_train, args.data_path_test, args.data_distribution_histogram_path,
                    args.class_json_path,
                    args.data_path_lists)