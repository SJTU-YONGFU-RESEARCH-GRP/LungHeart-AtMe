# coding=utf-8
import csv
import json
import os
import math
import shutil
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from Dataset import *
from utils import *
from Args import args

from PLE.PLE import PLE_base as create_model

# os.environ['CUDA_VISIBLE_DEVICES'] = '9'


if __name__ == '__main__':
    date_now = time.localtime()
    
    version = "AtMe"
    save_name = f'epochs' + str(args.epochs) + f'_bs' + str(args.batch_size) + f'_lr' + str(args.lr) + \
                f'_{date_now.tm_year}-{date_now.tm_mon}-{date_now.tm_mday}-{date_now.tm_hour}-{date_now.tm_min}-{date_now.tm_sec}'
    work_dir = os.path.join(args.work_dir, version, save_name)
    ensure_dir_exists(work_dir)

    weights_dir = os.path.join(work_dir, "weights")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    codes_dir = os.path.join(work_dir, "codes")
    if not os.path.exists(codes_dir):
        os.makedirs(codes_dir)

    source_code_to_save = \
        ['./Args.py', './Dataset.py', './train.py', './utils.py', './base_models.py',
         'PLE/PLE.py', 'PLE/PLE_tower.py', 'PLE/PLE_tower_lung.py', 'PLE/PLE_experts.py', 'PLE/attention_augment_conv.py']
    for source_code_to_save_temp in source_code_to_save:
        shutil.copy(source_code_to_save_temp, codes_dir)

    tb_writer = SummaryWriter()

    start_t = time.time()

    if not os.path.exists(args.data_path_lists):
        [train_images_path, train_images_label, val_images_path, val_images_label] = \
            read_split_data(args.data_path_train, args.data_path_test, args.data_distribution_histogram_path, args.class_json_path,
                            args.data_path_lists)
    else:
        [train_images_path, train_images_label, val_images_path, val_images_label] = read_pickle(args.data_path_lists)

    data_transform = {
        "train": transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 2
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    logger = get_logger(logpath=os.path.join(work_dir, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    gpu_usable = torch.cuda.is_available()
    device = torch.device(args.device if gpu_usable else "cpu")
    print("Device used now: ", device)

    model = create_model(args.num_spe_exp, args.num_sha_exp)

    pg = [p for p in model.parameters() if p.requires_grad]

    model = model.to(device)

    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    prepare_t = time.time()
    print('The preparation time:', prepare_t - start_t)

    confusion_matrix = []
    confusion_matrix_val = []

    # write Lung/Heart Acc,Se,Sp,Score to csv
    csv_path = os.path.join(work_dir, "result.csv")
    with open(csv_path, 'w') as f:
        csv_write = csv.writer(f)
        csv_head = ['epoch',
                    'train_loss_total',
                    'train_acc_total', 'train_se_total', 'train_sp_total', 'train_score_total',
                    'train_acc_lung', 'train_se_lung', 'train_sp_lung', 'train_score_lung',
                    'train_acc_heart', 'train_se_heart', 'train_sp_heart', 'train_score_heart',
                    'val_loss_total',
                    'val_acc_total', 'val_se_total', 'val_sp_total', 'val_score_total',
                    'val_acc_lung', 'val_se_lung', 'val_sp_lung', 'val_score_lung',
                    'val_acc_heart', 'val_se_heart', 'val_sp_heart', 'val_score_heart']
        csv_write.writerow(csv_head)

    minimum_loss = 1000
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc, [targ_list, pred_list] = train_one_epoch(model=model,
                                                                        optimizer=optimizer,
                                                                        data_loader=train_loader,
                                                                        device=device,
                                                                        epoch=epoch)
        Confusion_matrix = sk_confusion_matrix(targ_list.tolist(), pred_list.tolist())
        print('Confusion_matrix:')
        print(Confusion_matrix)
        confusion_matrix.append(Confusion_matrix)
        scheduler.step()

        # calculate se, sp, score for lung-classification and heart-classification specifically
        [[train_Se_total, train_Sp_total, train_Score_total],
         [train_acc_lung, train_Se_lung, train_Sp_lung, train_Score_lung],
         [train_acc_heart, train_Se_heart, train_Sp_heart, train_Score_heart]] = cal_scores(Confusion_matrix)


        # validate
        val_loss, val_acc, [targ_val, pred_val] = evaluate(
                                                    model=model,
                                                    data_loader=val_loader,
                                                    device=device,
                                                    epoch=epoch)
        Confusion_matrix_val = sk_confusion_matrix(targ_val.tolist(), pred_val.tolist())
        print('Valid Confusion_matrix:')
        print(Confusion_matrix_val)
        confusion_matrix_val.append(Confusion_matrix_val)

        # calculate se, sp, score for lung-classification and heart-classification specifically
        [[val_Se_total, val_Sp_total, val_Score_total],
         [val_acc_lung, val_Se_lung, val_Sp_lung, val_Score_lung],
         [val_acc_heart, val_Se_heart, val_Sp_heart, val_Score_heart]] = cal_scores(Confusion_matrix_val)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            "Epoch {:04d}  |  "
            "Train Loss {:.4f} | Train Acc {:.4f} | Train Lung Score {:.4f} | Train Heart Score {:.4f} | "
            "Valid Loss {:.4f} | Valid Acc {:.4f} | Valid Lung Score {:.4f} | Valid Heart Score {:.4f}".format(
                epoch,
                train_loss, train_acc, train_Score_lung, train_Score_heart,
                val_loss, val_acc, val_Score_lung, val_Score_heart
            )
        )

        # write Lung/Heart Acc,Se,Sp,Score to csv
        with open(csv_path, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [epoch,
                        train_loss,
                        train_acc, train_Se_total, train_Sp_total, train_Score_total,
                        train_acc_lung, train_Se_lung, train_Sp_lung, train_Score_lung,
                        train_acc_heart, train_Se_heart, train_Sp_heart, train_Score_heart,
                        val_loss,
                        val_acc, val_Se_total, val_Sp_total, val_Score_total,
                        val_acc_lung, val_Se_lung, val_Sp_lung, val_Score_lung,
                        val_acc_heart, val_Se_heart, val_Sp_heart, val_Score_heart]
            csv_write.writerow(data_row)

        model_path = os.path.join(weights_dir, "model.pt")
        if val_loss < minimum_loss:
            minimum_loss = val_loss
            print('Saving best model parameters with Test Loss = %.5f' % minimum_loss)
            torch.save(model.state_dict(), model_path)


    # write confusion matrix to json file
    con_mat_dict = dict()
    for idx, matrix in enumerate(confusion_matrix):
        con_mat_str = dict()
        for cls, row in enumerate(matrix.tolist()):
            con_mat_str[str(cls)] = str(row)
        con_mat_dict[str(idx)] = con_mat_str

    con_mat_dict_val = dict()
    for idx, matrix in enumerate(confusion_matrix_val):
        con_mat_str = dict()
        for cls, row in enumerate(matrix.tolist()):
            con_mat_str[str(cls)] = str(row)
        con_mat_dict_val[str(idx)] = con_mat_str

    json_cm = dict()
    json_cm["train"] = con_mat_dict
    json_cm["val"] = con_mat_dict_val
    json_cm = json.dumps(json_cm, indent=4)
    json_path = os.path.join(work_dir, "Confusion_Matrix.json")
    with open(json_path, 'w') as json_file:
        json_file.write(json_cm)

    plot_results(csv_path, work_dir)

    print("------------------- Training and Validation Finished! -------------------")
