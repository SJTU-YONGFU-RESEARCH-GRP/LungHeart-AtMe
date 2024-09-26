import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lrf', type=float, default=0.01)
# parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.3)
# for lung - weighted
# parser.add_argument('--pos-weight', type=list, default=[1, 2, 1, 1, 2, 1])
parser.add_argument('--pos-weight', type=list, default=[1, 1, 1, 1, 1, 1])
parser.add_argument('--weight-ratio', type=int, default=1)

#========== dataset ==========#
parser.add_argument('--denoise-threshold', type=float, default=0.2)
parser.add_argument('--ts-aug-ratio', type=list,
                    default=[4, 1, 0, 0, 0, 2])
parser.add_argument('--sc-aug-ratio', type=list,
                    default=[1, 0, 0, 0, 0, 1])
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--data-path-train', type=str,
                    default="Dataset/aug_denoise_stft_mfcc/train")
parser.add_argument('--data-path-test', type=str,
                    default="Dataset/aug_denoise_stft_mfcc/test")
parser.add_argument('--class-json-path', type=str,
                    default="Dataset/aug_denoise_stft_mfcc/classes.json")
parser.add_argument('--data-distribution-histogram-path', type=str,
                    default="Dataset/aug_denoise_stft_mfcc/data_distribution_histogram.png")
parser.add_argument('--data-path-lists', type=str,
                    default="Dataset/aug_denoise_stft_mfcc/train_val_lists.p")

#========== for model ==========#
parser.add_argument('--model-name', default='', help='create model name')
parser.add_argument('--num_classes', type=int, default=6)
# for PLE
parser.add_argument('--num_spe_exp', type=int, default=1)
parser.add_argument('--num_sha_exp', type=int, default=1)
# for AtMe
parser.add_argument('--shape', type=int, default=14)

#========== for environment ==========#
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu), for single gpu training')
parser.add_argument('--gpu', default=1, help='device number, for parallel or distributed training')
parser.add_argument('--work-dir', type=str, default="experiments")

# 是否debug
parser.add_argument('--DEBUG', type=bool, default=False)

args = parser.parse_args()
