import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import numpy as np

import argparse
import logging
from datetime import datetime

import os
import sys
import random

from enums import (
    ModelEnum,
    OptimizerEnum,
    LRSchedulerEnum,
    MixEnum
)
from preprocess.preprocess_data import DataPreProcessor
from preprocess.preprocess_model import ModelPreProcessor
from train import Trainer

import torch.autograd as autograd

# autograd.set_detect_anomaly(True)


def get_arg_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--test', type=int, help='테스트모드', default=0)
    parser.add_argument('-pm', '--model_path', type=str, help='모델 폴더 이름', default='')
    parser.add_argument('--amp', type=int, help='amp 옵션', default=0)
    
    parser.add_argument('-rs', '--random_seed', type=int, help='학습 랜덤 시드. -1은 랜덤 시드를 고정하지 않음.', default=4943872)
    parser.add_argument('-lf', '--log_file', type=int, help='로그 파일 출력 여부. 0=false, 1=true', default=1)
    parser.add_argument('-po', '--port', type=int, default=2033)
    
    parser.add_argument('-m', '--model', type=ModelEnum, help='학습 모델', choices=list(ModelEnum), default=ModelEnum.custom)
    parser.add_argument('-p', '--parallel', type=int, help='멀티 gpu 사용 여부. 0=false, 1=true', default=0)
    parser.add_argument('-op', '--optimizer', type=OptimizerEnum, help='옵티마이저', choices=list(OptimizerEnum), default=OptimizerEnum.sgd)
    parser.add_argument('-ls', '--lr_scheduler', type=LRSchedulerEnum, help='lr 스케쥴러', choices=list(LRSchedulerEnum), default=LRSchedulerEnum.custom_annealing)
    
    parser.add_argument('-ds', '--split_ratio', type=float, help='train/validation 분할 비율', default=0.2)
    parser.add_argument('-am', '--augmentation_mode', type=int, help='data augmentation mode', default=0)
    parser.add_argument('-asp', '--augment_split', type=float, help='augmentation 분할 비율', default=0.5)
    parser.add_argument('-w', '--num_worker', type=int, help='train/validation 분할 비율', default=0)
    parser.add_argument('-b', '--batch_size', type=int, help='학습 배치사이즈', default=128)
    parser.add_argument('-mc', '--mix_step', type=int, help='mix 적용시 몇 step마다 적용할지. 0은 모든 step에 적용.', default=0)
    parser.add_argument('-mt', '--mix_method', type=MixEnum, help='mix 방법', choices=list(MixEnum), default=MixEnum.mixup)
    
    parser.add_argument('-pd', '--p_depth', type=int, help='pyramnidnet depth', default=164)
    parser.add_argument('-pa', '--p_alpha', type=int, help='pyramnidnet alpha', default=84)
    parser.add_argument('-ps', '--p_shake', type=int, help='pyramnidnet shake', default=0)
    
    parser.add_argument('-e', '--epoch', type=int, help='epoch', default=100)
    parser.add_argument('-mlr', '--max_learning_rate', type=float, help='optimizer/scheduler max learning rate 설정 (custom cos scheduler는 반대)', default=0.1)
    parser.add_argument('-milr', '--min_learning_rate', type=float, help='optimizer/scheduler min learning rate 설정 (custom cos scheduler는 반대)', default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, help='optimizer weight decay 설정', default=5e-4)
    parser.add_argument('-gc', '--gradient_clip', type=float, help='gradient clip 설정. -1은 비활성화', default=0.1)
    parser.add_argument('-lsm', '--label_smoothing', type=float, help='label smoothing 설정', default=0)
    parser.add_argument('-es', '--early_stopping', type=int, help='ealry stoppin epoch 지정. -1은 비활성화', default=-1)
    parser.add_argument('-ad', '--adaptive', type=int, help="adaptive SAM 사용 여부", default=1)
    parser.add_argument('-snt', '--nesterov', type=int, help="nesterov sgd 사용 여부", default=1)
    parser.add_argument('--rho', type=int, help="SAM rho 파라미터", default=2.0)
    parser.add_argument('-cm', '--cos_max', type=int, help="cos annealing 주기", default=50)
    parser.add_argument('-cp', '--cut_p', type=float, help="cutmix 적용 확률", default=0.5)
    parser.add_argument('-sm', '--step_milestone', nargs='+', type=int, help='step lr scheduler milestone', default=[50])
    
    args = parser.parse_args()
    args.p_shake = args.p_shake == 1
    
    return args


def init_logger(args):
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    if args.log_file == 1:
        log_save_path = "./log"
        if not os.path.exists(log_save_path):
            os.makedirs(log_save_path)

        datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatter_file = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
        handler_file = logging.FileHandler(os.path.join(log_save_path, f'{datetime_now}.log'))
        handler_file.setLevel(logging.DEBUG)
        handler_file.setFormatter(formatter_file)
        logger.addHandler(handler_file)

    def catch_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger = logging.getLogger("main")
        
        logger.error(
            "Unexpected exception.",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = catch_exception


def fix_random(args):
    logger = logging.getLogger("main")
    
    if args.random_seed == -1:
        logger.debug('random seed not fix')
        return
    
    seed = args.random_seed
    deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if 1:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug('random seed fix')


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_training(rank, opts):
    opts.rank = rank
    opts.gpu = opts.rank % torch.cuda.device_count()
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    torch.distributed.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:' + str(opts.port),
                            world_size=opts.ngpus_per_node,
                            rank=opts.rank)

    torch.distributed.barrier()
    setup_for_distributed(opts.rank == 0)
    print('opts :',opts)


def main(rank, args):
    if args.parallel == 1:
        init_distributed_training(rank, args)
    
    
    logger = logging.getLogger('main')
    logger.debug(f'args: {vars(args)}')
    
    logger.debug(f'init data preprocessing')
    
    data_prep = DataPreProcessor(args.parallel, args.augmentation_mode)
    
    data_prep.transform_data()
    data_prep.split_data(args.split_ratio, args.augment_split)
    data_prep.get_data_loader(args.batch_size, args.num_worker)
    
    logger.debug(f'init model')
    if args.model is ModelEnum.custom:
        # Custom 모델 추가
        model_pre = ModelPreProcessor(args, None)
    else:
        model_pre = ModelPreProcessor(args)
    
    logger.debug(f'init trainer')
    trainer = Trainer(args, model_pre, data_prep)
    
    if args.test == 1:
        logger.debug(f'test start')
        trainer.model_save_path = os.path.join('.', 'model_dir', args.model_path)
        trainer.get_result()
    else:
        logger.debug(f'train start')
        trainer.train()
        trainer.get_result()
        trainer.save_history()

    logger.debug(f'finish process')

if __name__ == "__main__":
    args = get_arg_parse()
    init_logger(args)
    fix_random(args)
    
    if args.parallel == 1:
        args.ngpus_per_node = torch.cuda.device_count()
        args.gpu_ids = list(range(args.ngpus_per_node))
        args.num_worker = args.ngpus_per_node * 4
        args.batch_size = int(args.batch_size / args.ngpus_per_node)

        torch.multiprocessing.spawn(main,
                args=(args,),
                nprocs=args.ngpus_per_node,
                join=True)
    else:
        main(0, args)
