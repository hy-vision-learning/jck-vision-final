import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from preprocess.preprocess_model import ModelPreProcessor
from preprocess.preprocess_data import DataPreProcessor

import numpy as np

from custom_cosine import CosineAnnealingWarmUpRestarts
from optimizer.sam import SAM

from enums import (
    OptimizerEnum,
    LRSchedulerEnum,
    MixEnum
)

import mix.mixup as mixup
import mix.cutmix as cutmix

import logging
from utils import get_default_device, class_to_superclass

from datetime import datetime
import os
from tqdm import tqdm
import time
import pickle

import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, 
                 args, 
                 model_pre: ModelPreProcessor, 
                 data_pre: DataPreProcessor):
        self.logger = logging.getLogger('main')
        self.device = get_default_device()
        self.parallel = args.parallel
        self.amp = args.amp
        
        self.scaler = GradScaler()
        self.validation_loss = 1e10
        
        self.epoch = args.epoch
        self.gradient_clip = args.gradient_clip
        self.mix_step = args.mix_step
        self.mix_method = args.mix_method
        
        if args.parallel == 1:
            self.local_gpu_id = args.gpu
            self.model = model_pre.model.cuda(self.local_gpu_id)
            self.model = DistributedDataParallel(module=self.model, device_ids=[self.local_gpu_id])
        else:
            self.local_gpu_id = -1
            self.model = model_pre.model.to(self.device)
        
        self.model = model_pre.init_model_weight(self.model)
        self.logger.debug(summary(self.model, input_size=(1, 3, 32, 32)))

        self.data_pre = data_pre
        self.cut_p = args.cut_p
        
        self.optimizer = self.__init_optimizer(args)
        self.lr_scheduler = self.__init_scheduler(args)
        
        if args.parallel == 1:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=args.label_smoothing
            ).cuda(self.local_gpu_id)
            # self.criterion = self.__stable_cross_entropy_loss
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=args.label_smoothing
            )
        
        if args.model_path != '':
            datetime_now = args.model_path
        else:
            datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join('.', 'model_dir', datetime_now)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.logger.debug(f'save path: {self.model_save_path}')
        
        self.early_stopping = args.early_stopping
        
    def __init_optimizer(self, args):
        if args.optimizer == OptimizerEnum.sgd:
            self.logger.debug(f'optimizer init: sgd\tlr: {args.max_learning_rate}\tweight_decay: {args.weight_decay}')
            return optim.SGD(self.model.parameters(), lr=args.max_learning_rate, momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov == 1)
        
        if args.optimizer == OptimizerEnum.adam:
            self.logger.debug(f'optimizer init: adam\tlr: {args.max_learning_rate}\tweight_decay: {args.weight_decay}')
            return optim.Adam(self.model.parameters(), lr=args.max_learning_rate, weight_decay=args.weight_decay)
        
        if args.optimizer == OptimizerEnum.sam:
            self.logger.debug(f'optimizer init: sam\tlr: {args.max_learning_rate}\tweight_decay: {args.weight_decay}')
            return SAM(self.model.parameters(), base_optimizer=torch.optim.SGD, rho=args.rho, adaptive=args.adaptive,
                       lr=args.max_learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        return None
    
    def __init_scheduler(self, args):
        if args.lr_scheduler == LRSchedulerEnum.none:
            self.logger.debug(f'lr scheduler none')
            return None
        
        if args.lr_scheduler == LRSchedulerEnum.cycle:
            self.logger.debug(f'lr scheduler cycle')
            return optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-5, 
                    max_lr=args.max_learning_rate, step_size_up=len(self.data_pre.trainloader)*10, mode='triangular2')
            
        if args.lr_scheduler == LRSchedulerEnum.one_cycle:
            self.logger.debug(f'lr scheduler one cycle')
            return optim.lr_scheduler.OneCycleLR(self.optimizer, args.max_learning_rate, epochs=self.epoch,
                    steps_per_epoch=len(self.data_pre.trainloader))
            
        if args.lr_scheduler == LRSchedulerEnum.step_lr: 
            self.logger.debug(f'lr scheduler step cycle')
            # return optim.lr_scheduler.StepLR(self.optimizer, step_size=60, gamma=0.4)
            return optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.step_milestone, gamma=0.1)
        
        if args.lr_scheduler == LRSchedulerEnum.cos_annealing:
            self.logger.debug(f'lr scheduler cos_annealing cycle')
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.cos_max, eta_min=args.min_learning_rate)
        
        if args.lr_scheduler == LRSchedulerEnum.custom_annealing:
            self.logger.debug(f'lr scheduler custom cos annealing cycle')
            return CosineAnnealingWarmUpRestarts(self.optimizer, T_0=args.cos_max, T_mult=1, eta_max=args.min_learning_rate,  T_up=10, gamma=1)
        
        if args.lr_scheduler == LRSchedulerEnum.on_plateau:
            self.logger.debug(f'lr scheduler ReduceLROnPlateau')
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, threshold=1e-2,
                                                        patience=5, min_lr=args.min_learning_rate)
        
        # if args.lr_scheduler == LRSchedulerEnum.lambda_lr:
        #     self.logger.debug(f'lr scheduler one cycle')
        #     return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_scheduler_lambda,
        #             last_epoch=-1)
        
    def __stable_cross_entropy_loss(self, output, target, label_smoothing=0.1, epsilon=1e-8):
        num_classes = output.size(-1)
        log_probs = F.log_softmax(output.float(), dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - label_smoothing)

        loss = torch.sum(-true_dist * log_probs, dim=-1).mean()
        return loss + epsilon
        
    def save_best_model(self, type):
        if self.parallel == 1 and self.local_gpu_id != 0: return
        torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f'{type}_bset.pt'))
        self.logger.debug(f'{type} best model save')
        
    def load_model(self, type):
        return torch.load(os.path.join(self.model_save_path, f'{type}_bset.pt'))
        
    def __get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
    def __get_eval(self, data_loader, get_top5=True, get_superclass=True) :
        top1_pred = 0
        top5_pred = 0
        superclass_pred = 0
        running_loss = 0
        n = 0
        with torch.no_grad() :
            self.model.eval()
            for X, y_true in tqdm(data_loader, desc='val'):
                if self.local_gpu_id == -1:
                    X = X.to(self.device)
                    y_true = y_true.to(self.device)
                else:
                    X = X.to(self.local_gpu_id)
                    y_true = y_true.to(self.local_gpu_id)
                
                if self.amp == 1:
                    with autocast(device_type="cuda"):
                        y_hat, y_prob = self.model(X)
                else:
                    y_hat, y_prob = self.model(X)
                _, predicted_labels = torch.max(y_prob,1)
                
                loss = self.criterion(y_hat, y_true)
                running_loss += loss.item() * X.size(0)
                
                n += y_true.size(0)
                top1_pred += (predicted_labels == y_true).sum()
                
                if get_top5:
                    _, top5_pred_label = torch.topk(y_prob, 5, dim=1)
                    y_true_expanded = y_true.unsqueeze(1)
                    top5_pred += (top5_pred_label == y_true_expanded).any(dim=1).sum()

                if get_superclass:
                    y_true_super = torch.tensor([class_to_superclass[y.item()] for y in y_true])
                    pred_super = torch.tensor([class_to_superclass[y.item()] for y in predicted_labels])
                    superclass_pred += (y_true_super == pred_super).sum()
        
        epoch_loss = running_loss / len(data_loader.dataset)
        top1 = top5 = superclass = 0
        top1 = top1_pred.float() / n
        if get_top5:
            top5 = top5_pred.float() / n
        if get_superclass:
            superclass = superclass_pred.float() / n
        return top1, top5, superclass, epoch_loss
    
    
    def __forward(self, idx, criterion, model, X, y_true, device):
        r = np.random.rand(1)
        if self.mix_method == MixEnum.mixup and (self.mix_step == 0 or (idx + 1) % self.mix_step == 0):
            imgs, labels_a, labels_b, lambda_ = mixup.mixup(X, y_true, device)
            if self.amp == 1:
                with autocast(device_type="cuda"):
                    y_hat, _  = model(imgs)
                    loss = mixup.mixup_criterion(criterion, pred=y_hat, y_a=labels_a, y_b=labels_b, lam=lambda_)
            else:
                y_hat, _  = model(imgs)
                loss = mixup.mixup_criterion(criterion, pred=y_hat, y_a=labels_a, y_b=labels_b, lam=lambda_)
        elif self.mix_method == MixEnum.cutmix and r < self.cut_p:
            imgs, labels_a, labels_b, lambda_ = cutmix.cutmix(X, y_true, device)
            if self.amp == 1:
                with autocast(device_type="cuda"):
                    y_hat, _ = model(imgs)
                    loss = cutmix.cutmix_criterion(criterion, pred=y_hat, y_a=labels_a, y_b=labels_b, lam=lambda_)
            else:
                y_hat, _ = model(imgs)
                loss = cutmix.cutmix_criterion(criterion, pred=y_hat, y_a=labels_a, y_b=labels_b, lam=lambda_)
        else:                    
            if self.amp == 1:
                with autocast(device_type="cuda"):
                    y_hat, _  = model(X)
                    loss = criterion(y_hat, y_true)
            else:
                y_hat, _  = model(X)
                loss = criterion(y_hat, y_true)
        return loss, model
    
    def __feed(self, train_loader, model, criterion, optimizer, device):
        model.train()
        running_loss = 0
        update_time = isinstance(self.lr_scheduler, optim.lr_scheduler.OneCycleLR) or isinstance(self.lr_scheduler, optim.lr_scheduler.CyclicLR)
        
        for idx, (X, y_true) in tqdm(enumerate(train_loader), desc='train'):
            if self.local_gpu_id == -1:
                X = X.to(self.device)
                y_true = y_true.to(self.device)
            else:
                X = X.to(self.local_gpu_id)
                y_true = y_true.to(self.local_gpu_id)
            
            optimizer.zero_grad()
            
            if self.amp == 1:
                loss, model = self.__forward(idx, criterion, model, X, y_true, device)
                self.scaler.scale(loss).backward()
            else:
                loss, model = self.__forward(idx, criterion, model, X, y_true, device)
                loss.backward()
            
            if isinstance(optimizer, SAM):
                optimizer.first_step(zero_grad=True)
                
                if self.amp == 1:
                    loss_second, model = self.__forward(idx, criterion, model, X, y_true, device)
                    self.scaler.scale(loss_second).backward()
                    self.scaler.step(optimizer.base_optimizer) 
                    self.scaler.update()
                else:
                    loss_second, model = self.__forward(idx, criterion, model, X, y_true, device)
                    loss_second.backward()

                if self.gradient_clip != -1:
                    nn.utils.clip_grad_value_(self.model.parameters(), self.gradient_clip)

                optimizer.second_step(zero_grad=True)
            else:
                if self.gradient_clip != -1:
                    nn.utils.clip_grad_value_(self.model.parameters(), self.gradient_clip)   
                    
                if self.amp == 1:
                    self.scaler.step(optimizer) 
                    self.scaler.update() 
                else:
                    optimizer.step()
            
            running_loss += loss.item() * X.size(0)
            
            if update_time:
                self.lrs.append(self.__get_lr())
                self.lr_scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        self.validation_loss = epoch_loss
        
        lr = self.__get_lr()
        if not update_time:
            self.lrs.append(lr)
            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(epoch_loss)
            else:
                self.lr_scheduler.step()
            
        self.logger.debug(f'lr: {lr}')
        
        return model , optimizer, epoch_loss
    
    def train(self):
        best_loss = 1e10
        self.validation_loss = 1e10
        best_acc = 0
        
        count_stop_epoch = 0
        
        self.train_losses = []
        self.valid_losses = []
        self.train_acc_list = []
        self.valid_top1_list = []
        self.valid_top5_list = []
        self.valid_superclass_list = []
        self.lrs = []
        
        torch.cuda.empty_cache()
        start_time = time.time()
        
        trainloader = self.data_pre.trainloader
        validationloader = self.data_pre.validationloader

        for epoch in range(0, self.epoch):

            self.model, self.optimizer, train_loss = self.__feed(trainloader, self.model, self.criterion, self.optimizer, self.device)
            self.train_losses.append(train_loss)

            train_top1, _, _, _ = self.__get_eval(trainloader, get_top5=False, get_superclass=False)
            val_top1, val_top5, val_superclass, valid_loss = self.__get_eval(validationloader, get_top5=False, get_superclass=False)
            self.valid_losses.append(valid_loss)

            self.train_acc_list.append(train_top1)
            self.valid_top1_list.append(val_top1)
            self.valid_top5_list.append(val_top5)
            self.valid_superclass_list.append(val_superclass)

            self.logger.debug(f'Epoch: {epoch}\t'
                f'Train loss: {train_loss:.4f}\t'
                f'Valid loss: {valid_loss:.4f}\t'
                f'Train accuracy: {100 * train_top1:.2f}\t'
                f'Valid top1 accuracy: {100 * val_top1:.2f}\t'
                #   f'Valid top5 accuracy: {100 * val_top5:.2f}\t'
                #   f'Valid superclass accuracy: {100 * val_superclass:.2f}'
            )
            
            if best_acc < val_top1:
                best_acc = val_top1
                self.save_best_model('acc')
                self.logger.debug(f'best acc model: epoch {epoch}')
            if best_loss > valid_loss:
                best_loss = valid_loss
                count_stop_epoch = 0
                self.save_best_model('loss')
                self.logger.debug(f'best loss model: epoch {epoch}')
            else:
                count_stop_epoch += 1
                if self.early_stopping != -1:
                    self.logger.debug(f'{self.early_stopping - count_stop_epoch} left')
                    if count_stop_epoch == self.early_stopping:
                        self.logger.debug(f"Early Stopping")
                        break

        end_time = time.time()
        time_diff = end_time - start_time
        self.save_best_model('last')
        self.logger.debug(f'training time: {time_diff // 3600}h {time_diff % 3600 // 60}m {time_diff % 3600 % 60}')
        
    def get_result(self):
        best_sum = 0
        self.best_top1 = self.best_top5 = self.best_super = 0
        best_idx = ''
        
        for i in ['last', 'acc', 'loss']:
            try:
                self.model.load_state_dict(self.load_model(i))
            except:
                self.logger.debug(f'{i} model can not load')
                continue
            test_top1, test_top5, test_superclass, test_loss = self.__get_eval(self.data_pre.testloader)
            self.logger.debug(i + ': ' + str(test_top1.item()) + " " + str(test_top5.item()) + " " + str(test_superclass.item()))
            if test_top1 + test_top5 + test_superclass > best_sum:
                best_sum = test_top1 + test_top5 + test_superclass
                self.best_top1 = self.best_top5 = self.best_super = test_top1, test_top5, test_superclass
                best_idx = i
        self.logger.debug(f'{best_idx} is best model\t {self.best_top1} {self.best_top5} {self.best_super}')
        
    def save_history(self):
        info = {
            'train_acc_hist': self.train_acc_list,
            'valid_top1_hist': self.valid_top1_list,
            'valid_top5_hist': self.valid_top5_list,
            'valid_superclass_list': self.valid_superclass_list,
            'train_losses_hist': self.train_losses,
            'valid_losses_hist': self.valid_losses,
            'test_top1': self.best_top1,
            'test_top5': self.best_top5,
            'test_superclass': self.best_super,
        }

        with open(os.path.join(self.model_save_path, f'train_hist.pt'), 'wb') as f:
            pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)
            
        epoch_x = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epoch_x, self.train_losses, label='Train Loss')
        plt.plot(epoch_x, self.valid_losses, label='Validation Loss')

        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.model_save_path, f'loss.png'))
        
        epoch_x = range(1, len(self.train_acc_list) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epoch_x, [i.cpu() for i in self.train_acc_list], label='Train Acc')
        plt.plot(epoch_x, [i.cpu() for i in self.valid_top1_list], label='Validation Acc')

        plt.title('Training and Validation Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.savefig(os.path.join(self.model_save_path, f'acc.png'))
        
        lr_x = range(1, len(self.lrs) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(lr_x, self.lrs)

        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('LR')
        plt.legend()
        plt.savefig(os.path.join(self.model_save_path, f'lr.png'))
        
        self.logger.debug('histroy saved')
    
