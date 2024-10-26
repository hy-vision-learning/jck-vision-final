import torch.nn as nn

import model.resnet as resnet
import model.resnext as resnext
import model.wide_resnet as wide_resnet
import model.densenet as densenet
import model.pyramidnet as pyramidnet

from enums import ModelEnum
import logging


class ModelPreProcessor:
    def __init__(self, args, custom_model=None):
        self.logger = logging.getLogger('main')
        self.model = self.select_model(args.model, args)
        
        if self.model == None:
            assert custom_model is not None, "custom 모델을 지정하려면 custom_model이 필요합니다."
            self.model = custom_model
            
        # self.init_model_weight()
            
    def select_model(self, m: ModelEnum, args):
        if m == ModelEnum.resnet18:
            self.logger.debug('model init: resnet18')
            return resnet.resnet18(100)
        if m == ModelEnum.resnet34:
            self.logger.debug('model init: resnet34')
            return resnet.resnet34(100)
        if m == ModelEnum.resnet50:
            self.logger.debug('model init: resnet50')
            return resnet.resnet50(100)
        if m == ModelEnum.resnet101:
            self.logger.debug('model init: resnet101')
            return resnet.resnet101(100)
        if m == ModelEnum.resnet152:
            self.logger.debug('model init: resnet152')
            return resnet.resnet152(100)
        if m == ModelEnum.resnext50:
            self.logger.debug('model init: resnext50')
            return resnext.resnext50(100)
        if m == ModelEnum.wide_resnet_16_4:
            self.logger.debug('model init: wide_resnet_16_4')
            return wide_resnet.wide_resnet_16_4(100)
        if m == ModelEnum.wide_resnet_28_10_03:
            self.logger.debug('model init: wide_resnet_28_10_03')
            return wide_resnet.wide_resnet_28_10_03(100)
        if m == ModelEnum.densenet121:
            self.logger.debug('model init: densenet121')
            return densenet.densenet121(100)
        if m == ModelEnum.densenet161:
            self.logger.debug('model init: densenet161')
            return densenet.densenet161(100)
        if m == ModelEnum.densenet169:
            self.logger.debug('model init: densenet169')
            return densenet.densenet169(100)
        if m == ModelEnum.densenet201:
            self.logger.debug('model init: densenet201')
            return densenet.densenet201(100)
        if m == ModelEnum.pyramidnet100_84:
            self.logger.debug('model init: pyramidnet100_84')
            return pyramidnet.pyramidnet100_84(100, shake=args.p_shake)
        if m == ModelEnum.pyramidnet200_240:
            self.logger.debug('model init: pyramidnet200_240')
            return pyramidnet.pyramidnet200_240(100, shake=args.p_shake)
        if m == ModelEnum.pyramidnet236_220:
            self.logger.debug('model init: pyramidnet236_220')
            return pyramidnet.pyramidnet236_220(100, shake=args.p_shake)
        if m == ModelEnum.pyramidnet272_200:
            self.logger.debug('model init: pyramidnet272_200')
            return pyramidnet.pyramidnet272_200(100, shake=args.p_shake)
        if m == ModelEnum.pyramidnet_custom:
            self.logger.debug(f'model init: pyramidnet_custom {args.p_depth} {args.p_alpha}')
            return pyramidnet.pyramidnet_custom(100, args.p_depth, args.p_alpha, shake=args.p_shake)
        self.logger.debug('model custom')
        return None
    
    def init_model_weight(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        return model
