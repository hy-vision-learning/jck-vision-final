from enum import Enum


class ModelEnum(Enum):
    custom = 'custom'
    
    resnet18 = 'resnet18'
    resnet34 = 'resnet34'
    resnet50 = 'resnet50'
    resnet101 = 'resnet101'
    resnet152 = 'resnet152'
    
    resnext50 = 'resnext50'
    wide_resnet_16_4 = 'wide_resnet_16_4'
    wide_resnet_16_10 = 'wide_resnet_16_10'
    wide_resnet_28_10_03 = 'wide_resnet_28_10_03'
    
    densenet121 = 'densenet121'
    densenet169 = 'densenet169'
    densenet201 = 'densenet201'
    densenet161 = 'densenet161'
    
    pyramidnet100_84 = 'pyramidnet100_84'
    pyramidnet200_240 = 'pyramidnet200_240'
    pyramidnet236_220 = 'pyramidnet236_220'
    pyramidnet272_200 = 'pyramidnet272_200'
    pyramidnet_custom = 'pyramidnet_custom'

    def __str__(self):
        return self.value
    
    
class OptimizerEnum(Enum):
    sgd = 'sgd'
    adam = 'adam'
    sam = 'sam'

    def __str__(self):
        return self.value
    
    
class LRSchedulerEnum(Enum):
    none = 'none'
    lambda_lr = 'lambda_lr'
    step_lr = 'step_lr'
    cos_annealing = 'cos_annealing'
    custom_annealing = 'custom_annealing'
    one_cycle = 'one_cycle'
    cycle = 'cycle'
    on_plateau = 'on_plateau'

    def __str__(self):
        return self.value
    
    
class MixEnum(Enum):
    none = 'none'
    mixup = 'mixup'
    cutmix = 'cutmix'

    def __str__(self):
        return self.value
    