import torch
import torch.nn as nn

from spring.dirichlet.modules import QConv2dBias, QLinearBias
from spring.dirichlet.logger import logger


############################################################################
# Pattern and target module for merge conv + relu
class ConvReLUMatcher(nn.Module):
    """Matcher for torch.nn.Conv2d + nn.ReLU"""
    dummy_inputs = (torch.randn(1, 1, 2, 2),)

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out


class QConv2dBiasReLU(QConv2dBias):
    def __init__(self, conv_module, relu_module, **kwargs):
        super().__init__(raw_module=conv_module, **kwargs)
        self.relu = relu_module

    def after_forward(self, outputs):
        assert len(outputs) == 1
        outputs[0] = self.relu(outputs[0])
        super().after_forward(outputs)


def refactor_conv_relu(submodule, submodule_name, matched, attrs, qparams):
    def __join_name(attr):
        if submodule_name:
            return submodule_name + '.' + attr
        return attr
    logger.info(
        '[replace_factory] replace {} with QConv2dBiasReLU.'.format(
            ', '.join(map(__join_name, attrs))
        )
    )
    return Wrapper(QConv2dBiasReLU(matched.conv, matched.relu, **qparams))
############################################################################


############################################################################
# Pattern and target module for merge linear + relu
class LinearReLUMatcher(nn.Module):
    """Matcher for torch.nn.Linear + nn.ReLU"""
    dummy_inputs = (torch.randn(1, 500),)

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(500, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        return out


class QLinearBiasReLU(QLinearBias):
    def __init__(self, fc_module, relu_module, **kwargs):
        super().__init__(raw_module=fc_module, **kwargs)
        self.relu = relu_module

    def after_forward(self, outputs):
        assert len(outputs) == 1
        outputs[0] = self.relu(outputs[0])
        super().after_forward(outputs)


class Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def refactor_linear_relu(submodule, submodule_name, matched, attrs, qparams):
    def __join_name(attr):
        if submodule_name:
            return submodule_name + '.' + attr
        return attr
    logger.info(
        '[replace_factory] replace {} with QLinearBiasReLU.'.format(
            ', '.join(map(__join_name, attrs))
        )
    )
    return Wrapper(QLinearBiasReLU(matched.fc, matched.relu, **qparams))