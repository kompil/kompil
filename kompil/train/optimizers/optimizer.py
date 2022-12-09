import torch

from adabelief_pytorch import AdaBelief

from kompil.utils.factory import Factory

ADAM_OPT = "adam"
ADAMO_OPT = "adamo"  # Adam 'optimized' with custom parameters
ADAMS_OPT = "adams"  # Alias Adam + AMSGrad
ADAB_OPT = "adab"  # Ada belief
ADAMW_OPT = "adamw"  # Adam with weight regularization
ADAMSW_OPT = "adamsw"  # Alias AdamW(eight) + AMSGrad
SGD_OPT = "sgd"  # Vanilla Stochastic Gradient Descent
SGDM_OPT = "sgdm"  # SGD with mementum
NAG_OPT = "nag"  # Nesterov Accelerate Gradient

__FACTORY = Factory("optimizer")


def factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register_optimizer(name: str):
    return factory().register(name)


def get_optimizer(name: str, parameters, params):
    opt_lambda, opt_default_params = factory()[name]()
    opt_default_params.update(params)
    optimizer = opt_lambda(parameters, **opt_default_params)
    return optimizer


@register_optimizer(ADAMO_OPT)
def adamo_optimizer():
    # These empirically-chosen parameters will decrease historical mementum weight
    # It makes Adam more "curious" and inclined to explore a wider search space
    # Original values : (0.9, 0.999)
    return torch.optim.Adam, {
        "eps": 1e-8,
        "amsgrad": False,  # If True => Very bad results for overfitting
        "weight_decay": 0,  # If not 0 => Apply weight normalization, don't want
        "betas": (0.899, 0.99),
    }


@register_optimizer(ADAM_OPT)
def adam_optimizer():
    # https://arxiv.org/abs/1412.6980
    return torch.optim.Adam, {"eps": 1e-8, "amsgrad": False, "weight_decay": 0}


@register_optimizer(ADAMS_OPT)
def adams_optimizer():
    # http://www.satyenkale.com/papers/amsgrad.pdf
    return torch.optim.Adam, {"eps": 1e-8, "amsgrad": True, "weight_decay": 0}


@register_optimizer(ADAB_OPT)
def adabelief_optimizer():
    # https://github.com/juntang-zhuang/Adabelief-Optimizer
    return AdaBelief, {
        "eps": 1e-8,
        "weight_decay": 0,
        "fixed_decay": False,
        "amsgrad": False,
        "weight_decouple": False,
        "rectify": False,
        "print_change_log": False,
        "degenerated_to_sgd": False,
    }


@register_optimizer(ADAMW_OPT)
def adamw_optimizer():
    # https://arxiv.org/abs/1711.05101
    return torch.optim.AdamW, {"eps": 1e-8, "amsgrad": False, "weight_decay": 0}


@register_optimizer(ADAMSW_OPT)
def adamsw_optimizer():
    return torch.optim.AdamW, {"eps": 1e-8, "amsgrad": True, "weight_decay": 0}


@register_optimizer(SGD_OPT)
def sgd_optimizer():
    return torch.optim.SGD, {}


@register_optimizer(SGDM_OPT)
def sgdm_optimizer():
    return torch.optim.SGD, {"momentum": 0.9}


@register_optimizer(NAG_OPT)
def nesterov_accelerate_gradient_optimizer():
    return torch.optim.SGD, {"nesterov": True}
