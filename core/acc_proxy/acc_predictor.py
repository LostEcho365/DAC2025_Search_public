from functools import partial
from tabnanny import verbose
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from pyutils.general import logger
from torch import nn, Tensor
import random
import math

# from core.acc_proxy.gradnorm_score import compute_gradnorm_score
# from core.acc_proxy.params_score import compute_params_score
# from core.acc_proxy.robust_score import compute_exp_error_score
# from core.acc_proxy.sparsity_score import compute_sparsity_score
# from core.acc_proxy.zen_score import compute_zen_score
# from core.acc_proxy.zico_score import compute_zico_score

__all__ = ["AccuracyPredictor", "RobustnessPredictor"]

class AccuracyPredictor(nn.Module):

    _alg_list = {
        # "gradnorm": compute_gradnorm_score,
        # "zen": compute_zen_score,
        # "zico": compute_zico_score,
        # "params": compute_params_score,
        # "sparsity": compute_sparsity_score,
    }

    def __init__(self, model: nn.Module, alg_cfg: dict,SRAM_TEMP: int,RERAM_TEMP: int,memory: str) -> None:
        super().__init__()
        self.set_alg(alg_cfg)
        self.model = model
        self.STEMP=SRAM_TEMP
        self.RTEMP=RERAM_TEMP
        self.write_V=2#write_volts
        self.reram_gmax=1/6e3
        self.cim_f=1/(1.3e-8)
        self.memory=memory

    def reram_accuracy(self):
        q=1.6e-19
        kb=3.21e-21
        if self.memory=="reram":
            weight_flat=self.model.weight.cpu().data.numpy().reshape(-1)
            sigma_t=math.sqrt(4*self.reram_gmax*self.cim_f*kb*self.RTEMP)/self.write_V
            sigma_s=math.sqrt(2*self.reram_gmax*self.cim_f*q/self.write_V)
            noise_thermal=[np.random.normal(loc=0.0,scale=math.sqrt(4*max(abs(self.reram_gmax*i),self.reram_gmax/17)*self.cim_f*kb*self.RTEMP)/self.write_V) for i in weight_flat]
            noise_shot=[np.random.normal(loc=0.0,scale=math.sqrt(2*max(abs(self.reram_gmax*i),self.reram_gmax/17)*self.cim_f*q/self.write_V)) for i in weight_flat]
            total_error=math.sqrt(np.sum([i**2 for i in (noise_thermal+noise_shot)]))/self.reram_gmax
            return total_error
        else:
            return 0

    def set_alg(self, alg_cfg: dict):
        # alg_cfg: {
        #   "alg_name1": {"weight": 1, "config": kwargs}
        #   "alg_name2": {"weight": 1, "config": kwargs}
        # }

        assert all(alg in self._alg_list for alg in alg_cfg), logger.error(
            f"Only support accuracy proxy from {self._alg_list}, but got {alg_cfg}."
        )
        self.alg_cfg = alg_cfg
        self.acc_proxy_list = {
            alg_name: partial(self._alg_list[alg_name], **alg_cfg[alg_name]["config"])
            for alg_name in alg_cfg
        }

    def forward(self, arch_sol: Tuple | str) -> float:
        """evaluate accuracy proxy based on the arch_sol

        Args:
            arch_sol (Tuple | str): architecture solution

        Returns:
            float: acc proxy score
        """
        self.model.fix_arch_solution(arch_sol, verbose=False)
        score = 0
        for alg_name in self.acc_proxy_list:
            score += self.alg_cfg[alg_name]["weight"] * self.acc_proxy_list[alg_name](model=self.model)
        return score


class RobustnessPredictor(nn.Module):
    _alg_list = {"compute_exp_error_score": compute_exp_error_score}

    def __init__(
        self,
        model: nn.Module,
        alg: str = "compute_exp_error_score",
        num_samples: int = 10,
        phase_noise_std: float = 0.02,
        sigma_noise_std: float = 0.02,
        dc_noise_std: float = 0.02,
        cr_tr_noise_std: float = 0.02,
        cr_phase_noise_std: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__()
        self.alg = alg
        assert alg in self._alg_list, logger.error(
            f"Only support robustness proxy from {self._alg_list}, but got {alg}."
        )
        self.model = model
        self.robustness_proxy = partial(self._alg_list[alg], **kwargs)
        self.num_samples = num_samples
        self.super_ps_layers = self.model.super_layer.build_ps_layers(num_samples, 1)
        for i in range(len(self.super_ps_layers)):
            self.super_ps_layers[i].reset_parameters(alg="uniform")
        self.sigma = torch.randn(
            num_samples,
            1,
            self.model.super_layer.n_waveguides,
            dtype=torch.cfloat,
            device=model.device,
        )
        self.phase_noise_std = phase_noise_std
        self.sigma_noise_std = sigma_noise_std
        self.dc_noise_std = dc_noise_std
        self.cr_tr_noise_std = cr_tr_noise_std
        self.cr_phase_noise_std = cr_phase_noise_std

    def set_alg(self, alg: str, **kwargs):
        assert alg in self._alg_list, logger.error(
            f"Only support robustness proxy from {self._alg_list}, but got {alg}."
        )
        self.alg = alg
        self.robustness_proxy = partial(self._alg_list[alg], **kwargs)

    def forward(self, arch_sol: Tuple | str) -> float:
        """evaluate robustness proxy based on the arch_sol

        Args:
            arch_sol (Tuple | str): architecture solution

        Returns:
            float: acc proxy score
        """
        self.model.fix_arch_solution(arch_sol, verbose=False)
        score = self.robustness_proxy(
            self.model.super_layer,
            self.super_ps_layers,
            self.sigma,
            self.num_samples,
            # self.sigma_noise_std,
            self.phase_noise_std,
            self.dc_noise_std,
            self.cr_tr_noise_std,
            self.cr_phase_noise_std,
        )
        return score
