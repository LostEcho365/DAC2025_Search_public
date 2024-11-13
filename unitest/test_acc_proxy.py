"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-08-15 15:19:17
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-08-15 15:39:07
"""
"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-09-28 05:00:55
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-08-15 14:17:38
"""
import os
import sys

import numpy as np
import torch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.datasets.cifar10 import CIFAR10Dataset
from core.datasets.mnist import MNISTDataset
from core.builder import make_criterion, make_dataloader
from core.acc_proxy.gradnorm_score import compute_gradnorm_score
from core.acc_proxy.zen_score import compute_zen_score
from core.acc_proxy.zico_score import compute_zico_score
from core.acc_proxy.expressivity_score import ExpressivityScoreEvaluator, ParallelExpressivityScoreEvaluator
from core.acc_proxy.uniformity_score import compute_uniformity_score_kl, compute_uniformity_score_js
from core.optimizer.base import EvolutionarySearchBase
from core.acc_proxy.acc_predictor import AccuracyPredictor
from core.optimizer.utils import Converter, Evaluator
from core.cost.cost_predictor import CostPredictor
from core.models import SuperOCNN, ResNet20, VGG8, SuperResNet20
from core.models.layers.super_mesh import super_layer_name_dict
from core.optimizer.base import Evaluator, EvolutionarySearchBase, evo_args, generate_Butterfly_gene, generate_MZI_gene, generate_kxk_MMI_gene
from core.cost.ADC import ADC_list
from core.cost.DAC import DAC_list

sys.path.pop(0)

def process_gene(gene, required_length):
    
    dummy_block = [np.array([1,1,1,1,1,1,1,1]), np.array([0,1,2,3,4,5,6,7])]
    required_extention = required_length - len(gene[1:])

    # extend the gene
    for i in range(required_extention):
        gene.append(dummy_block)        
    return gene

def _build_model():
    device = torch.device("cuda:0")

    ps_cost = {'width': 100, 'height': 25, 'static_power': 0, 'dynamic_power':10, 'insertion_loss': 0.25} 
    y_branch_cost = {'width':27.7, 'length':2.4, 'insertion_loss':0.02}
    dc_cost = {'width': 200, 'length': 20, 'insertion_loss': 0.5}
    dc2_cost = {'width': 27.7, 'length': 2.4, 'insertion_loss': 0.33}
    dc3_cost = {'width': 41.55, 'length': 3.6, 'insertion_loss': 0.33}
    dc4_cost = {'width': 55.4, 'length': 4.8, 'insertion_loss': 0.33}
    dc5_cost = {'width': 69.25, 'length': 6, 'insertion_loss': 0.33}
    dc6_cost = {'width': 83.1, 'length': 7.2, 'insertion_loss': 0.33}
    dc7_cost = {'width': 96.95, 'length': 8.4, 'insertion_loss': 0.33}
    dc8_cost = {'width': 110.8, 'length': 9.6, 'insertion_loss': 0.33}
    cr_cost = {'width': 10, 'height': 10, 'cr_spacing':10, 'insertion_loss': 0.1}  
    # photodetector_cost = {'sensitivity': -5, 'power': 2.8, 'width': 40, 'length': 40, 'latency':10}
    photodetector_cost = {'sensitivity': -27, 'power': 6e-5, 'width': 180, 'length': 66, 'latency':10}
    TIA_cost = {'power': 3,'area': 50, 'latency':10}
    modulator_cost = {'static_power': 7e-5,'width': 250, 'length': 25,'insertion_loss': 6.4}

    arch = dict(
        n_waveguides=8,
        n_front_share_waveguides=8,
        n_front_share_ops=8,
        n_blocks=32,
        n_layers_per_block=2,
        n_front_share_blocks=4,
        share_ps="none",
        interleave_dc=True,
        device_cost=dict(
            ps_cost = ps_cost,
            dc_cost = dc_cost,
            dc2_cost = dc2_cost,
            dc3_cost = dc3_cost,
            dc4_cost = dc4_cost,
            dc5_cost = dc5_cost,
            dc6_cost = dc6_cost,
            dc7_cost = dc7_cost,
            dc8_cost = dc8_cost,
            cr_cost = cr_cost,
            photodetector_cost = photodetector_cost,
            TIA_cost = TIA_cost,
            modulator_cost = modulator_cost,
            adc_cost = ADC_list,
            dac_cost = DAC_list,
            laser_wall_plug_eff = 0.25,
            spacing = 250, # unit um
            h_spacing = 50,
            area_upper_bound=1000000000,
            area_lower_bound=1000,
            first_active_block=True,
            resolution = 4, 
            n_group = 4.5 # Group index
        )
    )

    model = SuperOCNN(
        28,
        28,
        in_channels=1,
        num_classes=10,
        kernel_list=[16,16],
        kernel_size_list=[3,3],
        stride_list=[2,1],
        padding_list=[1,1],
        hidden_list=[],
        block_list=[8,8,8],
        in_bit= 32,
        w_bit=32,
        photodetect=True,
        super_layer_name="ps_dc_cr_adeptzero",
        super_layer_config=arch,
        device = device,
    ).to(device)

    super_layer = model.super_layer
    converter = Converter(super_layer)
    # gene = [
    #     4,
    #     [np.array([2, 2, 2, 2]), np.array([0, 1, 2, 3, 4, 5, 6, 7])],
    #     [np.array([2, 2, 2, 2]), np.array([0, 1, 2, 3, 4, 5, 6, 7])],
    #     [np.array([1, 2, 2, 2, 1]), np.array([0, 1, 2, 3, 4, 5, 6, 7])],
    #     [np.array([1, 2, 2, 2, 1]), np.array([0, 1, 2, 3, 4, 5, 6, 7])],
    #     [np.array([2, 2, 2, 2]), np.array([0, 1, 2, 3, 4, 5, 6, 7])],
    #     [np.array([2, 2, 2, 2]), np.array([0, 1, 2, 3, 4, 5, 6, 7])],
    #     [np.array([1, 2, 2, 2, 1]), np.array([0, 1, 2, 3, 4, 5, 6, 7])],
    #     [np.array([1, 2, 2, 2, 1]), np.array([0, 1, 2, 3, 4, 5, 6, 7])]
    # ]
    # gene = generate_Butterfly_gene(k=8, n_blocks=8)
    arch_sol = []
    genes = []
    genes.append(generate_MZI_gene(k=16))
    genes.append(generate_Butterfly_gene(k=16,n_blocks=64))
    genes.append(generate_kxk_MMI_gene(k=16))

    arch_sol = [converter.gene2solution(gene, to_string=False) for gene in genes]
    # model.fix_arch_solution(arch_sol)
    return model, arch_sol


def test_zen_score():
    device = torch.device("cpu")
    model, _ = _build_model()
    zen_score = compute_zen_score(
        model,
        mixup_gamma=0.5,
        resolution=32,
        batch_size=4,
        repeat=2,
        fp16=True,
        device=device,
    )
    print(zen_score)


def test_gradnorm_score():
    device = torch.device("cpu")
    model, _ = _build_model()
    score = compute_gradnorm_score(
        model, resolution=32, batch_size=4, fp16=True, device=device
    )
    print(score)


def test_parallel_expressivity_score():
    device = torch.device("cuda:0")
    model, arch_sols = _build_model()
    checkpoint_path = "./checkpoint/mnist/cnn/train_16_MZI/SuperOCNN__acc-98.77_epoch-90.pt"
    solution_path = "./configs/mnist/genes/MZI_solution_16.txt"
    dataset = "mnist"
    num_samples = 200
    evaluator = ParallelExpressivityScoreEvaluator(checkpoint_path=checkpoint_path, solution_path=solution_path, device=device,
                                                   dataset=dataset, model=model, num_samples=num_samples)
    score = evaluator.compute_expressivity_score(arch_sols=arch_sols, num_samples=num_samples,num_steps=150,verbose=True)
    print(score)


def test_acc_predictor():
    device = torch.device("cpu")
    model, arch_sol = _build_model()
    calib_dataset = CIFAR10Dataset(
        root="./data",
        split="valid",
        train_valid_split_ratio=[0.95, 0.05],
        center_crop=32,
        resize=32,
        digits_of_interest=list(range(10)),
    )

    calib_dataloader = torch.utils.data.DataLoader(
        dataset=calib_dataset,
        batch_size=32,
        shuffle=0,
        pin_memory=True,
        num_workers=2,
    )
    criterion = make_criterion("ce")
    predictor = AccuracyPredictor(
        model, alg="gradnorm", resolution=32, batch_size=4, fp16=True, device=device
    )
    score = predictor(arch_sol)
    print(f"gradnorm = {score}")

    predictor.set_alg(
        alg="zen",
        mixup_gamma=0.5,
        resolution=32,
        batch_size=32,
        repeat=10,
        fp16=True,
        device=device,
    )
    score = predictor(arch_sol)
    print(f"zen = {score}")

    predictor.set_alg(
        alg="zico",
        calib_dataloader=calib_dataloader,
        criterion=criterion,
        fp16=True,
        device=device,
    )
    score = predictor(arch_sol)
    print(f"zico = {score}")

    predictor.set_alg(alg="params")
    score = predictor(arch_sol)
    print(f"params = {score}")


def test_zico_score():
    device = torch.device("cuda:0")
    model, _ = _build_model()
    calib_dataset = MNISTDataset(
        root="./data",
        split="valid",
        train_valid_split_ratio=[0.95, 0.05],
        center_crop=28,
        resize=28,
        resize_mode="bicubic",
        binarize=False,
        binarize_threshold=0.273,
        digits_of_interest=list(range(10)),
        n_test_samples= None,
        n_valid_samples= None,
    )

    calib_dataloader = torch.utils.data.DataLoader(
        dataset=calib_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    criterion = make_criterion("ce")
    score = compute_zico_score(
        model,
        calib_dataloader=calib_dataloader,
        criterion=criterion,
        fp16=False,
        device=device,
    )
    print("Zico-score:", score)

def test_uniformity_score():
    device = torch.device("cuda:0")
    model, _ = _build_model()
    # score = compute_uniformity_score_kl(model=model, device=device)
    # score_ls = []
    # for _ in range(5):
    score = compute_uniformity_score_js(model=model, device=device, num_samples=5)
    #     score_ls.append(score)
    # print(score_ls)
    # score_mean = sum(score_ls)/len(score_ls)
    # print(score_mean)
    print("Uniformity Score: ", score)


def test_num_param():
    device = torch.device("cuda:0")
    model, _ = _build_model()
    for i, layer in enumerate(model.features):
        if i == 0:
            print(layer)
            print(sum(p.numel() for p in layer.conv.parameters()))



if __name__ == "__main__":
    # test()
    # test_zen_score()
    # test_gradnorm_score()
    # test_zico_score()
    # test_acc_predictor()
    # test_expressivity_score()
    test_parallel_expressivity_score()
    # test_uniformity_score()
    # test_num_param()
