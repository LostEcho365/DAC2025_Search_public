'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-08-13 19:43:30
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-08-13 21:05:40
'''
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.evo_search import AccuracyPredictor, CostPredictor, Evaluator, EvolutionarySearch, Converter, args, run_es
from core.models.layers.super_mesh import super_layer_name_dict
from core.models import SuperOCNN
sys.path.pop(0)

def test_es():
    k = 8
    device = torch.device("cpu")
    # super_layer_config
    # arch = args.super_layer

    arch = dict(
        n_waveguides=k,
        n_front_share_waveguides=k,
        n_front_share_ops=k,
        n_blocks=4,
        n_layers_per_block=2,
        n_front_share_blocks=2,
        share_ps="row_col",
        interleave_dc=True,
        device_cost=dict(
            ps_width = 85, # width of each PS (unit um)
            ps_height = 80, # height of each PS (unit um)
            dc_width = 50, # width of each DC (unit um)
            dc_height = 30, # height of each DC (unit um)
            cr_width = 8, # width of each CR (unit um)
            cr_height = 8, # height of each CR (unit um)
            spacing = 50, # unit um
            area_upper_bound=300,
            area_lower_bound=200,
            first_active_block=True,
            ps_power = 50, # power of phase shifter, unit mW 
            pd_sensitivity = 1, ##
            resolution = 6, 
            ps_IL = 0.1, # insertion loss of ps, unit dB
            dc_IL = 0.1, # insertion loss of dc, unit dB
            cr_IL = 0.01, # insertion loss of cr, unit dB
            n_group = 4.5 # Group index
        ),
    )

    model = SuperOCNN(
        8,
        8,
        in_channels=3,
        num_classes=2,
        kernel_list=[3],
        kernel_size_list=[3],
        stride_list=[1],
        padding_list=[1],
        hidden_list=[],
        block_list=[8, 8],
        photodetect=True,
        super_layer_name="ps_dc_cr_adeptzero",
        super_layer_config=arch,
        device=device,
    ).to(device)

    # print(dir(model))
    # exit(0)
    super_layer = super_layer_name_dict["ps_dc_cr_adeptzero"](arch, device=device)
    # print(super_layer.arch)
    # print(super_layer)
    # print(type(super_layer))
    # exit(0)
    converter = Converter(super_layer)

    acc_predictor = AccuracyPredictor(model, alg="gradnorm", resolution=32, batch_size=4, fp16=True, device=device)

    cost_predictor = CostPredictor(model, cost_name="area")
    evaluator = Evaluator(
        acc_predictor, cost_predictor, score_mode="area", num_procs=1
    )

    model1 = EvolutionarySearch(
                               population_size=10,
                               parent_size=4,
                               matrix_size=8,
                               mutation_size=2,
                               mutation_rate_dc=1,
                               mutation_rate_cr=1,
                               mutation_rate_block=1,
                               crossover_size=4,
                               crossover_cr_split_ratio=0.5,
                               super_layer=super_layer,
                               constraints=dict(),
                               evaluator=evaluator)
        
    # population = model1.initialize_population()
    # individual1 = population[0]
    # individual2 = population[1]
    # individual3 = population[2]
    #print(individual1)
    
    #test DC mutation
    
    # print("Initial Gene before DC mutation:\n", individual1, "\n")
    # individual1 = model1.mutate(individual1)
    # print("Gene after DC Mutation:\n", individual1, "\n")
    # exit(0)
    
    # #test CR mutation
    
    # print("Initial Gene before CR mutation:\n", individual2, "\n")
    # individual2 = model2.mutate(individual2)
    # print("Gene after CR Mutation:\n", individual2, "\n")
    
    # #test Block Mutation
    
    # print("Initial Gene before block mutation:\n", individual3, "\n")
    # individual3 = model3.mutate(individual3)
    # print("Gene after block Mutation:\n", individual3, "\n")

    # #test Crossover
    # print("Initial Genes before Crossover:\n", individual1, "\n", individual2, "\n")
    # Crossovered_dc1, Crossovered_dc2 = model1.crossover(individual1, individual2)
    # print("Genes after Crossover:\n", Crossovered_dc1, "\n", Crossovered_dc2)
    # exit(0)
    # # TODO: ask
    # # TODO: tell
    # # scores = evaluator.evaluate_all(...)
    # # es_engine.tell(scores)

    model1.initialize_population()
    # # test ask
    genes = model1.ask(to_solution=False)
    # print(type(genes))
    # for gene in genes:
    #     print(gene, "\n")

    solutions = model1.ask(to_solution=True)
    print("Before selection: \n")
    for gene in genes:
        print(gene, "\n")
    # arch_sol = solutions[0]
    # print(arch_sol)
    # print(type(arch_sol))
    # arch_sol = eval(arch_sol)
    # print(arch_sol[0])
    # print(arch_sol[0][0])
    # print(type(arch_sol[0][0]))
    # print(arch_sol[1][1])
    # print(type(arch_sol[1][0]))
    # exit(0)
    # print(solutions)
    # print(type(solutions))
    # for solution in solutions:
        # print(solution, "\n")
    # area = []
    # for arch_sol in solutions:
    #     area.append(cost_predictor._evaluate_area(arch_sol))
    # print(area)
    # exit(0)
    # power = []
    # for arch_sol in solutions:
    #     power.append(cost_predictor._evaluate_power(arch_sol))
    # print(power)
    # exit(0)
    # latency = []
    # for arch_sol in solutions:
    #     latency.append(cost_predictor._evaluate_latency(arch_sol))
    # print(latency)
    # exit(0)

    #test tell
    scores, best_solution_cost_dict, best_solution_score = evaluator.evaluate_all(genes,solutions)
    print(scores)
    print(best_solution_cost_dict)
    print(best_solution_score)
    model1.tell(scores)
    print("After selection: \n")
    genes1 = model1.ask(to_solution=False)
    # print(genes1)
    # print(type(genes1[0]))
    # print(type(genes1[9]))
    # exit(0)
    for gene1 in genes1:
        print(gene1, "\n")
    solutions1 = model1.ask(to_solution=True)
    scores1, best_solution_cost_dict1, best_solution_score1 = evaluator.evaluate_all(genes1,solutions1)
    print(scores1)
    print(best_solution_cost_dict1)
    print(best_solution_score1)
    

    # run_es
    

    k = 4
    device = torch.device("cpu")
    # super_layer_config
    # arch = args.super_layer

    arch = dict(
        n_waveguides=k,
        n_front_share_waveguides=k,
        n_front_share_ops=k,
        n_blocks=4,
        n_layers_per_block=2,
        n_front_share_blocks=2,
        share_ps="row_col",
        interleave_dc=True,
        device_cost=dict(
            # ps_weight=6.8,
            # dc_weight=1.5,
            # cr_weight=0.064,
            ps_width = 2.0, # width of each PS
            ps_height = 2.0, # height of each PS
            ps_spacing = 3.0, # spacing between two PS(center to center)  
            ps_dc_spacing = 0.05, # spacing between PS array and DC array(boundary)
            dc_width = 2.0, # width of each DC
            dc_height = 2.0, # height of each DC
            dc_spacing = 3.0, # spacing between two DC(center to center)
            dc_cr_spacing = 0.05, # spacing between DC array and CR(boundary)
            cr_width = 2.0, # width of each CR
            cr_spacing_vertical = 0.05, # vertical spacing between two CR
            cr_spacing_horizontal = 0.05, # horizontal spacing between two CR
            cr_height = 2.0, # height of each CR
            cr_ps_spacing = 0.05, # spacing between CR and DC array in next block(boundary)
            area_upper_bound=120,
            area_lower_bound=70,
            first_active_block=True,
            ps_power = 50, # power of phase shifter, unit mW 
            pd_sensitivity = 1, ## Not sure about the unit!!!
            resolution = 6, 
            ps_IL = 0.1, # insertion loss of ps, unit dB
            dc_IL = 0.1, # insertion loss of dc, unit dB
            cr_IL = 0.01, # insertion loss of cr, unit dB
            n_group = 4.5 # Group index
        ),
    )
    model = SuperOCNN(
        8,
        8,
        in_channels=3,
        num_classes=2,
        kernel_list=[3],
        kernel_size_list=[3],
        stride_list=[1],
        padding_list=[1],
        hidden_list=[],
        block_list=[4, 4],
        photodetect=True,
        super_layer_name="ps_dc_cr_adeptzero",
        super_layer_config=arch,
        device=device,
    ).to(device)

    args = 1

if __name__ == "__main__":
    test_es()
    