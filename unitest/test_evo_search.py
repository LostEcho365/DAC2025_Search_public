import os
import sys

import numpy as np
np.random.seed(0)
import random
random.seed(0)

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.evo_search import AccuracyPredictor, CostPredictor, Evaluator, EvolutionarySearch, Converter, evo_args, run_es, NSGA2
from core.models.layers.super_mesh import super_layer_name_dict
from core.models import SuperOCNN
from core.cost.ADC import ADC_list
from core.cost.DAC import DAC_list

sys.path.pop(0)

def test_evo_search():
    k = 8

    device = torch.device("cuda:0")

    arg = evo_args(
        es_population_size=10,
        es_parent_size=4,
        matrix_size=k,
        es_mutation_size=2,
        es_mutation_rate_dc=0.5,
        es_mutation_rate_cr=0.5,
        es_mutation_rate_block=0.5,
        es_mutation_ops=["op1","op2","op3","op4"],
        es_crossover_size=4,
        es_crossover_cr_split_ratio=0.5,
        es_n_iterations=50,
        # es_score_mode= "area.power.latency.robustness",
        # es_score_mode= "area.power.latency", #objective: compute density(tera_ops(TOPS)/(mm^2)), energy efficiency(TOPS/(Watt)), include fixed power/latency
        es_score_mode = "compute_density.energy_efficiency",
        es_num_procs=1,
        es_constr={"area": [100,2000], "power":[100,1000], "latency":[250,350], "robustness":[0.005,0.01]},  # robustness as constraint
        # DC_port_number= 2
    )

    ps_cost = {'width': 85, 'height': 80, 'static_power': 14.8, 'dynamic_power':10, 'insertion_loss': 0.1} 
    dc_cost = {'width': 50, 'length': 30, 'insertion_loss': 0.3}
    dc2_cost = {'width': 50, 'length': 30, 'insertion_loss': 0.3}
    dc3_cost = {'width': 75, 'length': 45, 'insertion_loss': 0.3}
    dc4_cost = {'width': 100, 'length': 60, 'insertion_loss': 0.3}
    dc5_cost = {'width': 125, 'length': 75, 'insertion_loss': 0.3}
    dc6_cost = {'width': 150, 'length': 90, 'insertion_loss': 0.3}
    dc7_cost = {'width': 175, 'length': 105, 'insertion_loss': 0.3}
    dc8_cost = {'width': 200, 'length': 120, 'insertion_loss': 0.3}
    cr_cost = {'width': 8, 'height': 8, 'cr_spacing':10, 'insertion_loss': 0.1}  
    # photodetector_cost = {'sensitivity': -5, 'power': 2.8, 'width': 40, 'length': 40, 'latency':10}
    photodetector_cost = {'sensitivity': -25, 'power': 2.8, 'width': 40, 'length': 40, 'latency':10}
    TIA_cost = {'power': 3,'area': 5200, 'latency':10}
    modulator_cost = {'static_power': 10,'width': 50, 'length': 300,'insertion_loss': 0.8}
    attenuator_cost = {'insertion_loss': 0.1,'length': 7.5,'width': 7.5, 'static_power':2.5, 'dynamic_power':0}



    arch = dict(
        n_waveguides=k,
        n_front_share_waveguides=k,
        n_front_share_ops=k,
        n_blocks=8,
        n_layers_per_block=2,
        n_front_share_blocks=2,
        share_ps="row_col",
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
            attenuator_cost = attenuator_cost,
            adc_cost = ADC_list,
            dac_cost = DAC_list,
            laser_wall_plug_eff = 0.25,
            spacing = 50, # unit um
            area_upper_bound=arg.es_constr["area"][1],
            area_lower_bound=arg.es_constr["area"][0],
            first_active_block=True,
            resolution = 4, 
            n_group = 4.5 # Group index
        ),
        dc_port_candidates=[2],
    )

    model = SuperOCNN(
        8,
        8,
        in_channels=1,
        num_classes=10,
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

    # model = SuperOCNN(
    #     8,
    #     8,
    #     in_channels=1,
    #     num_classes=10,
    #     kernel_list=[16,16],
    #     kernel_size_list=[3,3],
    #     stride_list=[2, 1],
    #     padding_list=[1, 1],
    #     hidden_list=[],
    #     block_list=[8, 8, 8],
    #     photodetect=True,
    #     super_layer_name="ps_dc_cr_adeptzero",
    #     super_layer_config=arch,
    #     device=device,
    # ).to(device)

    best_solution, best_score, best_solution_cost_dict, best_score_values, avg_values_pack, best_values_pack = run_es(args=arg, model=model,verbose=True)
    # print(best_score_values)
    # exit(0)
    best_acc_values, best_area_values, best_power_values, best_latency_values = best_values_pack
    avg_score_values, avg_acc_values, avg_area_values, avg_power_values, avg_latency_values = avg_values_pack
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(list(range(1, arg.es_n_iterations + 1)), best_score_values, label = 'best_score_values', color='red', linestyle='--')
    axs[0,0].plot(list(range(1, arg.es_n_iterations + 1)), avg_score_values, label = 'average_score_values', color='blue', linestyle='-')
    axs[0,0].legend(loc='upper right')
    axs[0,0].set_xlabel("Number of Iterations")
    axs[0,0].set_ylabel("Score Value")

    axs[0,1].plot(list(range(1, arg.es_n_iterations + 1)), best_acc_values, label = 'best_acc_values', color='red', linestyle = "--")
    axs[0,1].plot(list(range(1, arg.es_n_iterations + 1)), avg_acc_values, label = 'average_acc_values', color='blue', linestyle = "-")
    axs[0,1].legend(loc='upper right')
    axs[0,1].set_xlabel("Number of Iterations")
    axs[0,1].set_ylabel("Acc Value")

    axs[1,0].plot(list(range(1, arg.es_n_iterations + 1)), best_area_values, label = 'best_area_values', color='red', linestyle = "--")
    axs[1,0].plot(list(range(1, arg.es_n_iterations + 1)), avg_area_values, label = 'average_area_values', color='blue', linestyle = "-")
    axs[1,0].legend(loc='upper right')
    axs[1,0].set_xlabel("Number of Iterations")
    axs[1,0].set_ylabel("area Value")

    axs[1,1].plot(list(range(1, arg.es_n_iterations + 1)), best_power_values, label = 'best_power_values', color='red', linestyle = "--")
    axs[1,1].plot(list(range(1, arg.es_n_iterations + 1)), avg_power_values, label = 'average_power_values', color='blue', linestyle = "-")
    axs[1,1].legend(loc='upper right')
    axs[1,1].set_xlabel("Number of Iterations")
    axs[1,1].set_ylabel("power Value")

    # axs[1,1].plot(list(range(1, arg.es_n_iterations + 1)), best_latency_values, label = 'best_latency_values', color='red', linestyle = "--")
    # axs[1,1].plot(list(range(1, arg.es_n_iterations + 1)), avg_latency_values, label = 'average_latency_values', color='blue', linestyle = "-")
    # axs[1,1].legend(loc='upper right')
    # axs[1,1].set_xlabel("Number of Iterations")
    # axs[1,1].set_ylabel("latency Value")

    # axs[1, 2].axis('off')
    
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    test_evo_search()