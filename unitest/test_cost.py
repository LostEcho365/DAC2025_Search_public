import os
import sys

import numpy as np
# np.random.seed(0)
import random
# random.seed(0)

import torch
# import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
from pyutils.plot import plt, set_ms
set_ms()
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.evo_search import AccuracyPredictor, CostPredictor, Evaluator, EvolutionarySearch, Converter, RobustnessPredictor, evo_args, run_es
from core.models.layers.super_mesh import super_layer_name_dict
from core.models import SuperOCNN
from core.cost.ADC import ADC_list
from core.cost.DAC import DAC_list
from core import builder

sys.path.pop(0)

def test_costpredictor():
    # make sure the costpredictor functions properly
    # And figure out the range of each cost by sampling a large number of solutions
    k = 8

    device = torch.device("cuda:0")

    arg = evo_args(
        es_population_size=500,
        es_parent_size=200,
        matrix_size=k,
        es_mutation_size=100,
        es_mutation_rate_dc=0.5,
        es_mutation_rate_cr=0.5,
        es_mutation_rate_block=0.5,
        es_mutation_ops=["op1","op2","op3","op4"],
        es_crossover_size=200,
        es_crossover_cr_split_ratio=0.5,
        es_n_iterations=40,
        # es_score_mode= "area.power.latency.robustness",
        # es_score_mode= "area.power.latency", #objective: compute density(tera_ops(TOPS)/(mm^2)), energy efficiency(TOPS/(Watt)), include fixed power/latency
        es_score_mode = "compute_density.energy_efficiency",
        es_num_procs=1,
        es_constr={"area": [100,20000], "power":[100,20000], "latency":[100,20000], "robustness":[0.005,0.01]}  # robustness as constraint
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
    photodetector_cost = {'sensitivity': 0.5, 'power': 2.8, 'width': 40, 'length': 40, 'latency':10}
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
        dc_port_candidates = [2,3,4,6,8],
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
            spacing = 250, # unit um
            area_upper_bound=arg.es_constr["area"][1],
            area_lower_bound=arg.es_constr["area"][0],
            first_active_block=True,
            resolution = 4, 
            n_group = 4.5 # Group index
        ),
    )

    model = SuperOCNN(
        img_height = 28,
        img_width = 28,
        in_channels=1,
        num_classes=10,
        kernel_list=[16],
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

    train_loader, validation_loader, test_loader = builder.make_dataloader(splits=["train","valid", "test"])
    criterion = builder.make_criterion("ce")

    acc_predictor = AccuracyPredictor(model, alg="gradnorm", resolution=32, batch_size=4, fp16=True)
    acc_predictor.set_alg(alg="zico", calib_dataloader=validation_loader, criterion=criterion, fp16=True, device=device) # Compute zico score

    cost_predictor = CostPredictor(model, cost_name="area.power.latency")

    robustness_predictor = RobustnessPredictor(model, alg="compute_exp_error_score", num_samples=16, phase_noise_std=0.15, sigma_noise_std=0.025,
                                               dc_noise_std=0.015, cr_tr_noise_std=0.02, cr_phase_noise_std=2*np.pi/180) # fix the value of standard deviation for different noises

    evaluator = Evaluator(
        arg, acc_predictor, cost_predictor, robustness_predictor, score_mode=arg.es_score_mode, num_procs=int(arg.es_num_procs)
    )

    es_engine = EvolutionarySearch(
        population_size=arg.es_population_size,
        parent_size=arg.es_parent_size,
        matrix_size=arg.matrix_size,
        mutation_size=arg.es_mutation_size,
        mutation_rate_dc=arg.es_mutation_rate_dc,
        mutation_rate_cr=arg.es_mutation_rate_cr,
        mutation_rate_block=arg.es_mutation_rate_block,
        mutation_ops=arg.es_mutation_ops,
        crossover_size=arg.es_crossover_size,
        crossover_cr_split_ratio=arg.es_crossover_cr_split_ratio,
        super_layer=model.super_layer,
        constraints=arg.es_constr,
        evaluator=evaluator
    )
    es_engine.initialize_population()

    solutions = es_engine.ask()
    genes = es_engine.ask(to_solution=False)
    # for gene in genes:
    #     print(gene, "\n")
    # for solution in solutions:
    #     print(solution, "\n")
    # (
    #     scores,
    #     best_solution_cost_dict,
    #     best_solution_score,
    #     acc_values,
    #     area_values,
    #     power_values,
    #     latency_values,
    # ) = evaluator.evaluate_all(genes, solutions, None, k, arg.es_population_size)
    area_values = []
    power_values = []
    latency_values = []
    for solution in solutions:
        area_solution = cost_predictor._evaluate_area(solution)
        power_solution = cost_predictor._evaluate_power(solution)
        latency_solution = cost_predictor._evaluate_latency(solution)
        area_values.append(area_solution)
        power_values.append(power_solution)
        latency_values.append(latency_solution)
    print("Area lower bound:", min(area_values))
    print("Area upper bound:", max(area_values))
    print("Power lower bound:", min(power_values))
    print("Power upper bound:", max(power_values))
    print("Latency lower bound:", min(latency_values))
    print("Latency upper bound:", max(latency_values))
    x_label = list(range(1,arg.es_population_size + 1))
    # print(x_label)
    
    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 6))
    axs1[0, 0].scatter(x_label, area_values)
    title_line = "Area values of solutions(1000*um2)"
    axs1[0, 0].set_title(title_line)

    axs1[0, 1].scatter(x_label, power_values)
    title_line = "Power values of solutions(mW)"
    axs1[0, 1].set_title(title_line)

    axs1[1, 0].scatter(x_label, latency_values)
    title_line = "Latency values of solutions(um)"
    axs1[1, 0].set_title(title_line)

    # fig1.tight_layout()
    fig1.savefig("figures/Range_cost_values.png", dpi=300)

if __name__ == "__main__":
    test_costpredictor()
    
