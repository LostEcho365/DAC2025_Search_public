import os
import sys

import numpy as np
import torch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.optimizer.base import generate_Butterfly_gene, generate_MZI_gene, generate_kxk_MMI_gene
import yaml

def save_gene_to_yaml(gene_function, k, n_blocks, file_path):
    gene = gene_function(k, n_blocks)
    converted_gene = [gene[0]] + [[sub_item.tolist() for sub_item in item] for item in gene[1:]]
    gene_str = str(converted_gene)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        yaml.dump(gene_str, file)

def main():
    k = 8
    n_blocks = 32
    folder_path = '/home/zjian124/Desktop/ADEPT_Zero/configs/mnist/genes/'
    save_gene_to_yaml(generate_Butterfly_gene, k, n_blocks, folder_path + 'Butterfly_gene.yaml')
    # save_gene_to_yaml(generate_MZI_gene, k, folder_path + 'MZI_gene.yaml')
    # save_gene_to_yaml(generate_kxk_MMI_gene, k, folder_path + 'kxk_MMI_gene.yaml')

if __name__ == '__main__':
    main()
