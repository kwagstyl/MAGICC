import pandas as pd
import numpy as np
from gene_mapping import get_indices

compartments=pd.read_csv('/data1/bigbrain/phate_testing/compartments/processed_compartments.csv')

compartment_names=list(np.unique(compartments['Compartment'])[1:])
compartment_bool = np.zeros((len(compartment_names),len(compartments)),dtype=bool)
gene_info_fn='/data1/bigbrain/phate_testing/all_gene_lists.csv'
genes_dataframe  = pd.read_csv(gene_info_fn)
for ci,compartment in enumerate(compartment_names):
    genes_dataframe['Compartment '+compartment]=compartments['Compartment']==compartment
    

syngo = np.loadtxt('/data1/bigbrain/phate_testing/syngo_database.txt',dtype=str)

vec = np.zeros(len(genes_dataframe),dtype=bool)
i,m = get_indices(syngo)
vec[i]=True
genes_dataframe['SynGO'] = vec

genes_dataframe.to_csv(gene_info_fn,index=False)
