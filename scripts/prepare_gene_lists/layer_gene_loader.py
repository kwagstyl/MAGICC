import numpy as np
import pandas as pd
from gene_mapping import get_indices

def load_layer_genes(paper='Maynard'):
    """load layer genes from paper of interest
    current options are Maynard,He and Zeng"""
    if paper=='He':
        gene_layers = load_he()
    elif paper =='Maynard':
        gene_layers = load_maynard()
    elif paper =='Zeng':
        gene_layers = load_zeng()
    else:
        print('paper not recognised')
    return gene_layers

def load_he():
    """load genes from he 2017, Superior frontal gyrus (Areas 9 or 10)"""
    gene_layers=np.zeros((6,20781),dtype=bool)
    layer_tables=pd.read_excel('/data1/bigbrain/phate_testing/von_economo_neurons/layer_markers.xlsx')
    for layer in np.arange(6):
        marker_list=np.array(layer_tables['Gene symbol'][layer_tables['Layer marker in human']=='L{}'.format(layer+1)])
        ensemble_ids = np.array(layer_tables['EnsemblID'][layer_tables['Layer marker in human']=='L{}'.format(layer+1)])
        ens=np.zeros(len(ensemble_ids),dtype=object)
        for ie,e in enumerate(ensemble_ids):
            ens[ie] = e.split('.')[0]
        gene_inds,bool_i=get_indices(marker_list,gene_ensembl=ens)
        gene_layers[layer,gene_inds] = True
    return gene_layers

def load_maynard():
    """load genes from maynard 2020, dlpfc"""
    gene_layers=np.zeros((6,20781),dtype=bool)
    sheet = pd.read_excel('/data1/bigbrain/phate_testing/von_economo_neurons/media-1.xlsx',sheet_name='Table S4B')
    for layer in np.arange(6):
        l_mask = np.logical_and(sheet['fdr_Layer{}'.format(layer+1)]<0.05,
                                         sheet['t_stat_Layer{}'.format(layer+1)]>0.0)
        gene_inds,_=get_indices(sheet['gene'][l_mask].values,
                          gene_ensembl=sheet['ensembl'][l_mask].values)
        gene_layers[layer,gene_inds] = True
    return gene_layers

def load_zeng():
    """load layer genes form zeng, visual and temporal, zeng 2012"""
    zeng_table=pd.read_excel('/data1/bigbrain/phate_testing/von_economo_neurons/zeng.xlsx')
    gene_layers=np.zeros((6,20781),dtype=bool)
    for layer in np.arange(6):
        genes_for_this_layer=[]
        for b,gene in enumerate(zeng_table['Gene symbol']):
            if str(layer+1) in str(zeng_table['Cortical marker (human)'][b]):
                genes_for_this_layer.append(gene)
        gene_inds,bool_i = get_indices(genes_for_this_layer)
        gene_layers[layer,gene_inds] = True
    return gene_layers



layer_genes=[]
papers=['Maynard','He']

for roi in np.arange(2): 
    layer_genes.append(load_layer_genes(paper=papers[roi]))
combined_layer_genes = layer_genes[0] + layer_genes[1]

gene_info_fn='/data1/bigbrain/phate_testing/all_gene_lists.csv'
genes_dataframe  = pd.read_csv(gene_info_fn)
for l in np.arange(6):
    genes_dataframe[f'Layer {l+1}'] = combined_layer_genes[l]
    
genes_dataframe.to_csv(gene_info_fn,index=False)

