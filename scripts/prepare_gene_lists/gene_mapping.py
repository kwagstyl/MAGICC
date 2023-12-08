import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib_surface_plotting as msp
import pandas as pd

data_dir= '/data1/bigbrain/phate_testing/weighted_island_vectors'

def get_indices(gene_names,gene_ensembl=None,gene_entrez=None,protein_id=None,
                 filter_mask=None):
    """get indices of genes based on names/aliases, entrez ids or gene_ensembles"""
    indices=[]
    mask = np.ones(len(gene_names),dtype=bool)
    print('matching gene ids and names...')
    aliases = np.array(load_aliases(),dtype=object)
    gene_info = pd.read_csv('/data1/bigbrain/phate_testing/gene_info.csv')
    if filter_mask is not None:
        aliases = aliases[filter_mask]
        gene_info = gene_info[filter_mask].reset_index()
    for i,gene_name in enumerate(gene_names):
        #first try ensembles
        if gene_ensembl is not None and np.sum(gene_info['ensemblgene']==gene_ensembl[i])>0:
            indices.append(np.where(gene_info['ensemblgene']==gene_ensembl[i])[0][0])
        #then try entrez
        elif gene_entrez is not None and np.sum(gene_info['gene.entrez_id']==gene_entrez[i])>0:
            indices.append(np.where(gene_info['gene.entrez_id']==gene_entrez[i])[0][0])
        elif protein_id is not None and np.sum(gene_info['protein_id']==protein_id[i])>0:
            indices.append(np.where(gene_info['protein_id']==protein_id[i])[0][0])
        elif np.sum(gene_info['gene.symbol']==gene_name)>0:
            indices.append(np.where(gene_info['gene.symbol']==gene_name)[0][0])
        else:
            found=False
            for k,alias in enumerate(aliases):
                if gene_name in alias:
                    indices.append(k)
                    found=True
                    break
            if not found:
                mask[i]=False
    #check for duplicates, if n gets really big
    indices_unique, counts=np.unique(indices,return_counts=True)
    if max(counts)>1:
#         duplicate_inds = indices_unique[counts>1]
#         mask_i = np.arange(len(mask))
#         mask_i=mask_i[mask]
#         for duple in duplicate_inds:
#             rows = mask_i[indices==duple]
        n_duplicates = np.sum(counts>1)
        print(f'warning,{n_duplicates} duplicates found')
    return np.array(indices),mask
 


    
def load_aliases():
    all_aliases=[]
    with open(os.path.join(data_dir,'all_aliases.txt'),'r') as f:
        for line in f:
            line=line[:-1]
            all_aliases.append(line.split(','))
    return all_aliases
    
    

    