import pandas as pd
import numpy as np
from gene_mapping import get_indices


#load werling tables (incorrectly named brainspan groups, should be brainvar
brainvar_groups=pd.read_csv('/data1/bigbrain/phate_testing/brainspan_groups.csv')
#put them into this
gene_info_fn='/data1/bigbrain/phate_testing/all_gene_lists.csv'
genes_dataframe  = pd.read_csv(gene_info_fn)

columns_of_interest=['Trajectory_group','PeakPeriod','PeakEpoch']
indices,mask=get_indices(brainvar_groups['GeneSymbol'],gene_ensembl=brainvar_groups['EnsemblID'])
brainvar_reduced = brainvar_groups[mask]
for column in columns_of_interest:
    classes = np.unique(brainvar_groups[column])
    for class_ in classes:
        name='BrainVar_'+column+'_'+str(class_)
        vec = np.zeros(20781,dtype=bool)
        subset=indices[brainvar_reduced[column]==class_]
        vec[subset] = True
        genes_dataframe[name] = vec
        
genes_dataframe.to_csv(gene_info_fn,index=False)
