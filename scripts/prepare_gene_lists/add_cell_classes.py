import pandas as pd
import json
from gene_mapping import get_indices
import numpy as np

#adult cell types
cell_type_list = ['Ex', 'In', 'Ast', 'End', 'Mic', 'OPC', 'Oli']
with open('/data1/bigbrain/phate_testing/cell_lists/cell_groups.json','r') as f:
    cell_types = json.load(f)
cell_df=pd.read_csv('/data1/bigbrain/phate_testing/cell_lists/cell_lists.csv')


gene_info_fn='/data1/bigbrain/phate_testing/all_gene_lists.csv'
genes_dataframe  = pd.read_csv(gene_info_fn)
for c,cell_type in enumerate(cell_type_list):
    genes_dataframe['Cell '+cell_type]= np.sum(cell_df[cell_types[cell_type]],axis=1)>0

genes_dataframe.to_csv(gene_info_fn,index=False)


#foetal cell types
fetal_cells = pd.read_excel('/home/kwagstyl/Downloads/1-s2.0-S0896627319305616-mmc5.xlsx',sheet_name='Cluster enriched genes')
fetal_cell_names = np.unique(fetal_cells['Cluster'])
fetal_cell_bool = np.zeros((len(fetal_cell_names),20781),dtype=bool)
for ci,cell in enumerate(fetal_cell_names):
    gene_na = fetal_cells['Gene'][fetal_cells['Cluster']==cell].values
    gen_inds,m = get_indices(gene_na,gene_ensembl=fetal_cells['Ensembl'][fetal_cells['Cluster']==cell].values)
    genes_dataframe['Fetal_cells ' + cell] = np.zeros(len(genes_dataframe),dtype=bool)
    genes_dataframe['Fetal_cells ' + cell][gen_inds] = True

genes_dataframe.to_csv(gene_info_fn,index=False)
