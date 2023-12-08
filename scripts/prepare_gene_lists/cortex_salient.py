import pandas as pd
import numpy as np



gene_info_fn='/data1/bigbrain/phate_testing/all_gene_lists.csv'
genes_dataframe  = pd.read_csv(gene_info_fn)
#create list of cortex genes
keys=['Cell','Layer','Protein']
vec = np.zeros(len(genes_dataframe),dtype=bool)
for c,column in enumerate(genes_dataframe.columns):
    for key in keys:
        if key in column:
            vec += genes_dataframe[column]
genes_dataframe['Combined_cortex'] = vec.astype(bool)
genes_dataframe.to_csv(gene_info_fn,index=False)

