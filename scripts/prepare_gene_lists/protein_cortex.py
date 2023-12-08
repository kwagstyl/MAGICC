import pandas as pd
from gene_mapping import get_indices
import numpy as np

#all_genes = pd.read_csv('/data1/bigbrain/phate_testing/gene_lists/proteinatlas_2dffabfa.tsv',delimiter='\t')
brain_expressed = pd.read_csv('/data1/bigbrain/phate_testing/gene_lists/NOT.tsv',delimiter='\t')

#all_genes,boo=get_indices(all_genes['Gene'],gene_ensembl=all_genes['Ensembl'])
cortex_genes,boo = get_indices(brain_expressed['Gene'],gene_ensembl=brain_expressed['Ensembl'])


gene_info_fn='/data1/bigbrain/phate_testing/all_gene_lists.csv'
genes_dataframe  = pd.read_csv(gene_info_fn)
genes_dataframe['Protein_cortex'] = np.zeros(len(genes_dataframe),dtype=bool)
genes_dataframe['Protein_cortex'][cortex_genes]= True
    
genes_dataframe.to_csv(gene_info_fn,index=False)

