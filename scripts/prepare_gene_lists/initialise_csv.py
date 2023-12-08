import pandas as pd


gene_info=pd.read_csv('/data1/bigbrain/phate_testing/gene_info.csv')

ex = gene_info[['gene.symbol']]

ex.to_csv('/data1/bigbrain/phate_testing/all_gene_lists.csv',index=False)
