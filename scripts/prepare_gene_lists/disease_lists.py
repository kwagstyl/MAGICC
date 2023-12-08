import pandas as pd
import numpy as np
from gene_mapping import get_indices

def ruzzo_list_load():
    t=pd.read_excel('/data1/bigbrain/phate_testing/gene_lists/1-s2.0-S0092867419307809-mmc3.xlsx',
                sheet_name = 'TADA_ASD-risk_genes')
    ruz=[]
    for g in t["Genes"].values:
        n=g.split(',')
        for ng in n:
            f= ng.strip(' ').strip('*')
            if f != ' ' and f != '':
                ruz.append(f)
    ruz= np.unique(ruz)
    ruz_inds,mask = get_indices(ruz)
    return ruz_inds

#pli first
pli_score=pd.read_csv('/data1/bigbrain/phate_testing/gene_lists/forweb_cleaned_exac_r03_march16_z_data_pLI.txt',delimiter='\t')
pli_indices=get_indices(list(pli_score['gene']))

pli_scores = pli_score['pLI'][pli_indices[1]]

vec = np.zeros(20781,dtype=bool)
vec[pli_indices[0]]=pli_scores>0.5

gene_info_fn='/data1/bigbrain/phate_testing/all_gene_lists.csv'
genes_dataframe  = pd.read_csv(gene_info_fn)
genes_dataframe['pLI'] = vec

#
rare=pd.read_csv('/data1/bigbrain/phate_testing/gene_lists/rare_disease_genes.csv')
columns = {'ASD_Satterstrom':'Rare ASD',
           'Schizophrenia_Schema':'Rare Schizophrenia',
           'DDD':'Rare DDD',
           'Epilepsy_Heyne':'Rare Epilepsy'}
for c in columns.keys():
    genes_dataframe[columns[c]]=rare[c]

ruz_inds = ruzzo_list_load()
genes_dataframe['Rare ASD'][ruz_inds] = True
    
#add GWAS here
sheets = {'GWAS_ASD_Grove':'GWAS ASD',
           'GWAS_Scz_Pardinas':'GWAS Schizophrenia',
           'GWAS_EA_Lee':'GWAS EA',
           'EpilepsyGWAS':'GWAS Epilepsy'}
for sheet in sheets.keys():
    gwas=pd.read_excel('/data1/bigbrain/phate_testing/disease_lists/disease_gene_list_sources.xlsx',
                      sheet_name=sheet)
    vec = np.zeros(20781,dtype=bool)
    gene_column=gwas.columns[0]
    if 'Scz' in sheet:
        gene_list = []
        for r in gwas[gene_column]:
            if ',' in r:
                gene_list.extend(r.split(','))
            else:
                gene_list.extend(r)
    else:
        gene_list = gwas[gene_column].values
    
    i,m=get_indices(gene_list)
    vec[i]=True
    genes_dataframe[sheets[sheet]]=vec
genes_dataframe.to_csv(gene_info_fn,index=False)
