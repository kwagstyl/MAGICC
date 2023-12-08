import pandas as pd
import numpy as np
import os
import h5py
import pandas as pd
from gene_mapping import get_indices
import pyreadr

gene_info_fn='/data1/bigbrain/phate_testing/all_gene_lists.csv'
def get_gene_rows(gene_set,indices,mask):
    rows=np.zeros(len(mask),dtype=bool)
    for gene in gene_set:
        rows[np.where(indices==gene)[0]]=True
    return rows

def get_expr_row(gene_set,indices,mask,row_order,gene_expr_data):
    gene_rows = get_gene_rows(gene_set,indices,mask)
    expr_row = np.mean(gene_expr_data[gene_rows][:,row_order],axis=0)
    return expr_row


def rect(pos,ax,text=None,):
    r = plt.Rectangle(pos-0.5, 1,1, facecolor="none",hatch="x ", edgecolor="k", linewidth=3)
    ax.add_patch(r)
    if text is not None:
        ax.text(pos[0],pos[1],
            text,
           ha='center',va='center')
        

#load in data
rdata = pyreadr.read_r('/data1/bigbrain/phate_testing/fetal_data/prenetal.allen_for.konrad.RData')
samples_df=rdata['for.konrad_donors_2566.12690_sample.metadata']
samples_df['row_number']=np.arange(len(samples_df))
gene_ids=rdata['for.konrad_donors_2566.12690_gene.metadata']
gene_expr_data=np.array(rdata['for.konrad_donors_2566.12690_gene.z_by_sample_expression'])
#get radial and tangential zone names
radial_zone = []
tangential_area=[]
for name in samples_df['structure_name']:
    split_name = name.split(' in ')
    if len(split_name)>1:
        radial_zone.append(split_name[0])
        tangential_area.append(split_name[1])
    else:
        radial_zone.append('NA')
        tangential_area.append('NA')

samples_df['radial_zone'] = radial_zone
samples_df['tangential_area'] = tangential_area
zones_of_interest = ['SG', 'MZ', 'outer CP','CP','inner CP', 'SP','IZ', 'outer SZ','SZ','inner SZ','VZ']
row_order = []
xticks=[]
xtick_labels=[]
b=0
zone_group=[]
for zi,zone in enumerate(zones_of_interest):
    subset = samples_df[samples_df['radial_zone']==zone]
    for structure in subset['structure_name']:
        area=subset['tangential_area'][subset['structure_name']==structure].values[0]
        row_order.append(subset['row_number'][subset['structure_name']==structure].values[0])
        zone_group.append(zi)
    xticks.append(b+len(subset)/2)
    xtick_labels.append(zone)
    b+=len(subset)
row_order = np.array(row_order) 
zone_group=np.array(zone_group)
indices,mask=get_indices(gene_ids.gene_symbol,gene_entrez=gene_ids.entrez_id)
#group into just 7 zones
reduced_zones = ['SG', 'MZ', 'CP', 'SP','IZ', 'SZ','VZ']

zone_group_reduced = np.zeros_like(zone_group)
for zi,zone in enumerate(zones_of_interest):
    for zri, zone_r in enumerate(reduced_zones):
        if zone_r in zone:
            zone_group_reduced[zone_group==zi]=zri
            
#test for enrichment
#hypergeometric test
fetal_expr_by_compartment = np.zeros((gene_expr_data.shape[0],len(reduced_zones)))
reordered=gene_expr_data[:,row_order]
for zi,zone in enumerate(np.unique(zone_group_reduced)):
    fetal_expr_by_compartment[:,zi] = np.mean(reordered[:,zone_group_reduced==zone],axis=1)
fetal_expr_by_compartment = fetal_expr_by_compartment[mask]

#find the top 5% of genes per compartment
masked_matrix=((-fetal_expr_by_compartment.T).argsort().argsort()<(5*len(indices)/100))

genes_dataframe  = pd.read_csv(gene_info_fn)
for zi,zone in enumerate(reduced_zones):
    genes_dataframe['Fetal layer '+zone] = np.zeros(len(genes_dataframe),dtype=bool)
    genes_dataframe['Fetal layer '+zone][indices] = masked_matrix[zi]
    
genes_dataframe.to_csv(gene_info_fn,index=False)



