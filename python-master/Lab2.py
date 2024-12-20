#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "chiehjul"
COLLABORATORS = ""


# ---

# # Lab 2 Introduction
# 
# 
# In this lab you will use parse a vcf file to extracts parts of it and load it into a database. 
# The fields you will be parsing from the vcf file are: 
# ```
# CHROM
# POS	
# ID	
# REF	
# ALT	
# QUAL	
# FILTER
# ```
# and from the INFO column, the following fields:
# ```
# 1000g2015aug_all
# ExAC_ALL
# FATHMM_pred
# LRT_pred
# MetaLR_pred
# MetaSVM_pred
# MutationAssessor_pred
# MutationTaster_pred
# PROVEAN_pred
# Polyphen2_HDIV_pred
# Polyphen2_HVAR_pred
# SIFT_pred
# fathmm-MKL_coding_pred.
# ```
# The fields:
# ```
# FATHMM_pred
# LRT_pred
# MetaLR_pred
# MetaSVM_pred
# MutationAssessor_pred
# MutationTaster_pred
# PROVEAN_pred
# Polyphen2_HDIV_pred
# Polyphen2_HVAR_pred
# SIFT_pred
# fathmm-MKL_coding_pred
# ```
# are predictor fields. They use a letter to indicate whether a given variation is harmful or not.
# 
# NOTE: use the helper functions in cell2. To use them, you have to run CELL 2! 
# 

# In[2]:


## Helper functions

import os
import sqlite3
from sqlite3 import Error
import gzip

def create_connection(db_file, delete_db=False):
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


# # Part 1 (10 pts)
# 
# In part1 you will open the gzipped vcf file using the python gzip module and figure out all the possible values for the predictor fields. 
# 
# You do not have to unzip the file. Use `with gzip.open(filename,'rt') as fp:` to read one line at a time. 
# 
# 

# In[3]:


def get_predictor_values(filename):
    """
    See part 1 description 
    """
    import gzip

   
    keys_save_value = [
        'FATHMM_pred',
        'LRT_pred',
        'MetaLR_pred',
        'MetaSVM_pred',
        'MutationAssessor_pred',
        'MutationTaster_pred',
        'PROVEAN_pred',
        'Polyphen2_HDIV_pred',
        'Polyphen2_HVAR_pred',
        'SIFT_pred',
        'fathmm-MKL_coding_pred',
    ]

    # YOUR CODE HERE
    import re
    
    # Init
    pred_val = {}
    for key in keys_save_value:
        pred_val[key] = []
    
    with gzip.open(filename,'rt') as fp:
        for line in fp:
            for key in keys_save_value:
                
                # Find the val of each key if exists 
                pattern = ';'+ key +'=[^;|^.]*;'
                val = re.findall(pattern, line)
                
                if val != []:
                    val = re.split(r'=',val[0].strip(';'))[-1]
                    if val not in pred_val[key]:
                        pred_val[key].append(val)

    return pred_val


# In[4]:



## Can take 30 seconds to run!
expected_solution = {'SIFT_pred': ['D', 'T'], 'Polyphen2_HDIV_pred': ['D', 'B', 'P'], 'Polyphen2_HVAR_pred': ['D', 'B', 'P'], 'LRT_pred': ['D', 'N', 'U'], 'MutationTaster_pred': ['D', 'P', 'N', 'A'], 'MutationAssessor_pred': ['H', 'N', 'L', 'M'], 'FATHMM_pred': ['T', 'D'], 'PROVEAN_pred': ['D', 'N'], 'MetaSVM_pred': ['D', 'T'], 'MetaLR_pred': ['D', 'T'], 'fathmm-MKL_coding_pred': ['D', 'N']}
filename = 'test_4families_annovar.vcf.gz'
predictor_values = get_predictor_values(filename)
assert predictor_values == expected_solution


# # Part 2 (No points)
# 
# You will now create 13 tables
# Tables 1-11 will be for :
# 
# FATHMM_pred
# LRT_pred 
# MetaLR_pred
# MetaSVM_pred
# MutationAssessor_pred
# MutationTaster_pred
# PROVEAN_pred
# Polyphen2_HDIV_pred
# Polyphen2_HVAR_pred
# SIFT_pred
# fathmm_MKL_coding_pred ## NOTE: I have replaced the dash with an underscore. 
# 
# The first column for each of these tables will be the name of the field + ID, e.g., FATHMM_predID. This
# column will be of type not null primary key. The second column will be called `prediction` and will be of
# type TEXT not null. The prediction values will be values you extracted for each of the fields in the previous step. 
# For example, 'FATHMM_pred' has two prediction values 'T' and 'D'. Name the tables after their field name, e.g.,  
# call table that will contains values of FATHMM_pred `FATHMM_pred`.  Make sure to sort the values, meaning, D should 
# be inserted before T. 
# 
# 
# Table 12 will be the Variants table. 
# The first column will be VariantID. 
# 
# The other columns will be:
# CHROM   
# POS 
# ID  
# REF 
# ALT 
# QUAL    
# FILTER
# thousandg2015aug_all ##NOTE: a column name cannot start with a number, so you have to rename!
# ExAC_ALL
# 
# 
# Table 12 will have 11 more columns that relate to each of prediction table. Consider writing a utility function
# to fetch the primary key for a given prediction from each of the prediction table. Name each of the 
# column should be the name of predictor + ID, e.g., FATHMM_predID. 
# 
# 
# You have already deteremined their data type, so use that to set their data type. 
# For integer use INTEGER. 
# For float use REAL. 
# For string use TEXT. 
# 
# Table 13  will be called PredictionStats. The first column will be PredictorStatsID INTEGER NOT NULL PRIMARY KEY. 
# The second column will be VariantID. The third column will be PredictorName. The fourth column will be PredictorValue. 
# This is not a normalized table!
# 
# The prediction value will be a float value mapped from the prediction text.  Use the following information to 
# assign values: 
# REF: https://brb.nci.nih.gov/seqtools/colexpanno.html#dbnsfp
# 
# FATHMM_pred
# - T = 0
# - D = 1
# 
# LRT_pred 
# - D = 1
# - N = 0
# - U = 0
# 
# MetaLR_pred
# - T = 0
# - D = 1
# 
# MetaSVM_pred
# - T = 0
# - D = 1
# 
# MutationAssessor_pred
# - H = 1
# - N = 0
# - L = 0.25
# - M = 0.5
# 
# MutationTaster_pred
# - D = 1
# - P = 0
# - N = 0
# - A = 1
# 
# PROVEAN_pred
# - D = 1
# - N = 0
# 
# Polyphen2_HDIV_pred
# - D = 1
# - B = 0
# - P = 0.5
# 
# Polyphen2_HVAR_pred
# - D = 1
# - B = 0
# - P = 0.5
# 
# SIFT_pred
# - D = 1
# - T = 0
# 
# fathmm-MKL_coding_pred
# - D = 1
# - N = 0
# 
# 
# ```
# 
# prediction_mapping = {
#     'FATHMM_pred': {'T': 0, 'D': 1},
#     'MetaLR_pred': {'T': 0, 'D': 1},
#     'MetaSVM_pred': {'T': 0, 'D': 1},
#     'SIFT_pred': {'T': 0, 'D': 1},
#     'fathmm_MKL_coding_pred': {'D': 1, 'N': 0},
#     'LRT_pred': {'U': 0, 'N': 0, 'D': 1},
#     'MutationAssessor_pred': {'H': 1, 'N': 0, 'L': 0.25, 'M': 0.5},  
#     'MutationTaster_pred': {'D': 1, 'P': 0, 'N': 0, 'A': 1},
#     'PROVEAN_pred': {'D': 1, 'N': 0},
#     'Polyphen2_HDIV_pred': {'D': 1, 'B': 0, 'P': 0.5},
#     'Polyphen2_HVAR_pred': {'D': 1, 'B': 0, 'P': 0.5},
# }
# 
# 
# ```
# 
# 
# The idea is that 11 predictors have been used to annotate all the variants in the file. This table combines all that information in one table. 
# By grouping the table based on variantid and summing the prediction values mapped from the prediction text, you can find which variants have a consensus on their being deterimental. 
# 
# IMPORTANT: instead of fathmm-MKL_coding_pred use fathmm_MKL_coding_pred everywhere. 
# HINT: You have to commit the changes you make to the table, otherwise your changes will not be saved. Consider wrapping all changes inside `with conn:`
# 
# 
# 

# In[5]:


# Creating lab2.db 
db_file = 'lab2.db'
conn = create_connection(db_file, delete_db=True)
conn.close()


# In[6]:


def create_tables_1_11(db_file):
    """
    Create tables 1 to 11.
    
    INPUT: db_file -- Name of database file -- eg. lab2.db
    
    """
    # YOUR CODE HERE
    
    # Init
    sql_state = ""
    predictor_values = {}
    
    str_list = "FATHMM_pred LRT_pred MetaLR_pred MetaSVM_pred MutationAssessor_pred MutationTaster_pred PROVEAN_pred Polyphen2_HDIV_pred Polyphen2_HVAR_pred SIFT_pred fathmm_MKL_coding_pred"
    table_list = str_list.split(' ')
    
    conn = create_connection(db_file)
    cur = conn.cursor()
    with conn:
        predictor_values = {
            'SIFT_pred': ['D', 'T'], 
            'Polyphen2_HDIV_pred': ['D', 'B', 'P'], 
            'Polyphen2_HVAR_pred': ['D', 'B', 'P'], 
            'LRT_pred': ['D', 'N', 'U'], 
            'MutationTaster_pred': ['D', 'P', 'N', 'A'], 
            'MutationAssessor_pred': ['H', 'N', 'L', 'M'], 
            'FATHMM_pred': ['T', 'D'], 
            'PROVEAN_pred': ['D', 'N'], 
            'MetaSVM_pred': ['D', 'T'], 
            'MetaLR_pred': ['D', 'T'], 
            'fathmm_MKL_coding_pred': ['D', 'N']
        }

    
        for table in table_list:
            sql_state = "CREATE TABLE IF NOT EXISTS [" + table + """] (
            [""" + table + """ID]  INTEGER  NOT NULL PRIMARY KEY,
            [prediction] TEXT NOT NULL);
            """
        
            #cur.execute("DROP TABLE " + table)
            
            create_table(conn, sql_state)
            
            for val in sorted(predictor_values[table]):
                sql_insert_statement = "INSERT INTO " + table + "('prediction')\n VALUES(?)"
                cur.execute(sql_insert_statement, val)

db_file = 'lab2.db'    
create_tables_1_11(db_file)


# # Part 3 (10 pts)
# 
# Write a function that returns a dictionary that maps for a given predictor its letter prediction to foreign key value. 
# Conver the dash in 'fathmm-MKL_coding_pred' to an underscore. 
# 

# In[7]:


def get_predictor_value_to_fk_map(db_file):
    """
    See part 3 description 
    
    INPUT: db_file -- Name of database file -- eg. lab2.db
    
    """
    # YOUR CODE HERE
    
    # Init 
    fk_map_dict = {}
    table_list = []
    
    str_list = "FATHMM_pred LRT_pred MetaLR_pred MetaSVM_pred MutationAssessor_pred MutationTaster_pred PROVEAN_pred Polyphen2_HDIV_pred Polyphen2_HVAR_pred SIFT_pred fathmm_MKL_coding_pred"
    table_list = str_list.split(' ')
    
    conn = create_connection(db_file)
    cur = conn.cursor()
    
    with conn:
        for table in table_list:
            #Init
            fk_map_dict[table] = {}
            
            # Get data list from table
            sql_stat = "select prediction, " + table + "ID from " + table
            val_list = cur.execute(sql_stat).fetchall()
            
            for val in val_list:
                fk_map_dict[table][val[0]] = val[1]
                
    return fk_map_dict


# In[8]:


expected_solution = {
    'FATHMM_pred': {'D': 1, 'T': 2}, 
    'LRT_pred': {'D': 1, 'N': 2, 'U': 3}, 
    'MetaLR_pred': {'D': 1, 'T': 2}, 
    'MetaSVM_pred': {'D': 1, 'T': 2}, 
    'MutationAssessor_pred': {'H': 1, 'L': 2, 'M': 3, 'N': 4}, 
    'MutationTaster_pred': {'A': 1, 'D': 2, 'N': 3, 'P': 4}, 
    'PROVEAN_pred': {'D': 1, 'N': 2}, 
    'Polyphen2_HDIV_pred': {'B': 1, 'D': 2, 'P': 3}, 
    'Polyphen2_HVAR_pred': {'B': 1, 'D': 2, 'P': 3}, 
    'SIFT_pred': {'D': 1, 'T': 2}, 
    'fathmm_MKL_coding_pred': {'D': 1, 'N': 2}}

db_file = 'lab2.db'
predictor_fk_map = get_predictor_value_to_fk_map(db_file)
assert predictor_fk_map == expected_solution


# # Part 4 (No Points)
# Create table 12 or the variants table. See description above
# 

# In[9]:


def create_variants_table(db_file):
    """
    Part 4
    """
    # YOUR CODE HERE
    # Init 
    
    str_list = "FATHMM_pred LRT_pred MetaLR_pred MetaSVM_pred MutationAssessor_pred MutationTaster_pred PROVEAN_pred Polyphen2_HDIV_pred Polyphen2_HVAR_pred SIFT_pred fathmm_MKL_coding_pred"
    table_list = str_list.split(' ')
    
    conn = create_connection(db_file)
    cur = conn.cursor()
    with conn:

        sql_state = """CREATE TABLE IF NOT EXISTS [Variants] (
        [VariantID]  INTEGER  NOT NULL PRIMARY KEY,
        [CHROM] TEXT NOT NULL,
        [POS] INTEGER NOT NULL,
        [ID] TEXT NOT NULL,
        [REF] TEXT NOT NULL,
        [ALT] TEXT NOT NULL,
        [QUAL] REAL NOT NULL,
        [FILTER] TEXT NOT NULL,
        [thousandg2015aug_all] INTEGER ,
        [ExAC_ALL] INTEGER """
    
        for table in table_list:
            sql_state = sql_state + """,\n\t[""" + table + """ID]  INTEGER"""
        sql_state = sql_state + """\n);"""
        
        #cur.execute("DROP TABLE Variants")
            
        create_table(conn, sql_state)

# create table
db_file = 'lab2.db'
create_variants_table(db_file)


# # Part 5 (No Points)
# 
# Create table 13 -- or the prediction stats table. See description above. 
# 

# In[10]:


def create_predictionstats_table(db_file):
    """
    Part 5   
    """
    # YOUR CODE HERE

    # Init 
    conn = create_connection(db_file)
    cur = conn.cursor()
    with conn:

        sql_state = """CREATE TABLE IF NOT EXISTS [PredictionStats] (
        [PredictorStatsID]  INTEGER  NOT NULL PRIMARY KEY,
        [VariantID] INTEGER ,
        [PredictorName] TEXT ,
        [PredictorValue] REAL
        );
        """
        
        #cur.execute("DROP TABLE PredictionStats")
            
        create_table(conn, sql_state)
        
    
db_file = 'lab2.db'
create_predictionstats_table(db_file)


# # Part 6 (10 Points)
# 
# Write a function to pull the following info fields given the whole info field. 
# ```
# values_to_pull = [
#         '1000g2015aug_all',
#         'ExAC_ALL',
#         'FATHMM_pred',
#         'LRT_pred',
#         'MetaLR_pred',
#         'MetaSVM_pred',
#         'MutationAssessor_pred',
#         'MutationTaster_pred',
#         'PROVEAN_pred',
#         'Polyphen2_HDIV_pred',
#         'Polyphen2_HVAR_pred',
#         'SIFT_pred',
#         'fathmm-MKL_coding_pred',
#     ]
# ```
# 

# In[11]:


def pull_info_values(info):
    """
    See part 6 description
    """
    # YOUR CODE HERE
    import re

    # Init
    info_dict = {}
    info_list = ['thousandg2015aug_all', 'ExAC_ALL', 'SIFT_pred', 'Polyphen2_HDIV_pred', 
                 'Polyphen2_HVAR_pred', 'LRT_pred','MutationTaster_pred', 'MutationAssessor_pred', 
                 'FATHMM_pred', 'PROVEAN_pred', 'MetaSVM_pred', 'MetaLR_pred', 'fathmm_MKL_coding_pred']
    
    for key in info_list:
        
        # Find the val of each key if exists 
        if key == 'thousandg2015aug_all':
            pattern = ';1000g2015aug_all=[^;]*;'
        elif key == 'fathmm_MKL_coding_pred':
            pattern = ';fathmm-MKL_coding_pred=[^;]*;'
        else:
            pattern = ';'+ key +'=[^;]*;'
        val = re.findall(pattern, info)
                
        if val != []:
            val = re.split(r'=',val[0].strip(';'))[-1]
            if val == '.':
                info_dict[key]= ''
            else:
                info_dict[key]= val
                
    return info_dict
    


# In[12]:


sample_info_input = "AC=2;AF=0.333;AN=6;BaseQRankSum=2.23;ClippingRankSum=0;DP=131;ExcessHet=3.9794;FS=2.831;MLEAC=2;MLEAF=0.333;MQ=60;MQRankSum=0;QD=12.06;ReadPosRankSum=-0.293;SOR=0.592;VQSLOD=21.79;culprit=MQ;DB;POSITIVE_TRAIN_SITE;ANNOVAR_DATE=2018-04-16;Func.refGene=exonic;Gene.refGene=MAST2;GeneDetail.refGene=.;ExonicFunc.refGene=nonsynonymous_SNV;AAChange.refGene=MAST2:NM_015112:exon29:c.G3910A:p.V1304M;Func.ensGene=exonic;Gene.ensGene=ENSG00000086015;GeneDetail.ensGene=.;ExonicFunc.ensGene=nonsynonymous_SNV;AAChange.ensGene=ENSG00000086015:ENST00000361297:exon29:c.G3910A:p.V1304M;cytoBand=1p34.1;gwasCatalog=.;tfbsConsSites=.;wgRna=.;targetScanS=.;Gene_symbol=.;OXPHOS_Complex=.;Ensembl_Gene_ID=.;Ensembl_Protein_ID=.;Uniprot_Name=.;Uniprot_ID=.;NCBI_Gene_ID=.;NCBI_Protein_ID=.;Gene_pos=.;AA_pos=.;AA_sub=.;Codon_sub=.;dbSNP_ID=.;PhyloP_46V=.;PhastCons_46V=.;PhyloP_100V=.;PhastCons_100V=.;SiteVar=.;PolyPhen2_prediction=.;PolyPhen2_score=.;SIFT_prediction=.;SIFT_score=.;FatHmm_prediction=.;FatHmm_score=.;PROVEAN_prediction=.;PROVEAN_score=.;MutAss_prediction=.;MutAss_score=.;EFIN_Swiss_Prot_Score=.;EFIN_Swiss_Prot_Prediction=.;EFIN_HumDiv_Score=.;EFIN_HumDiv_Prediction=.;CADD_score=.;CADD_Phred_score=.;CADD_prediction=.;Carol_prediction=.;Carol_score=.;Condel_score=.;Condel_pred=.;COVEC_WMV=.;COVEC_WMV_prediction=.;PolyPhen2_score_transf=.;PolyPhen2_pred_transf=.;SIFT_score_transf=.;SIFT_pred_transf=.;MutAss_score_transf=.;MutAss_pred_transf=.;Perc_coevo_Sites=.;Mean_MI_score=.;COSMIC_ID=.;Tumor_site=.;Examined_samples=.;Mutation_frequency=.;US=.;Status=.;Associated_disease=.;Presence_in_TD=.;Class_predicted=.;Prob_N=.;Prob_P=.;SIFT_score=0.034;SIFT_converted_rankscore=0.440;SIFT_pred=D;Polyphen2_HDIV_score=0.951;Polyphen2_HDIV_rankscore=0.520;Polyphen2_HDIV_pred=P;Polyphen2_HVAR_score=0.514;Polyphen2_HVAR_rankscore=0.462;Polyphen2_HVAR_pred=P;LRT_score=0.002;LRT_converted_rankscore=0.368;LRT_pred=N;MutationTaster_score=1.000;MutationTaster_converted_rankscore=0.810;MutationTaster_pred=D;MutationAssessor_score=1.67;MutationAssessor_score_rankscore=0.430;MutationAssessor_pred=L;FATHMM_score=1.36;FATHMM_converted_rankscore=0.344;FATHMM_pred=T;PROVEAN_score=-1.4;PROVEAN_converted_rankscore=0.346;PROVEAN_pred=N;VEST3_score=0.158;VEST3_rankscore=0.189;MetaSVM_score=-1.142;MetaSVM_rankscore=0.013;MetaSVM_pred=T;MetaLR_score=0.008;MetaLR_rankscore=0.029;MetaLR_pred=T;M-CAP_score=.;M-CAP_rankscore=.;M-CAP_pred=.;CADD_raw=4.716;CADD_raw_rankscore=0.632;CADD_phred=24.6;DANN_score=0.998;DANN_rankscore=0.927;fathmm-MKL_coding_score=0.900;fathmm-MKL_coding_rankscore=0.506;fathmm-MKL_coding_pred=D;Eigen_coding_or_noncoding=c;Eigen-raw=0.461;Eigen-PC-raw=0.469;GenoCanyon_score=1.000;GenoCanyon_score_rankscore=0.747;integrated_fitCons_score=0.672;integrated_fitCons_score_rankscore=0.522;integrated_confidence_value=0;GERP++_RS=4.22;GERP++_RS_rankscore=0.490;phyloP100way_vertebrate=4.989;phyloP100way_vertebrate_rankscore=0.634;phyloP20way_mammalian=1.047;phyloP20way_mammalian_rankscore=0.674;phastCons100way_vertebrate=1.000;phastCons100way_vertebrate_rankscore=0.715;phastCons20way_mammalian=0.999;phastCons20way_mammalian_rankscore=0.750;SiPhy_29way_logOdds=17.151;SiPhy_29way_logOdds_rankscore=0.866;Interpro_domain=.;GTEx_V6_gene=ENSG00000162415.6;GTEx_V6_tissue=Nerve_Tibial;esp6500siv2_all=0.0560;esp6500siv2_aa=0.0160;esp6500siv2_ea=0.0761;ExAC_ALL=0.0553;ExAC_AFR=0.0140;ExAC_AMR=0.0386;ExAC_EAS=0.0005;ExAC_FIN=0.0798;ExAC_NFE=0.0788;ExAC_OTH=0.0669;ExAC_SAS=0.0145;ExAC_nontcga_ALL=0.0541;ExAC_nontcga_AFR=0.0129;ExAC_nontcga_AMR=0.0379;ExAC_nontcga_EAS=0.0004;ExAC_nontcga_FIN=0.0798;ExAC_nontcga_NFE=0.0802;ExAC_nontcga_OTH=0.0716;ExAC_nontcga_SAS=0.0144;ExAC_nonpsych_ALL=0.0496;ExAC_nonpsych_AFR=0.0140;ExAC_nonpsych_AMR=0.0386;ExAC_nonpsych_EAS=0.0005;ExAC_nonpsych_FIN=0.0763;ExAC_nonpsych_NFE=0.0785;ExAC_nonpsych_OTH=0.0638;ExAC_nonpsych_SAS=0.0145;1000g2015aug_all=0.024361;1000g2015aug_afr=0.0038;1000g2015aug_amr=0.0461;1000g2015aug_eur=0.0795;1000g2015aug_sas=0.0041;CLNALLELEID=.;CLNDN=.;CLNDISDB=.;CLNREVSTAT=.;CLNSIG=.;dbscSNV_ADA_SCORE=.;dbscSNV_RF_SCORE=.;snp138NonFlagged=rs33931638;avsnp150=rs33931638;CADD13_RawScore=4.716301;CADD13_PHRED=24.6;Eigen=0.4614;REVEL=0.098;MCAP=.;Interpro_domain=.;ICGC_Id=.;ICGC_Occurrence=.;gnomAD_genome_ALL=0.0507;gnomAD_genome_AFR=0.0114;gnomAD_genome_AMR=0.0430;gnomAD_genome_ASJ=0.1159;gnomAD_genome_EAS=0;gnomAD_genome_FIN=0.0802;gnomAD_genome_NFE=0.0702;gnomAD_genome_OTH=0.0695;gerp++gt2=4.22;cosmic70=.;InterVar_automated=Benign;PVS1=0;PS1=0;PS2=0;PS3=0;PS4=0;PM1=0;PM2=0;PM3=0;PM4=0;PM5=0;PM6=0;PP1=0;PP2=0;PP3=0;PP4=0;PP5=0;BA1=1;BS1=1;BS2=0;BS3=0;BS4=0;BP1=0;BP2=0;BP3=0;BP4=0;BP5=0;BP6=0;BP7=0;Kaviar_AF=0.0552127;Kaviar_AC=8536;Kaviar_AN=154602;ALLELE_END"

expected_solution = {
    'thousandg2015aug_all': '0.024361', 
    'ExAC_ALL': '0.0553', 
    'SIFT_pred': 'D', 
    'Polyphen2_HDIV_pred': 'P', 
    'Polyphen2_HVAR_pred': 'P', 
    'LRT_pred': 'N',
    'MutationTaster_pred': 'D', 
    'MutationAssessor_pred': 'L', 
    'FATHMM_pred': 'T', 
    'PROVEAN_pred': 'N', 
    'MetaSVM_pred': 'T', 
    'MetaLR_pred': 'T', 
    'fathmm_MKL_coding_pred': 'D'
}


solution  = pull_info_values(sample_info_input)
assert solution == expected_solution


# # Part 7 (10 points)
# 
# Remember that to insert a record in SQLite, you have to use `cur.execute(sql, values)`, where `sql` is the insert statement and `values` is a list/tuple of values that will be substituted into the `sql` string wherever there is a question mark. 
# 
# Write a function that takes in as input: CHROM, POS, ID, REF, ALT, QUAL, FILTER, info_values and returns a list with the values in the following order:
# ```
# CHROM 
# POS
# ID
# REF
# ALT
# QUAL
# FILTER
# thousandg2015aug_all # 1000 has been replaced by the text thousand
# ExAC_ALL
# FATHMM_pred
# LRT_pred
# MetaLR_pred
# MetaSVM_pred
# MutationAssessor_pred
# MutationTaster_pred
# PROVEAN_pred
# Polyphen2_HDIV_pred
# Polyphen2_HVAR_pred
# SIFT_pred
# fathmm_MKL_coding_pred # note that the dash has been replaced by underscore
# ```
# 
# The info_values dictionary contains the predictor values. Use `None` for any empty/missing value for info fields, thousandg2015aug_all, and ExAC_ALL. 
# 
# 

# In[13]:


def build_values_list(CHROM, POS, ID, REF, ALT, QUAL, FILTER, info_values):
    """
    See part 7 description 
    
    """
    # YOUR CODE HERE
    import re 
    
    # Init
    val_list = []
    
    info_mapping = {
        'FATHMM_pred': {'D': 1, 'T': 2}, 
        'LRT_pred': {'D': 1, 'N': 2, 'U': 3}, 
        'MetaLR_pred': {'D': 1, 'T': 2}, 
        'MetaSVM_pred': {'D': 1, 'T': 2}, 
        'MutationAssessor_pred': {'H': 1, 'L': 2, 'M': 3, 'N': 4}, 
        'MutationTaster_pred': {'A': 1, 'D': 2, 'N': 3, 'P': 4}, 
        'PROVEAN_pred': {'D': 1, 'N': 2}, 
        'Polyphen2_HDIV_pred': {'B': 1, 'D': 2, 'P': 3}, 
        'Polyphen2_HVAR_pred': {'B': 1, 'D': 2, 'P': 3}, 
        'SIFT_pred': {'D': 1, 'T': 2}, 
        'fathmm_MKL_coding_pred': {'D': 1, 'N': 2}
    }
    
    var_str = "CHROM POS ID REF ALT QUAL FILTER" 
    for val in re.split('\W+', var_str):
        try:
            # Find the val in varable if exists 
            if eval(val) == '':
                val_list.append(None)
            else:
                val_list.append(eval(val))
        except: 
            continue
    
    info_str = """thousandg2015aug_all ExAC_ALL FATHMM_pred LRT_pred MetaLR_pred
    MetaSVM_pred MutationAssessor_pred MutationTaster_pred PROVEAN_pred Polyphen2_HDIV_pred Polyphen2_HVAR_pred
    SIFT_pred fathmm_MKL_coding_pred"""
        
    for val in re.split('\W+', info_str):
        # Get from info
        if val in info_values and info_values[val] != '':
            if val in info_mapping:
                val_list.append(info_mapping[val][info_values[val]])
            else:
                val_list.append(info_values[val])
        else:
            val_list.append(None)

    return val_list
    
    


# In[14]:


CHROM, POS, ID, REF, ALT, QUAL, FILTER = (7, 87837848, '.', 'C', 'A', 418.25, 'PASS') 
info_values = {'SIFT_pred': 'D', 'Polyphen2_HDIV_pred': 'D', 'Polyphen2_HVAR_pred': 'D', 'LRT_pred': 'D', 'MutationTaster_pred': 'D', 'MutationAssessor_pred': 'H', 'FATHMM_pred': 'T', 'PROVEAN_pred': 'D', 'MetaSVM_pred': 'D', 'MetaLR_pred': 'D', 'fathmm_MKL_coding_pred': 'D'}

results = build_values_list(CHROM, POS, ID, REF, ALT, QUAL, FILTER, info_values)
expected_results = [7, 87837848, '.', 'C', 'A', 418.25, 'PASS', None, None, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1]
assert results == expected_results


# # Part 8 (No Points)
# 
# Create a function that takes the database `conn` and `values` from the `build_values_list` function to insert a variant record. 
# 
# IMPORTANT: Function should return the id of the row inserted, which will be VariantId
# 

# In[15]:


def insert_variant(conn, values):
    """
    See description Part 8
    """
    # YOUR CODE HERE
    
    #Init 
    import re
    sql_stat = ""
    
    var_col_str = """CHROM POS ID REF ALT QUAL FILTER thousandg2015aug_all ExAC_ALL FATHMM_pred LRT_pred MetaLR_pred
    MetaSVM_pred MutationAssessor_pred MutationTaster_pred PROVEAN_pred Polyphen2_HDIV_pred Polyphen2_HVAR_pred
    SIFT_pred fathmm_MKL_coding_pred"""
    var_col_list = re.split('\W+', var_col_str)
    
    
    sql_stat = "INSERT INTO Variants("
    for col in var_col_list:
        if re.search(r'_pred',col):
            sql_stat = sql_stat + col + 'ID,'
        else:
            sql_stat = sql_stat + col + ','
    sql_stat = sql_stat[0:len(sql_stat)-1] + """)
               VALUES(?""" +", ?"*(len(var_col_list)-1) + " )"
    
    cur = conn.cursor()
    cur.execute(sql_stat, values)
    
    return cur.lastrowid


# In[16]:


#db_file = 'lab2.db'
#conn = create_connection(db_file)
#insert_variant(conn, expected_results)

#conn.close()


# # Part 9 (No Points)
# 
# Create a function that takes the database `conn` and `values` which is the tuple `(VariantId, PredictorName, PredictorValue)` Where `VariantId` will be the return value from the `insert_variant` function,  `PredictorName` is the name of the predictor, and `PredictorValue` is the mapping of the text value to a numeric value. 
# 

# In[17]:


def insert_predictionstat(conn, values):
    """
    See description in part 9
    """
    # YOUR CODE HERE
    
    #Init 
    import re
    sql_stat = ""
    
    var_col_str = "VariantId PredictorName PredictorValue"    
    var_col_list = re.split('\W+', var_col_str)
    
    sql_stat = "INSERT INTO PredictionStats("
    for col in var_col_list:
        sql_stat = sql_stat + col + ','
    sql_stat = sql_stat[0:len(sql_stat)-1] + """)
               VALUES(?""" +", ?"*(len(var_col_list)-1) + " )"

    cur = conn.cursor()
    cur.execute(sql_stat, values)
    
    return values[0]


# In[18]:


#db_file = 'lab2.db'
#conn = create_connection(db_file)
#insert_predictionstat(conn, ("apple", "banana", "cherry"))

#conn.close()


# # Part 10 (No Points)
# 
# Write a function to insert records into both the variants and predictor_stats table. 
# Hint:
# 1) Open connection to database  
# 2) Read file one line at a time using gzip read
# 3) Extract CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO
# 4) Use the pull_info_values function
# 5) Use build_values_list 
# 6) Use insert_variant -- save variant_id
# 7) Use insert_predictionstat -- insert each predictor at a time and remember to use `prediction_mapping` mapping. 
# 
# 
# 
# 
# ```
# 
# prediction_mapping = {
#     'FATHMM_pred': {'T': 0, 'D': 1},
#     'MetaLR_pred': {'T': 0, 'D': 1},
#     'MetaSVM_pred': {'T': 0, 'D': 1},
#     'SIFT_pred': {'T': 0, 'D': 1},
#     'fathmm_MKL_coding_pred': {'D': 1, 'N': 0},
#     'LRT_pred': {'U': 0, 'N': 0, 'D': 1},
#     'MutationAssessor_pred': {'H': 1, 'N': 0, 'L': 0.25, 'M': 0.5},  
#     'MutationTaster_pred': {'D': 1, 'P': 0, 'N': 0, 'A': 1},
#     'PROVEAN_pred': {'D': 1, 'N': 0},
#     'Polyphen2_HDIV_pred': {'D': 1, 'B': 0, 'P': 0.5},
#     'Polyphen2_HVAR_pred': {'D': 1, 'B': 0, 'P': 0.5},
# }
# 
# 
# ```
# 

# In[19]:


prediction_mapping = {
    'FATHMM_pred': {'T': 0, 'D': 1},
    'MetaLR_pred': {'T': 0, 'D': 1},
    'MetaSVM_pred': {'T': 0, 'D': 1},
    'SIFT_pred': {'T': 0, 'D': 1},
    'fathmm_MKL_coding_pred': {'D': 1, 'N': 0},
    'LRT_pred': {'U': 0, 'N': 0, 'D': 1},
    'MutationAssessor_pred': {'H': 1, 'N': 0, 'L': 0.25, 'M': 0.5},  
    'MutationTaster_pred': {'D': 1, 'P': 0, 'N': 0, 'A': 1},
    'PROVEAN_pred': {'D': 1, 'N': 0},
    'Polyphen2_HDIV_pred': {'D': 1, 'B': 0, 'P': 0.5},
    'Polyphen2_HVAR_pred': {'D': 1, 'B': 0, 'P': 0.5},
}


def populate_variants_predictorstats_tables(db_file, filename):
    """
    See description in part 10
    """
    # YOUR CODE HERE
    
    # Init
    import re

    sql_state = ""
    predictor_values = {}
    
    var_col_str = "CHROM POS ID REF ALT QUAL FILTER" 
    var_col_list = re.split('\W+', var_col_str)

    info_col_list = """thousandg2015aug_all ExAC_ALL FATHMM_pred LRT_pred MetaLR_pred
    MetaSVM_pred MutationAssessor_pred MutationTaster_pred PROVEAN_pred Polyphen2_HDIV_pred Polyphen2_HVAR_pred
    SIFT_pred fathmm_MKL_coding_pred"""
    info_col_list = re.split('\W+', info_col_list)
    
    conn = create_connection(db_file)
    cur = conn.cursor()
    
    with conn:
        with gzip.open(filename,'rt') as fp:
            for line in fp:
                if re.match(r'#', line):
                    continue
                else:
                    # Init
                    var_list = [] #CHROM POS ID REF ALT QUAL FILTER
                    var_info_dict = {}
                
                    # Variants part
                    ##CHROM POS ID REF ALT QUAL FILTER
                    line_list = line.split('\t',7)
                    
                    for index in range(7):
                        var_list.append(line_list[index])
                
                    ## Info part and mapping val
                    var_info_dict = pull_info_values(line_list[7])

                    ## Insert data to table
                    CHROM, POS, ID, REF, ALT, QUAL, FILTER = [line_list[index] for index in range(7)]
                    values = build_values_list(CHROM, POS, ID, REF, ALT, QUAL, FILTER, var_info_dict)
                
                    var_rid = insert_variant(conn, values)
                
                    #Perdiction part
                    per_info_dict = pull_info_values(line_list[7])
                
                    for key in per_info_dict:
                        if per_info_dict[key] != '' and key in prediction_mapping:
                            per_info_dict[key] = prediction_mapping[key][var_info_dict[key]]
                            per_rid = insert_predictionstat(conn, (var_rid,key,prediction_mapping[key][var_info_dict[key]]))

                        elif per_info_dict[key] == '':
                            per_info_dict[key] = None
                        
                    
try:
    conn.close()
except:
    pass

db_file = 'lab2.db'
filename = 'test_4families_annovar.vcf.gz'
populate_variants_predictorstats_tables(db_file, filename)


# # Part 11 (10 Points)
# 
# Write a function that returns the total number of variants
# 
# 

# In[20]:


def num_of_total_variants(conn):
    # YOUR CODE HERE
    import pandas as pd
    
    sql = "select * from Variants;"
    df = pd.read_sql_query(sql, conn)
    return list(df.shape)[0]


# In[21]:


db_file = 'lab2.db'
conn = create_connection(db_file)
assert num_of_total_variants(conn) == 50001
conn.close()


# # Part 12 (10 Points)
# Write a function returns the total number of variant predictions -- the count of the predictiostats table. 

# In[22]:


def num_of_total_variant_predictions(conn):
    """
    Part 12
    """
    # YOUR CODE HERE
    import pandas as pd
    
    sql = "select * from PredictionStats;"
    df = pd.read_sql_query(sql, conn)
    return list(df.shape)[0]


# In[23]:


db_file = 'lab2.db'
conn = create_connection(db_file)
assert num_of_total_variant_predictions(conn) == 1324
conn.close()


# # Part 13 (10 Points)
# Return the total number of variant predictions that have value greater than zero. Number of values from the predictionstats table that are greater than 0. 

# In[24]:


def num_of_total_variant_predictions_with_value_gt_zero(conn):
    """
    See part 13 description
    """
    # YOUR CODE HERE
    import pandas as pd
    
    sql = "select * from PredictionStats where PredictorValue > 0;"
    df = pd.read_sql_query(sql, conn)

    return list(df.shape)[0]


# In[25]:


db_file = 'lab2.db'
conn = create_connection(db_file)
assert num_of_total_variant_predictions_with_value_gt_zero(conn) == 219
conn.close()


# # Part 14 (10 Points)
# 
# Write a function that given `CHROM, POS, ID, REF, ALT` returns a variant's info with the following columns (column order is important) :
# ```
# Variants.CHROM,
# Variants.POS,
# Variants.ID,
# Variants.REF,
# Variants.ALT,
# Variants.QUAL,
# Variants.FILTER,
# Variants.thousandg2015aug_all,
# Variants.ExAC_ALL,
# FATHMM_pred.prediction,
# LRT_pred.prediction,
# MetaLR_pred.prediction,
# MetaSVM_pred.prediction,
# MutationAssessor_pred.prediction,
# MutationTaster_pred.prediction,
# PROVEAN_pred.prediction,
# Polyphen2_HDIV_pred.prediction,
# Polyphen2_HVAR_pred.prediction,
# SIFT_pred.prediction,
# fathmm_MKL_coding_pred.prediction,
# sum(PredictionStats.PredictorValue)
# ```
# 
# For the predictions, return the actual text value. And the last column is the sum of all the mapped prediction scores for a given variant!
# 
# 

# In[26]:


def fetch_variant(conn, CHROM, POS, ID, REF, ALT):
    """
    See Part 14 description
    """
    # YOUR CODE HERE
    import pandas as pd
    
    fetch_list = ["CHROM", "POS", "ID", "REF", "ALT"]
    
    str_list = "FATHMM_pred LRT_pred MetaLR_pred MetaSVM_pred MutationAssessor_pred MutationTaster_pred PROVEAN_pred Polyphen2_HDIV_pred Polyphen2_HVAR_pred SIFT_pred fathmm_MKL_coding_pred"
    table_list = str_list.split(' ')
    
    sql = """select Variants.CHROM,
    Variants.POS,
    Variants.ID,
    Variants.REF,
    Variants.ALT,
    Variants.QUAL,
    Variants.FILTER,
    Variants.thousandg2015aug_all,
    Variants.ExAC_ALL,"""
    
    for table in table_list:
        sql = sql + "\n    " + table + ".prediction,"
    
    sql = sql + "\n    sum(PredictionStats.PredictorValue) \nfrom\n    Variants"
    for table in table_list:
        col_name = table + "ID" 
        sql = sql + "\n    LEFT OUTER JOIN " + table + " ON " + table + "." + col_name + " = Variants." + col_name 
    sql = sql + "\n    LEFT OUTER JOIN PredictionStats ON PredictionStats.VariantID = Variants.VariantID" 
    sql = sql + "\n where"
    for fetch in fetch_list:
        if type(eval(fetch)) == str:
            sql = sql + '(' + str(fetch) + " like '" + str(eval(fetch)) + "') AND "
        else:
            sql = sql + '(' + str(fetch) + " = " + str(eval(fetch)) + ') AND ' 
    sql = sql[0:len(sql)-4] + ";"
    
    
    df = pd.read_sql_query(sql, conn)
    return tuple(df.values[0])
    


# In[27]:


db_file = 'lab2.db'
conn = create_connection(db_file)
assert fetch_variant(conn, '22', 25599849, 'rs17670506', 'G', 'A') == ('22', 25599849, 'rs17670506', 'G', 'A', 3124.91, 'PASS', 0.0251597, 0.0425, 'D', 'D', 'T', 'T', 'M', 'D', 'D', 'D', 'D', 'D', 'D', 8.5)
conn.close()


# In[28]:


db_file = 'lab2.db'
conn = create_connection(db_file)
assert fetch_variant(conn, 'X', 2836184, 'rs73632976', 'C', 'T') == ('X', 2836184, 'rs73632976', 'C', 'T', 1892.12, 'PASS', None, 0.0427, 'D', 'U', 'D', 'T', 'M', 'P', 'D', 'P', 'P', 'D', 'D', 6.5)
conn.close()


# In[29]:


db_file = 'lab2.db'
conn = create_connection(db_file)
assert fetch_variant(conn, '5', 155935708, 'rs45559835', 'G', 'A') == ('5', 155935708, 'rs45559835', 'G', 'A', 1577.12, 'PASS', 0.0189696, 0.0451, 'D', 'D', 'T', 'T', 'L', 'D', 'D', 'P', 'B', 'T', 'D', 5.75)
conn.close()


# In[30]:


db_file = 'lab2.db'
conn = create_connection(db_file)
assert fetch_variant(conn, '4', 123416186, '.', 'A', 'G') == ('4', 123416186, '.', 'A', 'G', 23.25, 'PASS', None, None, None, None, None, None, None, None, None, None, None, None, None, None)
conn.close()


# # Part 15 (10 Points)
# Write a function that returns the variant with the highest predictor score sum. 
# 
# Return the variant info the following order:
# ```
# Variants.CHROM,
# Variants.POS,
# Variants.ID,
# Variants.REF,
# Variants.ALT,
# Variants.QUAL,
# Variants.FILTER,
# Variants.thousandg2015aug_all,
# Variants.ExAC_ALL,
# FATHMM_pred.prediction,
# LRT_pred.prediction,
# MetaLR_pred.prediction,
# MetaSVM_pred.prediction,
# MutationAssessor_pred.prediction,
# MutationTaster_pred.prediction,
# PROVEAN_pred.prediction,
# Polyphen2_HDIV_pred.prediction,
# Polyphen2_HVAR_pred.prediction,
# SIFT_pred.prediction,
# fathmm_MKL_coding_pred.prediction,
# sum(PredictionStats.PredictorValue)
#         
# ```
# 
# Again, return the predictor text values and the last column is the sum of the prediction values. 

# In[31]:


def variant_with_highest_sum_of_predictor_value(conn):
    """
    See part 15 description 
    """
    # YOUR CODE HERE
    import pandas as pd
    
    fetch_list = ["CHROM", "POS", "ID", "REF", "ALT"]
    
    str_list = "FATHMM_pred LRT_pred MetaLR_pred MetaSVM_pred MutationAssessor_pred MutationTaster_pred PROVEAN_pred Polyphen2_HDIV_pred Polyphen2_HVAR_pred SIFT_pred fathmm_MKL_coding_pred"
    table_list = str_list.split(' ')
    
    sql = """select Variants.CHROM,
    Variants.POS,
    Variants.ID,
    Variants.REF,
    Variants.ALT,
    Variants.QUAL,
    Variants.FILTER,
    Variants.thousandg2015aug_all,
    Variants.ExAC_ALL,"""
    
    for table in table_list:
        sql = sql + "\n    " + table + ".prediction,"
    
    sql = sql + "\n    sum(PredictionStats.PredictorValue) \nfrom\n    Variants"
    for table in table_list:
        col_name = table + "ID" 
        sql = sql + "\n    LEFT OUTER JOIN " + table + " ON " + table + "." + col_name + " = Variants." + col_name 
    sql = sql + "\n    LEFT OUTER JOIN PredictionStats ON PredictionStats.VariantID = Variants.VariantID" 
    sql = sql + "\n group by PredictionStats.VariantID order by sum(PredictionStats.PredictorValue) DESC limit 1;"    
    
    df = pd.read_sql_query(sql, conn)
    return tuple(df.values[0])


# In[32]:


db_file = 'lab2.db'
conn = create_connection(db_file)
assert variant_with_highest_sum_of_predictor_value(conn) == ('7', 87837848, '.', 'C', 'A', 418.25, 'PASS', None, None, 'T', 'D', 'D', 'D', 'H', 'D', 'D', 'D', 'D', 'D', 'D', 10.0)
conn.close()


# In[ ]:




