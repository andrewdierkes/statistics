#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-success">
# <b> Test results and statistical analysis of all cards sent by the screen printing team. This includes MESH 20, 31, 43, INK types A, E & I. With 3 bags of each Ink type (A/E) and two bags of I for each mesh type. <b> <\div>

# In[1]:


pip install pingouin


# In[2]:


pip install scikit_posthocs


# In[3]:


import re
import numpy as np
import pathlib as path
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats
from datetime import date
import pingouin as pg
from itertools import combinations


# In[62]:


_path = path.Path.cwd()

assay_name_dataset = []

dataset_dict = {}
qos_dataset_dict = {}
spatial_list = []


for _filepath in _path.iterdir():
    if _filepath.suffix != r'.tsv':
        continue
    elif _filepath.suffix == r'.tsv':
        
        with open(_filepath, 'r') as file:
            
            #individual assay data across 4 channels
            assay_dict = {}
            wl_index = []
            
            data = file.read()
            
            assay_name_finder = re.compile(r'@DES:.*')
            qos_finder = re.compile(r'(?<=QOS Optical Signal: )[0-9]+.[0-9]+')        
            
            assay_name = re.findall(assay_name_finder, data)
            qos_value = qos_finder.findall(data)
        
            assay_name_remove = re.compile(r'@DES:.')
            assay_list = [assay_name_remove.sub('', string) for string in assay_name]
            assay = assay_list[0]            
            assay_name_dataset.append(assay)
            
            position_finder = re.compile(r'[a-z]+(?=_[0-9]+)')
            position = position_finder.findall(assay)
            #qos data
            qos_dict = {}
            qos_dict['assay_name'] = assay
            qos_dict['position'] = ''.join(position)
            qos_dict['channels'] = (1400,2200,3000,3800)
            qos_dict['qos_value'] = [float(var) for var in qos_value]
            

            #find assays which are not correct
            for var in qos_dict['qos_value']:
                if var < 10000:
                    print(file,var)
                    
            qos_dataset_dict[f'{assay}'] = qos_dict

            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)      
          

            #else:
             #   print(f'number of scans if off from 80, check file {file}')


# In[63]:


def delist(args):
    '''delists a list of lists into 1 list'''
    delist = [var for small_list in args for var in small_list]
    return(delist) 


# In[64]:


#extract data on QOS values
qos_summary = []
for key, value in qos_dataset_dict.items():
    df_qos = pd.DataFrame(value)
    qos_summary.append(df_qos)


# In[73]:


#kruskal-wallis
def kruskal_array(df, groupby, feature):
    '''df = dataframe, groupby=iv,
    feature = dv'''
    regex = df.iloc[:,groupby].unique()
    
    groupby_name = df.columns[groupby]
    
    #sort dataframe to align with regex iterator below(min,max... basically need groupings in one space)
    df.sort_values(by=[groupby_name], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    #display(df)
    display(df.head(), df.tail())
    
    print(f'grouped by {groupby_name}')
    
    #find indexes of regex patterns
    idx_list = []
    for var in regex:
        idx = df[df.iloc[:,groupby] == var].index.to_list()
        idx_list.append(idx)
    
    for var in idx_list:
        print(f'length of group = {len(var)}')
        
    #iterate thru df using idx_list
    kw_list = []
    kw_dict = {}
    for var in idx_list:
        if len(var) == 0:
            pass
        else:
            _min = min(var)
            _max = max(var)
            chunk = len(var)
            offset = 0
            print(_min,_max,chunk)
            kw_list.append(np.array(df.iloc[_min:_max, feature]))
            kw_dict[f'{df.iloc[offset, groupby]}'] = np.array(df.iloc[_min:_max, feature])
          
    return kw_list
        


# In[65]:


#raw QOS channel data
df_dataset_qos = pd.concat(qos_summary).sort_index().reset_index(drop=True)
df_dataset_qos.sort_values(by=['assay_name', 'channels'], inplace=True)
display(df_dataset_qos)


# <div class="alert alert-warning alert-warning">
# <b> Analysis considering the raw qos value of each strip <b>
# </div>

# In[69]:


df_dataset_qos.boxplot(column=['qos_value'], by=['position'], ylabel='raw_qos_value')


# In[67]:


normality.__doc__
normality(df_dataset_qos,1,3)


# In[72]:


#qos values by position not guassian distributed
pg.normality(df_dataset_qos,dv='qos_value',group='position',method='shapiro')


# In[75]:


kruskal_array.__doc__
kruskal_array(df_dataset_qos,1,3)


# In[88]:


kw_qos = kruskal_array(df_dataset_qos,1,3)

result = stats.kruskal(kw_qos[0],kw_qos[1], kw_qos[2])
print(result)

if result[1] < 0.05:
    print(f'our p-val of {result[1]} is less than our significance value of 0.05. This means we can reject our null & know that the medians of k groups differ')
if result[1] > 0.05:
     print(f'our p-val of {result[1]} is greater than our significance value of 0.05. This means we cant reject our null & know that the medians of k groups come from similar populations')


# In[95]:


#POST HOC for non parametric test: Drunns with a bonferroni correction to reduce family wise error
import scikit_posthocs as sp

mesh_dunn = sp.posthoc_dunn(kw_qos, p_adjust = 'bonferroni')
display(mesh_dunn)

#if p-val < 0.05 -> reject null and disprove likeness... difference in groups median
print('sig-dif between 1 & 3 medians')


# In[9]:


def chunk_cv(df_col, chunk=4):
    '''This function will iterate over a df_col and return the average and stdev, using chunk_iterator.. so if you'd like the average & stdev values occuring every 3 rows... use 3 as your chunk iterator'''

    offset_mean = 0
    offset_stdev = 0
    
    number_list = [var for var in range(len(df_col))]
    
    dataset_average = []
    dataset_stdev = []
    
    
    while offset_mean < len(number_list):
        i_mean = number_list[offset_mean:chunk+offset_mean]
        average = df_col.iloc[i_mean].mean(axis=0)
        
        dataset_average.append(average)
        #dataset_array.append(_array)
        
        offset_mean += chunk
    
    while offset_stdev < len(number_list):
        i_stdev = number_list[offset_stdev:chunk+offset_stdev]
        stdev = df_col.iloc[i_stdev].std(ddof=1)
        
        dataset_stdev.append(stdev)
        
        offset_stdev += chunk
    
    return dataset_average, dataset_stdev


# In[10]:


def unique(df_col, chunk=4):
    '''This function will iterate over a df_col and return only unique values using chunk_iterator.. so if you'd like the unique value of something occuring every 3 rows... use 3 as your chunk iterator'''

    offset = 0
    
    number_list = [var for var in range(len(df_col))]
    
    dataset_array = []
    dataset_list = []
    
    while offset < len(number_list):
        i = number_list[offset:chunk+offset]
        _array = df_col.iloc[i].unique()
        
        dataset_array.append(_array)
        
        offset += chunk
    
    for var in dataset_array:
        dataset_list.append(var.tolist())
    
    unique = delist(dataset_list)
    
    return unique


# In[11]:


#find unique names from Assay Name row, slice each name to get rid of the rep number and then pass through unique function
#name_len = length of names we want to slice

def slice_name(df, row, name_len, end_offset):
    '''Function SLICE_NAME slices strings (usually assay name) that are similar in nature (df row). It works to 
    remove the ends (end_offset) which are different (usually rep number) to allow the UNIQUE function to work
    df = dataframe to use
    row = the index of the column you want to slice
    name_len = the length of each assay name, they all should be the same
    end_offset = what position you want to end with'''
    
    assay__name = [var for var in df.iloc[:,row]]

    unique_name = []


    for var in assay__name:
        if len(var) == name_len:
            unique_name.append(var[0:end_offset])

        else:
            print(file)
            print(f'labeling error for assay:{var}, length {len(var)}')
            for count, var2 in enumerate(var):
                print(count, var2)
    
    return pd.DataFrame(unique_name)


# In[43]:


def regex_cv(df,groupby,feature):
    '''regex_cv applies the inputted patterns on df_feature.
    df = dataframe, groupby = col # for where to iloc and search for the regex,
    feature = col # which you want to aggregate mean, stdev & cv'''
    
    regex = df.iloc[:,groupby].unique()
    
    groupby_name = df.columns[groupby]
    
    #sort dataframe to align with regex iterator below(min,max... basically need groupings in one space)
    df.sort_values(by=[groupby_name], inplace=True)
    df.reset_index(drop=True, inplace=True)

    display(df.head(), df.tail())
    
    print(f'grouped by {groupby_name}')
    #find indexes of regex patterns
    idx_list = []
    for var in regex:
        idx = df[df.iloc[:,groupby] == var].index.to_list()
        idx_list.append(idx)

    #iterate thru df using idx_list
    cv_list = []
    mean_list = []
    median_list = []
    stdev_list = []
    
    for var in idx_list:
        #print(var)
        if len(var) == 0:
            pass
        else:
            _min = min(var)
            _max = max(var)
            #print(_min,_max)
            chunk = len(var)
            offset = 0

            mean = df.iloc[_min:_max, feature].mean()
            median = df.iloc[_min:_max, feature].median()
            stdev = df.iloc[_min:_max, feature].std()
            
            cv = (stdev/mean)*100
    
            
            mean_list.append(mean)
            median_list.append(median)
            stdev_list.append(stdev)
            cv_list.append(cv)
            
    final = list(zip(cv_list,median_list,mean_list,stdev_list))
    df_final = pd.DataFrame(final, index=[regex], columns=['cv','mean','median','stdev'])
    display(df_final)
    
    return df_final


# In[28]:


def normality(df,groupby,feature):
    '''df=dataframe name, groupby=col # different categories were comparing(IV),
    feature=DV,'''
    
    import pylab
    
    regex = df.iloc[:,groupby].unique()
    
    groupby_name = df.columns[groupby]
    
    #sort dataframe to align with regex iterator below(min,max... basically need groupings in one space)
    df.sort_values(by=[groupby_name], inplace=True)
    df.reset_index(drop=True, inplace=True)

    display(df.head(),df.tail())
    
    print(f'grouped by {groupby_name}')
    #find indexes of regex patterns
    idx_list = []
    for var in regex:
        idx = df[df.iloc[:,groupby] == var].index.to_list()
        idx_list.append(idx)

    #iterate thru df using idx_list
    cv_list = []
    i =0
    for var in idx_list:
        if len(var) == 0:
            pass
        else:
            _min = min(var)
            _max = max(var)
            chunk = len(var)
            offset = 0
            print(_min,_max,type(_min))
                    #KDE & QQ plots
            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1, title=f"Grouping {''.join(str(regex[i]))}")
            sns.kdeplot(df.iloc[_min:_max,feature])
            plt.subplot(1,2,2)
            stats.probplot(df.iloc[_min:_max,feature],plot=pylab)
            plt.show()
            i += 1


# In[44]:


#based on qos data
q_avg, q_stdev = chunk_cv(df_dataset_qos.iloc[:,2], 4)
q_name = unique(df_dataset_qos.iloc[:,0],4)

qos_strip_dict = {}
qos_strip_dict['assay'] = q_name
qos_strip_dict['average_qos'] = q_avg
qos_strip_dict['stdev'] = q_stdev


df_qos_strip = pd.DataFrame(qos_strip_dict)


position_list = []
for var in df_qos_strip.iloc[:,0]:
    position_finder = re.compile(r'[a-z]+(?=_[0-9]+)')
    position = position_finder.findall(var)
    position_list.append(''.join(position))

df_qos_strip.insert(1, 'position', position_list)
df_qos_strip.insert(4, 'intrastrip_cv', round((df_qos_strip.iloc[:,3]/df_qos_strip.iloc[:,2])*100,4))
display(df_qos_strip)


# In[101]:


regex_cv.__doc__
df_qos = regex_cv(df_qos_strip, 1,4)
df_qos.drop(columns=['cv'],inplace=True)
df_qos.index.set_names('intrastrip_cv', inplace=True)
display(df_qos)


# In[58]:


df_dataset_qos.boxplot(column=['qos_value'],by='channels', figsize=(8,6), ylabel='qos_value')
df_qos_strip.boxplot(column=['average_qos'],by='position', figsize=(8,6), ylabel='qos_value')
df_qos_strip.boxplot(column=['intrastrip_cv'],by='position', figsize=(8,6), ylabel='intrastrip_cv')


# <div class="alert alert-warning alert-warning">
# <b> Analysis considering the intrastrip CV value of each strip <b>
# </div>

# In[96]:


#visualize distribution
normality(df_qos_strip,1,4)


# In[97]:


#test for guassian distribution
pg.normality(df_qos_strip, dv='intrastrip_cv',group='position',method='shapiro')


# In[33]:


#test for equal variance 
pg.homoscedasticity(df_qos_strip, dv='intrastrip_cv',group='position',method='levene')


# In[98]:


#Welch ANOVA
df_anova = pg.welch_anova(df_qos_strip, dv='intrastrip_cv',between='position')
display(df_anova)

for var in df_anova.iloc[:,4]:
    if var < 0.05:
        print('Reject Null, there is a stat-significant difference between means')
    else:
        print('Accept Null, no difference')


# In[99]:


#games-howell post-hoc
df_gh = pg.pairwise_gameshowell(df_qos_strip, dv='intrastrip_cv', between='position').round(5)
display(df_gh)


row_gh = 0
chunk_gh = 1

while row_gh < len(df_gh.iloc[:,0]):
    if df_gh.iloc[row_gh,8] < 0.05:
        print(f'The difference in group means of {df_gh.iloc[row_gh,0]} and {df_gh.iloc[row_gh,1]} are statistically significant')
    else:
        print(f'The difference in group means of {df_gh.iloc[row_gh,0]} and {df_gh.iloc[row_gh,1]} are not statistically significant')
    row_gh += chunk_gh


# <div class="alert alert-warning alert-warning">
# <b> Analysis considering the average qos value of each strip <b>
# </div>

# In[83]:


normality(df_qos_strip,1,2)
pg.normality(df_qos_strip,dv='average_qos',group='position',method='shapiro')


# In[85]:


#test for equal variance 
pg.homoscedasticity(df_qos_strip, dv='average_qos',group='position',method='levene')


# In[86]:


pg.anova(df_qos_strip,dv='average_qos',between='position')


# In[87]:


pg.pairwise_tukey(data=df_qos_strip,dv='average_qos',between='position')

