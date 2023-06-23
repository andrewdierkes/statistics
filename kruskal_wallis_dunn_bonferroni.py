#!/usr/bin/env python
# coding: utf-8

# In[3]:


#kruskal-wallis
def kruskal_array(df, groupby, feature):
    '''df = dataframe, groupby=iv,
    feature = dv'''
    import pandas as pd
    from scipy import stats
    
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
        


# <div class="alert alert-block alert-warning">
# <b> Next we take a look at the KW hypothesis test seeing if there is a difference in medians for the mesh categories <b>
# </div>

# In[122]:


kw_mesh = kruskal_array(df_spatial_strip, 2, 8)

result = stats.kruskal(kw_mesh[0],kw_mesh[1], kw_mesh[2])
print(result)

if result[1] < 0.05:
    print(f'our p-val of {result[1]} is less than our significance value of 0.05. This means we can reject our null & know that the medians of k groups differ')
if result[1] > 0.05:
     print(f'our p-val of {result[1]} is greater than our significance value of 0.05. This means we cant our null & know that the medians of k groups come from similar populations')


# <div class="alert alert-block alert-warning">
# <b> We get a H value of 19.29 and a pval < 0.05. Because we have a low chance of the medians being from the same population distribution; we reject the null hypothesis, we know there are groups which differ and we'd like to see which groups differ from others so we run the dunn post hoc with a bonferroni p-value adjustment (to reduce family wise error) <b>
# </div>

# In[123]:


#POST HOC for non parametric test: Drunns with a bonferroni correction to reduce family wise error
import scikit_posthocs as sp

mesh_dunn = sp.posthoc_dunn(kw_mesh, p_adjust = 'bonferroni')
display(mesh_dunn)


#if p-val < 0.05 -> reject null and disprove likeness... difference in groups median
print('significant difference between groups 1 & 2 as well as 3 & 2. Proving 2 is the odd one out')


# In[ ]:




