#!/usr/bin/env python
# coding: utf-8

# In[167]:


pip install pingouin


# In[168]:


pip install numpy-indexed


# In[169]:


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


# In[170]:


_path = path.Path.cwd()

assay_name_dataset = []
dataset_dict = {}
spatial_list = []

for _filepath in _path.iterdir():
    if _filepath.suffix != r'.log':
        continue
    elif _filepath.suffix == r'.log':
        
        with open(_filepath, 'r') as file:
            
            #individual assay data across 4 channels
            assay_dict = {}
            wl_index = []
            
            data = file.read()
            
            assay_name_finder = re.compile(r'@DES:.*')
            pixelData_finder = re.compile(r'pixelData.*]')
            numberPixel_finder = re.compile(r'numberPixels=[0-9][0-9]')           
            beginPosition_finder = re.compile(r'beginPosition.*[0-9]')
            numberOfSteps_finder = re.compile(r'numberOfSteps.*[0-9]')
            stepSize_finder = re.compile(r'stepSize.*[0-9]')
            spd_finder = re.compile(r'COMMAND.*Sent:.*INS.*SPD.*')
            windowStartWavelength_finder = re.compile(r'windowStartWavelength.*[0-9] ')
            windowEndWavelength_finder = re.compile(r'Sent INS SSCPARAM.*[0-9]')
                                                      
            
            assay_name = re.findall(assay_name_finder, data)
            pixelData = re.findall(pixelData_finder, data)
            numberPixel = re.findall(numberPixel_finder, data)
            beginPosition = re.findall(beginPosition_finder, data)
            numberOfSteps = re.findall(numberOfSteps_finder, data)
            stepSize = re.findall(stepSize_finder, data)
            windowStartWavelength = re.findall(windowStartWavelength_finder, data)
            windowEndWavelength = re.findall(windowEndWavelength_finder, data)
            spd = re.findall(spd_finder, data)
            
            #find specific names of assay
            #print(assay_name, file)
            
            
            #number of scans you hope to take
            #scrub & transform data into float
            #if len(pixelData) == 80:    
            pixel_data_remove = re.compile(r'pixelData:.\[')
            pixel_data_removal = re.compile(r'\]')

            number_pixel_removal = re.compile(r'numberPixels=')
            begin_position_removal = re.compile(r'beginPosition.*=..')
            number_of_steps_removal = re.compile(r'numberOfSteps.*=..')
            step_size_removal = re.compile(r'stepSize.*=..')
            window_start_wl_removal = re.compile(r'windowStartWavelength=')
            spd_removal = re.compile('COMMAND\tSent: INS ')
            assay_name_remove = re.compile(r'@DES:.')


            pixel_data_semi = [pixel_data_remove.sub('', string) for string in pixelData]
            pixel_data = [var.split(', ') for var in [pixel_data_removal.sub('', string) for string in pixel_data_semi]]

            pixel_dataset = []

            for scan in pixel_data:
                scan_float = []
                for var in scan:
                    try:
                        scan_float.append(float(var))
                    except ValueError as e:
                            f'Cannot convert {var} to float'
                pixel_dataset.append(scan_float)
            
            

            #converting into floats and strings for parsing 
            number_pixel_list = [int(var) for var in [number_pixel_removal.sub('', string) for string in numberPixel]]
            #print(number_pixel_list, file)
            begin_position = [int(var) for var in [begin_position_removal.sub('', string) for string in beginPosition]]
            number_of_steps_list = [int(var) for var in [number_of_steps_removal.sub('', string) for string in numberOfSteps]]
            step_size_list = [int(var) for var in [step_size_removal.sub('', string) for string in stepSize]]
            window_start_wl_list = [float(var) for var in [window_start_wl_removal.sub('', string) for string in windowStartWavelength]]
            SPD = [spd_removal.sub('', string) for string in spd]
            window_end_wl_string = ''.join(windowEndWavelength)
            #ending WL edited as changing IT will effect the result (Old code: int(window_end_wl_string[40:43])
            window_end_wl = 650
            assay_list = [assay_name_remove.sub('', string) for string in assay_name]
                
            number_pixel = int(number_pixel_list[0])
            
            
            
            number_of_steps = int(number_of_steps_list[0])
            step_size = int(step_size_list[0])
            window_start_wl = int(window_start_wl_list[0])
            assay = assay_list[0]
            
            assay_name_dataset.append(assay)

            #using generator; divide scans into groups of 20 for each channel
            def chunks(lst, chunk):
                for i in range(0, len(lst), chunk):
                    yield lst[i:i+chunk]

            #unpack with generator into a list with *
            pixel_channels = [*chunks(pixel_dataset, 20)]
            pixel_ch0 = pixel_channels[0]
            pixel_ch1 = pixel_channels[1]
            pixel_ch2 = pixel_channels[2]
            pixel_ch3 = pixel_channels[3]
            

            #create spatial read values from data within assay
            def channel_chunk(ch_pos, step_size, num_steps):
                '''iterate thru spatial coords using ch_pos we start with, step size and num_steps from assay.. range(start,stop,interval)'''
                spatial_lister = []
                for i in range(ch_pos, ch_pos+(step_size*num_steps), step_size):
                                spatial_lister.append(i)
                return spatial_lister

            #spatial locations of assay used for lambda plotter...
            spatial_list.append(channel_chunk(int(begin_position[0]), step_size, number_of_steps))
            spatial_list.append(channel_chunk(int(begin_position[1]), step_size, number_of_steps))
            spatial_list.append(channel_chunk(int(begin_position[2]), step_size, number_of_steps))
            spatial_list.append(channel_chunk(int(begin_position[3]), step_size, number_of_steps))

            #create index using starting and ending wl and number of pixels in each scan array; adding outside of loop
            wl_step = ((window_end_wl-window_start_wl)/number_pixel)
            start_wl = (window_start_wl-wl_step)
            #ending WL edited as changing IT will effect the result (Old code=int(window_end_wl_string[40:43])-wl_step)
            end_wl = 650-wl_step

            offset = start_wl 

            while offset < end_wl:

                i = offset + wl_step
                wl_index.append(i)
                offset += wl_step

                #add spectral and spatial data to outside of log reading script
            assay_dict[f'{assay} ch0'] = dict(zip(spatial_list[0], pixel_ch0))
            assay_dict[f'{assay} ch1'] = dict(zip(spatial_list[1], pixel_ch1))
            assay_dict[f'{assay} ch2'] = dict(zip(spatial_list[2], pixel_ch2))
            assay_dict[f'{assay} ch3'] = dict(zip(spatial_list[3], pixel_ch3))

            dataset_dict[f'{assay}'] = assay_dict

            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)      
            

            #else:
             #   print(f'number of scans if off from 80, check file {file}')


# In[171]:


data_dict = {}

def lambda_plotter(df, wl_index, channel_sp, graph_channel):
    ch_max = [df.max(axis=1)]
    ch_spatial = channel_sp
    
    #iterate thru max list 
    for var in ch_max:
        lambda_max = (max(var))

    lmax_index = []

    #iterate thru df columns looking for the lamda max value
    for col in df.columns:
        index_lookup = df[df[col] == lambda_max].index.tolist()
        try:
        #if a variable exists... appended
            lmax_index.append(index_lookup[0])
        except:
            pass

    var = float(lmax_index[0])
    
    lambda_max_row = wl_index.index(var)
    
    #avg function
    def average(data, length):
        avg = sum(data)/len(data)
        return avg
    
    #create an average list similar to plateau if it is within 10% of the RFU
    RFU_max_list = []
    for z in df.iloc[lambda_max_row]:
        if z >= lambda_max*.92:
            RFU_max_list.append(z)
    
    num_spatial_pos = len(RFU_max_list)
    
    average = average(RFU_max_list, num_spatial_pos)

    
    #take the spatial data across the max wavelength (df.iloc) and assign variables as arrays for find_peaks
    x = np.array(channel_sp)
    y = np.array(df.iloc[lambda_max_row])

    #scipy peak finder; use 1d np.array and height = min value for peak 
    peaks = find_peaks(y, height = 2000, distance=10)
    peak_pos = x[peaks[0]]
    height = peaks[1]['peak_heights']
    
    
    
    #plot spatial vs RFU at max wl & maxima via scipy
    #fig, ax = plt.subplots()
    #plt.scatter(ch_spatial, y)
    #plt.scatter(peak_pos, height, color = 'r', s = 30, marker = 'D', label = 'maxima')
    

    #ax.set(xlabel='Spatial Position', ylabel='RFU', title= f'Formulation {graph_channel} Spatial RFU & Lambda Max at {round(var, 3)}')
    #ax.legend()
    #ax.grid()
    
    data_dict['Assay Name'] = graph_channel
    data_dict['Max RFU'] = height
    data_dict['Average +/- 8%'] = average 
    data_dict['Spatial Positions Taken for Average'] = num_spatial_pos 
    data_dict['Lambda Max'] = var
    data_dict['Spatial Position for Max RFU'] = peak_pos
    
    #return f'Max RFU output {lambda_max}, seen at wavelength: {var}'


# In[172]:


def heat_map(dataframe):
    hm = sns.heatmap(dataframe, cmap='YlOrRd')


# In[173]:


def delist(args):
    delist = [var for small_list in args for var in small_list]
    return(delist) 

#generate a list of the assay names pairing number of repeats with number of channels
assay_single = []
for assay in dataset_dict.keys():
    assay_single.append(np.repeat(assay, 4))
    
assay_list = delist(assay_single)

#for lambda plotter title
channel = [1400,2200,3000,3800]
channel_assay = channel*(len(dataset_dict.keys()))


#create an iterator to go thru assay_list and pair the assay_name with the four corresponding dataframes
assay_chunk = 1
offset = 0


summary_list = []

#iterate thru dataset_dict getting into specific channels to create dataframes & link assay_name with DF
for assay in dataset_dict.values():
    for channel in assay.values():
        
        df = pd.DataFrame(channel, index=wl_index)
        df.index.name = assay_list[offset]
        
        
        #list_of_data.append(df)
        
        #display(df)
        lambda_plotter(df, wl_index, spatial_list[offset], assay_list[offset])
        df_summary = pd.DataFrame(data_dict)
        summary_list.append(df_summary)
        
        offset += assay_chunk
        
        #display(heat_map(df))
        
        #display(hm = sns.heatmap(df, cmap='YlOrRd'))
        

        


# In[174]:


#generate dataframe of individual channels 
df_dataset_sum = pd.concat(summary_list).sort_index().reset_index(drop=True)
df_dataset_sum.sort_values(by=['Assay Name', 'Spatial Position for Max RFU'], inplace=True)
#display(df_dataset_sum)


# In[175]:


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


# In[176]:


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


# In[177]:


#generate intrastrip data, avg, stdev & cv using previous dataframe with all channel values
strip = unique(df_dataset_sum['Assay Name'])
strip_avg, strip_stdev = chunk_cv(df_dataset_sum['Max RFU'], 4)
strip_avg_avg, strip_avg_stdev = chunk_cv(df_dataset_sum['Average +/- 8%'], 4)

#generate dictionary for pd.df
strip_dict = {}
strip_dict['Strip Name'] = strip
strip_dict['Intra Strip Average RFU'] = strip_avg
strip_dict['Intra Strip StDev'] = strip_stdev
strip_dict['Intra Strip Average +/- 8% Average'] = strip_avg_avg
strip_dict['Intra Strip Average +/- 8% StDev'] = strip_avg_stdev



# In[178]:


#df for intrastrip avg & stdev, inserting CVs
df_strip = pd.DataFrame(strip_dict)
df_strip.insert(3, 'Intra Strip CV, n=5', round(((df_strip['Intra Strip StDev']/df_strip['Intra Strip Average RFU'])*100),4))
df_strip.insert(6, 'Intra Strip Average +/- 8% CV, n=5', round(((df_strip['Intra Strip Average +/- 8% StDev']/df_strip['Intra Strip Average +/- 8% Average'])*100),4))
display(df_strip)


# In[179]:


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


# In[180]:


#take avg CV for each strip

unique_strip_slicer = slice_name(df_strip, 0, 16, -3)

unique_strip = unique(unique_strip_slicer.iloc[:,0], 3)

df_strip_cv_avg, df_strip_cv_stdev = chunk_cv(df_strip.iloc[:,3], 3)

df_strip_cv_data = {}
df_strip_cv_data['Strip Name'] = unique_strip
df_strip_cv_data['Average CV'] = df_strip_cv_avg


pd.DataFrame(df_strip_cv_data)


# In[181]:


#should equal number of assays ran
print(len(df_strip['Strip Name']))


# In[182]:


#generate pd.df for repeat averages for strips ran

#pass previous df through functions to generate mean & stdev
repeat_strip_avg, repeat_strip_stdev = chunk_cv(df_strip['Intra Strip Average RFU'], 3)
repeat_strip_avg_avg, repeat_strip_avg_stdev = chunk_cv(df_strip['Intra Strip Average +/- 8% Average'], 3)

#generate dictionary for pd.df
repeat_strip_dict = {}
repeat_strip_dict['Strip Name'] = unique_strip
repeat_strip_dict['Strip Repeat Average RFU'] = repeat_strip_avg
repeat_strip_dict['Strip Repeat StDev'] = repeat_strip_stdev
repeat_strip_dict['Strip Repeat Average +/- 8% Average'] = repeat_strip_avg_avg
repeat_strip_dict['Strip Repeat Average +/- 8% StDev'] = repeat_strip_avg_stdev

#create pd.df & insert cv values
df_repeat_strip = pd.DataFrame(repeat_strip_dict)
df_repeat_strip.insert(3, 'Strip Repeat CV', round(((df_repeat_strip['Strip Repeat StDev']/df_repeat_strip['Strip Repeat Average RFU'])*100),4))
df_repeat_strip.insert(6, 'Strip Repeat Average +/- 8% CV', round(((df_repeat_strip['Strip Repeat Average +/- 8% StDev']/df_repeat_strip['Strip Repeat Average +/- 8% Average'])*100),4))
display(df_repeat_strip)


# In[183]:


formulation_repeat = slice_name(df_repeat_strip, 0, 13, 5)
df_repeat_strip['Formulation'] = formulation_repeat
display(df_repeat_strip)

display(df_repeat_strip.boxplot(column = 'Strip Repeat CV', by = 'Formulation', ylabel = 'CV', figsize=(8,6)))
display(sns.kdeplot(data=df_repeat_strip, x= 'Strip Repeat CV', hue= 'Formulation',  fill = False, ax = ax))


# In[184]:


#test & visualize for normal distribution
fig, ax = plt.subplots()
sns.kdeplot(data=df_repeat_strip, x= 'Strip Repeat CV', hue= 'Formulation',  fill = False, ax = ax);
ax.grid()


pg.normality(df_repeat_strip, dv='Strip Repeat CV', group= 'Formulation', method='shapiro')


# In[185]:


#test for equal varience
pg.homoscedasticity(data=df_repeat_strip, dv='Strip Repeat CV', group='Formulation', method='levene')


# In[186]:


#normal anova bc normally distributed and equal varience
df_repeat_strip_anova = pg.anova(data=df_repeat_strip, dv='Strip Repeat CV', between = 'Formulation')
display(df_repeat_strip_anova)


for var in df_repeat_strip_anova.iloc[:,4]:
    if var < 0.05:
        print('Reject Null, there is a stat-significant difference between means')
    else:
        print('Accept Null, meaning...no difference in the average strip repeat CVs amongst different levels of CB%')

# In[189]:


#Tukeys: best for equal varience and normal distribution, post hoc, with no control
df_tukey= pg.pairwise_tukey(data=df_repeat_strip, dv='Strip Repeat CV', between='Formulation')
display(df_tukey)
row_tukey = 0
chunk_tukey = 1

while row_tukey < len(df_tukey.iloc[:,0]):
    if df_tukey.iloc[row_tukey,7] < alpha_tukey:
        print(f'The difference in group means of {df_tukey.iloc[row_tukey,0]} and {df_tukey.iloc[row_tukey,1]} are statistically significant')
    else:
        print(f'The difference in group means of {df_tukey.iloc[row_tukey,0]} and {df_tukey.iloc[row_tukey,1]} are not statistically significant')
    
    row_tukey += chunk_tukey


# In[190]:


#using SLICE_NAME and UNIQUE we can create bag labeling column

bag_name = slice_name(df_repeat_strip, 0, 13, 10)
unique_bag = unique(bag_name.iloc[:,0], 5)
#display(bag_name)


# In[191]:


#generate pd.df for repeat averages for strips ran

#pass previous df through functions to generate mean & stdev
bag_avg, bag_stdev = chunk_cv(df_repeat_strip['Strip Repeat Average RFU'], 5)
bag_avg_avg, bag_avg_stdev = chunk_cv(df_repeat_strip['Strip Repeat Average +/- 8% Average'], 5)

#generate dictionary for pd.df
bag_dict = {}
bag_dict['Bag Name'] = unique_bag
bag_dict['Bag Average RFU'] = bag_avg
bag_dict['Bag StDev'] = bag_stdev
bag_dict['Bag Average +/- 8% Average'] = bag_avg_avg
bag_dict['Bag Average +/- 8% StDev'] = bag_avg_stdev

#for var in bag_dict.values():
 #   print(len(var), var)

#create pd.df & insert cv values
df_bag = pd.DataFrame(bag_dict)
df_bag.insert(3, 'Bag CV', round(((df_bag['Bag StDev']/df_bag['Bag Average RFU'])*100),4))
df_bag.insert(6, 'Strip Repeat Average +/- 8% CV', round(((df_bag['Bag Average +/- 8% StDev']/df_bag['Bag Average +/- 8% Average'])*100),4))
#display(df_bag)


# In[192]:


#using SLICE_NAME and UNIQUE we can create bag labeling column
formulation_name = slice_name(df_bag, 0, 10, -5)
unique_formulation = unique(formulation_name.iloc[:,0], 2)


# In[193]:


#prep for anova test
#Ho = groups have no difference in avg CV
#Ha = groups have difference in avg CV
df_bag['Formulation Name'] = formulation_name
#display(df_bag)
iv = 'Formulation Name'
dv = 'Intra Strip CV, n=5'
#df_strip.groupby(iv).mean(numeric_only=True)


# In[194]:


#Generate histogram to display data 
form_name = slice_name(df_strip, 0, 16, -11)
df_strip['Formulation Name'] = form_name

df_strip.boxplot(column = 'Intra Strip CV, n=5', by= 'Formulation Name', figsize = (8,6));


# In[195]:


#generate KDE plot to visualize distribution of data, x being the DV and hue being the different conditions
fig, ax = plt.subplots()
sns.kdeplot(data= df_strip, x = 'Intra Strip CV, n=5', hue = 'Formulation Name', fill = False, ax = ax)

#test for normality using Shapiro-Wilk method to ensure data is normally distributed
#W = test stat, pval = pval, normal = True if data is normally distributed
nm_pg = pg.normality(df_strip, dv = 'Intra Strip CV, n=5', group = 'Formulation Name', method = 'shapiro')
display(nm_pg)


#sp_shapiro = stats.shapiro(df_strip


# In[242]:


sns.histplot(x='Intra Strip CV, n=5', data=df_strip, hue='Formulation Name', bins=len(df_strip), stat="density",
             element="step", fill=False, cumulative=True, common_norm=False);
plt.title("Cumulative distribution function");


# In[196]:


#test for homoscedasticity (equal variance) using levene test, ANOVA assumes homoscedascity
pg.homoscedasticity(df_strip, dv = 'Intra Strip CV, n=5', group = 'Formulation Name', method = 'levene')


# In[197]:


#homosedasticity false.. use welch anova which is more robust
df_anova = pg.welch_anova(df_strip, dv = 'Intra Strip CV, n=5', between = 'Formulation Name')


display(df_anova)

for var in df_anova.iloc[:,4]:
    if var < 0.05:
        print('Reject Null, there is a stat-significant difference between means')
    else:
        print('Accept Null, no difference')


# In[209]:


gh = pg.pairwise_gameshowell(df_strip, dv = 'Intra Strip CV, n=5', between = 'Formulation Name').round(5)
display(gh)

row_gh = 0
chunk_gh = 1

while row_gh < len(gh.iloc[:,0]):
    if gh.iloc[row_gh,8] < 0.05:
        print(f'The difference in group means of {gh.iloc[row_gh,0]} and {gh.iloc[row_gh,1]} are statistically significant')
    else:
        print(f'The difference in group means of {gh.iloc[row_gh,0]} and {gh.iloc[row_gh,1]} are not statistically significant')
    row_gh += chunk_gh


# In[237]:


#post-hoc testing to determine which group differences are statistically significant
from statsmodels.stats.libqsturng import qsturng, psturng
import math 


#Confidence Interval
i = 0
chunk = 1

group_comparison = []
lower_CI = []
upper_CI = []

#k = total num groups

k = len(df_strip['Formulation Name'].unique())
alpha = 0.05

while i < len(gh.iloc[:,0]):
    row = i
    
    lower_conf = gh.iloc[row, 4] - qsturng(1-alpha, k, gh.iloc[row,7])
    upper_conf = gh.iloc[row, 4] + qsturng(1-alpha, k, gh.iloc[row,7])
    lower_CI.append(lower_conf)
    upper_CI.append(upper_conf)
    group = gh.iloc[row, 0] + ' : ' + gh.iloc[row, 1]
    group_comparison.append(group)
    
    i += chunk

#make a y placeholder for each CI
y_graph = range(len(group_comparison))

#zip all of the CI data
zip_CI = zip(group_comparison, y_graph, lower_CI, upper_CI)

#make plot to display CI; if passes thru 0 
fig, ax = plt.subplots()

for g, y, l, u in zip_CI:
    x = [l,u]
    yf = np.repeat(y,2)
    
    plt.plot(x,yf,label=g)
    ax.set(xlabel='Confidence Interval', title='95% CI for CB% Intrastrip Group CV Mean Comparison')
    ax.legend(bbox_to_anchor=(1,1.05))
    
##WRONG->"Having zero in one's confidence interval implies that a treatment effect has no effect." : RIGHT -> This is often how such confidence intervals are interpreted, but this is a mistake. A confidence interval that contains zero is not certainty that there is no treatment effect, but that it is uncertain whether there is a treatment effect. Having zero in one's confidence interval implies \ that a treatment effect could have a positive or negative effect on the outcome of interest. (DataCamp).


# In[220]:


formulation_avg, formulation_stdev = chunk_cv(df_bag.iloc[:,1],2)

formulation_dict = {}
formulation_dict['Formulation Name'] = unique_formulation
formulation_dict['Formulation Average (3 bags)'] = formulation_avg
formulation_dict['Formulation StDev (3 bags)'] = formulation_stdev

df_formulation = pd.DataFrame(formulation_dict)
df_formulation.insert(3, 'Formulation CV', round(((df_formulation['Formulation StDev (3 bags)']/df_formulation['Formulation Average (3 bags)'])*100), 4))
display(df_formulation)


# In[ ]:





# In[19]:


date = date.today()
with pd.ExcelWriter(f'{date}.xlsx') as writer:
    df_dataset_sum.to_excel(writer, sheet_name='Channels')
    df_strip.to_excel(writer, sheet_name='Strip')
    df_repeat_strip.to_excel(writer, sheet_name='Strip Repeat')
    df_bag.to_excel(writer, sheet_name='Bag')
    df_formulation.to_excel(writer, sheet_name='Formulation')


# In[ ]:




