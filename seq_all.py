# takes a csv file 'data_sorted.csv' containing SPAT messages as an input from the same folder
# and gives 'seq_G2R_output.csv' as output file in the output folder
# the output file contains a ranking of every signal group of the input
# and its performance data based on green to red predictions

# these libraries are necessary
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os
import pathlib

warnings.filterwarnings('ignore')

# get current path
path = os.path.dirname(os.path.realpath(__file__))

print('Reading all .csv files, this can take a while...')

# get all csv files in current path
all_files = glob.glob(os.path.join(path, "*.csv"))

# concatenate all data and load as dataframe
df_from_each_file = (pd.read_csv(f) for f in tqdm(all_files))
df = pd.concat(df_from_each_file, ignore_index=True)

#filtering out useless data
df = df[df.state != 'unavailable']
df = df[df.state != 'caution-Conflicting-Traffic']

# convert to datetime format
df['received_timestamp'] = pd.to_datetime(df['received_timestamp'], unit='ms')
df['state_timestamp'] = pd.to_datetime(df['state_timestamp'], unit='ms')
df['state_end_min'] = pd.to_datetime(df['state_end_min'], unit='ms')
df['state_end_max'] = pd.to_datetime(df['state_end_max'], unit='ms')
df['state_end_likely'] = pd.to_datetime(df['state_end_likely'], unit='ms')

# grouping of signalgroups
signalgroups = [d for _, d in df.groupby('id')]

# resetting index
signalgroups = [d.reset_index() for d in signalgroups]

# tijd function makes a new column containing the timestamp of the next sequence change
def tijd(signalgroup):
    
    # make dataframe with a new interval each row
    df1 = signalgroup.loc[(signalgroup.ischanged == True)]
    
    # create real time column and fill timestamps inbetween
    df1['time_s'] = np.nan
    df1["time_shift"] = df1["state_timestamp"].shift(-1).replace()
    signalgroup['time_s'] = np.nan    
    signalgroup.loc[df1.index,'time_s'] = df1['time_shift']
    signalgroup['time_s'] = signalgroup['time_s'].fillna(method='ffill')
    signalgroup['time_s'] = pd.to_datetime(signalgroup['time_s'])

    # check that the sequence time is not later than the state timestamp
    signalgroup['difference'] = (signalgroup['time_s'] - signalgroup['state_timestamp']).dt.total_seconds()
    signalgroup.drop(signalgroup[signalgroup.difference < 0].index, inplace=True)
    signalgroup.drop('difference', axis = 1, inplace=True)
    return signalgroup

print()
print("Preparing DataFrames...")
# creating dataframes per signalgroup
signals = ['permissive-clearance', 
           'caution-Conflicting-Traffic', 
           'protected-Movement-Allowed', 
           'permissive-Movement-Allowed', 
           'pre-Movement', 
           'protected-clearance']

id_list = []
groups = []
minmaxgroups = []
for signalgroup in tqdm(signalgroups):
    # make a list with all signal group IDs for later use
    id_list.append(signalgroup['id'].to_list()[0])

    # change all state values to "green" and "red"
    signalgroup['state'] = np.where(signalgroup['state'].isin(signals), "green", "red")

    # shift the states to search for sequences    
    signalgroup["ischanged"] = signalgroup[
        "state"
            ].shift(1, fill_value=signalgroup["state"].head(1)) != signalgroup["state"]

    # put every signalgroup through the tijd() function
    signalgroup = tijd(signalgroup)
    minmax = signalgroup
    minmaxgroups.append(minmax)

    # drop all row with no state_end_likely and create column indicating new sequence
    signalgroup = signalgroup.dropna(subset=['state_end_likely'])
    signalgroup["new_sequence"] = signalgroup["time_s"].shift(1) != signalgroup["time_s"]
    groups.append(signalgroup)

#resetting index
groups = [d.reset_index() for d in groups]

# adding weights to the timestamp
def weights(group):
    # create arrays for weights
    group['weights'] = group['weights'].fillna(0)
    vector = group['weights'].to_numpy()
    indeces = group.index[group['new_sequence'] == True].tolist()

    # check if there are sequences
    if len(indeces) == 0:
        return group

    # remove the first indecece if 0
    if indeces[0] == 0:
        indeces = np.delete(indeces, 0)

    # split the weights per sequence
    sequences = np.split(vector, indeces, axis=0)
    weights = []

    # normalize weights per sequence
    for seq in sequences:
        n = seq/seq.sum()
        weights.append(n)

    # add weights to dataframe
    weights = np.array(weights)
    weights = np.concatenate(weights).ravel()
    group['weights'] = weights

    return group


# initiate columns
errors = {}
stds = {}
means = {}
nums = {}
seq = {}
maxaccs = {}
minaccs = {}

print()
print("Ranking data")
# Calculate difference and error
for i, signal in tqdm(enumerate(groups), total=len(groups)):

    # check if signal has predictions
    if len(signal.index) < 1:
        error = float('inf')
        signal_id = id_list[i]
        errors[signal_id] = error
        standard_D = float('inf')
        stds[signal_id] = standard_D
        means[signal_id] = float('inf')
        nums[signal_id] = 0
        maxaccs[signal_id] = 0
        minaccs[signal_id] = 0
    else:
        # difference time, switch time.
        signal["weights"] = signal['time_s'] - signal['state_timestamp']
        signal['weights'] = (abs(signal["weights"].dt.total_seconds()))
        signal.drop(signal[signal.weights > 300].index, inplace=True)
        signal["weights"] =  1/ signal["weights"]
        
        # update weights
        signal = weights(signal)
        
        # difference pred, real
        signal["difference"] = (signal['time_s'] - signal['state_end_likely']) 
        signal["difference"] = signal["difference"].dt.total_seconds() * signal["weights"]
        signal.drop(signal[abs(signal.difference) > 3000].index, inplace=True)

        
        # number of sequences used
        signal_id = signal['id'].to_list()[0]
        seq[signal_id] = (signal.new_sequence).sum()
        
        #compute error
        arr = signal["difference"]**2
        error = arr.sum()/seq[signal_id]
        errors[signal_id] = error
        
        # standard deviation of difference
        standard_D = signal['difference'].std()
        stds[signal_id] = standard_D
        
        # mean of difference
        means[signal_id] = signal['difference'].mean()
        
        # number of messages used
        nums[signal_id] = len(signal['difference'])
        
        
        
        # calculate the amount of min and max predictions that were true
        df = minmaxgroups[i].dropna(subset=['state_end_min'])
        if len(df.index)<1:
            minaccs[signal_id] = 0
        else:
            df['min'] = (df['state_end_min'] <= df['time_s'])
            minaccs[signal_id] = f"{df['min'].value_counts()[True]}/{len(df.index)}"
        df = minmaxgroups[i].dropna(subset=['state_end_max'])
        if len(df.index)<1:
            maxaccs[signal_id] = 0
        else:
            df['max'] = (df['state_end_max'] >= df['time_s'])
            maxaccs[signal_id] = f"{df['max'].value_counts()[True]}/{len(df.index)}"

# ranking
ranked = dict(sorted(errors.items(), key=lambda item: item[1]))

# final dataframe with outputs
outputdf = pd.DataFrame.from_dict(ranked, orient='index', columns=['MSE'])
outputdf["STD diff"] = pd.Series(stds)
outputdf["Mean diff"] = pd.Series(means)
outputdf["Good"] = abs(outputdf["Mean diff"])+2*outputdf["STD diff"]
outputdf["NumPred"] = pd.Series(nums)
outputdf["NumSeq"] = pd.Series(seq)
outputdf["MaxAcc"] = pd.Series(maxaccs)
outputdf["MinAcc"] = pd.Series(minaccs)

#data to csv
folderName = 'Output'
if not os.path.exists(os.getcwd() + '/' + folderName):
    os.makedirs(os.getcwd() + '/' + folderName, exist_ok=True) 

outputdf.to_csv('Output/seq_all_output.csv')

print("Done")
