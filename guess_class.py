import pandas as pd
import numpy as np

def guess_topics (word_matrix):
    word_matrix = word_matrix.iloc[:24]
    folders = ['C1','C4','C7']
    classes = dict()

    grouper = [next(p for p in folders if p in c) for c in word_matrix.index]
    u = word_matrix.groupby(grouper, axis=0).count()

    for folder in folders:
        topics = list(u.loc[folder][u.loc[folder] > 4].index)

        classes[folder] = topics

    return classes

def majority_class (predicted_labels, actual_labels): 
    
    uniq = np.unique(predicted_labels)

    majority_corres = dict()

    for val in uniq:
        indices = list(np.where(predicted_labels==val)[0])
        
        counts = {label: 0 for label in uniq}
        

        for i in indices:
            label = actual_labels[i]
            counts[label] += 1
        
        
        max_count = -1
        max_index = 0
        for label in counts.keys():
            if counts[label] > max_count:
                max_count = counts[label]
                max_index = label
        
        
        majority_corres[val] = max_index   
         
    #print(majority_corres)    
    modified_predicted = [majority_corres[label] for label in predicted_labels]
        
    return modified_predicted


#actual_labels = [0]*8 + [1]*8 + [2]*8

#predicted_labels = [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

#print(majority_class (predicted_labels, actual_labels))