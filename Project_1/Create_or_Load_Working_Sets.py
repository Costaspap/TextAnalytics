# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 00:14:26 2019

@author: Kostas
"""

from sklearn.model_selection import train_test_split



total = len(corpus_clean)

sets = list(range(0, total))

# 60% for train
training_idx, tuning_idx = train_test_split(sets,train_size = .6, random_state = 2019)

# 20% & 20% for validation and test
validation_idx, test_idx = train_test_split(tuning_idx,train_size = .5, random_state = 2019)

training_set = [corpus_clean[i] for i in training_idx]
validation_set = [corpus_clean[i] for i in validation_idx]
test_set = [corpus_clean[i] for i in test_idx]


del training_idx, validation_idx, tuning_idx, test_idx
gc.collect()

print('Training Size: ', len(training_set))
print('Validation Size: ', len(validation_set))
print('Test Size: ', len(test_set))


##########################################################
# Dont need to do this but here it is anyway##############
##########################################################



#words = word_tokenize(' '.join(training_set))
#print('-------------------------')
#print('Words Tokenized.')
#vocabulary = set(words)
#print('-------------------------')
#print('Vocabulary Created.')
#WordCounts = Counter(words)
#print('-------------------------')
#print('WordCounts Calculated.')
#print('-------------------------')
#
#with open('AllWords', 'wb') as f:
#    pickle.dump(words, f) 
#
#with open('vocabulary', 'wb') as f:
#    pickle.dump(vocabulary, f) 
#    
#with open('WordCounts', 'wb') as f:
#    pickle.dump(WordCounts, f) 


#valid_vocabulary = [k for k,v in WordCounts.items() if v > 10]
#invalid_vocabulary = [k for k,v in WordCounts.items() if v <= 10]
#print("valid voc", len(valid_vocabulary))
#print("invalid voc", len(invalid_vocabulary))

#with open('valid_vocabulary', 'wb') as f:
#    pickle.dump(valid_vocabulary, f) 
#    
#with open('invalid_vocabulary', 'wb') as f:
#    pickle.dump(invalid_vocabulary, f)


##########################################################
########### Replace OOV words in sentences################
##########################################################



#import re

#substitutions = {}

#for word in invalid_vocabulary:
#    substitutions[word] = 'UNK'
#
#print('Substitutions formed.')    
#    
#def replace_all(text,dict_sub):
#    for i, j in dict_sub.items():
#        text = text.replace(i, j)
#    return text
#    
#
#Train_Clean = replace_all('|'.join(training_set),substitutions).split('|') 
#print('Training set cleaned.')
#Validation_Clean = replace_all('|'.join(validation_set),substitutions).split('|') 
#print('Validation set cleaned.')
#Test_Clean = replace_all('|'.join(test_set),substitutions).split('|') 
#print('Test set cleaned.')
#



#with open('Train_Clean', 'wb') as f:
#    pickle.dump(Train_Clean, f) 
#
#with open('Validation_Clean', 'wb') as f:
#    pickle.dump(Validation_Clean, f) 
#    
#with open('Test_Clean', 'wb') as f:
#    pickle.dump(Test_Clean, f) 
#    
#with open('valid_vocabulary', 'wb') as f:
#    pickle.dump(valid_vocabulary, f) 

##########################################################
###################### Load the sets!#####################
##########################################################


with open('Train_Clean', 'rb') as f:
    Train_Clean = pickle.load(f)
    
with open('Validation_Clean', 'rb') as f:
    Validation_Clean = pickle.load(f)
    
with open('Test_Clean', 'rb') as f:
    Test_Clean = pickle.load(f)
    
with open('valid_vocabulary', 'rb') as f:
    valid_vocabulary = pickle.load(f)



