import os
import re

import pandas as pd

# Basic/Naive approach

# reading the text file
file = open("text.txt", "r")
sentences = file.readlines()
sentences = [sentence.strip() for sentence in sentences]

df = pd.DataFrame(sentences, columns=['sentence'])

Q_WORD = ['what', 'where','when','why','who','how']
QQ_WORD = ['if','were','was','am','is','are','has','have','can','could','will','would','shall','should','do','does']

# function to detect if the sentence is a question
def detect_inquiry(sentences):
    result = []
    for sentence in sentences:
        flag = 0
        # if last character is a question mark
        # result is a question
        if(sentence[-1] == '?'):
            result.append(1)
            continue
        else:
            # we will check if the sentence contains any wh word
            # if yes, then it is a question
            for word in Q_WORD:
                if(word in sentence.split(' ')):
                    result.append(1)
                    flag = 1
                    break
            
            # we will check if the first word is QQ_WORD
            # if yes, then it is a question
            for word in QQ_WORD:
                # another check if the sentence is already marked as a question
                if(flag == 1):
                    break;
                
                if(word == sentence.split(' ')[0]):
                    result.append(1)
                    flag = 1
                    break

            # if the sentence is not a question
            # append 0 in result
            if(flag != 1):
                result.append(0)

            
    return result


result = detect_inquiry(sentences)

df['type'] = result
df['type'] = ['yes' if x == 1 else 'no' for x in df['type']]

# got a count of 1004 inquiry in text file

# saving the dataframe to csv file
df.to_csv('result1.csv', index=False)