#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import time

def random_batch():
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # target
        random_labels.append(skip_grams[i][1])  # context word

    return random_inputs, random_labels

# Model
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        # W and WT is not Traspose relationship
        self.W = nn.Linear(voc_size, embedding_size, bias=False) # voc_size > embedding_size Weight
        self.WT = nn.Linear(embedding_size, voc_size, bias=False) # embedding_size > voc_size Weight

    def forward(self, X):
        # X : [batch_size, voc_size]
        hidden_layer = self.W(X) # hidden_layer : [batch_size, embedding_size]
        output_layer = self.WT(hidden_layer) # output_layer : [batch_size, voc_size]
        return output_layer

if __name__ == '__main__':

    #read data
    path = './data/segmented_data/'
    
    file = 'NYC_hour_6.0seg.dat'
    f = open(path+file,'r')

    document = []
    count = 0
    lines = f.readlines()
    for i in range(len(lines)):
        if count > 15000:
            break
        line = lines[i]

        rw = line.strip().split(' ')
        rw1 = ['</s>'] + rw[1:]

        document.append(' '.join(rw1))
        count += 1
    f.close()

    print(len(document),document[1],
            '\n','data reading finished')

    ######
    time_start =time.time()
    ##
    batch_size = 16 # mini-batch size
    embedding_size = 250 # embedding size

    #sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit", "dog cat animal", "cat monkey animal", "monkey dog animal"]
    sentences = document

    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    # Make skip gram of one size window
    skip_grams = []
    for i in range(5, len(word_sequence) - 5):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 5]], word_dict[word_sequence[i + 5]]]
        for w in context:
            skip_grams.append([target, w])

    model = Word2Vec()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(5000):
        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y,s=0.1)
        #plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
    plt.savefig(path + file.split('.', 1)[
        0] + '_vec' + '.png', transparent=True)
    plt.show()
    ##
    time_end=time.time()

    print('total time_cost:', time_end-time_start)

    output_path = './data/word2vec/'
    data=[]
    for i, w in enumerate(word_list):
        data.append([w,WT[i].tolist()])

    textfile = open(output_path+file.split('.',1)[0]+'_cbow.dat', "w")

    textfile.write(str(len(data)) + ' '+'250'+'\n')
    
    for j in range(len(data)):
    
        if j % 10000 == 0:
            print(data[j][0])
    
        textfile.write(str(data[j][0]) + ' ')
    
        for i in range(len(data[j][1])-1):
        
            textfile.write(str(round(data[j][1][i], 5)) + ' ')
    
        textfile.write(str(round(data[j][1][-1],5))+'\n')   
        textfile.flush()
    textfile.close()
    print('file writing finished!!')









