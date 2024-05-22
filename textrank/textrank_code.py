
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from res_selfatt import NyAttentioin
from gensim.models import KeyedVectors
wo2vec = KeyedVectors.load_word2vec_format("./smart300_6.bin") #your word2vec.bin model

def get_input(x):
    x_train = []
    x = x.split(',')
    len_x = len(x)

    padding = 128 - (len_x % 128)
    pad = torch.zeros((padding, 300))
    for token in x:
        x_train.append(wo2vec[token])
    x_train = np.array(x_train)
    x_train = torch.tensor(x_train)
    x_train = torch.cat((x_train, pad))
    batch = int((len_x + padding) / 128)
    x_train_split = x_train.view(batch, 128, 300)
    out = torch.mean(x_train_split,dim=1)  #Word vectors are converted to sentence vectors
    # out = att(x_train_split)

    out = out.numpy()
    return out


def get_siminx(filepath):
    df = pd.read_csv(filepath)
    df['ops_abs'] = 0
    # print(similarity_matrix)
    err_index = []
    for i in range(len(df['ops_new'])):
        x = df.loc[i,'ops_new']
        len_x = len(x.split(','))
        print(len_x)
        if len_x <= 2048:
            df.loc[i, 'ops_abs'] = x
        else:
            sentence = get_input(df.loc[i,'ops_new'])
            similarity_matrix = np.zeros((len(sentence), len(sentence)))
            for j in range(len(sentence)):
                for k in range(len(sentence)):
                    # Sort yourself and calculate the similarity yourself
                    if j != k:
                        # Travers the sentences_vectors sentence vectors, 
                        #such as the first line and the second line to calculate the similarity, 
                        #the first line to the third line to calculate the similarity, and so on, 
                        #and then deposit
                        # similarity_matrix
                        # reshape(1,-1)n*n-->n*1
                        similarity_matrix[j][k] = cosine_similarity(
                            sentence[j].reshape(1,-1),sentence[k].reshape(1,-1)
                        )
            score = use_pagerank(similarity_matrix)

            if score == None:
                err_index.append(i)
                continue
            index = abstr_extra(score)

            result = file_abstr(index,df.loc[i,'ops_new'])

            df.loc[i,'ops_abs'] = result
    return df

def file_abstr(index,doc):  #
    doc_list = doc.split(',')
    # split
    step = 128
    split_arrays = []
    for i in range(0,len(doc_list),step):
        if i + step < len(doc_list):
            split_arrays.append(doc_list[i:i+step])
        else:
            split_arrays.append(doc_list[i:len(doc_list)])

    doc_end = []

    for i in range(len(split_arrays)):
        if i in index:
            doc_end.append(split_arrays[i])
        else:
            continue
    # Use list derivation and join methods to combine integers into a string
    result = ",".join(num for sublist in doc_end for num in sublist)

    return result


def use_pagerank(simi_matrix):
    try:
        nx_graph = nx.from_numpy_array(simi_matrix)
        scores = nx.pagerank(nx_graph,max_iter=1000,alpha=0.6)
        return scores
    except nx.PowerIterationFailedConvergence as e:
        print(f"error：{e}")
        return None


def abstr_extra(scores):
    # Get the keys of the 16 items with the highest values in the dictionary 128
    top_four_indices = sorted(scores, key=scores.get, reverse=True)[:16]
    return top_four_indices



if __name__ == '__main__':
    filepath_train = './train_complie1.csv' #Training set, cleaned opcode file
    df = get_siminx(filepath_train)
    df.to_csv('./train_complie_zy.csv')
    print(df)




