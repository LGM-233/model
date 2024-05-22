# 手把手 | 基于TextRank算法的文本摘要（附Python代码）
# http://blog.itpub.net/31562039/viewspace-2286669/
#消除警告
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# 1、导入所需的库
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from res_selfatt import NyAttentioin
from gensim.models import KeyedVectors
wo2vec = KeyedVectors.load_word2vec_format("D:/PycharmProjects/pythonProject/Ming21/只能合约/词嵌入训练/smart300_6.bin")
att = NyAttentioin(hidden_size=768,attensize_size=768)
# def get_input(x):
#     x_train = []
#     x = x.split(' ')
#     len_x = len(x)
#
#     padding = 512 - (len_x % 512)
#     pad = torch.zeros((padding, 100))
#     for token in x:
#         x_train.append(wo2vec[token])
#     x_train = np.array(x_train)
#     x_train = torch.tensor(x_train)
#     x_train = torch.cat((x_train, pad))
#     batch = int((len_x + padding) / 512)
#     x_train_split = x_train.view(batch, 512, 100)
#     out = torch.mean(x_train_split,dim=1)
#     out = out.numpy()
#     return out

def get_input(x):
    x_train = []
    x = x.split(',')
    len_x = len(x)

    padding = 256 - (len_x % 256)
    pad = torch.zeros((padding, 300))
    for token in x:
        x_train.append(wo2vec[token])
    x_train = np.array(x_train)
    x_train = torch.tensor(x_train)
    x_train = torch.cat((x_train, pad))
    batch = int((len_x + padding) / 256)
    x_train_split = x_train.view(batch, 256, 300)
    out = torch.mean(x_train_split,dim=1)
    # out = att(x_train_split)

    out = out.numpy()
    return out

# def sentence_split(filepath):
#     df = pd.read_csv(filepath)
#     sentence = np.array([])
#     for i in range(len(df['ops_new'])):
#         str_split = get_input(df['ops_new'][i])
#         sentence = np.append(sentence,str_split)
#
#     return sentence

def get_siminx(filepath):
    df = pd.read_csv(filepath)
    df['ops_abs'] = 0
    # print(similarity_matrix)
    err_index = []
    # err_index_val = [1666,2821,2953,3887,4793,5914,6428,7170]
    # df = df1.drop(1153)
    # 用余弦相似度初始化相似度矩阵（全零矩阵）
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
                    # 这里的if用于排序自己与自己计算相似度
                    if j != k:
                        # 遍历sentences_vectors的句子向量，例如第一行与第二行计算相似度，第一行与第三行计算相似度等等，然后存入
                        # similarity_matrix中的指定位置
                        # reshape(1,-1)将n*n的矩阵变为n*1的
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

def file_abstr(index,doc):
    doc_list = doc.split(',')
    # 分割数组
    step = 256
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
    # 使用列表推导式和 join 方法将整数合并为一句字符串
    result = ",".join(num for sublist in doc_end for num in sublist)

    return result


def demo():
    # # 8、相似矩阵准备
    # # 下一步是找出句子之间的相似性，我们将使用余弦相似性来解决这个问题。
    # # 让我们为这个任务创建一个空的相似度矩阵，并用句子的余弦相似度填充它。
    #
    # # 首先定义一个n乘n的零矩阵，然后用句子间的余弦相似度填充矩阵，这里n是句子的总数。
    # similarity_matrix = np.zeros((len(clean_sentences),len(clean_sentences)))
    # # print(sentences_vectors)
    # # print(similarity_matrix)
    # # 用余弦相似度初始化相似度矩阵（全零矩阵）
    # for i in range(len(clean_sentences)):
    #     for j in range(len(clean_sentences)):
    #         # 这里的if用于排序自己与自己计算相似度
    #         if i != j:
    #             # 遍历sentences_vectors的句子向量，例如第一行与第二行计算相似度，第一行与第三行计算相似度等等，然后存入
    #             # similarity_matrix中的指定位置
    #             # reshape(1,-1)将n*n的矩阵变为n*1的
    #             similarity_matrix[i][j] = cosine_similarity(
    #                 sentences_vectors[i].reshape(1,-1),sentences_vectors[j].reshape(1,-1)
    #             )
    # print(similarity_matrix)
    #
    pass

# def use_pagerank(simi_matrix):
#     nx_graph = nx.from_numpy_array(simi_matrix)
#     scores = nx.pagerank(nx_graph)
#     return scores

def use_pagerank(simi_matrix):
    try:
        nx_graph = nx.from_numpy_array(simi_matrix)
        scores = nx.pagerank(nx_graph,max_iter=1000,alpha=0.6)
        return scores
    except nx.PowerIterationFailedConvergence as e:
        print(f"收敛失败：{e}")
        return None


def abstr_extra(scores):
    # 获取字典中值最大的8个项的键 256
    top_four_indices = sorted(scores, key=scores.get, reverse=True)[:8]
    return top_four_indices



if __name__ == '__main__':
    filepath_train = 'D:/PycharmProjects/pythonProject/Ming21/只能合约/big-mult/train/train_complie1.csv'
    df = get_siminx(filepath_train)
    df.to_csv('D:/PycharmProjects/pythonProject/Ming21/只能合约/词嵌入训练/train_complie_zy.csv')
    print(df)

    # # 9. 应用PageRank算法
    # # 在进行下一步之前，我们先将相似性矩阵sim_mat转换为图结构。这个图的节点为句子，边用句子之间的相似性分数表示。
    # # 在这个图上，我们将应用PageRank算法来得到句子排名。
    # nx_graph = nx.from_numpy_array(similarity_matrix)
    # scores = nx.pagerank(nx_graph)
    #
    # # 10. 摘要提取
    # # 遍历sentences数组，i是当前的位置角标，s是当前的句子
    # # scores[i]：从scores中取出第i个位置的分数与当前句子组成一对
    # # 将所有的分数，句子信息组成的list赋值给ranked_sentences
    # # sorted：并排序，reverse=True降序
    # ranked_sentences = sorted(
    #     ((scores[i],s) for i,s in enumerate(sentences)),reverse=True
    # )
    # # 排序
    # for i in range(10):
    #     print(ranked_sentences[i][1])
    # # 打印得分最高的前10个句子，即为摘要



