from gensim.models import Word2Vec, KeyedVectors
from prepare_data1 import read_data


filepath_train_list = 'train.csv'  #train_data

sentence_list = []
sentence,_ = read_data(filepath_train_list)

for sen in sentence:
        sen = list(sen.split(' '))
        sentence_list.append(sen)



#旧模型

model = Word2Vec(sentence_list, vector_size=300, negative=5, sample=0.001, window=10, min_count=0, sg=1, workers=4, alpha=0.01,epochs=30) #sg=1:skip
model.wv.save_word2vec_format("smart300.bin")
model = KeyedVectors.load_word2vec_format("smart300.bin")
print(model['push'])
