import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from BiGRU import BiGRU
import numpy as np
import datetime
import time
import copy
from sklearn.metrics import accuracy_score,hamming_loss,recall_score
from torch.optim.lr_scheduler import StepLR
from PosiEnc import PositionalEncoding

from ASL_loss import AsymmetricLoss


import pandas as pd

wo2vec = KeyedVectors.load_word2vec_format("smart300.bin")
loss_func = nn.BCEWithLogitsLoss()
loss_ASL = AsymmetricLoss()
use_cuda = torch.cuda.is_available() 
ngpu = 1
device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")
metric_name = 'acc'
metric_name_hm = 'hanming_loss'
metric_func = lambda y_true, y_pred: accuracy_score(y_true, y_pred)
metric_hm = lambda y_true, y_pred: hamming_loss(y_true, y_pred)

#print time
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "=========="*8 + '%s'%nowtime)

def pred_sigmoid(logits):

    threshold = 0.5
    predicted_labels = (logits > threshold).float()
    return predicted_labels.long()

class TextCNN(nn.Module):
    def __init__(self, embedding_dim, num_classes, filter_sizes, num_filters, dropout_rate):
        super(TextCNN, self).__init__()
        self.embedding_dim = embedding_dim
        # Convolutional Layers with different sizes
        self.convolutional_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_filters,
                      kernel_size=k,
                      padding = int((k - 1) / 2)) for k in filter_sizes
        ])

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters * 4, num_classes)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, embedding_dim]
        # we transpose the second and the third dimension
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, sequence_length]

        # Apply convolutional layers & global max pooling
        convs = [F.relu(conv(x)) for conv in
                 self.convolutional_layers]  # shape of each conv: [batch_size, num_filters, <variable length>]
        # print(convs[0].shape)
        pools = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in
                 convs]  # shape of each pool: [batch_size, num_filters]
        # print(pools[0].shape)

        x = torch.cat(pools, 1)  # concatenate all the pools, [batch_size, len(filter_sizes) * num_filters]
        x = x.view(1,x.shape[0] * self.embedding_dim)

        x = self.dropout(x)  # apply dropout
        #
        # # Apply fully connected layer
        logits = self.fc(x)  # [batch_size, num_classes]

        return logits

class textcnnmodel(nn.Module):
    def __init__(self, num_classes, d_model):
        super(textcnnmodel, self).__init__()
        self.textcnn = TextCNN(embedding_dim=d_model,num_classes=num_classes,filter_sizes=[3,5,7],num_filters=100,dropout_rate=0.5)
        self.bigru_fc = nn.Linear(d_model, 4)
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self,x):
        # join position
        # positon1 = PositionalEncoding(max_num_seg=100, d_model=768, num_seg=512, seg_len=512).to(device)

        x_out = self.textcnn(x)
        logits = torch.squeeze(x_out) 

def get_split_input(x):
    x_train = []
    x = x.split(',')
    len_x = len(x)
    # padding = 512 - (len_x % 512)
    padding = 2048 - len_x
    pad = torch.zeros((padding, 300))
    for token in x:
        x_train.append(wo2vec[token])
    x_train = np.array(x_train)
    x_train = torch.tensor(x_train)
    x_train = torch.cat((x_train, pad))
    batch = int((len_x + padding) / 512)
    x_train_split = x_train.view(batch, 512, 300)
    inputs = x_train_split.clone().detach()      #.requires_grad_(True)
    inputs.to(device) 
    return inputs


def train_step(model, inps, labs, optimizer):
    inputs = get_split_input(inps)
    inputs = inputs.to(device)
    # labs = labs.replace(',',' ')
    numbers = [int(num) for num in labs.strip('[]').split()]
    labs = torch.tensor(numbers,dtype=torch.float32)

    labs = labs.to(device)
    model.train()  
    optimizer.zero_grad()  

    # forward
    logits = model(inputs)
    logits_e = torch.sigmoid(logits)  

    loss = loss_ASL(logits_e, labs)
    pred = pred_sigmoid(logits_e)
    metric = metric_hm(labs.cpu().numpy(), pred.cpu().numpy()) 
    # print(pred.cpu().numpy(), labs.cpu().numpy())
    # recall = recall_sc(pred.cpu().numpy(), labs.cpu().numpy())
    # backward
    loss.backward()  
    optimizer.step()  

    return loss.item(), metric.item()

def validate_step(model, inps, labs):
    inputs = get_split_input(inps)
    inputs = inputs.to(device)
    # labs = labs.replace(',', ' ')
    numbers = [int(num) for num in labs.strip('[]').split()]
    labs = torch.tensor(numbers, dtype=torch.float32)
    labs = labs.to(device)

    model.eval()  

    # forward
    with torch.no_grad():
        logits = model(inputs)
        logits_e = torch.sigmoid(logits)  
        loss = loss_ASL(logits_e, labs)
        pred = pred_sigmoid(logits_e)

        metric = metric_hm(labs.cpu().numpy(), pred.cpu().numpy())  
        # recall = recall_sc(pred.cpu().numpy(), labs.cpu().numpy())
        tp = np.sum((pred.cpu().numpy() == 1) & (labs.cpu().numpy() == 1))
        fn = np.sum((pred.cpu().numpy() == 0) & (labs.cpu().numpy() == 1))
        fp = np.sum((pred.cpu().numpy() == 1) & (labs.cpu().numpy() == 0))
    return loss.item(), metric.item(), tp, fn, fp

def train_model(model, train_dloader, val_dloader, optimizer, num_epochs, scheduler_1r, init_epoch=0,  print_every=600):

    starttime = time.time()
    print('*' * 27, 'start training...')
    printbar()

    best_metric, min_hm = 0., 10.

    for epoch in range(init_epoch+1, init_epoch+num_epochs+1):

        print(f'epoch = {epoch}')
        loss_sum, metric_sum = 0., 0.
        for step, row in train_dloader.iterrows():

            inps = row['ops_abs']
            labs = row['label']
            torch.cuda.empty_cache()
            loss, metric = train_step(model, inps, labs, optimizer)
            loss_sum += loss
            metric_sum += metric


            if (step+1) % print_every == 0:
                print('*'*27, f'[step = {step+1}] loss: {loss_sum/(step+1):.3f}, {metric_name_hm}: {metric_sum/(step+1):.3f}')

        val_loss_sum, val_metric_sum = 0., 0.
        tp_sum, fn_sum ,fp_sum = 0., 0., 0.
        for val_step, row in val_dloader.iterrows():

            inps = row['ops_abs']
            labs = row['label']
            torch.cuda.empty_cache()
            loss, metric, tp, fn ,fp = validate_step(model, inps, labs)
            val_loss_sum += loss
            val_metric_sum += metric
            tp_sum += tp
            fn_sum += fn
            fp_sum += fp

        # if scheduler_1r:
        #     scheduler_1r.step()
        scheduler_1r.step()

        mar_R = tp_sum / (tp_sum + fn_sum)
        mar_P = tp_sum / (tp_sum + fp_sum)
        mar_F1 = (2.0 * mar_P * mar_R) / (mar_P + mar_R)
        record = (epoch, loss_sum / step, metric_sum / (step+1), val_loss_sum / (val_step+1), val_metric_sum / (val_step+1), mar_R, mar_P, mar_F1)

        # log_file.write(
        #     f'Epoch {epoch}, Loss: {val_loss_sum / (val_step + 1) :.3f}, Hanming {metric_name_hm}: {val_metric_sum / (val_step + 1):.3f},mar_R:{mar_R:.3f},mar_P:{mar_P:.3f},mar_F1{mar_F1:.3f}\n')


        print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}, mar-R: {:.3f}, mar-P: {:.3f}, mar-F1: {:.3f}'.format(
            record[0], record[1], metric_name_hm, record[2], record[3], metric_name_hm, record[4], record[5], record[6],record[7]))
        printbar()



        endtime = time.time()
        time_elapsed = endtime - starttime
        print('*' * 27, 'training finished...')
        print('*' * 27, 'and it costs {} h {} min {:.2f} s'.format(int(time_elapsed // 3600),
                                                                   int((time_elapsed % 3600) // 60),
                                                                   (time_elapsed % 3600) % 60))

        print('Best val Acc: {:4f}'.format(best_metric))


if __name__ == '__main__':
    EPOCHS = 20

    filepath_train = 'train_part_opcode.csv'

    filepath_val = 'valid_part_opcode.csv'
    train_data = pd.read_csv(filepath_train)
    val_data = pd.read_csv(filepath_val)

    model = textcnnmodel(num_classes=6, d_model=300)
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=2e-7) 
    scheduler_1r = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda epoch: 0.1 if epoch > EPOCHS * 0.8 else 1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    train_model(model, train_dloader=train_data, val_dloader=val_data, optimizer=optimizer, num_epochs=EPOCHS, scheduler_1r=scheduler)


