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
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


import pandas as pd


wo2vec = KeyedVectors.load_word2vec_format("./smart768.bin")
loss_func = nn.BCEWithLogitsLoss()
loss_ASL = AsymmetricLoss()
use_cuda = torch.cuda.is_available() 
ngpu = 1
device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")
metric_name = 'acc'
metric_name_hm = 'hanming_loss'
metric_func = lambda y_true, y_pred: accuracy_score(y_true, y_pred)
metric_hm = lambda y_true, y_pred: hamming_loss(y_true, y_pred)


df_history = pd.DataFrame(columns=["epoch", "loss", metric_name_hm, "val_loss", "val_"+metric_name_hm, "micro-R", "micro-P", "micro-F1"])

# print time
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')#print time
    print('\n' + "=========="*8 + '%s'%nowtime)

def pred_sigmoid(logits):
    # sigmoid_output = torch.sigmoid(logits)
    threshold = 0.5
    predicted_labels = (logits > threshold).float()
    return predicted_labels.long()


class word2vec_add_bigru(nn.Module):
    def __init__(self, num_class, d_model, num_layer, input_size, hidden_size):
        super(word2vec_add_bigru, self).__init__()
        self.gru = BiGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, output_size=d_model)
        self.bigru_fc = nn.Linear(d_model, num_class)
        self.att_res = NyAttentioin(hidden_size=768, attensize_size=768)
        self.fc = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.2)
        self.transformerencoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=6, batch_first=True,dropout=0.2), 
            num_layers=6)


    def forward(self,x):
        # join position
        # positon1 = PositionalEncoding(max_num_seg=100, d_model=768, num_seg=512, seg_len=512).to(device)

        x_res = x.view(1, -1, 768) #[num_seg, 512, 768] => [1, num_seg*512, 768]
        position_res = PositionalEncoding(max_num_seg=100, d_model=768, num_seg=2048, seg_len=2048).to(device)
        x_in = position_res(x_res)
        # x_in_res = self.att_res(x_in)
        x_input = x_in.reshape((4,512,768))

        input_att = self.transformerencoder(x_input)  # [num_seg, 512, 768]
        # input_att, _ = self.attention1(x,x,x)  #[num_seg, 512, 768]  
        input_att.to(device)
        split_out = self.gru(input_att)  # [num_seg, 512, 768] => [num_seg, 768]
        split_out = split_out.unsqueeze(dim=0)  # [num_seg, 768] => [1, num_seg, 768]
        # 加入position
        positon2 = PositionalEncoding(max_num_seg=100, d_model=768, num_seg=len(split_out[0]), seg_len=0).to(device)
        split_out_pos = positon2(split_out) #[1, num_seg, 768] 
        split_out_ts = self.transformerencoder(split_out_pos)  # [1, num_seg, 768] 
        split_out_ts.to(device)
        end_out = self.gru(split_out_ts)  # [1, num_seg, 768] => [1, 768]
        # out_and_res = end_out + x_in_res
        logits = self.bigru_fc(self.dropout(end_out))  # [1, 768] => [1, 6]
        logits = torch.squeeze(logits)  # [6]
        return logits

def get_split_input(x):
    x_train = []
    x = x.split(',')
    len_x = len(x)
    # padding = 512 - (len_x % 512)
    padding = 2048 - len_x
    pad = torch.zeros((padding, 768))
    for token in x:
        x_train.append(wo2vec[token])
    x_train = np.array(x_train)
    x_train = torch.tensor(x_train)
    x_train = torch.cat((x_train, pad))
    batch = int((len_x + padding) / 512)
    x_train_split = x_train.view(batch, 512, 768)
    inputs = x_train_split.clone().detach()      #.requires_grad_(True)
    inputs.to(device) 
    return inputs


def train_step(model, inps, labs, optimizer):
    inputs = get_split_input(inps)
    inputs = inputs.to(device)
    labs = labs.replace(',',' ')
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
    labs = labs.replace(',', ' ')
    numbers = [int(num) for num in labs.strip('[]').split()]
    labs = torch.tensor(numbers, dtype=torch.float32)
    labs = labs.to(device)

    model.eval()  

    # forward
    with torch.no_grad():
        logits = model(inputs)
        logits = torch.sigmoid(logits)  
        loss = loss_ASL(logits, labs)
        pred = pred_sigmoid(logits)

        metric = metric_hm(labs.cpu().numpy(), pred.cpu().numpy()) 
        # recall = recall_sc(pred.cpu().numpy(), labs.cpu().numpy())
        tp = np.sum((pred.cpu().numpy() == 1) & (labs.cpu().numpy() == 1))
        fn = np.sum((pred.cpu().numpy() == 1) & (labs.cpu().numpy() == 0))
        fp = np.sum((pred.cpu().numpy() == 0) & (labs.cpu().numpy() == 1))
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
        
        # log_file.write(
        #     f'Epoch {epoch}, Loss: {val_loss_sum / (val_step+1) :.3f}, Hanming {metric_name_hm}: {val_metric_sum / (val_step+1):.3f}\n')
        # if scheduler_1r:
        #     scheduler_1r.step()
        scheduler_1r.step()

        mar_R = tp_sum / (tp_sum + fn_sum)
        mar_P = tp_sum / (tp_sum + fp_sum)
        mar_F1 = (2.0 * mar_P * mar_R) / (mar_P + mar_R)
        record = (epoch, loss_sum / step, metric_sum / (step+1), val_loss_sum / (val_step+1), val_metric_sum / (val_step+1), mar_R, mar_P, mar_F1)
        df_history.loc[epoch] = record
        print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}, mar-R: {:.3f}, mar-P: {:.3f}, mar-F1: {:.3f}'.format(
            record[0], record[1], metric_name_hm, record[2], record[3], metric_name_hm, record[4], record[5], record[6],record[7]))
        printbar()

        current_metric_avg = val_metric_sum / val_step
        if current_metric_avg < min_hm:
            min_hm = current_metric_avg
            checkpoint = save_dir + f'epoch{epoch:03d}_valhm{current_metric_avg:.3f}_ckpt.tar'
            if device.type == 'cuda' and ngpu > 1:
                model_sd = copy.deepcopy(model.module.state_dict())
            else:
                model_sd = copy.deepcopy(model.state_dict())
            # save
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
            }, checkpoint)
        endtime = time.time()
        time_elapsed = endtime - starttime
        print('*' * 27, 'training finished...')
        print('*' * 27, 'and it costs {} h {} min {:.2f} s'.format(int(time_elapsed // 3600),
                                                                   int((time_elapsed % 3600) // 60),
                                                                   (time_elapsed % 3600) % 60))

        print('Best val Acc: {:4f}'.format(best_metric))
    return df_history

# draw
# def plot_metric(df_history, metric):
#     plt.figure()

#     train_metrics = df_history[metric]
#     val_metrics = df_history['val_' + metric]  #

#     epochs = range(1, len(train_metrics) + 1)

#     plt.plot(epochs, train_metrics, 'bo--')
#     plt.plot(epochs, val_metrics, 'ro-')  #

#     plt.title('Training and validation ' + metric)
#     plt.xlabel("Epochs")
#     plt.ylabel(metric)
#     plt.legend(["train_" + metric, 'val_' + metric])

#     plt.savefig(imgs_dir + 'ttg_'+ metric + '.png')  
#     plt.show()

if __name__ == '__main__':
    EPOCHS = 20
    filepath_train = './train_udrequ_shuffle_2048.csv'

    filepath_val = './val_udrequ_shuffle_2048.csv'
    train_data = pd.read_csv(filepath_train)
    val_data = pd.read_csv(filepath_val)

    model = word2vec_add_bigru(num_class=6, d_model=768, num_layer=1, input_size=768, hidden_size=768) 
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=2e-7) 
    scheduler_1r = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda epoch: 0.1 if epoch > EPOCHS * 0.8 else 1)
    # define StepLR
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
    log_file = open(log_dir,'w')
    train_model(model, train_dloader=train_data, val_dloader=val_data, optimizer=optimizer, num_epochs=EPOCHS, scheduler_1r=scheduler)
    # log_file.close()
    # plot_metric(df_history, 'loss')
    # plot_metric(df_history, metric_name_hm)
