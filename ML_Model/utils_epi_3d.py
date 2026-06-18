import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

def loss_function(x, x_hat,y_true, mean, sph_err_encoder, mean_decoder,sph_err_decoder,model_type,num_properties):
    var_loss = torch.mean(sph_err_encoder) + torch.mean(sph_err_decoder)
    # print("var_loss: {}".format(var_loss))
    # print("max x_hat: {}, min x_hat: {}".format(torch.max(x_hat), torch.min(x_hat)))
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat,x, reduction='mean')
    prediction_loss = nn.functional.mse_loss(mean[:,:num_properties].squeeze(), y_true, reduction='mean')
    reproduction_mid_value_loss = nn.functional.mse_loss(mean_decoder.squeeze(), mean.squeeze(), reduction='mean')
    total_loss = reproduction_loss + var_loss  + prediction_loss + reproduction_mid_value_loss
    return total_loss, reproduction_loss,prediction_loss, torch.mean(sph_err_encoder),torch.mean(sph_err_decoder), reproduction_mid_value_loss

def train_model(data_loader, model,device,optimizer,x_dim,model_type,num_properties):
    model.train()
    overall_loss = 0
    rep_loss = 0
    m_loss = 0
    err_e = 0
    err_d = 0
    pred_loss = 0
    for batch_idx, (input, y_true) in enumerate(data_loader):
        #x = x.view(batch_size, x_dim)
        input = input.to(device)
        y_true = y_true.float()
        y_true = y_true.to(device)
        #y_true = torch.squeeze(y_true, dim=1)  # Ensure y_true is 1D if it has a single dimension

        optimizer.zero_grad()

        pred, mean, sph_err_encoder, mean_decoder, sph_err_docoder = model(input)

        loss,reproduction_loss, prediction_loss, sph_err_e, sph_err_d, mean_loss = loss_function(input.view(-1,x_dim), pred.view(-1,x_dim), y_true, mean, sph_err_encoder,mean_decoder, sph_err_docoder,model_type,num_properties)
        
        overall_loss += loss.item()
        rep_loss += reproduction_loss.item()
        pred_loss += prediction_loss.item()
        err_e += sph_err_e.item()
        err_d += sph_err_d.item()
        m_loss += mean_loss.item()
        loss.backward()
        optimizer.step()
    return overall_loss / (batch_idx+1), rep_loss/(batch_idx+1), pred_loss/(batch_idx+1), err_e/(batch_idx+1), err_d/(batch_idx+1), m_loss/(batch_idx+1)
            
def test_model(data_loader, model,device,x_dim,model_type,num_properties):
    model.eval()  
    overall_loss = 0
    rep_loss = 0
    m_loss = 0
    err_e = 0
    err_d = 0
    pred_loss = 0
    for batch_idx, (input, y_true) in enumerate(tqdm(data_loader)):
        input = input.to(device)
        y_true = y_true.float()
        y_true = y_true.to(device)
        pred, mean, sph_err_encoder, mean_decoder, sph_err_docoder = model(input)
        loss,reproduction_loss, prediction_loss, sph_err_e, sph_err_d, mean_loss = loss_function(input.view(-1,x_dim), pred.view(-1,x_dim), y_true, mean, sph_err_encoder, mean_decoder,sph_err_docoder,model_type,num_properties)
        overall_loss += loss.item()
        rep_loss += reproduction_loss.item()
        pred_loss += prediction_loss.item()
        err_e += sph_err_e.item()
        err_d += sph_err_d.item()
        m_loss += mean_loss.item()
        
    return overall_loss / (batch_idx+1), rep_loss/(batch_idx+1), pred_loss/(batch_idx+1), err_e/(batch_idx+1),err_d/(batch_idx+1), m_loss/(batch_idx+1)
            

class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

class CombinedDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
    
    def __len__(self):
        return min(len(self.input_data), len(self.output_data))
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]
