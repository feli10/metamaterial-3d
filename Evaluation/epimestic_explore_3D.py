import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scipy.spatial import distance

from models_epi import Model
from utils_epi import loss_function, load_mat, show_image, CombinedDataset, show_image_group, plot_ternary 
from hom_FE import StructuralFE

if __name__ == '__main__':
    torch.manual_seed(42)
    cuda = False
    device = torch.device("cuda" if cuda else "cpu")
    im_x = 50
    im_y = 50 
    num_properties = 5  # Number of properties in the dataset
    modes1 = 10
    modes2 = 6
    model_type = 'Freq_FNO'

    dataset_path = '/Users/achez/datasets/Chen/rho_data.csv'
    property_path = '/Users/achez/datasets/Chen/property_data.csv'
    results_dir = './results'
    plot_data_dir = './plot_data'
    figure_dir = './figures'
    model_file = osp.join("checkpoints","gpu_"+model_type+"_Chen_data_32_64_model_epi3.pth")

    batch_size = 64
    x_dim  = 2500
    hidden_dim = 64
    latent_dim = 32
    lr = 1e-3
    epochs = 1000
     
    X = np.loadtxt(dataset_path, delimiter=',').astype(np.float32)
    y = np.loadtxt(property_path, delimiter=',').astype(np.float32)  

    hom_FE_solver = StructuralFE()
    hom_FE_solver.initializeSolver(data_type=torch.float32, nelx=im_x, nely=im_y, penal=2,Emin=1e-6, Emax=1.0, nu=0.3)


    X = X.reshape(-1, im_x, im_y)
    X = np.transpose(X, (0, 2, 1))  
    
    dataset = CombinedDataset(X,y)

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=42)


    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = False,
        drop_last = True)
    
    
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle  = True,
        drop_last = True)

    loaded_model = torch.load(model_file,map_location=torch.device('cpu'))
    loaded_model.device = device
    loaded_model.to(device)
    loaded_model.eval()

    it = iter(train_loader)
    (input, y_true) = next(it)
    input = input.to(device)
    y_true = y_true.float()
    y_true = y_true.to(device)

    print("Original property: {}".format(y_true[0].squeeze()))
    Q_input = hom_FE_solver.solve(input[0].permute(1,0).reshape(im_x*im_y))
    print("Input property:C11:{}, C12:{}, C22:{}, C33:{}, rho:{}".format(Q_input[0,0], Q_input[0,1],Q_input[1,1],Q_input[2,2], torch.mean(input[0])))

    pred, mean, err_e, mean_d, err_d = loaded_model(input)
    print("err_e: {}, err_d:{}".format(torch.mean(err_e), torch.mean(err_d)))

    print("Predicted latent property: {}".format(mean[0,:num_properties].squeeze()))
    pred_density = np.mean(pred[0].cpu().detach().numpy().reshape(im_x,im_y))
    pred[pred<0.5] = 0
    pred[pred>0.5] = 1
    pred = pred.view(-1, im_x, im_y)
    Q = hom_FE_solver.solve(pred[0].permute(1,0).reshape(-1))
    print("Predicted property:C11:{}, C12:{}, C22:{}, C33:{}, rho:{}".format(Q[0,0],Q[0,1],Q[1,1],Q[2,2], pred_density))

    mean2 = mean
    mean2[:,0] = mean[:,0] + torch.rand_like(mean[:,0]).to(device)
    z = loaded_model.Encoder.reparameterization(mean2,err_e)
    pred2, mean_d, err_d2 = loaded_model.Decoder(z)
    print("err_d2:{}".format(torch.mean(err_d2)))

    ### study the relationship between the spherical error and the distance between a test data from the training dataset
    training_latent_data_path = osp.join(plot_data_dir, model_type+'_training_latent_data.txt')
    if osp.exists(training_latent_data_path):
        training_latent_data = np.loadtxt(training_latent_data_path, delimiter=',').astype(np.float32)
    else:
        training_latent_data = []
        for batch_idx, (input, y_true) in enumerate(tqdm(train_loader)):
            input = input.to(device)
            y_true = y_true.float()
            y_true = y_true.to(device)
            _, mean, err_e, _, _ = loaded_model(input)
            training_latent_data.append(mean.cpu().detach().numpy())
        training_latent_data = np.vstack(training_latent_data)
        training_latent_data = np.squeeze(training_latent_data)
        np.savetxt(osp.join(plot_data_dir, model_type+'_training_latent_data.txt'), training_latent_data, delimiter=',', fmt='%.6f')

    test_latent_data = []
    test_sph_err = []
    test_reproduction_loss_list = []
    test_prediction_loss_list = []
    for batch_idx, (input, y_true) in enumerate(tqdm(test_loader)):
        input = input.to(device)
        y_true = y_true.float()
        y_true = y_true.to(device)
        x_hat, mean, err_e, mean_d, err_d = loaded_model(input)
        reproduction_loss = torch.mean( (input.view(-1,x_dim)-x_hat.view(-1,x_dim))**2,dim=1).flatten()
        prediction_loss = torch.mean( (mean[:,:num_properties].squeeze()-y_true)**2, dim=1).flatten()
        test_latent_data.append(mean.cpu().detach().numpy())
        test_sph_err.append(err_d.cpu().detach().numpy())
        test_reproduction_loss_list.append(reproduction_loss.cpu().detach().numpy())
        test_prediction_loss_list.append(prediction_loss.cpu().detach().numpy())
        
        ### repreating mean by tile
        min_noise = 0 
        for _ in range(3):
            noise = torch.zeros_like(mean,dtype=torch.float32).to(device)
            noise_indices = torch.randint(0, mean.shape[1], (mean.shape[0],)).to(device)
            sample_indices = torch.arange(mean.shape[0]).to(device)
            noise[sample_indices,noise_indices,0,0] = 0.1*torch.rand(mean.shape[0]).to(device)+min_noise
            mean2 = mean + noise
            z = loaded_model.Encoder.reparameterization(mean2,err_e)
            pred2, mean_d, err_d2 = loaded_model.Decoder(z)
            test_latent_data.append(mean2.cpu().detach().numpy())
            test_sph_err.append(err_d2.cpu().detach().numpy())
            min_noise += 0.1

    test_latent_data = np.concatenate(test_latent_data, axis=0)
    test_latent_data = np.squeeze(test_latent_data)
    test_sph_err = np.concatenate(test_sph_err, axis=0)  
    test_sph_err = np.squeeze(test_sph_err)
    confidence_score = np.exp(-np.mean(test_sph_err-0.04, axis=1))
    test_reproduction_loss_list = np.concatenate(test_reproduction_loss_list, axis=0)
    test_reproduction_loss_list = np.squeeze(test_reproduction_loss_list)
    test_prediction_loss_list = np.concatenate(test_prediction_loss_list, axis=0)
    test_prediction_loss_list = np.squeeze(test_prediction_loss_list)
    print("test_latent_data shape: {}, test_sph_err shape: {}".format(test_latent_data.shape, test_sph_err.shape))
    print("confidence_score shape: {}, test_reproduction_loss shape: {}, test_prediction_loss shape: {}".format(confidence_score.shape, test_reproduction_loss_list.shape, test_prediction_loss_list.shape))
    
    min_dist = []
    for i in range(test_latent_data.shape[0]):
        dist = distance.cdist(test_latent_data[i:i+1,:], training_latent_data, 'euclidean')
        min_dist.append(np.min(dist))
    min_dist = np.array(min_dist)
    print("min_dist shape: {}".format(min_dist.shape))
    ### plot the relationship between the confidence score and the min distance
    plt.figure()
    plt.scatter(min_dist, confidence_score, alpha=0.5)
    plt.xlabel('Min Distance to Training Data in Latent Space')
    plt.ylabel('Confidence Score (exp(-spherical error))')
    plt.grid(True)
    plt.savefig(osp.join(figure_dir, model_type+'_confidence_vs_min_dist.png'))
    ### save plot data
    np.savetxt(osp.join(plot_data_dir, model_type+'_confidence_vs_min_dist.txt'), np.column_stack((min_dist, confidence_score)), delimiter=',', fmt='%.6f')
    plt.show()
    # ### plot the relationship between the confidence score and the reproduction loss
    # plt.figure()
    # plt.scatter(test_reproduction_loss_list, confidence_score, alpha=0.5)
    # plt.xlabel('Reproduction Loss')
    # plt.ylabel('Confidence Score (exp(-spherical error))')
    # plt.grid(True)
    # plt.savefig(osp.join(results_dir, model_type+'_confidence_vs_reproduction_loss.png'))
    # plt.show()
    # ### plot the relationship between the confidence score and the prediction loss
    # plt.figure()
    # plt.scatter(test_prediction_loss_list,confidence_score, alpha=0.5)
    # plt.xlabel('Prediction Loss')
    # plt.ylabel('Confidence Score (exp(-spherical error))')
    # plt.grid(True)
    # plt.savefig(osp.join(results_dir, model_type+'_confidence_vs_prediction_loss.png'))
    # plt.show()


