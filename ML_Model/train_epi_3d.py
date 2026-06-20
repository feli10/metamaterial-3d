# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import os.path as osp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from models_epi_3d import Model
from utils_epi_3d import CombinedDataset, train_model, test_model,Logger

import os
import argparse

if __name__ == '__main__':
    cuda = False
    device = torch.device("cuda" if cuda else "cpu")

    repo = osp.dirname(osp.dirname(__file__))
    dataset_path = os.environ.get("DATASET_PATH", osp.join(repo, "dataset.npz"))
    d = np.load(dataset_path)

    model_type = 'Freq_FNO' #'Spherical_FNO'
    experiment_name = model_type

    out_dir = os.environ.get("OUTPUT_DIR", repo)
    results_dir = osp.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(osp.join(out_dir, "checkpoints"), exist_ok=True)
    model_file = osp.join(out_dir, "checkpoints", experiment_name + ".pth")

    im_x = 10
    im_y = 10
    im_z = 10 
    x_dim  = im_x*im_y*im_z
    hidden_dim = 64
    latent_dim = 32
    num_properties = 22  # 1 + 21, volume fraction + 21 independent properties (6+5+4+3+2+1) becuase elasticity tensor is a symmetric matrix
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    jobid = os.environ.get("SLURM_JOB_ID", "local")
    
    ## setup logger
    train_logger = Logger(
        osp.join(results_dir, f"{experiment_name}_{jobid}_train.log"),
        ['ep', 'train_loss','train_rep','train_pred','train_sph_err_e','train_sph_err_d','train_mean']
    )
    test_logger = Logger(
        osp.join(results_dir, f"{experiment_name}_{jobid}_test.log"),
        ['ep', 'test_loss','test_rep', 'test_pred','test_sph_err_e','test_sph_err_d','test_mean']
    )

    kwargs = {'num_workers': 1, 'pin_memory': False} 

    # X is the 3D shape data, in the shape of [n,im_x, im_y, im_z]
    X = d["cells"].astype(np.float32) 
    iu = np.triu_indices(6)
    C_flat = d["C"][:, iu[0], iu[1]]
    # y is the material property data, in the shape of [n, 22]
    y = np.concatenate([d["vfs"][:, None], C_flat], axis=1).astype(np.float32) 
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Compute min/max per property
    prop_min = np.min(y, axis=0)
    prop_max = np.max(y, axis=0)
    print("Property ranges:")
    for i, (lo, hi) in enumerate(zip(prop_min, prop_max), 1):
        print(f"  Property {i}:  min = {lo:.4f},  max = {hi:.4f}")

    
    dataset = CombinedDataset(X,y)

    print(f"dataset shape: {len(dataset)}")
    
    from torch.utils.data import Subset
    
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=42) ## randomly split the datasedt into training and test datasets
    
    # ── ADD THESE LINES FOR A SMALL DEBUG SUBSET 
    # debug_n = 200
    # train_dataset = Subset(train_dataset, list(range(debug_n)))
    # print(f"  Debug mode: training on only {debug_n} samples "  f"→ {len(train_dataset)/batch_size:.0f} batches/epoch")

    # debug_n = 100
    # test_dataset = Subset(test_dataset, list(range(debug_n)))
    # print(f"  Debug mode: testing on only {debug_n} samples "  f"→ {len(test_dataset)/batch_size:.0f} batches/epoch")


    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        **kwargs )
    
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        drop_last = True,
        **kwargs
        )
    
    print("ShapeSpace min:", X.min(), "max:", X.max())
    print("PropertySpace min:", y.min(), "max:", y.max())
    
    print(f"train_loader shape: {len(train_loader)}")

    use_old_model = True
    if osp.exists(model_file) and use_old_model:
        print("Loading model from", model_file)
        model = torch.load(model_file, map_location=device, weights_only=False)
        model.device = device
    else:
        model = Model(batch_size,num_properties,x_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y,im_z, modes1=5, modes2=5,modes3=5).to(device)

    model.to(device)
    print("Model created with type:", model_type)
    optimizer = Adam(model.parameters(), lr=lr)
    # New Scheduler to reduce loss 
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',        # we want to minimize the loss
        factor=0.5,        # multiply LR by 0.5 whenever we trigger
        patience=2,        # wait 2 epochs without improvement
        )

    print("Start training EAAE...")
    print(f" Training for {epochs} epochs with batch size {batch_size}, lr {lr:.0e}")
    for epoch in range(epochs):
        overall_loss, rep_loss, pred_loss, sph_err_e, sph_err_d, m_loss = train_model(train_loader,model,device,optimizer,x_dim,model_type,num_properties)
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Train Loss: ", overall_loss)
        train_logger.log({
        'ep': epoch,             
        'train_loss': overall_loss,
        'train_rep': rep_loss,
        'train_pred': pred_loss,
        'train_sph_err_e': sph_err_e,
        'train_sph_err_d': sph_err_d,
        'train_mean': m_loss
        })
        if epoch % 10 == 0:
            torch.save(model, model_file)
            overall_loss, rep_loss, pred_loss, sph_err_e, sph_err_d, m_loss = test_model(test_loader, model,device,x_dim,model_type,num_properties)
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Test Loss: ", overall_loss)
            test_logger.log({
            'ep': epoch,             
            'test_loss': overall_loss,
            'test_rep': rep_loss,
            'test_pred': pred_loss,
            'test_sph_err_e': sph_err_e,
            'test_sph_err_d': sph_err_d,
            'test_mean': m_loss
            })
        
    print("Finish!!")
    torch.save(model, model_file)



