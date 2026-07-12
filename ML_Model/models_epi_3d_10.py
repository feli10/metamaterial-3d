import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self,batchsize,num_properties, x_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y,im_z,modes1, modes2,modes3):
        super(Model, self).__init__()
        self.num_properties = num_properties # 5   # since there are 5 material dims
        self.device = device
        self.im_x = im_x
        self.im_y = im_y
        self.im_z = im_z
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        if model_type == 'Freq_FNO':
            self.Encoder = FNO_Encoder(num_properties,input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,im_x=im_x, im_y=im_y,im_z=im_z,modes1=modes1,modes2=modes2,modes3=modes3)
            self.Decoder = FreqFNO_Decoder(batchsize, device,latent_dim=latent_dim//2, hidden_dim = hidden_dim, output_dim = x_dim,im_x=im_x, im_y=im_y,im_z=im_z,modes1=modes1,modes2=modes2,modes3=modes3)

    def forward(self, x):
        mid_value, sph_err_encoder = self.Encoder(x)
        z = self.Encoder.reparameterization(mid_value,sph_err_encoder,self.modes1,self.modes2,self.modes3)
        x_hat, mid_value_decoder, sph_err_decoder = self.Decoder(z)
        return x_hat, mid_value, sph_err_encoder, mid_value_decoder, sph_err_decoder

class FNO_Encoder(nn.Module):
    def __init__(self,num_properties,input_dim, hidden_dim, latent_dim,im_x,im_y,im_z,modes1, modes2,modes3):
        super(FNO_Encoder, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.im_x = im_x
        self.im_y = im_y
        self.im_z = im_z
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_properties = num_properties
        self.activation_function = nn.LeakyReLU(0.2)
        self.p = nn.Linear(4, self.hidden_dim) # input channel is 4: (a(x, y), x, y,z)
        self.conv0 = SpectralConv3d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.hidden_dim, self.latent_dim, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.latent_dim, self.latent_dim, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d(self.latent_dim, self.latent_dim, self.modes1, self.modes2, self.modes3)
        self.conv5 = SpectralConv3d(self.latent_dim, self.latent_dim, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.mlp1 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.mlp2 = MLP(self.latent_dim, self.latent_dim, self.latent_dim)
        self.mlp3 = MLP(self.latent_dim, self.latent_dim, self.latent_dim)
        self.mlp4 = MLP(self.latent_dim, self.latent_dim, self.latent_dim)
        self.mlp5 = MLP(self.latent_dim, self.latent_dim, self.latent_dim)
        self.w0 = nn.Conv3d(self.hidden_dim, self.hidden_dim, 1)
        self.w1 = nn.Conv3d(self.hidden_dim, self.hidden_dim, 1)
        self.w2 = nn.Conv3d(self.hidden_dim, self.latent_dim, 1)
        self.w3 = nn.Conv3d(self.latent_dim, self.latent_dim, 1)
        self.w4 = nn.Conv3d(self.latent_dim, self.latent_dim, 1)
        self.w5 = nn.Conv3d(self.latent_dim, self.latent_dim, 1)

    def forward(self, x):
        x = x.view(-1,self.im_x,self.im_y,self.im_z,1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.activation_function(self.p(x))
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.activation_function(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.activation_function(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.activation_function(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.activation_function(x)

        x1 = self.conv4(x)
        x1 = self.mlp4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        x = self.activation_function(x)

        x1 = self.conv5(x)
        x1 = self.mlp5(x1)
        x2 = self.w5(x)
        x = x1 + x2

        mean = torch.mean(x,dim=(2,3,4),keepdim=True)
        var = torch.var(x,dim=(2,3,4),keepdim=True)
        return mean, var
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
    
    def reparameterization(self,mean,var,modes1,modes2, modes3):
        device = mean.device
        latent_dim = mean.shape[1] 
        epsilon_real = torch.randn(mean.shape[0], latent_dim//2, modes1, modes2,modes3).to(device)* torch.sqrt(var[:,:latent_dim//2,:,:,:])
        z_real = mean[:,:latent_dim//2,:,:,:] + epsilon_real
        epsilon_image = torch.randn(mean.shape[0], latent_dim//2, modes1, modes2,modes3).to(device)* torch.sqrt(var[:,latent_dim//2:,:,:,:])
        z_image = mean[:,latent_dim//2:,:,:,:] + epsilon_image
        z = torch.complex(z_real, z_image)
        return z
    

    
class FreqFNO_Decoder(nn.Module):
    def __init__(self, batchsize, device,latent_dim, hidden_dim, output_dim,im_x,im_y,im_z, modes1,modes2, modes3):
        super(FreqFNO_Decoder, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.im_x = im_x
        self.im_y = im_y
        self.im_z = im_z
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.p = LocalMLP_Complex_3d(latent_dim,hidden_dim, self.modes1, self.modes2, self.modes3)
        self.conv0 = SpectralConv3d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.hidden_dim, self.hidden_dim, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.mlp1 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.mlp2 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.mlp3 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2)
        self.w0 = nn.Conv3d(self.hidden_dim, self.hidden_dim, 1)
        self.w1 = nn.Conv3d(self.hidden_dim, self.hidden_dim, 1)
        self.w2 = nn.Conv3d(self.hidden_dim, self.hidden_dim, 1)
        self.w3 = nn.Conv3d(self.hidden_dim, self.hidden_dim, 1)
        self.q = MLP(self.hidden_dim, self.latent_dim*2+1, self.hidden_dim*2) # output channel is 1: u(x, y)
        self.complex_activation_function = ComplexReLU(0.2)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.x_grid, self.y_grid, self.dx, self.dy = self.get_spherical_grid(batchsize, latent_dim*2, 25, 40, device)

    def forward(self, x, output_im_x = 10, output_im_y = 10, output_im_z = 10):
        x = self.complex_activation_function(self.p(x))

        x = torch.fft.irfftn(x, s=(output_im_x, output_im_y, output_im_z),dim=(-3,-2,-1))

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.LeakyReLU(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.LeakyReLU(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.LeakyReLU(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.q(x)
  
        x_hat = x[:,0:1,:,:,:]
        x_hat = torch.sigmoid(x_hat)
        x_hat = x_hat.permute(0, 2, 3, 4, 1)

        x_epi = x[:,1:,:,:,:]
        ### rearragle 3D array to 2D array
        x_epi = x_epi.reshape(x_epi.shape[0], x_epi.shape[1], 25, 40)  
        x_epi = 2*torch.sigmoid(x_epi)-1
        mid_value = x_epi[:,:,25//2,40//2]
        mid_value = mid_value[:,:,None,None]
        sph_err = self._calculate_spherical_error(x_epi,mid_value,self.x_grid,self.y_grid,self.dx,self.dy)
        return x_hat,mid_value, sph_err

    def get_spherical_grid(self, batchsize, channels, size_x, size_y, device):
        x_coords = torch.linspace(-1, 1, size_x).to(device)
        y_coords = torch.linspace(-1, 1, size_y).to(device)
        l0 = 1/np.sqrt(2)
        x_coords = x_coords * l0
        y_coords = y_coords * l0
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        x_grid = x_grid.unsqueeze(0).unsqueeze(0).repeat(batchsize,channels,1,1)
        y_grid = y_grid.unsqueeze(0).unsqueeze(0).repeat(batchsize,channels,1,1)
        dx = x_coords[1]-x_coords[0]
        dy = y_coords[1]-y_coords[0]
        return x_grid, y_grid, dx, dy

    def _calculate_spherical_error(self, x,mid_value,x_grid,y_grid,dx,dy):
        #batchsize, channels, size_x, size_y = x.shape
        ## radical error on sphereical surface
        ## normalize height field
        z0 = 0
        r0_sq = 1
        K0 = 1
        # original_sign = torch.sign(mid_value)
        small_value_mask = torch.abs(mid_value) < 1e-4
        # mid_value = mid_value.masked_fill(small_value_mask, 1e-3)
        # mid_value = torch.abs(mid_value)*original_sign
        # small mid_value guard
        mid_value = mid_value.sign() * mid_value.abs().clamp(min=1e-3) 
        x = x/mid_value
        ## assume the sphere center is (0,0,0)
        z_grid = x - z0
        radius_sq = x_grid**2 + y_grid**2 + z_grid**2
        radical_error = torch.mean((radius_sq - r0_sq)**2,dim=[2,3],keepdim=True)
        radical_error = radical_error.masked_fill(small_value_mask, 0.0)
        # ## curvature error on spherical surface
        K = _principal_curvatures_heightfield(z_grid, dx, dy)
        curvature_err = torch.mean((K[:,:,5:-5,5:-5] - K0)**2,dim=[2,3],keepdim=True)
        curvature_err = curvature_err.masked_fill(small_value_mask, 0.0)
        #print("radical_error: {}, curvature_err: {}".format(torch.mean(radical_error), torch.mean(curvature_err)))
        sph_err = radical_error + curvature_err*1e-1
        return sph_err

def _principal_curvatures_heightfield(z_grid,dx,dy):
    # First derivatives
    fx = torch.gradient(z_grid, spacing=dx, dim=2)[0]
    fy = torch.gradient(z_grid, spacing=dy, dim=3)[0]
    # Second derivatives
    fxx = torch.gradient(fx, spacing=dx, dim=2)[0]
    fyy = torch.gradient(fy, spacing=dy, dim=3)[0]
    fxy = torch.gradient(fx, spacing=dy, dim=3)[0]
    eps = 1e-8
    K = (fxx*fyy - fxy*fxy)/((1 + fx*fx + fy*fy)**2 + eps)
    return K

################################################################
# fourier layer
################################################################ 
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class LocalMLP(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(LocalMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.float32))
    
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        out = self.compl_mul2d(x, self.weights1)
        return out
    
class LocalMLP_Complex_3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(LocalMLP_Complex_3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.complex64))
    
    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,z ), (in_channel, out_channel, x,y,z) -> (batch, out_channel, x,y,z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        out = self.compl_mul3d(x, self.weights1)
        return out
    
class MLP_Complex(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP_Complex, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1, dtype=torch.complex64)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1, dtype=torch.complex64)
        self.LeakyReLU = ComplexReLU(0.2)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.LeakyReLU(x)
        x = self.mlp2(x)
        return x
    
class ComplexReLU(nn.Module):
    def __init__(self, negative_slope):
        super(ComplexReLU, self).__init__()
        self.negative_slope = negative_slope
    def forward(self, x):
        LeakyReLU = nn.LeakyReLU(self.negative_slope)
        return torch.complex(LeakyReLU(x.real), LeakyReLU(x.imag))
    
class ComplexTanh(nn.Module):
    def __init__(self):
        super(ComplexTanh, self).__init__()
    def forward(self, x):
        return torch.complex(F.tanh(x.real), F.tanh(x.imag))
    
