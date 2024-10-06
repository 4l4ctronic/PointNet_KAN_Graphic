# Libraries
import torch
import torch.utils.data as data
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import linecache
from operator import itemgetter
from numpy import zeros

#Parameter setup
NUM_POINTS = 1024 
poly_degree = 4 # Polynomial degree of Jacaboi Polynomial
ALPHA = 1.0 # \alpha in Jacaboi Polynomial
BETA = 1.0 # \beta in Jacaboi Polynomial
SCALE = 1.0 # To control the size of tensor A in the manuscript
Feature = 3 # (x,y,z)

###### Object: KANshared (i.e., shared KAN) ######
class KANshared(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=ALPHA, b=BETA):
        super(KANshared, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        batch_size, input_dim, num_points = x.shape
        x = x.permute(0, 2, 1).contiguous() 
        x = torch.tanh(x) 

        jacobi = torch.ones(batch_size, num_points, self.input_dim, self.degree + 1, device=x.device)

        if self.degree > 0:
            jacobi[:, :, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        for i in range(2, self.degree + 1):
            A = (2*i + self.a + self.b - 1)*(2*i + self.a + self.b)/((2*i) * (i + self.a + self.b))
            B = (2*i + self.a + self.b - 1)*(self.a**2 - self.b**2)/((2*i)*(i + self.a + self.b)*(2*i+self.a+self.b-2))
            C = -2*(i + self.a -1)*(i + self.b -1)*(2*i + self.a + self.b)/((2*i)*(i + self.a + self.b)*(2*i + self.a + self.b -2))
            jacobi[:, :, :, i] = (A*x + B)*jacobi[:, :, :, i-1].clone() + C*jacobi[:, :, :, i-2].clone()

        jacobi = jacobi.permute(0, 2, 3, 1)  
        y = torch.einsum('bids,iod->bos', jacobi, self.jacobi_coeffs) 
        return y

###### Object: KAN ######
class KAN(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=ALPHA, b=BETA):
        super(KAN, self).__init__()
        self.inputdim = input_dim
        self.outdim   = output_dim
        self.a        = a
        self.b        = b
        self.degree   = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim)) 
        
        x = torch.tanh(x)
        
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        for i in range(2, self.degree + 1):
            A = (2*i + self.a + self.b - 1)*(2*i + self.a + self.b)/((2*i) * (i + self.a + self.b))
            B = (2*i + self.a + self.b - 1)*(self.a**2 - self.b**2)/((2*i)*(i + self.a + self.b)*(2*i+self.a+self.b-2))
            C = -2*(i + self.a -1)*(i + self.b -1)*(2*i + self.a + self.b)/((2*i)*(i + self.a + self.b)*(2*i + self.a + self.b -2))
            jacobi[:, :, i] = (A*x + B)*jacobi[:, :, i-1].clone() + C*jacobi[:, :, i-2].clone()

        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs) 
        y = y.view(-1, self.outdim)
        return y

###### Deep PointNet KAN For Segmentation ###### 
class DeepPointNetKAN(nn.Module):
    def __init__(self, input_channels, output_channels, scaling=SCALE):
        super(PointNetKAN, self).__init__()

        #Shared KAN (64, 64, 128)
        self.jacobikan1 = KANshared(input_channels, int(64 * scaling), poly_degree)
        self.jacobikan2 = KANshared(int(64 * scaling), int(64 * scaling), poly_degree)
        self.jacobikan3 = KANshared(int(64 * scaling), int(128 * scaling), poly_degree)

        #Shared KAN (128, 512, max pool 2048)
        self.jacobikan4 = KANshared(int(128 * scaling), int(512 * scaling), poly_degree)
        self.jacobikan5 = KANshared(int(512 * scaling), int(2048 * scaling), poly_degree)
        
        #Shared KAN (512, 256, 128, output_channels)
        self.jacobikan6 = KANshared(int(2048 * scaling) + 14*int(64 * scaling) + int(num_objects), int(512 * scaling), poly_degree)
        self.jacobikan7 = KANshared(int(512 * scaling), int(256 * scaling), poly_degree)
        self.jacobikan8 = KANshared(int(256 * scaling), int(128 * scaling), poly_degree)
        self.jacobikan9 = KANshared(int(128 * scaling), output_channels, poly_degree)
        
        #Batch Normalization
        self.bn1 = nn.BatchNorm1d(int(64 * scaling))
        self.bn2 = nn.BatchNorm1d(int(64 * scaling))
        self.bn3 = nn.BatchNorm1d(int(128 * scaling))
        self.bn4 = nn.BatchNorm1d(int(512 * scaling))
        self.bn5 = nn.BatchNorm1d(int(2048 * scaling))
        self.bn6 = nn.BatchNorm1d(int(512 * scaling))
        self.bn7 = nn.BatchNorm1d(int(256 * scaling))
        self.bn8 = nn.BatchNorm1d(int(128 * scaling))
 
        #input_transform
        self.ITjacobikan1 = KANshared(input_channels, int(64 * scaling), poly_degree)
        self.ITjacobikan2 = KANshared(int(64 * scaling), int(128 * scaling), poly_degree)
        self.ITjacobikan3 = KANshared(int(128 * scaling), int(1024 * scaling), poly_degree)

        self.ITbn1 = nn.BatchNorm1d(int(64 * scaling))
        self.ITbn2 = nn.BatchNorm1d(int(128 * scaling))
        self.ITbn3 = nn.BatchNorm1d(int(1024 * scaling))

        self.ITjacobikan4 = KAN(int(1024 * scaling), int(512 * scaling), poly_degree)
        self.ITjacobikan5 = KAN(int(512 * scaling), int(256 * scaling), poly_degree)

        self.ITbn4 = nn.BatchNorm1d(int(512 * scaling))
        self.ITbn5 = nn.BatchNorm1d(int(256 * scaling))

        self.ITjacobikan6 = KAN(int(256 * scaling), Feature * Feature,  poly_degree)

        self.ITbn6 = nn.BatchNorm1d(Feature * Feature)

        #feature_transform
        self.FTjacobikan1 = KANshared(int(128 * scaling), int(128 * scaling), poly_degree)
        self.FTjacobikan2 = KANshared(int(128 * scaling), int(128 * scaling), poly_degree)
        self.FTjacobikan3 = KANshared(int(128 * scaling), int(1024 * scaling), poly_degree)

        self.FTbn1 = nn.BatchNorm1d(int(128 * scaling))
        self.FTbn2 = nn.BatchNorm1d(int(128 * scaling))
        self.FTbn3 = nn.BatchNorm1d(int(1024 * scaling))

        self.FTjacobikan4 = KAN(int(1024 * scaling), int(512 * scaling), poly_degree)
        self.FTjacobikan5 = KAN(int(512 * scaling), int(256 * scaling), poly_degree)

        self.FTbn4 = nn.BatchNorm1d(int(512 * scaling))
        self.FTbn5 = nn.BatchNorm1d(int(256 * scaling))

        self.FTjacobikan6 = KAN(int(256 * scaling), 128 * 128, poly_degree)

        self.FTbn6 = nn.BatchNorm1d(128 * 128)

    def forward(self, x, class_label):

        The_intput = x

        # Input Transform
        batch_size = x.size(0)
        x = self.ITjacobikan1(x)
        x = self.ITbn1(x)
        x = self.ITjacobikan2(x)
        x = self.ITbn2(x)
        x = self.ITjacobikan3(x)
        x = self.ITbn3(x)

        The_feature = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)

        x = self.ITjacobikan4(The_feature)
        x = self.ITbn4(x)
        x = self.ITjacobikan5(x)
        x = self.ITbn5(x)
        x = self.ITjacobikan6(x)
        x = self.ITbn6(x)

        x = x.view(batch_size, Feature, Feature)
        x = torch.bmm(The_intput.transpose(1, 2), x).transpose(1, 2)
        # End of Input Transform

        # Shared KAN (64, 64, 128)
        x = self.jacobikan1(x)
        x = self.bn1(x)
        
        local_1 = x

        x = self.jacobikan2(x)
        x = self.bn2(x)

        local_2 = x

        x = self.jacobikan3(x)
        x = self.bn3(x)

        local_3 = x

        The_intermeidate = x

        # Feature Transform
        x = self.FTjacobikan1(x)
        x = self.FTbn1(x)
        x = self.FTjacobikan2(x)
        x = self.FTbn2(x)
        x = self.FTjacobikan3(x)
        x = self.FTbn3(x)

        The_feature = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)

        x = self.FTjacobikan4(The_feature)
        x = self.FTbn4(x)
        x = self.FTjacobikan5(x)
        x = self.FTbn5(x)
        x = self.FTjacobikan6(x)
        x = self.FTbn6(x)

        x = x.view(batch_size, 128, 128)
        x = torch.bmm(The_intermeidate.transpose(1, 2), x).transpose(1, 2)
        # End of Input Transform

        # Shared KAN (128, 512, 2048)
  
        local_4 = x

        x = self.jacobikan4(x)
        x = self.bn4(x)

        local_5 = x

        x = self.jacobikan5(x)
        x = self.bn5(x)

        class_label = class_label.view(-1, num_objects, 1)
        class_label = class_label.view(-1, class_label.size(1), 1).expand(-1, -1, num_points)

        # Max pooling to get the global feature
        global_feature = F.max_pool1d(x, kernel_size=x.size(-1))
        global_feature = global_feature.view(-1, global_feature.size(1), 1).expand(-1, -1, num_points)

        # Concatenate local and global features
        x = torch.cat([local_1 , local_2, local_3, local_4, local_5, global_feature, class_label], dim=1)

        # Shared KAN (256, 256, 128, output_channels)
        x = self.jacobikan6(x)
        x = self.bn6(x)
        x = self.jacobikan7(x)
        x = self.bn7(x)
        x = self.jacobikan8(x)
        x = self.bn8(x)
        x = self.jacobikan9(x)

        return x
