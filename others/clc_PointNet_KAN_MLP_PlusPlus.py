# Libraries
import torch
import torch.utils.data as data
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import linecache
from operator import itemgetter
from numpy import zeros
from torchsummary import summary
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import h5py

# Parameter setup
NUM_POINTS = 2048 #1024 
NUM_CLASSES = 40  # ModelNet40
BATCH_SIZE = 64 
poly_degree = 2   # Polynomial degree of Jacobi Polynomial
FEATURE = 6       # (x,y,z) + (nx,ny,nz)
ALPHA = -0.5       #  in Jacobi Polynomial
BETA = -0.5        #  in Jacobi Polynomial
SCALE = 1.0       # To control the size of tensor A in the manuscript
MAX_EPOCHS = 100
direction = '/scratch/users/kashefi/KANmlp/ModelNet40'

###### Function: parse_dataset ######
def parse_dataset(num_points=NUM_POINTS):
    train_points_with_normals = []
    train_labels = []
    test_points_with_normals = [] 
    test_labels = []
    class_map = {}

    DATA_DIR = direction 
    folders = glob.glob(os.path.join(DATA_DIR, "*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        class_map[i] = folder.split("/")[-1]
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files  = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            mesh = trimesh.load(f)
            points, face_indices = mesh.sample(num_points, return_index=True)
            normals = mesh.face_normals[face_indices]
            points_with_normals = np.concatenate([points, normals], axis=1)
            train_points_with_normals.append(points_with_normals)
            train_labels.append(i)

        for f in test_files:
            mesh = trimesh.load(f)
            points, face_indices = mesh.sample(num_points, return_index=True)
            normals = mesh.face_normals[face_indices]  
            points_with_normals = np.concatenate([points, normals], axis=1)
            test_points_with_normals.append(points_with_normals)
            test_labels.append(i)

    train_points = torch.tensor(np.array(train_points_with_normals), dtype=torch.float32)
    test_points  = torch.tensor(np.array(test_points_with_normals),  dtype=torch.float32)
    train_labels = torch.tensor(np.array(train_labels), dtype=torch.long)
    test_labels  = torch.tensor(np.array(test_labels),  dtype=torch.long)

    return train_points, test_points, train_labels, test_labels, class_map

###### Object: PointCloudDataset ######
class PointCloudDataset(Dataset):
    def __init__(self, points, labels):
        self.points = points
        self.labels = labels
        self.normalize()

    def normalize(self):
        for i in range(self.points.shape[0]):
            spatial_coords = self.points[i, :, :3]  
            normals        = self.points[i, :, 3:]  
            centroid       = spatial_coords.mean(axis=0, keepdims=True)
            spatial_coords -= centroid
            furthest_distance = torch.max(torch.sqrt(torch.sum(spatial_coords ** 2, axis=1, keepdims=True)))
            spatial_coords /= furthest_distance
            self.points[i] = torch.cat((spatial_coords, normals), dim=1)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point = self.points[idx]
        label = self.labels[idx]
        return point, label

###### Object: KANshared (shared KAN) ######
class KANshared(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=ALPHA, b=BETA):
        super(KANshared, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.a          = a
        self.b          = b
        self.degree     = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        batch_size, input_dim, num_points = x.shape
        x = x.permute(0, 2, 1).contiguous()  # [B, N, input_dim]
        x = torch.tanh(x)

        jacobi = torch.ones(batch_size, num_points, input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            jacobi[:, :, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        for i in range(2, self.degree + 1):
            A = (2*i + self.a + self.b - 1)*(2*i + self.a + self.b)/((2*i) * (i + self.a + self.b))
            B = (2*i + self.a + self.b - 1)*(self.a**2 - self.b**2)/((2*i)*(i + self.a + self.b)*(2*i+self.a+self.b-2))
            C = -2*(i + self.a -1)*(i + self.b -1)*(2*i + self.a + self.b)/((2*i)*(i + self.a + self.b)*(2*i + self.a + self.b -2))
            jacobi[:, :, :, i] = (A*x + B)*jacobi[:, :, :, i-1].clone() + C*jacobi[:, :, :, i-2].clone()

        jacobi = jacobi.permute(0, 2, 3, 1)  # [B, input_dim, degree+1, N]
        y = torch.einsum('bids,iod->bos', jacobi, self.jacobi_coeffs)
        return y

###### Object: KAN ######
class KAN(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=ALPHA, b=BETA):
        super(KAN, self).__init__()
        self.inputdim      = input_dim
        self.outdim        = output_dim
        self.a             = a
        self.b             = b
        self.degree        = degree

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

###### Object: PointNetKAN (original) ######
class PointNetKAN(nn.Module):
    def __init__(self, input_channels, output_channels, scaling=SCALE):
        super(PointNetKAN, self).__init__()
        self.jacobikan5 = KANshared(input_channels, int(1024 * scaling), poly_degree)
        self.jacobikan6 = KAN(int(1024 * scaling), output_channels, poly_degree)
        self.bn5        = nn.BatchNorm1d(int(1024 * scaling))

    def forward(self, x):
        x = self.jacobikan5(x)
        x = self.bn5(x)
        global_feature = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
        x = self.jacobikan6(global_feature)
        return x


# ----------- Begin PointNet++ Integration ------------

def square_distance(src, dst):
    """Calculate squared Euclid distance between each two points."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Selects points from `points` according to `idx`.
    points: (B, N, C)
    idx:    (B, S1, S2, ..., Sk) indices into the N dimension
    returns: (B, S1, S2, ..., Sk, C)
    """
    B, N, C = points.shape
    # make a batch index that has the same shape as idx
    # e.g. if idx is (B, S) we want batch_indices of shape (B, S)
    batch_shape = idx.shape  # (B, S1, S2, ..., Sk)
    # we create a tensor [0,1,...,B-1] of shape (B,)
    # then view it as (B,1,1,...,1) with k ones, and expand to batch_shape
    device = points.device
    batch_indices = torch.arange(B, device=device) \
                        .view(B, *([1] * (idx.dim()-1))) \
                        .expand(batch_shape)
    # now both batch_indices and idx are the same shape
    # we can do pointwise gather:
    new_points = points[batch_indices, idx, :]  # (B, S1, S2, ..., Sk, C)
    return new_points

def farthest_point_sample(xyz, npoint):
    """Farthest point sampling."""
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """Group points within radius."""
    sqrdists = square_distance(new_xyz, xyz)
    idx = sqrdists.sort(dim=-1)[1][:, :, :nsample]  # (B, S, nsample)
    # handle out-of-radius points
    group_first = idx[:, :, 0:1].repeat(1, 1, nsample)
    mask = sqrdists.gather(2, idx) > radius ** 2
    idx[mask] = group_first[mask]
    return idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """Perform sampling and grouping for SA layer."""
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)                # (B, npoint, 3)
    idx     = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)                 # (B, npoint, nsample, 3)
    grouped_xyz -= new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)       # (B, npoint, nsample, D)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B, npoint, nsample, 3+D)
    else:
        new_points = grouped_xyz
    # reshape to (B, 3+D, nsample, npoint)
    new_points = new_points.permute(0, 3, 2, 1)
    return new_xyz.permute(0, 2, 1), new_points  # new_xyz: (B,3,npoint)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint    = npoint
        self.radius    = radius
        self.nsample   = nsample
        self.group_all = group_all

        # build shared MLPs via KANshared + BN1d
        self.mlps = nn.ModuleList()
        self.bns  = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlps.append(
                KANshared(input_dim=last_channel,
                          output_dim=out_channel,
                          degree=poly_degree)               # THIS IS SHARED
            )
            self.bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz:    (B, 3, N)
        points: (B, D, N) or None
        returns:
          new_xyz:    (B, 3, S)
          new_points: (B, mlp[-1], S)
        """
        if self.group_all:
            # group_all: use all N points into one region
            B, _, N = xyz.shape
            new_xyz = torch.zeros(B, 3, 1, device=xyz.device)
            grouped_xyz    = xyz.unsqueeze(2)        # (B, 3, 1, N)
            if points is not None:
                grouped_pts = points.unsqueeze(2)    # (B, D, 1, N)
                new_points  = torch.cat([grouped_xyz, grouped_pts], dim=1)
            else:
                new_points = grouped_xyz             # (B, C+D, 1, N)
        else:
            # sampling + grouping
            new_xyz, grouped = sample_and_group(
                self.npoint, self.radius, self.nsample,
                xyz.permute(0,2,1),
                None if points is None else points.permute(0,2,1)
            )
            # grouped: (B, C+D, nsample, S)
            new_points = grouped

        # now new_points is (B, C_in, K, S)  with K = nsample or 1
        B, C_in, K, S = new_points.shape

        # rearrange so each local patch is fed into KANshared
        new_points = new_points.permute(0,3,1,2).reshape(B*S, C_in, K)
        for kan, bn in zip(self.mlps, self.bns):
            new_points = kan(new_points)            # THIS IS SHARED
            new_points = bn(new_points)             # BatchNorm1d
            #new_points = F.relu(new_points) # no need to activation!

        # reshape back to (B, out_channel, K, S)
        out_ch = new_points.size(1)
        new_points = new_points.reshape(B, S, out_ch, K).permute(0,2,3,1)

        # correct pooling:
        if self.group_all:
            # pool over the S axis (the original N points)
            new_points = torch.max(new_points, dim=-1)[0]   # (B, out_ch, 1)
        else:
            # pool over the K axis (nsample neighbors)
            new_points = torch.max(new_points, dim=2)[0]    # (B, out_ch, S)

        return new_xyz, new_points


class PointNetPlusPlus(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(PointNetPlusPlus, self).__init__()
        # input_channels = 6 => normals included
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32,
                                          in_channel=input_channels-3, mlp=[64,64,128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                          in_channel=128, mlp=[128,128,256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                          in_channel=256, mlp=[256,512,1024], group_all=True)
        self.fc1   = nn.Linear(1024, 512)
        self.bn1   = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2   = nn.Linear(512, 256)
        self.bn2   = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3   = nn.Linear(256, output_channels)

    def forward(self, x):
        B, _, _ = x.size()
        xyz    = x[:, :3, :]     # (B,3,N)
        points = x[:, 3:, :]     # (B,3,N)
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, -1)               # (B,1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x

###### Loading data, setting devices ######
train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(NUM_POINTS)

train_dataset = PointCloudDataset(train_points, train_labels)
test_dataset  = PointCloudDataset(test_points,  test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### Model setup ######
# Replace original PointNetKAN with PointNet++:
model = PointNetPlusPlus(input_channels=FEATURE, output_channels=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

###### Training ######
for epoch in range(MAX_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for points, labels in train_loader:
        points, labels = points.to(device), labels.to(device)
        points = points.transpose(1, 2)  # (B,FEATURE,NUM_POINTS)
        optimizer.zero_grad()
        outputs = model(points)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * points.size(0)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss     = running_loss / len(train_loader.dataset)
    epoch_accuracy= 100 * correct / total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total   = 0
    class_correct_val = np.zeros(NUM_CLASSES)
    class_total_val   = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)
            points = points.transpose(1, 2)
            outputs = model(points)
            loss    = criterion(outputs, labels)

            val_loss += loss.item() * points.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total_val[label]   += 1
                class_correct_val[label] += (predicted[i] == label).item()

    val_loss     /= len(test_loader.dataset)
    val_accuracy = 100 * val_correct / val_total
    class_accuracy_val = 100 * np.divide(
        class_correct_val, class_total_val, out=np.zeros_like(class_correct_val), where=class_total_val!=0
    )
    average_class_accuracy_val = np.mean(class_accuracy_val)

    scheduler.step()

    print(f"Epoch {epoch+1}/{MAX_EPOCHS}, "
          f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, "
          f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.2f}%, "
          f"Avg Class Accuracy Test: {average_class_accuracy_val:.2f}%")
