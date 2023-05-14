import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib as plt
from PIL import Image

## SpatialSoftmax implementation taken from https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834
class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = torch.nn.Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

#Implementation of Network from Figure 3 (on pg 4) of paper
class VRNet(nn.Module):
    def __init__(self):
        super(VRNet, self).__init__()
        # Convolution 1 160x120x4 -> 77x57x80
        self.conv1_rgb = nn.Conv2d(3, 64, 7, padding='valid', stride=2)
        self.conv1_depth = nn.Conv2d(1, 16, 7, padding='valid', stride=2)
        # Convolution 2 77x57x80 -> 77x57x32
        self.conv2 = nn.Conv2d(80, 32, 1, padding='same')
        self.conv2_bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.99)
        # Convolution 3 77x57x43 -> 75x55x32
        self.conv3 = nn.Conv2d(32, 32, 3, padding='valid')
        self.conv3_bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.99)
        # Convolution 4 75x55x32 -> 73x53x32
        self.conv4 = nn.Conv2d(32, 32, 3, padding='valid')
        self.conv4_bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.99)

        # spatial softmax
        self.spatialSoftmax = SpatialSoftmax(53, 73, 32, temperature=1, data_format='NCHW')

        self.flatten = nn.Flatten()

        #fully connected layers
        self.fc1 = nn.Linear(64, 50)
        self.fc1_bn = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_bn = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.fc3 = nn.Linear(50, 50)
        self.fc3_bn = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.fc4 = nn.Linear(50, 7) # Vx, Vy, Vz, Wx, Wy, Wz, grabber open

        #set conv1_rgb weights to be first layer from pretrained model
        googlenet = torchvision.models.googlenet(pretrained=True)
        self.conv1_rgb.weight.data = googlenet.conv1.conv.weight.data

        #weights should be uniformly sampled from [-0.1, 0.1]
        self.conv1_depth.weight.data.uniform_(-0.1, 0.1)
        self.conv2.weight.data.uniform_(-0.1, 0.1)
        self.conv3.weight.data.uniform_(-0.1, 0.1)
        self.conv4.weight.data.uniform_(-0.1, 0.1)

        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        self.fc4.weight.data.uniform_(-0.1, 0.1)

    def forward(self, rgbImg, depthImg):
        #conv layers
        x_rgb = F.relu(self.conv1_rgb(rgbImg))
        x_depth = F.relu(self.conv1_depth(depthImg))
        x = torch.cat((x_rgb, x_depth), 1)
        
        #implement convulutional layers with batch normalization
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.spatialSoftmax(x)
        x = self.flatten(x)

        #fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class VRDataLoader(Dataset):
    def __init__(self, data_dir, startRun, lastRun, batch_size=1):
        self.data_dir = data_dir
        self.startRun = startRun
        self.lastRun = lastRun
        self.batch_size = batch_size
        self.rgb_images, self.depth_images, self.states = self.load_data()
        self.arrayIndicies = list([i for i in range(len(self.rgb_images))])
        print(len(self.rgb_images), len(self.depth_images), len(self.states))
        assert(len(self.rgb_images) == len(self.depth_images) == len(self.states))

    def load_data(self):
        rgbs = []
        depths = []
        states = []
        
        for i in range(self.lastRun - self.startRun):
            rgb_dir = os.path.join(self.data_dir, f'{i+self.startRun}', 'rgb')
            depth_dir = os.path.join(self.data_dir, f'{i+self.startRun}', 'depth')
            state_dir = os.path.join(self.data_dir, f'{i+self.startRun}', 'states')
            
            state_names = os.listdir(state_dir) #get all files in the directory
            state_names = [state_name for state_name in state_names if state_name.endswith('.csv')] #only get the csv files
            num_points = len(state_names)
            lastState = None

            this_run_states = []
            for i in range(num_points):
                rgb_path = os.path.join(rgb_dir, f'rgb{i}.png')
                depth_path = os.path.join(depth_dir, f'depth{i}.png')
                state_path = os.path.join(state_dir, f'states{i}.csv')

                with open(state_path, 'r') as f:
                    data = f.readlines()
                    isOpen = int(data[0].split(',')[6])
                    data = data[1].split(',')
                    data.append(isOpen)
                    #convert to pytorch tensor
                    state = np.array([float(x) for x in data])
                    this_state = np.zeros(7)
                    this_state[0:6] = state[0:6]
                    this_state[6] = state[6]
                    this_run_states.append(this_state)

                if i == 0:
                    continue
                
                # rgb = torchvision.io.read_image(rgb_path)
                rgb = Image.open(rgb_path)
                # depth = torchvision.io.read_image(depth_path)
                depth = Image.open(depth_path)
                rgb = torchvision.transforms.ToTensor()(rgb)
                depth = torchvision.transforms.ToTensor()(depth) 
                rgbs.append(rgb)
                depths.append(depth)

            #smooth out velocities with a gaussian for this run
            this_run_states = np.array(this_run_states, dtype=np.float32)

            size = 15
            sigma = 4
            filter_range = np.linspace(-int(size/2),int(size/2),size)
            gaussian_filter = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
            
            #smooth out positions with a gaussian
            for i in range(6):
                this_run_states[:, i] = np.convolve(this_run_states[:, i], gaussian_filter, mode='same')

            #take numerical derivative of position to get velocity
            this_run_states[:-1, 0:6] = np.diff(this_run_states[:, 0:6], axis=0)
            this_run_states = this_run_states[1:, :]

            #smooth out velocities with a gaussian
            # for i in range(6):
            #     this_run_states[:, i] = np.convolve(this_run_states[:, i], gaussian_filter, mode='same')

            #add to total list of states
            states.extend(this_run_states)

        #smooth out velocities with a moving average
        states = np.array(states, dtype=np.float32)
        
        #plot x velocity after smoothing
        states = torch.tensor(states).to('cuda')
        
        #compute mean and std of images
        rgbs = torch.stack(rgbs).float() / 255
        depths = torch.stack(depths).float() / 255
        rgb_mean = torch.mean(rgbs, dim=(0, 2, 3))
        depth_mean = torch.mean(depths, dim=(0, 2, 3))
        
        #compute std
        rgb_std = torch.std(rgbs, dim=(0, 2, 3))
        depth_std = torch.std(depths, dim=(0, 2, 3))
        #normalize images
        rgbs[:,0,:,:] = (rgbs[:,0,:,:] - rgb_mean[0]) / rgb_std[0]
        rgbs[:,1,:,:] = (rgbs[:,1,:,:] - rgb_mean[0]) / rgb_std[1]
        rgbs[:,2,:,:] = (rgbs[:,2,:,:] - rgb_mean[0]) / rgb_std[2]
        depths = (depths - depth_mean) / depth_std

        print('rgb mean: ', rgb_mean)
        print('rgb std: ', rgb_std)
        print('depth mean: ', depth_mean)
        print('depth std: ', depth_std)
        print('states mean: ', torch.mean(states, dim=0))
        print('states std: ', torch.std(states, dim=0))

        #normalize states
        for i in range(6):
            states[:, i] = (states[:, i] - torch.mean(states[:, i])) / torch.std(states[:, i])

        return rgbs, depths, states
    
    def __len__(self):
        return len(self.states) // self.batch_size

    def __getitem__(self, idx):
        #shuffle array index mapping
        # if idx == 0:
        np.random.shuffle(self.arrayIndicies)
            
        idx = idx * self.batch_size
        desiredIndexes = self.arrayIndicies[idx:idx+self.batch_size]

        rgb_img = []
        depth_img = []
        state = []
        
        for i in desiredIndexes:
            rgb_img.append(self.rgb_images[i])
            depth_img.append(self.depth_images[i])
            state.append(self.states[i])

        rgb_img = torch.stack(rgb_img)
        depth_img = torch.stack(depth_img)
        state = torch.stack(state)

        return rgb_img, depth_img, state

class DataPreprocessor():
    def __init__(self, rgb_mean, rgb_std, depth_mean, depth_std, state_mean, state_std):
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.state_mean = state_mean
        self.state_std = state_std

    def normalizeRgb(self, rgb):
        rgb[:,0,:,:] = (rgb[:,0,:,:] - self.rgb_mean[0]) / self.rgb_std[0]
        rgb[:,1,:,:] = (rgb[:,1,:,:] - self.rgb_mean[1]) / self.rgb_std[1]
        rgb[:,2,:,:] = (rgb[:,2,:,:] - self.rgb_mean[2]) / self.rgb_std[2]
        return rgb
    
    def normalizeDepth(self, depth):
        return (depth - self.depth_mean) / self.depth_std
    
    def normalizeState(self, state):
        for i in range(6):
            state[:, i] = (state[:, i] - self.state_mean[i]) / self.state_std[i]
        return state
    
    def denormalizeState(self, state):
        for i in range(6):
            state[:, i] = (state[:, i] * self.state_std[i]) + self.state_mean[i]
        return state
    

