import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch.utils.data as data
import torch

NN = 1000

class NumpyDataset(data.Dataset):

    def __init__(self, array, transform=None):
        super().__init__()
        self.array = array
        self.transform = transform

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        x = self.array[index]
        if self.transform:
            x = self.transform(x)
        return x

def generate_1d_flow_data(n):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n//2,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n//2,))
    return np.concatenate([gaussian1, gaussian2])

def load_flow_demo_0(n, visualize = False):
    x = np.random.uniform(size=n)
    y = np.random.normal(loc=0, scale=1, size=n)
    train_data = np.concatenate([x.reshape(n, 1), y.reshape(n, 1)], axis=1)

    if visualize:
        plt.figure()
        plt.scatter(x, y, marker='o', alpha=0.1)
        plt.show()
    return train_data


def load_flow_demo_1(n_train, n_test, loader_args, visualize=True, train_only=False):
    # 1d distribution, mixture of two gaussians
    train_data, test_data = generate_1d_flow_data(n_train), generate_1d_flow_data(n_test)

    if visualize:
        plt.figure()
        x = np.linspace(-3, 3, num=100)
        densities = 0.5 * norm.pdf(x, loc=-1, scale=0.25) + 0.5 * norm.pdf(x, loc=0.5, scale=0.5)
        plt.figure()
        plt.plot(x, densities)
        plt.show()
        plt.figure()
        plt.hist(train_data, bins=50)
        # plot_hist(train_data, bins=50, title='Train Set')
        plt.show()

    train_dset, test_dset = NumpyDataset(train_data), NumpyDataset(test_data)
    train_loader, test_loader = data.DataLoader(train_dset, **loader_args), data.DataLoader(test_dset, **loader_args)

    if train_only:
        return train_loader
    return train_loader, test_loader



def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def visualize_demo1_flow(train_loader, initial_flow, final_flow):
    plt.figure(figsize=(10,5))
    train_data = torch.FloatTensor(train_loader.dataset.array)

    # before:
    plt.subplot(231)
    plt.hist(get_numpy(train_data), bins=50)
    plt.title('True Distribution of x')

    plt.subplot(232)
    x = torch.FloatTensor(np.linspace(-3, 3, NN))
    z, _ = initial_flow.flow(x)
    plt.plot(get_numpy(x), get_numpy(z))
    plt.title('Flow x -> z')

    plt.subplot(233)
    z_data, _ = initial_flow.flow(train_data)
    plt.hist(get_numpy(z_data), bins=50)
    plt.title('Empirical Distribution of z')

    # after:
    plt.subplot(234)
    plt.hist(get_numpy(train_data), bins=50)
    plt.title('True Distribution of x')

    plt.subplot(235)
    x = torch.FloatTensor(np.linspace(-3, 3, NN))
    z, _ = final_flow.flow(x)
    plt.plot(get_numpy(x), get_numpy(z))
    plt.title('Flow x -> z')

    plt.subplot(236)
    z_data, _ = final_flow.flow(train_data)
    plt.hist(get_numpy(z_data), bins=50)
    plt.title('Empirical Distribution of z')

    plt.tight_layout()

def plot_train_curves(epochs, train_losses, test_losses, title=''):
    x = np.linspace(0, epochs, len(train_losses))
    plt.figure()
    plt.plot(x, train_losses, label='train_loss')
    if test_losses:
        plt.plot(x, test_losses, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()