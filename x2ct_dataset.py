import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch
import params
from options import options

# options
opt = options().opt

# Transformer
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(256),
        transforms.ToTensor(),
        normalize
    ])

# Train & validation path for X-ray datasets
x_train_path = '~/../yy-volume/datasets/x2ct/covid_x/train/'
x_val_path = '~/../yy-volume/datasets/x2ct/covid_x/val/'
# Train & validation path for CT datasets
ct_train_path = '~/../yy-volume/datasets/x2ct/covid_ct/train/'
ct_val_path = '~/../yy-volume/datasets/x2ct/covid_ct/val/'

# X-ray dataset
# Train
x_dataset_train = datasets.ImageFolder(
    x_train_path,
    transform)
# Validation
x_dataset_val = datasets.ImageFolder(
    x_val_path,
    transform)

# CT dataset
# Train
ct_dataset_train = datasets.ImageFolder(
    ct_train_path,
    transform)
# Validation
ct_dataset_val = datasets.ImageFolder(
    ct_val_path,
    transform)

# Sampler
# print('start')
# x_weights = [155 if label == 0 else 1 for data, label in x_dataset_train]

# x_sampler = WeightedRandomSampler(x_weights, num_samples=192, replacement=True)

# ct_weights = [1 if label == 0 else 1.84 for data, label in ct_dataset_train]
# ct_sampler = WeightedRandomSampler(ct_weights,\
#                                 num_samples=400,\
#                                 replacement=True)


# Dataloader
print('Start load x_train_loader_train...')
x_train_loader_train = DataLoader(
    x_dataset_train, batch_size=opt.batch_size, shuffle=True,
    num_workers=opt.workers, pin_memory=True, sampler=None)

print('Start load x_train_loader_val...')
x_train_loader_val = DataLoader(
    x_dataset_val, batch_size=opt.batch_size, shuffle=True,
    num_workers=opt.workers, pin_memory=True, sampler=None)

print('Start load ct_train_loader_train...')
ct_train_loader_train = DataLoader(
    ct_dataset_train, batch_size=opt.batch_size, shuffle=True,
    num_workers=opt.workers, pin_memory=True, sampler=None)

print('Start load ct_train_loader_val...')
ct_train_loader_val = DataLoader(
    ct_dataset_val, batch_size=opt.batch_size, shuffle=True,
    num_workers=opt.workers, pin_memory=True, sampler=None)




# mnist_train_dataset = datasets.MNIST(root='data/pytorch/MNIST', train=True, download=True,
#                                      transform=transform)
# mnist_valid_dataset = datasets.MNIST(root='data/pytorch/MNIST', train=True, download=True,
#                                      transform=transforms)
# mnist_test_dataset = datasets.MNIST(root='data/pytorch/MNIST', train=False, transform=transform)

# indices = list(range(len(mnist_train_dataset)))
# validation_size = 5000
# train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
# train_sampler = SubsetRandomSampler(train_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)

# mnist_train_loader = DataLoader(
#     mnist_train_dataset,
#     batch_size=params.batch_size,
#     sampler=train_sampler,
#     num_workers=params.num_workers
# )

# mnist_valid_loader = DataLoader(
#     mnist_valid_dataset,
#     batch_size=params.batch_size,
#     sampler=train_sampler,
#     num_workers=params.num_workers
# )

# mnist_test_loader = DataLoader(
#     mnist_test_dataset,
#     batch_size=params.batch_size,
#     num_workers=params.num_workers
# )


# mnist_train_all = (mnist_train_dataset.train_data[5000:].reshape(55000, 28, 28, 1))
# mnist_concat = torch.cat((mnist_train_all, mnist_train_all, mnist_train_all), 3)
# print(mnist_test_dataset.test_labels.shape, mnist_test_dataset.test_labels)


def one_hot_embedding(labels, num_classes=10):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


