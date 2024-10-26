#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os
import io
import s3fs
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import argparse

class S3ImageDataset(Dataset):
    """
        I wanted to test something different. There are a couple of way to download / sync the data from s3.
        1 - using aws s3 sycn .... ( this approach actuallt is the easy one the most cost efficiency)
        2 - using boto3 to simulate s3 sync ( this is an overkill )
        3 - using `from torch.utils.data import Dataset` and create our own Dataset streaming directly from S3
        I would like to test 3er approach and thats the reason of this class
    """
    def __init__(self, s3_bucket, prefix):
        self.s3 = s3fs.S3FileSystem()
        self.s3_bucket = s3_bucket
        self.prefix = prefix
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # pretrained modules are ussually 224x224 size
            transforms.ToTensor()           # Convert images to PyTorch tensors
        ])

        self.files = self.s3.glob(f'{self.s3_bucket}/{self.prefix}/*/*.jpg')
        
        self.classes = []
        self.class_to_idx = {}

        for file in self.files:
            # print("about to check: " + file.split('/')[-2])
            idx, cls = (file.split('/')[-2]).split('.')
            if cls not in self.class_to_idx:
                self.class_to_idx[cls] = int(idx)
                self.classes.append(cls)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]

        with self.s3.open(file_path, 'rb') as f:
            image = Image.open(io.BytesIO(f.read())).convert('RGB')

        # By defualt we are using `transforms.ToTensor()`
        #   default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
        #  and transforms.Resize as Images have multiples sizes

        if self.transform:
            image = self.transform(image)

        directory = file_path.split('/')[-2]
        idx, cls = directory.split('.')
        label = int(idx)

        return image, label


def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    device = 'cuda' if tourch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model.to(device)
    model.eval()



def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    device = 'cuda' if tourch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    # Average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total_samples

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy


def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model.to(device)
    model.train()
    total_loss = 0.0
    total_batches = len(train_loader)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # print(f"{batch_idx=}, {inputs=} and {labels=}")
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Print progress every 10 batches (you can change this)
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{total_batches}, Loss: {loss.item():.4f}')

        # Calculate the average loss over all batches in the epoch
        avg_loss = total_loss / total_batches
        print(f'model trained with {avg_loss=}')

        return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # there are 49 types of dogs in the training data
    # for now, lets keep it hard coded, we might want it
    # parametized????....
    # it seems num_class needs to be indexed starting 0 so, I hace 49 classes
    # but, its complaining about the class 49, I assume, it expect starting 0
    num_classes = 50

    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

def create_data_loaders(source, batch_size=None):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    dataset = S3ImageDataset(s3_bucket='legc-deep-learning', prefix=source)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''

    optimizer = None
    # let check for the parameters
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else: #should be sgd, otherwise an error will be trigged
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader = create_data_loaders(source='train', batch_size=None if args.train_batch_size == -1 else args.train_batch_size)
    loss_criterion = nn.CrossEntropyLoss()
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    # test_loader = create_data_loaders(source='test', batch_size=None if args.test_batch_size == -1 else args.test_batch_size)
    # test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, args.path_to_save)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''

    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=64,
        help='train batch size input for training (default: 64)'
    )

    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help='test batch size input for training (default: None)'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.5,
        help='learning rate input for training (default: 0.5)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help="number of epochs to train (default 5)"
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['sgd', 'adam', 'rmsprop'],
        help='Optimizer to train (default: adam)'
    )

    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help="Momentum for SGD if selected (defualt 0.9)"
    )

    parser.add_argument(
        '--path_to_save',
        type=str,
        default='/tmp/default_output_model',
        help="output model path (default /tmp/default_output_model)"
    )

    args=parser.parse_args()
    
    main(args)
