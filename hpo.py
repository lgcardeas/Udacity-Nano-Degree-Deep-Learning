#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch

import s3fs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse

import os
import io

from torch.utils.data import DataLoader, Dataset
from PIL import Image

class S3ImageDataset(Dataset):
    """
        I wanted to test something different. There are a couple of way to download / sync the data from s3.
        1 - using aws s3 sycn .... ( this approach actuallt is the easy one the most cost efficiency)
        2 - using boto3 to simulate s3 sync ( this is an overkill )
        3 - using `from torch.utils.data import Dataset` and create our own Dataset streaming directly from S3
        I would like to test 3er approach and thats the reason of this class
    """
    def __init__(self, s3_bucket, prefix, mode="training"):
        self.s3 = s3fs.S3FileSystem()
        self.s3_bucket = s3_bucket
        self.prefix = prefix
        
        # Define transformations with different behavior for training/testing
        if mode == "training":
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:  # For testing or validation
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.files = self.s3.glob(f'{self.s3_bucket}/{self.prefix}/*/*.jpg')
        
        self.classes = []
        self.class_to_idx = {}

        for file in self.files:
            # print("about to check: " + file.split('/')[-2])
            idx, cls = (file.split('/')[-2]).split('.')
            if cls not in self.class_to_idx:
                self.class_to_idx[cls] = int(idx) - 1
                self.classes.append(cls)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]

        try:
            with self.s3.open(file_path, 'rb') as f:
                image = Image.open(io.BytesIO(f.read())).convert('RGB')
            if self.transform:
                image = self.transform(image)

            directory = file_path.split('/')[-2]
            idx, cls = directory.split('.')
            label = int(idx) - 1  # Adjust to zero-indexing if needed
            return image, torch.tensor(label)

        except (OSError, IOError) as e:
            print(f"Warning: Skipping corrupted image at {file_path} - {e}")
            return self.__getitem__((index + 1) % len(self.files))

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def test(model, test_loader, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def train(model, train_loader, criterion, optimizer, epochs=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0
        print(f'Starting Epoch {epoch + 1}/{epochs}')

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.fc.parameters(), max_norm=2.0)

            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / total_samples
        print(f'Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}')

    return model

def net():
    num_classes = 133
    model = models.resnet18(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final layer for the new number of classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),  # Intermediate layer
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)    # Output layer
    )

    # Make only the final layer trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    # Apply weight initialization to the new layers in `fc`
    model.fc.apply(initialize_weights)

    return model

def create_data_loaders(source, batch_size=None, mode="training"):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    dataset = S3ImageDataset(s3_bucket='legc-deep-learning', prefix=source, mode=mode)
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
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    else: #should be sgd, otherwise an error will be trigged
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.learning_rate, momentum=args.momentum)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader = create_data_loaders(source='train', batch_size=1 if args.train_batch_size == None else args.train_batch_size, mode="training")
    loss_criterion = nn.CrossEntropyLoss()
    model=train(model, train_loader, loss_criterion, optimizer, args.epochs)
    
    # '''
    # TODO: Test the model to see its accuracy
    # '''
    test_loader = create_data_loaders(source='test', batch_size=1 if args.test_batch_size == None else args.test_batch_size, mode="testing")
    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), os.path.join(os.environ["SM_MODEL_DIR"], "model.pth"))

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
        default=1,
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

    args=parser.parse_args()
    
    main(args)
