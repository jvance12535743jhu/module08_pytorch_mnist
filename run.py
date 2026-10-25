import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from utilities import ConvNet, myargs, training, plot

def main():
    # Parse command-line arguments
    args = myargs() # from utilities.py

    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST dataset
    # Download & load the training dataset, applying a transformation to convert images to PyTorch tensors
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    # Download & load the test dataset, applying a transformation to convert images to PyTorch tensors
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    # DataLoader provides an iterable over the dataset with support for batching, shuffling, & parallel data loading
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size, # replaced by command line argument and parser
                                               shuffle=True)

    # DataLoader for the test dataset, used for evaluating the model
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size, # replaced by command line argument and parser
                                              shuffle=False)

    # Create an instance of the model & move it to the configured device (GPU/CPU)
    model = ConvNet().to(device)

    # Print the model for observation
    # You can use the following libraries to observe the flowchart of the models similar to TensorFlow
    # from torchview import draw_graph; from torchviz import make_dot
    # see more at:
    # https://github.com/mert-kurttutan/torchview
    # https://github.com/szagoruyko/pytorchviz
    print(model)

    # Loss & optimizer
    # CrossEntropyLoss combines nn.LogSoftmax & nn.NLLLoss in one single class
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer with the specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) # learning_rate from command line argument and parser

    # Training the model
    model, loss_list, acc_train, acc_test = training(model, args.num_epochs, # num_epochs from command line argument and parser
                                                    train_loader, test_loader, criterion, optimizer, device) # also needs test_loader and device
    # Make some plots!
    plot(loss_list, acc_train, acc_test) # saves to ./figure.png

if __name__ == '__main__':
    main()
