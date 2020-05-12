# Python standard library
from pathlib import Path
from random import shuffle
# Third-party libraries, installable via pip
import numpy as np
from tifffile import imread, imwrite
# Install instructions: https://pytorch.org/get-started/locally/#start-locally
# Make sure to install pytorch with CUDA support, and have a CUDA-able GPU.
print('Importing pytorch...', end='')
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
assert torch.cuda.is_available(), 'No GPU available for pytorch.'
print(' done.')

# Set input/output behavior
input_dir = Path('./2_random_forest_annotations')
output_dir = Path('./3_neural_network_annotations')
other_output_dir = Path('./neural_network_intermediate_files')
saved_state_path = other_output_dir / 'neural_network_state.pt'
save_debug_imgs = True
max_label = 3

# Sanity checks on input/output
assert input_dir.is_dir(), "Input directory does not exist"
img_filenames = [x for x in input_dir.iterdir() if x.suffix == '.tif']
assert len(img_filenames) > 0, "No annotated images to process"
example_image = imread(str(img_filenames[0]))
num_input_channels = example_image.shape[0] - 1
assert all(label in range(max_label + 1)
           for label in np.unique(example_image[-1, :, :]))
assert max_label < 2**8 # Bro you don't need more
output_dir.mkdir(exist_ok=True)
if save_debug_imgs:
    other_output_dir.mkdir(exist_ok=True)

# Pick a neural network to train.
print("Initialzing model...", end='')
model = torchvision.models.segmentation.fcn_resnet50(
    num_classes=max_label, # Only guess the annotated pixels
    pretrained=False, pretrained_backbone=True)
model.backbone.conv1 = nn.Conv2d( # fcn_resnet50 assumes RGB input
    num_input_channels, # The first layer should be N-channel, not RGB
    64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model = model.cuda()
print(" done.")

# NB this learning rate and weight decay might be hilariously wrong. Be careful!
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-2, weight_decay=1e-3, amsgrad=True)

# Pick up where we left off, if we've already done some training:
starting_epoch = 0
if saved_state_path.is_file():
    print("Loading saved model and optimizer state...", end='')
    checkpoint = torch.load(saved_state_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    starting_epoch = 1 + checkpoint['epoch']
    model.train()
    print(' done.')

def loss_fn(output, target):
    # Our target has an extra slice at the beginning for unannotated pixels:
    output, target = output['out'], target[:, 1:, :, :]
    assert output.shape == target.shape
    # Each class gets an equal vote, and each annotated pixel in a given
    # class class gets an equal vote:
    annotated_pixels_per_label = torch.sum(target, dim=(2, 3), keepdim=True)
    label_weight = 1e-9 + max_label * annotated_pixels_per_label
    probs = F.softmax(output, dim=1)
    return torch.sum(probs * target / (-label_weight))

def load_data(img_path):
    img = imread(str(img_path))
    assert img.shape[0] == 1 + num_input_channels
    assert len(img.shape) == 3
    # First channels hold the raw images. Match shape and dtype to what
    # torch expects: (batch_size, num_input_channels, y, x) and float32
    input_ = torch.cuda.FloatTensor(img[np.newaxis, :-1, ...].astype('float32'))
    input_.requires_grad = True
    # Last channel holds our annotations of the raw image. Annotation
    # values are ints ranging from 0 to max_label; each different
    # annotation value signals a different label. We unpack these into a
    # "1-hot" representation called 'target'.
    labels = img[np.newaxis, -1:, :, :].astype('uint8')
    assert labels.max() <= max_label
    # We pass a small dtype to the GPU, but the on-GPU dtype has to be
    # Long to work with .scatter_():
    labels = torch.cuda.LongTensor(labels)
    # An empty Boolean array to be filled with our "1-hot" representation:
    target = torch.cuda.BoolTensor(
        1, max_label + 1, img.shape[1], img.shape[2]
        ).zero_() # Initialization to zero is not automatic!
    # Copy each class into its own boolean image:
    target.scatter_(dim=1, index=labels.data, value=True)
    return input_, target

def save_output(output, img_path):
    guess = F.softmax(output['out'].cpu().data, dim=1).numpy().astype('float32')
    imwrite(img_path,
            guess,
            photometric='MINISBLACK',
            imagej=True,
            ijmetadata={'Ranges:', (0, 1)*guess.shape[1]})    

for epoch in range(starting_epoch, 100000): # Basically forever
    img_paths = [x for x in input_dir.iterdir() if x.suffix == '.tif']
    shuffle(img_paths)
    loss_list = []
    print("\nEpoch", epoch)
    for i, img_path in enumerate(img_paths):
        print('.', sep='', end='')
        input_, target = load_data(img_path)
        output = model(input_)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Outputs for inspection
        loss_list.append(loss.detach().item())
        save_output(output, output_dir / img_path.name)
        if save_debug_imgs and i == 0:
            save_output(output,
                        other_output_dir / ('e%06i_'%epoch + img_path.name))
    print('\nLosses:')
    print(''.join('%0.5f '%x for x in loss_list))
    if saved_state_path.is_file():
        saved_state_path.replace(saved_state_path.parent /
                                 (saved_state_path.stem + '_backup.pt'))
    torch.save(
        {'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict()},
        saved_state_path)




