import os

import numpy as np
import tifffile

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

assert torch.cuda.is_available(), 'No GPU available'

num_input_channels = 1
num_labels = 4 # Including 0, the "unannotated" label
assert num_labels < 2**8 # Bro you don't need more
model = torchvision.models.segmentation.fcn_resnet50(
    num_classes=num_labels - 1, # Only guess the annotated classes
    pretrained=False, pretrained_backbone=True)
model.backbone.conv1 = nn.Conv2d( # Input probably isn't RGB
    num_input_channels, # Leave the other parameters unchanged
    64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model = model.cuda()

optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-2, amsgrad=True, weight_decay=1e-3)

saved_state_filename = './saved_model.pt'
backup_saved_state_filename = './backup_saved_model.pt'
starting_epoch = 0
if os.path.exists(saved_state_filename):
    print("Loading saved model and optimizer state.")
    checkpoint = torch.load(saved_state_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    starting_epoch = 1 + checkpoint['epoch']
    model.train()

def loss_fn(output, target):
    # Our target has an extra slice at the beginning for unannotated pixels:
    output, target = output['out'], target[:, 1:, :, :]
    assert output.shape == target.shape
    annotated_pixels_per_label = torch.sum(target, dim=(2, 3), keepdim=True)
    label_weight = 1e-9 + (num_labels - 1) * annotated_pixels_per_label
    probs = F.softmax(output, dim=1)
    return torch.sum(probs * target / (-label_weight))

img_dir = './2_human_annotations'

def load_data(img_name):
    assert len(os.listdir(img_dir)) > 0, 'no human-annotated images found'
    img_path = os.path.join(img_dir, img_name)
    img = tifffile.imread(img_path)
    assert img.shape[0] == 1 + num_input_channels
    assert len(img.shape) == 3
    # First channels hold the raw images. Match shape and dtype to what
    # torch expects: (batch_size, num_input_channels, y, x) and float32
    input_ = torch.cuda.FloatTensor(img[np.newaxis, :-1, ...].astype('float32'))
    input_.requires_grad = True
    # Last channel holds our annotations of the raw image. Annotation
    # values are ints ranging from 0 to num_labels; each different
    # annotation value signals a different label. We unpack these into a
    # "1-hot" representation called 'target'.
    labels = img[np.newaxis, -1:, :, :].astype('uint8')
    assert labels.max() < num_labels
    # Might as well pass a small dtype to the GPU, but the on-GPU dtype
    # has to be Long to work with .scatter_:
    labels = torch.cuda.LongTensor(labels)
    # An empty Boolean array to be filled with our "1-hot" representation:
    target = torch.cuda.BoolTensor(
        1, num_labels, img.shape[1], img.shape[2]
        ).zero_() # Initialization to zero is not automatic!
    # Copy each class into its own boolean image:
    target.scatter_(dim=1, index=labels.data, value=True)
    return input_, target

def save_output(output, filename, dir_='./3_machine_annotations'):
    guess = F.softmax(output['out'].cpu().data, dim=1).numpy().astype('float32')
    tifffile.imwrite(
        os.path.join(dir_, filename),
        guess,
        photometric='MINISBLACK',
        imagej=True,
        ijmetadata={'Ranges:', (0, 1)*guess.shape[1]})    

for epoch in range(starting_epoch, 100000): # Basically forever
    img_names = [i for i in os.listdir(img_dir) if i.endswith('.tif')]
    loss_list = []
    print("\nEpoch", epoch)
    for i, img_name in enumerate(img_names):
        print('.', sep='', end='')
        input_, target = load_data(img_name)
        output = model(input_)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Outputs for inspection and debugging:
        loss_list.append(loss.detach().item())
        save_output(output, img_name)
        if i == 0:
            if not os.path.isdir('./convergence'): os.mkdir('./convergence')
            save_output(output, 'e%06i_'%(epoch)+img_name, dir_='./convergence')
    print('\nLosses:')
    print(''.join('%0.5f '%x for x in loss_list))
    if os.path.exists(saved_state_filename):
        os.replace(saved_state_filename, backup_saved_state_filename)
    torch.save(
        {'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict()},
        saved_state_filename)




