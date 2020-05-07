import os
import numpy as np
from tifffile import imread, imwrite, TiffFile
import napari

# Rewrite this function (and instructions!) as needed to load your data.
instructions = """\n
By default, this code is written to load a 5D ImageJ hyperstack in tzcyx
order, named './1_to_be_annotated/annotate_me.tif'. Each t,z slice will
get its own set of labels; all channels share a common set of labels.
"""
def load():
    input_filename = './1_to_be_annotated/annotate_me.tif'
    assert os.path.exists(input_filename), "Please create %s"%input_filename
    data = imread(input_filename)
    if len(data.shape) in (2, 3, 4): # Careful not to confuse c with t or z!
        with TiffFile(input_filename) as tif:
            md = tif.imagej_metadata
            n_c = md.get('channels', 1)
            n_z = md.get('slices', 1)
            n_t = md.get('frames', 1)
            data = data.reshape(n_t, n_z, n_c, data.shape[-2], data.shape[-1])
    print('annotating data with shape', data.shape, 'and dtype', data.dtype)
    assert len(data.shape) == 5, instructions
    new_shape = list(data.shape)
    new_shape[2] += 2 # Two extra channels to hold human and machine labels
    data_with_labels = np.zeros(new_shape, dtype=data.dtype)
    data_with_labels[:, :, :-2, :, :] = data
    for t in range(data.shape[0]):
        for z in range(data.shape[1]):
            human_labels_filename = (
                './2_human_annotations/t%06i_z%06i.tif'%(t, z))
            if os.path.exists(human_labels_filename):
                human_labels = imread(human_labels_filename)[-1, :, :]
                data_with_labels[t, z, -2, :, :] = human_labels
            machine_labels_filename = (
                './3_machine_annotations/t%06i_z%06i.tif'%(t, z))
            if os.path.exists(machine_labels_filename):
                machine_labels = imread(machine_labels_filename)
                # Convert from softmax 1-hot to labels:
                machine_labels = 1 + np.argmax(machine_labels, axis=0)
                data_with_labels[t, z, -1, :, :] = machine_labels
    return data_with_labels

with napari.gui_qt():
    data_with_labels = load()
    viewer = napari.Viewer()
    for i in reversed(range(data_with_labels.shape[2]-2)):
        layer = viewer.add_image(data_with_labels[:, :, i, :, :],
                                 name='Image ch %i'%i)
        layer.selected = False
        layer.blending = 'additive'
        if i in range(1, 4):
            layer.colormap = ('red', 'green', 'blue')[i-1]
    viewer.add_labels(data_with_labels[:, :, -1, :, :], name='Machine labels')
    viewer.layers['Machine labels'].opacity = 0.25
    viewer.layers['Machine labels'].visible = False
    viewer.layers['Machine labels'].editable = False
    viewer.add_labels(data_with_labels[:, :, -2, :, :], name='Human labels')
    viewer.layers['Human labels'].opacity = 0.25
    viewer.layers['Human labels'].selected = True
    viewer.dims.set_axis_label(0, 't')
    viewer.dims.set_axis_label(1, 'z')
    
    @viewer.bind_key('s')
    def save_layer(viewer):
        t, z = viewer.dims.point[0], viewer.dims.point[1]
        x = data_with_labels[t, z, :-1, :, :]
        filename = './2_human_annotations/t%06i_z%06i.tif'%(t, z)
        print("Saving", x.shape, x.dtype, "as", filename)
        ranges = []
        for ch in x: # Autoscale each channel
            ranges.extend((ch.min(), ch.max()))
        imwrite(filename, x, imagej=True, ijmetadata={'Ranges': tuple(ranges)})

    @viewer.bind_key('r')
    def reload(viewer):
        print('Reloading human and machine annotations')
        data_with_labels[:] = load()
        


        
