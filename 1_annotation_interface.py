import os
import numpy as np
from tifffile import imread, imwrite
import napari

# Preprocess your image to be a 3D stack; every 2D slice will be
# annotated via the same model.
def load():
    input_filename = './1_to_be_annotated/annotate_me.tif'
    assert os.path.exists(input_filename), "Please create %s"%input_filename
    data = imread(input_filename)
    print('annotating data with shape', data.shape, 'and dtype', data.dtype)
    assert len(data.shape) == 3 and data.dtype == np.uint16
    data_with_labels = np.zeros((3,) + data.shape, dtype='uint16')
    data_with_labels[0, :, :] = data
    for i in range(data.shape[0]):
        human_labels_filename = './2_human_annotations/%06i.tif'%i
        if os.path.exists(human_labels_filename):
            human_labels = imread(human_labels_filename)[1, :, :]
            data_with_labels[1, i, :, :] = human_labels
        machine_labels_filename = './3_machine_annotations/%06i.tif'%i
        if os.path.exists(machine_labels_filename):
            machine_labels = imread(machine_labels_filename)
            machine_labels = 1 + np.argmax(machine_labels, axis=0)
            data_with_labels[2, i, :, :] = machine_labels
    return data_with_labels

with napari.gui_qt():
    data_with_labels = load()
    viewer = napari.Viewer()
    viewer.add_image(data_with_labels[0, :, :, :], name='Image')
    viewer.layers['Image'].selected = False
    viewer.add_labels(data_with_labels[2, :, :, :], name='Machine labels')
    viewer.layers['Machine labels'].opacity = 0.25
    viewer.layers['Machine labels'].visible = False
    viewer.layers['Machine labels'].editable = False
    viewer.add_labels(data_with_labels[1, :, :, :], name='Human labels')
    viewer.layers['Human labels'].opacity = 0.25
    viewer.layers['Human labels'].selected = True
    
    @viewer.bind_key('s')
    def save_layer(viewer):
        i = viewer.dims.point[0]
        x = data_with_labels[0:2, i, :, :]
        filename = './2_human_annotations/%06i.tif'%i
        print("Saving", x.shape, x.dtype, "as", filename)
        imwrite(
            filename,
            x.reshape(1, 1, *x.shape),
            imagej=True,
            ijmetadata={'Ranges':(x[0, :, :].min(), x[0, :, :].max(),
                                  x[1, :, :].min(), x[1, :, :].max())})

    @viewer.bind_key('r')
    def reload(viewer):
        print('Reloading human and machine annotations')
        data_with_labels[:] = load()
        


        
