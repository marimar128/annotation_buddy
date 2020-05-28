#Python standard library
from pathlib import Path
# Third-party libraries, installable via pip
import numpy as np
from tifffile import imread, imwrite, TiffFile
# Install as described at napari.org/tutorials/fundamentals/installation
import napari

# Set input/output behavior
data_dir = Path.cwd() # The current working directory
data_filename = data_dir / 'annotate_me.tif'
human_labels_dir = data_dir / '1_human_annotations'
rf_labels_dir = data_dir / '2_random_forest_annotations'
nn_labels_dir = data_dir / '3_neural_network_annotations'

# Sanity checks on input/output
instructions = """\n
The data to be annotated should be a 5D ImageJ hyperstack in `tzcyx`
order, named %s.
"""%(data_filename)
assert data_filename.is_file(), instructions
human_labels_dir.mkdir(exist_ok=True)
rf_labels_dir.mkdir(exist_ok=True)
nn_labels_dir.mkdir(exist_ok=True)

def load():
    assert data_filename.is_file(), instructions
    data = imread(str(data_filename)) # Hopefully, already 5D tzcyx order...
    if len(data.shape) in (2, 3, 4): # Careful not to confuse c with t or z!
        # Read the tif ImageJ metadata to figure out which dimension is which:
        with TiffFile(data_filename) as tif:
            md = tif.imagej_metadata
        n_c = md.get('channels', 1)
        n_z = md.get('slices', 1)
        n_t = md.get('frames', 1)
        data = data.reshape(n_t, n_z, n_c, data.shape[-2], data.shape[-1])
    print('Annotating data with shape', data.shape, 'and dtype', data.dtype)
    assert len(data.shape) == 5, instructions
    new_shape = list(data.shape)
    # Add three extra channels to hold human, random forest, and NN labels
    new_shape[2] += 3
    data_with_labels = np.zeros(new_shape, dtype=data.dtype)
    data_with_labels[:, :, :-3, :, :] = data
    for t in range(data.shape[0]):
        for z in range(data.shape[1]):
            tif_name = 't%06i_z%06i.tif'%(t, z)
            # Load human-generated labels:
            human_labels_path = human_labels_dir / tif_name
            if human_labels_path.is_file():
                human_labels = imread(str(human_labels_path))[-1, :, :]
                data_with_labels[t, z, -3, :, :] = human_labels
            # Load RandomForest-generated labels:
            rf_labels_path = rf_labels_dir / tif_name
            if rf_labels_path.is_file():
                random_forest_labels = imread(str(rf_labels_path))[-1, :, :]
                data_with_labels[t, z, -2, :, :] = random_forest_labels
            # Load neural network-generated labels:
            nn_labels_path = nn_labels_dir / tif_name
            if nn_labels_path.is_file():
                nn_output = imread(str(nn_labels_path))
                # Convert from softmax 1-hot to labels:
                neural_network_labels = 1 + np.argmax(nn_output, axis=0)
                data_with_labels[t, z, -1, :, :] = neural_network_labels
    return data_with_labels

with napari.gui_qt():
    data_with_labels = load()
    viewer = napari.Viewer(
        title="Mister Clicky instructions: " +
        "Draw annotations in the 'Human labels' layer. " +
        "Press 'S' to save annotations, and  'R' to reload annotations.")
    for i in reversed(range(data_with_labels.shape[2]-3)):
        layer = viewer.add_image(data_with_labels[:, :, i, :, :],
                                 name='Image ch %i'%i)
        layer.selected = False
        layer.blending = 'additive'
        if i in range(4):
            layer.colormap = ('gray', 'red', 'green', 'blue')[i]
    viewer.add_labels(data_with_labels[:, :, -1, :, :],
                      name='Neural net labels')
    viewer.layers['Neural net labels'].opacity = 0.25
    viewer.layers['Neural net labels'].visible = False
    viewer.layers['Neural net labels'].editable = False
    viewer.add_labels(data_with_labels[:, :, -2, :, :],
                      name='Rand. forest labels')
    viewer.layers['Rand. forest labels'].opacity = 0.25
    viewer.layers['Rand. forest labels'].visible = False
    viewer.add_labels(data_with_labels[:, :, -3, :, :], name='Human labels')
    viewer.layers['Human labels'].opacity = 0.25
    viewer.layers['Human labels'].selected = True
    viewer.layers['Human labels'].mode = 'paint'
    viewer.layers['Human labels'].selected_label = 1
    viewer.layers['Human labels'].brush_size = 3
    viewer.active_layer = viewer.layers['Human labels']
    viewer.dims.set_axis_label(0, 't')
    viewer.dims.set_axis_label(1, 'z')
    
    @viewer.bind_key('s')
    def save_slice(viewer):
        t, z = viewer.dims.point[0], viewer.dims.point[1]
        # Save the human-annotated labels from the current slice
        filename = human_labels_dir / ('t%06i_z%06i.tif'%(t, z))
        x = data_with_labels[t, z, :-2, :, :]
        print("Saving", x.shape, x.dtype, "as", filename)
        ranges = []
        for ch in x: # Autoscale each channel
            ranges.extend((ch.min(), ch.max()))
        imwrite(filename, x, imagej=True, ijmetadata={'Ranges': tuple(ranges)})
        # Also force the random-forest annotated labels to agree with
        # the human labels, and resave them to disk.
        filename = rf_labels_dir / ('t%06i_z%06i.tif'%(t, z))
        # Agreement on screen
        human_labels = data_with_labels[t, z, -3, :, :]
        rf_labels = data_with_labels[t, z, -2, :, :]
        overruled = (human_labels != 0)
        rf_labels[overruled] = human_labels[overruled]
        viewer.layers['Rand. forest labels'].refresh()
        # Agreement on disk
        data_and_rf_slices = (*range(data_with_labels.shape[2] - 3), -2)
        x = data_with_labels[t, z, data_and_rf_slices, :, :]
        print("Saving", x.shape, x.dtype, "as", filename)
        ranges = []
        for ch in x: # Autoscale each channel
            ranges.extend((ch.min(), ch.max()))
        imwrite(filename, x, imagej=True, ijmetadata={'Ranges': tuple(ranges)})

    @viewer.bind_key('a')
    def save_all_slices(viewer):
        for t in range(data_with_labels.shape[0]):
            for z in range(data_with_labels.shape[1]):
                # Save the human-annotated labels from all slices
                filename = human_labels_dir / ('t%06i_z%06i.tif'%(t, z))
                x = data_with_labels[t, z, :-2, :, :]
                print("Saving", x.shape, x.dtype, "as", filename)
                ranges = []
                for ch in x: # Autoscale each channel
                    ranges.extend((ch.min(), ch.max()))
                imwrite(filename, x, imagej=True,
                        ijmetadata={'Ranges': tuple(ranges)})
                # Also force the random-forest annotated labels to agree with
                # the human labels, and resave them to disk.
                filename = rf_labels_dir / ('t%06i_z%06i.tif'%(t, z))
                # Agreement on screen
                human_labels = data_with_labels[t, z, -3, :, :]
                rf_labels = data_with_labels[t, z, -2, :, :]
                overruled = (human_labels != 0)
                rf_labels[overruled] = human_labels[overruled]
                viewer.layers['Rand. forest labels'].refresh()
                # Agreement on disk
                data_and_rf_slices = (*range(data_with_labels.shape[2] - 3), -2)
                x = data_with_labels[t, z, data_and_rf_slices, :, :]
                print("Saving", x.shape, x.dtype, "as", filename)
                ranges = []
                for ch in x: # Autoscale each channel
                    ranges.extend((ch.min(), ch.max()))
                imwrite(filename, x, imagej=True,
                        ijmetadata={'Ranges': tuple(ranges)})

    @viewer.bind_key('r')
    def reload(viewer):
        print('Reloading human and machine annotations')
        data_with_labels[:] = load()
        viewer.layers['Human labels'].refresh()
        viewer.layers['Rand. forest labels'].refresh()
        viewer.layers['Neural net labels'].refresh()
