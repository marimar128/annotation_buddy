# Python standard library
from pathlib import Path
# Third-party libraries, installable via pip
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from tifffile import imread, imwrite

# Set input/output behavior
input_dir = Path('./1_human_annotations')
output_dir = Path('./2_random_forest_annotations')
debug_output_dir = Path('./random_forest_intermediate_images')
probability_threshold = 0.6 # Only annotate where you're confident
save_debug_imgs = True

# Sanity checks on input/output
assert input_dir.is_dir()
output_dir.mkdir(exist_ok=True)
if save_debug_imgs:
    debug_output_dir.mkdir(exist_ok=True)
input_filenames = [x for x in input_dir.iterdir() if x.suffix == '.tif']
assert len(input_filenames) > 0, "No annotated images to process"


def calculate_features(data):
    assert len(data.shape) == 3 and data.shape[0] > 1 # N-channel, 2D images
    features = []
    for channel in range(data.shape[0] - 1): # Ignore the labels channel
        im = data[channel, :, :].astype('float32')
        # NB our features are chosen in a subjective and unprincipled way:
        features.extend((im,
                         ndi.gaussian_filter(im, sigma=2),
                         ndi.gaussian_filter(im, sigma=4),
                         ndi.gaussian_gradient_magnitude(im, sigma=2),
                         ndi.gaussian_gradient_magnitude(im, sigma=4),
                         ndi.sobel(ndi.gaussian_filter(im, sigma=2), axis=-1),
                         ndi.sobel(ndi.gaussian_filter(im, sigma=2), axis=-2),
                         ndi.laplace(im)
                         ))
    return np.stack(features, axis=2) # Not my usual byteorder

print("Loading images and calculating 'features' to train our",
      "random forest model...")
flattened_features, flattened_labels = [], []
for fn in input_filenames:
    data = imread(str(fn))
    features = calculate_features(data)
    labels = data[-1, :, :].astype('uint32') # Classification, not regression
    if save_debug_imgs: # Inspection is the path to insight
        x = np.moveaxis(features, -1, 0) # Convert to my usual byteorder
        ranges = []
        for ch in x: # Set display range for each channel
            ranges.extend((ch.min(), ch.max()))
        imwrite(debug_output_dir / (fn.stem + '_features.tif'), x,
                imagej=True, ijmetadata={'Ranges': tuple(ranges)})        
    # Only keep pixels that have labels:
    labels_exist = labels.ravel() != 0
    flattened_features.append(features.reshape(-1, features.shape[-1]
                                               )[labels_exist])
    flattened_labels.append(labels.ravel()[labels_exist])
# This step is a potential memory-hog:
flattened_features = np.concatenate(flattened_features)
flattened_labels = np.concatenate(flattened_labels)
if len(np.unique(flattened_labels)) < 2:
    print('*'*10, "WARNING", '*'*10)
    print("Only", len(np.unique(flattened_labels)),
          "unique label(s) are currently annotated.")
    print("Useful random forest classification requires at least two labels.")
    print('*'*29)
print("Done calculating.\n")

print("Training a random forest classifier with",
      flattened_features.shape[0], "annotated pixels\nand",
      flattened_features.shape[1], "features calculated per pixel... ")
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(flattened_features, flattened_labels)
print("Done training.\n")

print("Loading images, re-calculating 'features', and saving predictions...")
for fn in input_filenames:
    data = imread(str(fn))
    features = calculate_features(data)
    probabilities = clf.predict_proba(features.reshape(-1, features.shape[-1])
                                      ).reshape(*data.shape[1:], clf.n_classes_)
    predictions = clf.classes_[np.argmax(probabilities, axis=-1)]
    confident = probabilities.max(axis=-1) > probability_threshold
    unannotated = data[-1, :, :] == 0
    new_annotations = confident & unannotated
    data[-1, :, :][new_annotations] = predictions[new_annotations]
    # Save as tif with per-channel display ranges:
    ranges = []
    for ch in data:
        ranges.extend((ch.min(), ch.max()))
    imwrite(output_dir / fn.name,
            data,
            imagej=True,
            ijmetadata={'Ranges': tuple(ranges)})
print("Done calculating and saving.\n")
