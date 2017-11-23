from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np

from helper import parse_driving_log, \
    evenout_training_distribution, preprocess_image, plot_image_gray


def batch_generater(np_training_paths, np_training_angles, batch_size=32):
    num_samples = len(np_training_paths)

    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_training_paths = np_training_paths[offset:offset+batch_size]
            batch_training_angles = np_training_angles[offset:offset+batch_size]
            batch_training_images = []
            for training_path in batch_training_paths:
                img = preprocess_image(training_path)
                batch_training_images.append(img)
            yield shuffle(np.array(batch_training_images), batch_training_angles)


# ==================================================================
#
# Training code starts here
#
# ==================================================================

all_samples, all_angles = parse_driving_log()
train_samples, validation_samples, training_angles, validation_angles = train_test_split(all_samples, all_angles, test_size=0.2)

train_samples, training_angles = shuffle(train_samples, training_angles)

np_training_paths = np.array(train_samples)
np_training_angles = np.array(training_angles)
np_validation_paths = np.array(validation_samples)
np_validation_angles = np.array(validation_angles)

print("Training shape: {}, {}".format(np_training_paths.shape, np_training_angles.shape))
print("Validation shape: {}, {}".format(np_validation_paths.shape, np_validation_angles.shape))

# Plot distribution of angles
np_training_paths, np_training_angles = evenout_training_distribution(np_training_paths, np_training_angles)
np_training_paths, np_training_angles = shuffle(np_training_paths, np_training_angles)

train_generator = batch_generater(np_training_paths, np_training_angles, batch_size=32)
validation_generator = batch_generater(np_validation_paths, np_validation_angles, batch_size=32)

for i in range(1):
    images, angles = (next(train_generator))
    plot_image_gray(images[:10], angles[:10], 'plot/generated_process.png')
