'''
Atomic script for training
'''

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

import argparse
import os
from datetime import datetime

from albumentations import (
    Compose, RandomBrightness, RandomContrast, RandomGamma, Normalize
)

from model import default_categorical
from generators import TortueInMemoryGenerator
from training_utils import newtub_to_array

# Training parameters
epochs = 100
steps = 100
verbose = 1
min_delta = .0005
patience = 8
use_early_stop=True
batch_size=64
save_model_path = os.path.join(os.path.dirname(__file__), '../trained_models', 'convnet_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.hdf5')

model = default_categorical()

# Image augmentation setup
AUGMENTATIONS_TRAIN = Compose([
    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
    Normalize()
])

AUGMENTATIONS_TEST = Compose([Normalize()])


if __name__ == '__main__':

    print(os.path.join(os.path.dirname(__file__), '../sample_data/train'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", dest="train_dir",  type=str, default=os.path.join(os.path.dirname(__file__), '../sample_data/train'))
    parser.add_argument("--valid_dir", dest="valid_dir",  type=str, default=os.path.join(os.path.dirname(__file__), '../sample_data/validation'))
    parser.add_argument("--model_path_output",  dest="model_path_output", type=str, default=save_model_path)
    args = parser.parse_args()

    # checkpoint to save model after each epoch
    save_best = ModelCheckpoint(save_model_path,
                                monitor='val_loss',
                                verbose=verbose,
                                save_best_only=True,
                                mode='min')

    # stop training if the validation error stops improving.
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=min_delta,
                               patience=patience,
                               verbose=verbose,
                               mode='auto')
    callbacks_list = [save_best]

    if use_early_stop:
        callbacks_list.append(early_stop)

    # import data
    print('import data...')
    X_train, Y_train = newtub_to_array(args.train_dir, n_class=3)
    train_gen = TortueInMemoryGenerator(X_train, Y_train, batch_size, augmentations=AUGMENTATIONS_TRAIN, flip_proportion=.3)
    X_val, Y_val = newtub_to_array(args.valid_dir, n_class=3)
    valid_gen = TortueInMemoryGenerator(X_val, Y_val, batch_size, augmentations=AUGMENTATIONS_TEST, flip_proportion=.3)

    model.summary()

    print('fit model...')
    # fit from numpy array
    hist = model.fit_generator(train_gen,
                               epochs=epochs,
                               validation_data=valid_gen,
                               use_multiprocessing=False,
                               callbacks=callbacks_list)