from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

import argparse

from src.model import default_categorical
from training_utils import newtub_to_array
from src.generators import TortueInMemoryGenerator

save_model_path = '/home/projects/tortue-rapide/tortue-rapide-training-training/test_models'

parser = argparse.ArgumentParser()

model = default_categorical()

# augmentation
from albumentations import (
    Compose, RandomBrightness, RandomContrast, RandomGamma, Normalize
)

AUGMENTATIONS_TRAIN = Compose([
    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
    Normalize()
])

AUGMENTATIONS_TEST = Compose([Normalize()])

# fit model
epochs = 100
steps = 100
verbose = 1
min_delta = .0005
patience = 8
use_early_stop=True
batch_size=64

# train / validation split
X_train, Y_train = newtub_to_array('/home/sample_data/tortue-rapide/sets/raw/ysance_abbey_epingle_20190513/', n_class=3)
train_gen = TortueInMemoryGenerator(X_train, Y_train, batch_size, augmentations=AUGMENTATIONS_TRAIN, flip_proportion=.3)

# Validation sample_data
X_val, Y_val = newtub_to_array('/home/sample_data/tortue-rapide/sets/raw/lesquare_NECTAR_20190518', n_class=3)
valid_gen = TortueInMemoryGenerator(X_val, Y_val, batch_size, augmentations=AUGMENTATIONS_TEST, flip_proportion=.3)


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

model.summary()

print('fit model...')
# fit from numpy array
hist = model.fit_generator(train_gen,
                           epochs=epochs,
                           validation_data=valid_gen,
                           use_multiprocessing=False,
                           callbacks=callbacks_list)

if __name__ == '__main__':

    parser.add_argument("--model_path", help="increase output verbosity")
    parser.add_argument("--training", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--test", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()

    print('kek')
