# Credits to donkeycar

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dropout, Flatten, Dense, Cropping2D, Convolution2D

def default_categorical():
    'default_categorical model from donkey car'
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in
    x = Cropping2D(cropping=((40, 10), (0, 0)))(x) # crop 40 pixels off top and 10 off bottom
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.1)(x)

    # categorical output of the angle
    angle_out = Dense(3, activation='softmax', name='angle_cat_out')(x)

    # continous output of throttle
    # throttle_out = Dense(1, activation='relu', name='throttle_out')(x)  # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
