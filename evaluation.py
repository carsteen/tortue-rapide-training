# utils functions for performance evaluation and model explanation. Intented to be used in notebooks.

import matplotlib.pyplot as plt
import cv2
import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import to_categorical
from skimage.segmentation import mark_boundaries
from lime import lime_image

def predicted_right(y_val, y_probas):
  y_hat = to_categorical(np.argmax(y_probas, axis=1))
  right_val = [np.array_equal(yhat, y) for yhat, y in zip(y_hat, y_val)]
  return np.array(right_val)

def viz_validation(x_val, y_val, y_hat):
  plt.figure(figsize=(13,40))

  for n in range(len(y_val)):
    plt.subplot(1+(len(y_val) // 4), 4, n+1)
    plt.imshow(x_val[n]+.5)
    plt.title(str(y_val[n]) + '\n' + np.array2string(np.around(y_hat[n], 3), separator=' | '))
    plt.axis('off')

def lime_explanation(model, img, pred_index):
    """Visualize LIME explanation of a model decision

    img : image np array
    pred_index: label index to show outputs for
    model: keras model
    """
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(img / 255 - .5,
                                             classifier_fn=model.predict,
                                             hide_color=0, num_samples=2000)

    temp, mask = explanation.get_image_and_mask(pred_index, positive_only=False, num_features=2, hide_rest=False)

    plt.imshow(mark_boundaries(temp + .5, mask))
    plt.title('proba(left, straight, right) : ' + str(np.round(model.predict(img.reshape((1,) + img.shape) / 255 - .5), 3)))


def viz_layer_output_cropmodel(img, layer_idx, pred_idx, crop_layer_idx, model, top_crop=30, bottom_crop=10):
    """Vizualize specific areas of the input image that trigger the network decision
    to be used for network with crop2D layer

    img : image np array
    layer_idx: layer outputs to show
    pred_idx: label index to show outputs for
    crop_layer_idx: index of cropping layer in keras model
    model: keras model
    top_crop: top pixels to trim off
    bottom_crop: bottom pixels to trim off"""

    cropped_img = img[top_crop : -bottom_crop, :, :]

    # compute prediction for specific image
    pred = model.predict(img.reshape((1,) + img.shape) / 255 - .5)

    output = model.output[:, pred_idx]
    last_conv_layer = model.layers[layer_idx]

    crop_layer = model.layers[crop_layer_idx]

    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([crop_layer.output], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([croped_img.reshape((1,) + croped_img.shape) / 255 - .5])

    n_filters = conv_layer_output_value.shape[-1]

    for i in range(n_filters):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (cropped_img.shape[1], cropped_img.shape[0]))
    cropped_img = cropped_img / 255
    cropped_img = cropped_img.astype(np.float32)
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    added_image = cv2.addWeighted(gray_img, 0.5, heatmap, 0.1, 0)

    plt.imshow(added_image)
    plt.title(str(np.round(pred, 3)))