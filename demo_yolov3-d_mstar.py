from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from keras import backend as K
from models.yolov3 import YOLOv3_Darknet53, YOLOv3_Darknet53_MSTAR
from PIL import Image
from tog.attacks import *
import os
from matplotlib import pyplot as plt

K.clear_session()

weights = 'model_weights/yolov3-d/ep078-loss16.562-val_loss14.732.h5'  # TODO: Change this path to the victim model's weights

detector = YOLOv3_Darknet53_MSTAR(weights=weights)

eps = 8 / 255.       # Hyperparameter: epsilon in L-inf norm
eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
n_iter = 10          # Hyperparameter: number of attack iterations

fpath = './assets/MSTAR/000899.jpg'    # TODO: Change this path to the image to be attacked

noise_shift = 1
noise_shift_pix = 20

input_img = Image.open(fpath)
x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)
visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes)})

### TOG-untargeted Attack ###

# Generation of the adversarial example
x_adv_untargeted = tog_untargeted(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)

# Visualizing the detection results on the adversarial example and compare them with that on the benign input
detections_adv_untargeted = detector.detect(x_adv_untargeted, conf_threshold=detector.confidence_thresh_default)
visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
                      'TOG-untargeted': (x_adv_untargeted, detections_adv_untargeted, detector.model_img_size, detector.classes)})

if noise_shift == 1:
    noise = x_adv_untargeted - x_query
    if len(noise.shape) == 4:
        noise = noise[0]
    noise[noise_shift_pix:, noise_shift_pix:, :] = noise[:noise.shape[0]-noise_shift_pix, :noise.shape[1]-noise_shift_pix, :]
    noise[:noise_shift_pix, :, :] = 0
    noise[:, :noise_shift_pix, :] = 0
    new_x_adv_untargeted = x_query + noise

    detections_adv_untargeted = detector.detect(new_x_adv_untargeted, conf_threshold=detector.confidence_thresh_default)
    visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
                        'Noise shifted TOG-untargeted': (new_x_adv_untargeted, detections_adv_untargeted, detector.model_img_size, detector.classes)})

### TOG-vanishing Attack ###

# Generation of the adversarial example
x_adv_vanishing = tog_vanishing(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)

# Visualizing the detection results on the adversarial example and compare them with that on the benign input
detections_adv_vanishing = detector.detect(x_adv_vanishing, conf_threshold=detector.confidence_thresh_default)
visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
                      'TOG-vanishing': (x_adv_vanishing, detections_adv_vanishing, detector.model_img_size, detector.classes)})

if noise_shift == 1:
    noise = x_adv_vanishing - x_query
    if len(noise.shape) == 4:
        noise = noise[0]
    noise[noise_shift_pix:, noise_shift_pix:, :] = noise[:noise.shape[0]-noise_shift_pix, :noise.shape[1]-noise_shift_pix, :]
    noise[:noise_shift_pix, :, :] = 0
    noise[:, :noise_shift_pix, :] = 0
    new_x_adv_vanishing = x_query + noise

    detections_adv_untargeted = detector.detect(new_x_adv_vanishing, conf_threshold=detector.confidence_thresh_default)
    visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
                        'Noise shifted TOG-vanishing': (new_x_adv_vanishing, detections_adv_untargeted, detector.model_img_size, detector.classes)})

### TOG-fabrication Attack ###

x_adv_fabrication = tog_fabrication(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)

# Visualizing the detection results on the adversarial example and compare them with that on the benign input
detections_adv_fabrication = detector.detect(x_adv_fabrication, conf_threshold=detector.confidence_thresh_default)
visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
                      'TOG-fabrication': (x_adv_fabrication, detections_adv_fabrication, detector.model_img_size, detector.classes)})

if noise_shift == 1:
    noise = x_adv_fabrication - x_query
    if len(noise.shape) == 4:
        noise = noise[0]
    noise[noise_shift_pix:, noise_shift_pix:, :] = noise[:noise.shape[0]-noise_shift_pix, :noise.shape[1]-noise_shift_pix, :]
    noise[:noise_shift_pix, :, :] = 0
    noise[:, :noise_shift_pix, :] = 0
    new_x_adv_fabrication = x_query + noise

    detections_adv_untargeted = detector.detect(new_x_adv_fabrication, conf_threshold=detector.confidence_thresh_default)
    visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
                        'Noise shifted TOG-fabrication': (new_x_adv_fabrication, detections_adv_untargeted, detector.model_img_size, detector.classes)})

### TOG-mislabeling Attack ###

# Generation of the adversarial examples
x_adv_mislabeling_ml = tog_mislabeling(victim=detector, x_query=x_query, target='ml', n_iter=n_iter, eps=eps, eps_iter=eps_iter)
x_adv_mislabeling_ll = tog_mislabeling(victim=detector, x_query=x_query, target='ll', n_iter=n_iter, eps=eps, eps_iter=eps_iter)

# Visualizing the detection results on the adversarial examples and compare them with that on the benign input
detections_adv_mislabeling_ml = detector.detect(x_adv_mislabeling_ml, conf_threshold=detector.confidence_thresh_default)
detections_adv_mislabeling_ll = detector.detect(x_adv_mislabeling_ll, conf_threshold=detector.confidence_thresh_default)
visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
                      'TOG-mislabeling (ML)': (x_adv_mislabeling_ml, detections_adv_mislabeling_ml, detector.model_img_size, detector.classes),
                      'TOG-mislabeling (LL)': (x_adv_mislabeling_ll, detections_adv_mislabeling_ll, detector.model_img_size, detector.classes)})

if noise_shift == 1:
    noise = x_adv_mislabeling_ml - x_query
    if len(noise.shape) == 4:
        noise = noise[0]
    noise[noise_shift_pix:, noise_shift_pix:, :] = noise[:noise.shape[0]-noise_shift_pix, :noise.shape[1]-noise_shift_pix, :]
    noise[:noise_shift_pix, :, :] = 0
    noise[:, :noise_shift_pix, :] = 0
    new_x_adv_mislabeling_ml = x_query + noise

    noise = x_adv_mislabeling_ll - x_query
    if len(noise.shape) == 4:
        noise = noise[0]
    noise[noise_shift_pix:, noise_shift_pix:, :] = noise[:noise.shape[0]-noise_shift_pix, :noise.shape[1]-noise_shift_pix, :]
    noise[:noise_shift_pix, :, :] = 0
    noise[:, :noise_shift_pix, :] = 0
    new_x_adv_mislabeling_ll = x_query + noise

    detections_adv_untargeted = detector.detect(new_x_adv_mislabeling_ml, conf_threshold=detector.confidence_thresh_default)
    detections_adv_untargeted = detector.detect(new_x_adv_mislabeling_ll, conf_threshold=detector.confidence_thresh_default)

    visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
                        'Noise shifted TOG-mislabeling (ML)': (new_x_adv_mislabeling_ml, detections_adv_mislabeling_ml, detector.model_img_size, detector.classes),
                        'Noise shifted TOG-mislabeling (LL)': (new_x_adv_mislabeling_ll, detections_adv_mislabeling_ll, detector.model_img_size, detector.classes)})
