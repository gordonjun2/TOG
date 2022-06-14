weights = 'model_weights/yolov3-d/ep078-loss16.562-val_loss14.732.h5'  # TODO: Change this path to the victim model weights

from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from models.yolov3 import YOLOv3_MobileNetV1, YOLOv3_Darknet53_MSTAR
from matplotlib import pyplot as plt
from keras import backend as K
from PIL import Image
from tqdm import tqdm
import numpy as np
import datetime
import random
import os
K.clear_session()

detector = YOLOv3_Darknet53_MSTAR(weights=weights)

MSTAR_path = '../keras-yolo3/datasets/VOCdevkit2007/VOC2007'  # TODO: Change this path to your VOC2007
#VOC12_path = '/research/datasets/VOCdevkit/VOC2012'  # TODO: Change this path to your VOC2012

eps = 8 / 255.        # Hyperparameter: epsilon in L-inf norm
eps_iter = 0.0001     # Hyperparameter: attack learning rate
n_epochs = 50         # Hyperparameter: number of attack iterations
n_samples = 12800      # Hyperparameter: number of training samples

fpaths_train = []  # Load image paths
for prefix in [MSTAR_path]:
    with open('%s/ImageSets/Main/trainval.txt' % prefix, 'r') as f:
        fpaths_train += [os.path.join(prefix, 'JPEGImages', '%s.jpg' % fname.strip()) for fname in f.readlines()]
random.shuffle(fpaths_train)  # Shuffle the image paths for random sampling
fpaths_train = fpaths_train[:n_samples]  # Select only n_samples images

eta = np.random.uniform(-eps, eps, size=(*detector.model_img_size, 3))

for epoch in range(n_epochs):
    pbar = tqdm(fpaths_train)
    pbar.set_description('Epoch %d/%d' % (epoch + 1, n_epochs))
    
    for fpath in pbar:
        input_img = Image.open(fpath)
        x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)

        x_adv = np.clip(x_query + eta, 0.0, 1.0)                  # Step 1: Apply the current eta
        grad = detector.compute_object_vanishing_gradient(x_adv)  # Step 2: Conduct one-step SGD
        signed_grad = np.sign(grad[0])
        eta = np.clip(eta - eps_iter * signed_grad, -eps, eps)    # Step 3: Extract the new eta

    random.shuffle(fpaths_train)

plt.clf()
plt.title('Trained TOG-universal: eta')
plt.imshow((eta - eta.min()) / (eta.max() - eta.min()))
plt.axis('off')
plt.tight_layout()
plt.show()

input_img = Image.open('./assets/MSTAR/000056.jpg')
x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
x_adv_vanishing = np.clip(x_query + eta, 0.0, 1.0)
detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)
detections_adv_vanishing = detector.detect(x_adv_vanishing, conf_threshold=detector.confidence_thresh_default)
visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
                      'TOG-universal': (x_adv_vanishing, detections_adv_vanishing, detector.model_img_size, detector.classes)})

input_img = Image.open('./assets/MSTAR/000604.jpg')
x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
x_adv_vanishing = np.clip(x_query + eta, 0.0, 1.0)
detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)
detections_adv_vanishing = detector.detect(x_adv_vanishing, conf_threshold=detector.confidence_thresh_default)
visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
                      'TOG-universal': (x_adv_vanishing, detections_adv_vanishing, detector.model_img_size, detector.classes)})