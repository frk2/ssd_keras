from keras import backend as K
from keras.optimizers import Adam

from keras_ssd7 import build_model
from keras_ssd_loss import SSDLoss
from ssd_batch_generator import BatchGenerator
from ssd_box_encode_decode_utils import SSDBoxEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from math import ceil
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from matplotlib import pyplot as plt
from pathlib import  Path
### Set up the model

# 1: Set some necessary parameters


def train(epochs, batch_size, train_generator, n_train_samples, val_generator, n_val_samples ):
  history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=ceil(n_train_samples / batch_size),
                                epochs=epochs,
                                callbacks=[ModelCheckpoint('./ssd7_0_weights_epoch{epoch:02d}_loss{loss:.4f}.h5',
                                                           monitor='val_loss',
                                                           verbose=1,
                                                           save_best_only=True,
                                                           save_weights_only=True,
                                                           mode='auto',
                                                           period=1),
                                           EarlyStopping(monitor='val_loss',
                                                         min_delta=0.001,
                                                         patience=2),
                                           ReduceLROnPlateau(monitor='val_loss',
                                                             factor=0.5,
                                                             patience=0,
                                                             epsilon=0.001,
                                                             cooldown=0)],
                                validation_data=val_generator,
                                validation_steps=ceil(n_val_samples / batch_size))

  model_name = 'ssd7_bosch'
  model.save('./{}.h5'.format(model_name))
  model.save_weights('./{}_weights.h5'.format(model_name))

  print()
  print("Model saved as {}.h5".format(model_name))

  print("Weights also saved separately as {}_weights.h5".format(model_name))
  print()

def predict(model, X):
  y_pred = model.predict(X)
  print(y_pred)
  y_pred_decoded = decode_y2(y_pred,
                            confidence_thresh=0.1,
                            iou_threshold=0.4,
                            top_k='all',
                            input_coords='centroids',
                            normalize_coords=False,
                            img_height=None,
                            img_width=None)

  print("Decoded predictions (output format is [class_id, confidence, xmin, xmax, ymin, ymax]):\n")
  print(y_pred_decoded)
  return y_pred_decoded

# 5: Draw the predicted boxes onto the image

def predictAndDraw(model, X):
  i = 0
  y_pred_decoded = predict(model, X)
  plt.figure(figsize=(20,12))
  plt.imshow(X[i])

  current_axis = plt.gca()

  classes = ['background', 'red', 'yellow', 'green' ]

  # Draw the predicted boxes in blue
  for box in y_pred_decoded[i]:
      label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
      current_axis.add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))
      current_axis.text(box[2], box[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

  # Draw the ground truth boxes in green (omit the label for more clarity)
  # for box in y_true[i]:
  #     label = '{}'.format(classes[int(box[0])])
  #     current_axis.add_patch(plt.Rectangle((box[1], box[3]), box[2]-box[1], box[4]-box[3], color='green', fill=False, linewidth=2))
  #     #current_axi

if __name__ == '__main__':
  img_height = 300  # Height of the input images
  img_width = 480  # Width of the input images
  img_channels = 3  # Number of color channels of the input images
  n_classes = 4  # Number of classes including the background class
  min_scale = 0.08  # The scaling factor for the smallest anchor boxes
  max_scale = 0.96  # The scaling factor for the largest anchor boxes
  scales = [0.08, 0.16, 0.32, 0.64,
            0.96]  # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
  aspect_ratios = [0.5, 1.0, 2.0]  # The list of aspect ratios for the anchor boxes
  two_boxes_for_ar1 = True  # Whether or not you want to generate two anchor boxes for aspect ratio 1
  limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
  variances = [1.0, 1.0, 1.0, 1.0]  # The list of variances by which the encoded target coordinates are scaled
  coords = 'centroids'  # Whether the box coordinates to be used should be in the 'centroids' or 'minmax' format, see documentation
  normalize_coords = False  # Whether or not the model is supposed to use relative coordinates that are within [0,1]

  # 2: Build the Keras model (and possibly load some trained weights)

  K.clear_session()  # Clear previous models from memory.
  # The output `predictor_sizes` is needed below to set up `SSDBoxEncoder`
  model, predictor_sizes = build_model(image_size=(img_height, img_width, img_channels),
                                       n_classes=n_classes,
                                       min_scale=min_scale,
                                       max_scale=max_scale,
                                       scales=scales,
                                       aspect_ratios_global=aspect_ratios,
                                       aspect_ratios_per_layer=None,
                                       two_boxes_for_ar1=two_boxes_for_ar1,
                                       limit_boxes=limit_boxes,
                                       variances=variances,
                                       coords=coords,
                                       normalize_coords=normalize_coords)
  ### Set up training

  batch_size = 32

  # 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

  adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)

  ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

  model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

  # 4: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function

  ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                  img_width=img_width,
                                  n_classes=n_classes,
                                  predictor_sizes=predictor_sizes,
                                  min_scale=min_scale,
                                  max_scale=max_scale,
                                  scales=scales,
                                  aspect_ratios_global=aspect_ratios,
                                  aspect_ratios_per_layer=None,
                                  two_boxes_for_ar1=two_boxes_for_ar1,
                                  limit_boxes=limit_boxes,
                                  variances=variances,
                                  pos_iou_threshold=0.5,
                                  neg_iou_threshold=0.2,
                                  coords=coords,
                                  normalize_coords=normalize_coords)

  # 5: Create the training set batch generator

  train_dataset = BatchGenerator(images_path='./bosch/',
                                 include_classes='all',
                                 box_output_format=['class_id', 'xmin', 'xmax', 'ymin',
                                                    'ymax'])  # This is the format in which the generator is supposed to output the labels. At the moment it **must** be the format set here.

  i,l = train_dataset.parse_bosch_yaml(yaml_file='./bosch/train.yaml', ret=True)

  #  XML parser will be helpful, check the documentation.

  # Change the online data augmentation settings as you like
  train_generator = train_dataset.generate(batch_size=batch_size,
                                           train=True,
                                           ssd_box_encoder=ssd_box_encoder,
                                           equalize=False,
                                           brightness=(0.5, 2, 0.5),
                                           # Randomly change brightness between 0.5 and 2 with probability 0.5
                                           flip=0.5,  # Randomly flip horizontally with probability 0.5
                                           translate=((5, 50), (3, 30), 0.5),
                                           # Randomly translate by 5-50 pixels horizontally and 3-30 pixels vertically with probability 0.5
                                           scale=(0.75, 1.3, 0.5),
                                           # Randomly scale between 0.75 and 1.3 with probability 0.5
                                           random_crop=False,
                                           crop=False,
                                           resize=(img_width, img_height),
                                           gray=False,
                                           limit_boxes=True,
                                           include_thresh=0.4,
                                           diagnostics=False)

  n_train_samples = train_dataset.get_n_samples()

  # 6: Create the validation set batch generator (if you want to use a validation dataset)

  val_dataset = BatchGenerator(images_path='./bosch/',
                               include_classes='all',
                               box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

  i,l = val_dataset.parse_bosch_yaml(yaml_file='./bosch/test.yaml', ret=True, force_dir='rgb/test')

  # val_dataset.parse_csv(labels_path='./udacity_data/labels.csv',
  #                       input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])

  val_generator = val_dataset.generate(batch_size=batch_size,
                                       train=True,
                                       ssd_box_encoder=ssd_box_encoder,
                                       equalize=False,
                                       brightness=False,
                                       flip=False,
                                       translate=False,
                                       scale=False,
                                       random_crop=False,
                                       crop=False,
                                       resize=(img_width, img_height),
                                       gray=False,
                                       limit_boxes=True,
                                       include_thresh=0.4,
                                       diagnostics=False)

  n_val_samples = val_dataset.get_n_samples()

  ### Make predictions

  # 1: Set the generator

  predict_generator = train_dataset.generate(batch_size=1,
                                           train=False,
                                           equalize=False,
                                           brightness=False,
                                           flip=False,
                                           translate=False,
                                           scale=False,
                                           random_crop=False,
                                           crop=False,
                                           resize=(img_width, img_height),
                                           gray=False,
                                           limit_boxes=True,
                                           include_thresh=0.4,
                                           diagnostics=False)

  print(n_train_samples)

  if Path('ssd7_bosch.h5').is_file():
    print('Using existing model!')
    # model = load_model('ssd7_bosch.h5')
    model.load_weights('ssd7_0_weights_epoch16_loss0.0896.h5')
    X, y_true, filenames = next(predict_generator)
    predictAndDraw(model, X)
  else:
    train(30, 128, train_generator, n_train_samples, val_generator, n_val_samples)