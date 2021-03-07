import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.losses import mean_squared_error
import glob
import matplotlib.patches as patches
import json
import numpy as np
from matplotlib.path import Path
import pydicom as dicom
import cv2
# from PyPI import pydicom
import matplotlib.pyplot as plt

def get_roi(image, contour, shape_out = 32):
    """
    Create a binary mask with ROI from contour.
    Extract the maximum square around the contour.
    :param image: input image (needed for shape only)
    :param contour: numpy array contour (d, 2)
    :return: numpy array mask ROI (shape_out, shape_out)
    """
    X_min, Y_min = contour[:,0].min(), contour[:,1].min()
    X_max, Y_max = contour[:,0].max(), contour[:,1].max()
    w = X_max - X_min
    h = Y_max - Y_min
    mask_roi = np.zeros(image.shape)
    if w > h :
        mask_roi[int(Y_min - (w -h)/2):int(Y_max + (w -h)/2), int(X_min):int(X_max)] = 1.0
    else :
        mask_roi[int(Y_min):int(Y_max), int(X_min - (h-w)/2):int(X_max + (h -w)/2)] = 1.0
    return cv2.resize(mask_roi, (shape_out, shape_out), interpolation = cv2.INTER_NEAREST)

def create_dataset(image_shape=64, n_set='train', original_image_shape=256,
                   roi_shape=32, data_path='data/'):
    """
    Creating the dataset from the images and the contour for the CNN.
    :param image_shape: image dataset desired size
    :param original_image_shape: original image size
    :param roi_shape: binary ROI mask shape
    :param data_path: path for the dataset
    :return: correct size image dataset, full size image dataset, label (contours) dataset
    """

    if n_set == 'train':
        number_set = 3
        name_set = 'Training'

    elif n_set == 'test':
        number_set = 1
        name_set = 'Online'

        # Create dataset
    series = json.load(open('series_case.json'))[n_set]
    print("series:", series)

    images, images_fullsize, contours, contour_mask = [], [], [], []

    # Loop over the series
    for case, serie in series.items():
        image_path_base = data_path + 'challenge_%s/%s/DICOM/' % (name_set.lower(), case)
        print("image_path_base:", image_path_base)

 #        contour_path_base = data_path + 'Sunnybrook Cardiac MR Database ContoursPart%s/\
 # %sDataContours/%s/contours-manual/IRCCI-expert/' % (number_set, name_set, case)

        contour_path_base = data_path + 'Sunnybrook Cardiac MR Database ContoursPart%s/\
%sDataContours/%s/contours-manual/IRCCI-expert/' % (number_set, name_set, case)

        print("contour_path_base:", contour_path_base)

        contours_list = glob.glob(contour_path_base + '*')
        print("contours_list:", contours_list)

        # windows 的路径不一样
        # contours_list_series = [k.split('/')[7].split('-')[2] for k in contours_list]
        contours_list_series = [k.split('\\')[1].split('-')[2] for k in contours_list]
        # Loop over the contours/images
        for c in contours_list_series:
            # Get contours and images path
            idx_contour = contours_list_series.index(c)
            print("c:", c, "  idx_contour:", idx_contour)
            image_path = image_path_base + 'IM-0001-%s.dcm' % c

            print("image_path:", image_path)
            contour_path = contours_list[idx_contour]

            # open image as numpy array and resize to (image_shape, image_shape)
            image_part = dicom.read_file(image_path).pixel_array

            # open contours as numpy array
            contour = []
            file = open(contour_path, 'r')
            for line in file:
                contour.append(tuple(map(float, line.split())))
            contour = np.array(contour)
            # append binary ROI mask
            contours.append(get_roi(image_part, contour))

            # create mask contour with experts contours
            x, y = np.meshgrid(np.arange(256), np.arange(256))  # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            p = Path(contour)  # make a polygon
            grid = p.contains_points(points)
            mask_contour = grid.reshape(256, 256)
            mask_contour = mask_contour * 1
            contour_mask.append(mask_contour)

            # Open image and resize it
            images.append(cv2.resize(image_part, (image_shape, image_shape)))
            images_fullsize.append(cv2.resize(image_part, (original_image_shape, original_image_shape)))

    X_fullsize = np.array(images_fullsize)
    X = np.reshape(np.array(images), [len(images), image_shape, image_shape, 1])
    Y = np.reshape(np.array(contours), [len(contours), 1, roi_shape, roi_shape])

    print('Dataset shape :', X.shape, Y.shape)
    return X, X_fullsize, Y, contour_mask

X, X_fullsize, Y, contour_mask = create_dataset(n_set='train')



f, ax = plt.subplots(ncols = 3, figsize=(10,10))
ax[0].imshow(X[30].reshape(64,64))

ax[0].set_title('Input image')
ax[1].imshow(Y[30].reshape(32, 32))
_ = ax[1].set_title('Input binary ROI mask')

ax[2].imshow(contour_mask[30])
plt.show()

def create_model(input_shape=(64, 64)):
    """
    Simple convnet model : one convolution, one average pooling and one fully connected layer:
    :return: Keras model
    """
    model = Sequential()
    model.add(Conv2D(100, (11,11), padding='valid', strides=(1, 1), input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(AveragePooling2D((6,6)))
    model.add(Reshape([-1, 8100]))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([-1, 32, 32]))
    return model

m = create_model()
m.compile(loss='mean_squared_error',
          optimizer='adam',
          metrics=['accuracy'])
print('Size for each layer :\nLayer, Input Size, Output Size')
for p in m.layers:
    print(p.name.title(), p.input_shape, p.output_shape)

def training(m, X, Y, batch_size=16, epochs= 10, data_augm=False):
    """
    Training CNN with the possibility to use data augmentation
    :param m: Keras model
    :param X: training pictures
    :param Y: training binary ROI mask
    :return: history
    """
    if data_augm:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=50,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)
        datagen.fit(X)
        history = m.fit_generator(datagen.flow(X, Y,
                                    batch_size=batch_size),
                                    steps_per_epoch=X.shape[0] // batch_size,
                                    epochs=epochs)
    else:
        history = m.fit(X, Y, batch_size=batch_size, epochs=epochs)
    return history


h = training(m, X, Y, batch_size=16, epochs= 20, data_augm=False)

metric = 'loss'
plt.plot(range(len(h.history[metric])), h.history[metric])
plt.ylabel(metric)
plt.xlabel('epochs')
plt.title("Learning curve")
plt.show()

y_pred = m.predict(X, batch_size=16)

def compute_roi_pred(y_pred, idx, roi_shape=32):
    """
    Computing and cropping a ROI from the original image for further processing in the next stage
    :param y_pred: predictions
    :param idx: desired image prediction index
    :param roi_shape: shape of the binary mask
    """
    # up sampling from 32x32 to original MR size
    pred = cv2.resize(y_pred[idx].reshape((roi_shape, roi_shape)), (
                      256,256), cv2.INTER_NEAREST)
    # select the non null pixels
    pos_pred = np.array(np.where(pred > 0.5))
    # get the center of the mask
    X_min, Y_min = pos_pred[0, :].min(), pos_pred[1, :].min()
    X_max, Y_max = pos_pred[0, :].max(), pos_pred[1, :].max()
    X_middle = X_min + (X_max - X_min) / 2
    Y_middle = Y_min + (Y_max - Y_min) / 2
    # Find ROI coordinates
    X_top = int(X_middle - 50)
    Y_top = int(Y_middle - 50)
    X_down = int(X_middle + 50)
    Y_down = int(Y_middle + 50)
    # crop ROI of size 100x100
    mask_roi = np.zeros((256, 256))
    mask_roi = cv2.rectangle(mask_roi, (X_top, Y_top), (X_down, Y_down), 1, -1)*255
    return X_fullsize[idx][X_top:X_down, Y_top:Y_down], mask_roi, contour_mask[idx][X_top:X_down, Y_top:Y_down]

pred2, mask_roi, mask_contour = compute_roi_pred(y_pred, 234)

f, ax = plt.subplots(ncols=3, figsize=(10, 10))
ax[0].imshow(X_fullsize[234])
ax[1].imshow(mask_roi)
ax[2].imshow(pred2)
plt.show()

cv2.imwrite('./Rapport/images/mask_roi.png', mask_roi)
cv2.imwrite('./Rapport/images/predic_roi.png', pred2)

plt.imshow(mask_contour), mask_contour.shape







