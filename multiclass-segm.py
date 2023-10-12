

!pip install segmentation-models #needs to be extra installed in Google Colab

#@title Import libraries
import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, jaccard_score

#@title Import folders with images and masks from Google Drive
from google.colab import drive
drive.mount('/content/gdrive')
!ls gdrive/MyDrive

!unzip gdrive/MyDrive/path_images.zip
!unzip gdrive/MyDrive/path_masks.zip

#@title Set new size of images
new_size = (384,384) #Set new size of image if needed

#@title Assign a variable with classes to be segmented
#c = ['background', 'bone', 'metal'] #In case of multiclass
c = ['background', 'bone', 'metal']
print(f"Number of Classes: {len(c)}")
num_classes = len(c)

#@title Set model parameters and evaluation metrics
BACKBONE = 'resnet34' #Set an encoder for model architecture
preprocess_input = sm.get_preprocessing(BACKBONE)
BATCH_SIZE = 10
CLASSES = c # Only needed if multiclass segmentation
LR = 0.0002 #Learning Rate
EPOCHS = 20
optimizer = tf.keras.optimizers.legacy.Adam(LR)
metrics = [sm.metrics.FScore(),  # DICE score
           sm.metrics.IOUScore(),  # Jaccard - Intersection over Unit score
           sm.metrics.Precision(),  # true positive - false positive ratio
           sm.metrics.Recall()]  # true positive - false negative ratio

# Define network parameters, the number of classes and type of activation function
n_classes = (len(CLASSES))  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#Training/Validation/Testing = 60/15/25. If other split is needed, change accordingly.

#@title Define folder with images for training and turn them into numpy array
train_images = []

for directory_path in glob.glob("path_images/"):
    index = 0  # Initialize index variable to 0
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        if index % 4 < 3:  # Check if index is divisible by 4 (i.e., chooses every first three images)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, new_size) #if resizing needed due to RAM
            train_images.append(img)
        index += 1  # Increment index variable by 1
    print("Number of train images:", len(train_images))

train_images = np.array(train_images)

#@title Define folder with masks for training and turn them into numpy array
train_masks = []

for directory_path in glob.glob("path_masks/"):
    index = 0  # Initialize index variable to 0
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        if index % 4 < 3:  # Check if index is even (i.e., first mask)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, new_size) #if resizing needed due to RAM
            mask = np.expand_dims(mask, axis=-1)
            train_masks.append(mask)
        index += 1  # Increment index variable by 1
    print("Number of train masks:", len(train_masks))

train_masks = np.array(train_masks)

print("The dimensions of train_images array is:", train_images.shape)
print("The dimensions of train_masks array is:", train_masks.shape)

#@title Define X as a variable for training images and Y as training masks
X = train_images
Y = train_masks

#@title Training/Validation split
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

#@title Preprocess the target labels (y_train and y_val) into one-hot encoded format
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

#@title Due to type error change type to float32
x_train = np.array(x_train)
y_train = np.array(y_train).astype('float32')
x_val = np.array(x_val)
y_val = np.array(y_val).astype('float32')

#@title Preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

#@title ----LOSS-----
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# Set class weights for dice_loss
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 2, 1]))
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

#@title Create model
model = sm.Unet(BACKBONE, encoder_weights = "imagenet")

#Remove the sigmoid activation layer from the model (if there's one)
# Check if the last activation is sigmoid and remove it if multiclass segmentation
if isinstance(model.layers[-1], tf.keras.layers.Activation) and model.layers[-1].activation.__name__ == 'sigmoid':
    model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

# Add the Dense layer as the penultimate layer
dense_layer = tf.keras.layers.Dense(num_classes)(model.layers[-1].output)

# Add the softmax activation to the last layer for multi-class classification
softmax_output = tf.keras.layers.Activation('softmax')(dense_layer)

# Create the new model by specifying the inputs and outputs
model = tf.keras.Model(inputs=model.input, outputs=softmax_output)

# Print model summary
model.compile(optimizer, loss= total_loss, metrics=metrics)
print(model.summary())

#@title Train model
history = model.fit(x_train,
                    y_train,
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS,
                    verbose = 1,
                    validation_data = (x_val, y_val))

# Calculate the mean training score
train_metrics = model.evaluate(x_train, y_train)

for metric, value in zip(metrics, train_metrics[1:]):
    print("mean training {}: {:.5}".format(metric.__name__, value))

# Calculate the mean val score
val_metrics = model.evaluate(x_val, y_val)

for metric, value in zip(metrics, val_metrics[1:]):
    print("mean val {}: {:.5}".format(metric.__name__, value))

# Save the trained model
model.save("/content/gdrive/MyDrive/model1.h5")

from sklearn.metrics import classification_report, confusion_matrix

# Predict on training data
train_pred = model.predict(x_train)

# Reshape ground truth and predictions to 1D arrays
y_train_flat = y_train.argmax(axis=-1).flatten()
train_pred_flat = train_pred.argmax(axis=-1).flatten()

# Define class labels (target names) explicitly
class_labels = ['0', '1', '2'] #pixel value can be put here

# Print classification report for training data with specified labels
print("Classification Report (Training):")
print(classification_report(y_train_flat, train_pred_flat, labels=[0, 1, 2], target_names=class_labels))

# Calculate confusion matrix for training data
conf_matrix_train = confusion_matrix(y_train_flat, train_pred_flat)

# Print confusion matrix for training data
print("Confusion Matrix (Training):")
print(conf_matrix_train)

# Predict on validation data
val_pred = model.predict(x_val)

# Reshape ground truth and predictions to 1D arrays
y_val_flat = y_val.argmax(axis=-1).flatten()
val_pred_flat = val_pred.argmax(axis=-1).flatten()

# Print classification report for validation data with specified labels
print("Classification Report (Validation):")
print(classification_report(y_val_flat, val_pred_flat, labels=[0, 1, 2], target_names=class_labels))

# Calculate confusion matrix for validation data
conf_matrix_val = confusion_matrix(y_val_flat, val_pred_flat)

# Print confusion matrix for validation data
print("Confusion Matrix (Validation):")
print(conf_matrix_val)

#@title Evaluation - Plot training and validation loss at each epoch
# Plot the training and validation loss at each epoch
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "y", label = "Training loss")
plt.plot(epochs, val_loss, "r", label = "Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("/content/gdrive/MyDrive/lossgraph_training43.png")

#@title Evaluation - Plot the iou score of training and validation at each epoch
iou_score = history.history["iou_score"]
val_iou_score = history.history["val_iou_score"]
epochs = range(1, len(iou_score) + 1)

plt.figure()
plt.plot(epochs, iou_score, "y", label="Training IOU Score")
plt.plot(epochs, val_iou_score, "r", label="Validation IOU Score")
plt.title("Training and validation IOU Score")
plt.xlabel("Epochs")
plt.ylabel("IOU Score")
plt.legend()
plt.savefig("/content/gdrive/MyDrive/iougraph_training1.png")

#@title Import test images
test_images = []
for directory_path in glob.glob("path_images/"):
    index = 0  # Initialize index variable to 0
    for testimg_path in glob.glob(os.path.join(directory_path, "*.png")):
        if index % 4 == 3:  # Check if index is odd (i.e., each second image)
            testimg = cv2.imread(testimg_path, cv2.IMREAD_COLOR)
            test_images.append(testimg)
        index += 1  # Increment index variable by 1
    print("Number of test images:", len(test_images))

test_images = np.array(test_images)
test_images = np.array(test_images).astype('float32')  #make format uniform

#@title Import test masks
test_masks = []
for directory_path in glob.glob("path_masks/"):
    index = 0  # Initialize index variable to 0
    for testmask_path in glob.glob(os.path.join(directory_path, "*.png")):
        if index % 4 == 3:  # Check if index is even (i.e., first image)
            testmask = cv2.imread(testmask_path, cv2.IMREAD_GRAYSCALE)
            testmask = np.expand_dims(testmask, axis=-1)
            test_masks.append(testmask)
        index += 1  # Increment index variable by 1
    print("Number of test masks:", len(test_masks))

test_masks = np.array(test_masks)
test_masks = np.array(test_masks).astype('float32')    #make format uniform

test_masks = tf.keras.utils.to_categorical(test_masks, num_classes=num_classes)

#@title Evaluate model with test images and test masks using F-Score, IOU-Score, TP-FN ratio and TP-FP ratio
test_metrics = model.evaluate(test_images, test_masks)
#print("Loss: {:.5}".format(test_metrics[0]))
for metric, value in zip(metrics, test_metrics[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

# Predict on test data
test_pred = model.predict(test_images)

# Reshape ground truth and predictions to 1D arrays
y_test_flat = np.argmax(test_masks, axis=-1).flatten()
test_pred_flat = np.argmax(test_pred, axis=-1).flatten()

# Define class labels (target names) explicitly
class_labels = ['0', '1', '2']  # Use appropriate class names

# Print classification report for test data with specified labels
print("Classification Report (Test):")
print(classification_report(y_test_flat, test_pred_flat, labels=[0, 1, 2], target_names=class_labels))

# Calculate confusion matrix for test data
conf_matrix_test = confusion_matrix(y_test_flat, test_pred_flat)

# Print confusion matrix for test data
print("Confusion Matrix (Test):")
print(conf_matrix_test)

#@title Visualize test images, ground-truth mask, prediction for n-number of images or given indices
n = 5
ids = np.random.choice(np.arange(len(test_images)), size=n)

# Helper function for data visualization
def visualize(**images):
    #Plot images in one row.
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap='gray')  # Use 'gray' colormap for grayscale visualization
    plt.show()

# Helper function for data visualization
def denormalize(x):
    #Scale image to range 0..1 for correct plot
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

# Visualize image, ground truth, and prediction
for i in ids:
    image, groundtruth_mask = test_images[i], test_masks[i]
    image = np.expand_dims(image, axis=0)
    prediction_mask = model.predict(image)#.round() it makes the prediction mask binary
    prediction_mask = prediction_mask.squeeze()  # Remove the first dimension

    visualize(
        image=denormalize(image.squeeze()),
        groundtruth_mask=groundtruth_mask,
        prediction_mask=prediction_mask,
    )
    plt.show()