# Food Vision 101 (100% data), with tfds built in dataset
    # Data Augmentation
        # SNS Heatmap Confusion Matrix

# Ways to improve -> learning rate set to 0.01, use EfficientNet_v2 latest
# try to get atleast 75% accuracy

import tensorflow as tf
import tensorflow_datasets as tfds
from keras.applications.efficientnet import EfficientNetB0
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
# view available datasets
# print('\nList of Available Datasets:')
# print(tdfs.list_builders())

(train_dataset, test_dataset), info = tfds.load(
    name='food101',
    split=['train', 'validation'], # train, test or validation
    shuffle_files=False,
    as_supervised=True, # tuple -> (img, label)
    with_info=True # returns (dataset, dataset_info) store it in info
)

# examine the data from tdfs
print('\n')
print('Labels:\n')
print(info.features['label'].names)
unique_labels = info.features['label'].names
print('\n')
print(f'Length of Training Data : {len(train_dataset)}')
print(f'Length of Testing Data : {len(test_dataset)}')
print('\n')
print(train_dataset)
print(test_dataset)
print('\n')

# try with and without preprocessing model scores ....
def preprocess(image, label):
    image = tf.image.resize(image, size=[224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, label

train_dataset = train_dataset.map(preprocess)
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(batch_size=32)
test_dataset = test_dataset.map(preprocess)
test_dataset = test_dataset.batch(batch_size=32)

print(train_dataset)
print(test_dataset)

# view random images after preprocessing
i = 0
for img, label in test_dataset.unbatch().as_numpy_iterator():
    if i == 5:
        break
    print()
    print(f'Image :\n {img}')
    print(f'Label : {label}')
    plt.imshow(img)
    plt.show()
    i += 1

base_model = EfficientNetB0(include_top=False)
base_model.trainable = False
input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomHeight(factor=0.2),
    tf.keras.layers.experimental.preprocessing.RandomWidth(factor=0.2),
    tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
])

augmented_layers = data_augmentation(input_layer)
base_model_augmented = base_model(augmented_layers, training=False)
pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model_augmented)
output_layer = tf.keras.layers.Dense(units=101, activation='softmax')(pooling_layer)
model = tf.keras.Model(input_layer, output_layer)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), # onehotencode labels
    optimizer=tf.keras.optimizers.Adam(),
    metrics = ['accuracy']
)
print('\nRunning Model...')
history1 = model.fit(
    x=train_dataset,
    validation_data=test_dataset,
    epochs=5,
    verbose=2
)

# all layers trainable
model.layers[2].trainable = True
# freeze top 10 layers
for layer in model.layers[2].layers[:-10]:
    layer.trainable = False
print()
# check if top 10 layers are freezed
for layer in model.layers[2].layers:
    print(f'{layer.name} {layer.trainable}')

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)
print('\nRunning Model After Unfreeze...')
history2 = model.fit(
    x=train_dataset, 
    validation_data=test_dataset, 
    verbose=2, 
    epochs=15,
    initial_epoch=history1.epoch[-1]
)

# view history1, history 2 aka. losses and accuracies of with & without fine tuning
history1_df = pd.DataFrame(history1.history)
history2_df = pd.DataFrame(history2.history)
fig, (ax1, ax2) = plt.subplots(ncols=2)

total_epochs = len(history1_df) + len(history2_df)
total_epochs_range = range(0, total_epochs)
total_loss = history1_df['loss'].to_list() + history2_df['loss'].to_list()
total_val_loss = history1_df['val_loss'].to_list() + history2_df['val_loss'].to_list()
total_accuracy = history1_df['accuracy'].to_list() + history2_df['accuracy'].to_list()
total_val_accuracy = history1_df['val_accuracy'].to_list() + history2_df['val_accuracy'].to_list()

ax1.plot(total_epochs_range, total_loss, label='Total Loss')
ax1.plot(total_epochs_range, total_val_loss, label='Total Validation Loss')
ax1.axvline(4, linestyle='--', color='lime', label='Started Fine Tuning')
ax1.set(xlabel='Epoch', ylabel='Loss', title='Loss vs Validation Loss')
ax1.legend()

ax2.plot(total_epochs_range, total_accuracy, label='Total Accuracy')
ax2.plot(total_epochs_range, total_val_accuracy, label='Total Validation Accuracy')
ax2.axvline(4, linestyle='--', color='lime', label='Started Fine Tuning')
ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy vs Validation Accuracy')
ax2.legend()
plt.show()

# Predicition on our image/images
our_img = 'Machine Learning 3/FoodVision/my_imgs_for_testing/03-pizza-dad.jpeg'
our_img, label = preprocess(our_img, 'placeholder_label')
our_img = tf.reshape(our_img, shape=(-1, 224, 224, 3))
pred = model.predict(our_img)[0]
pred = np.argmax(pred)
print(f'Prediction : {unique_labels[pred]}')
plt.imshow(our_img)
plt.title(f'Prediction : {unique_labels[pred]}')
plt.show()

# heatmap of confusion matrix
preds = model.predict(test_dataset)
preds_edit = []
confidences = []
for pred in preds:
    preds_edit.append(unique_labels[np.argmax(pred)])
    confidences.append(np.max(pred))
actuals = []
for img, label in test_dataset.unbatch().as_numpy_iterator():
    actuals.append(unique_labels[np.argmax(label)])
sns.heatmap(data=confusion_matrix(y_pred=preds_edit, y_true=actuals), annot=True, cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.show()






# Epoch 1/5
# 2368/2368 - 1593s - loss: 2.1092 - accuracy: 0.4905 - val_loss: 1.3441 - val_accuracy: 0.6383 - 1593s/epoch - 673ms/step
# Epoch 2/5
# 2368/2368 - 1560s - loss: 1.6278 - accuracy: 0.5859 - val_loss: 1.2225 - val_accuracy: 0.6708 - 1560s/epoch - 659ms/step
# Epoch 3/5
# 2368/2368 - 1560s - loss: 1.5122 - accuracy: 0.6105 - val_loss: 1.1873 - val_accuracy: 0.6760 - 1560s/epoch - 659ms/step
# Epoch 4/5
# 2368/2368 - 1567s - loss: 1.4362 - accuracy: 0.6281 - val_loss: 1.1722 - val_accuracy: 0.6799 - 1567s/epoch - 662ms/step
# Epoch 5/5
# 2368/2368 - 1555s - loss: 1.3864 - accuracy: 0.6401 - val_loss: 1.1415 - val_accuracy: 0.6901 - 1555s/epoch - 657ms/step
