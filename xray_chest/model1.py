from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.layers import BatchNormalization
#from tensorflow.keras.utils import plot_model


'''
deep learning model using VGG16 network
result 25 epochs, 500 steps per epoch, batch size 8

500/500 [==============================] - 582s 1s/step - 
loss: 0.5160 - accuracy: 0.7515 - val_loss: 0.3889 - val_accuracy: 0.8160

75% ACCURACY ON TRAINING set
'''

# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(128, 128, 3)))
headModel = baseModel.output

headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = BatchNormalization(axis=1)(headModel)
headModel = Dropout(0.25)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] compiling model...")
INIT_LR = 0.001
EPOCHS = 51
BS = 32

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
print(model.summary())
#plot_model(model, to_file="model.png",show_shapes=True, show_layer_names=True)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# load data
print("[INFO] loading data...")
data = pd.read_csv("C:\\Users\Korisnik\Desktop\ori\\xray_chest\chest_xray_data_set\metadata\chest_xray_metadata.csv")
dfd = pd.DataFrame(data=data)
dfd["Label_1_Virus_category"] = dfd["Label_1_Virus_category"].replace({None: 'Normal', '': 'Normal'})
train_df, validate_df = train_test_split(dfd, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=15,
    fill_mode="nearest",
    rescale=1./255,
    width_shift_range=0.1)

train_generator = trainAug.flow_from_dataframe(
    train_df,
    "C:\\Users\Korisnik\Desktop\ori\\xray_chest\chest_xray_data_set",
    x_col='X_ray_image_name',
    y_col='Label_1_Virus_category',
    target_size=(128,128),
    class_mode='categorical',
    batch_size=BS,
)

val_set = trainAug.flow_from_dataframe(
    validate_df,
    "C:\\Users\Korisnik\Desktop\ori\\xray_chest\chest_xray_data_set",
    x_col='X_ray_image_name',
    y_col='Label_1_Virus_category',
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=BS)

# train the head of the network
print("[INFO] training head...")
H = model.fit(
    train_generator,
    steps_per_epoch=4000 // BS,
    validation_data=val_set,
    validation_steps=1000 // BS,
    epochs=EPOCHS)

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Pneumonia Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

