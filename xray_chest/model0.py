import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense,BatchNormalization,SpatialDropout2D, ZeroPadding2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
'''
results

'''

model=Sequential()


model.add(Conv2D(32,(3,3), padding='same',activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))
model.add(SpatialDropout2D(0.1))


model.add(Conv2D(32,(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.2))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.2))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.3))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.3))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.3))

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.5))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))
INIT_LR = 0.001
EPOCHS = 101
adam=keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

print(model.summary())
#plot_model(model, to_file="model.png",show_shapes=True, show_layer_names=True)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
print("[INFO] loading data...")
data = pd.read_csv("C:\\Users\Korisnik\Desktop\ori\\xray_chest\chest_xray_data_set\metadata\chest_xray_metadata.csv")
dfd = pd.DataFrame(data=data)
dfd["Label_1_Virus_category"] = dfd["Label_1_Virus_category"].replace({None: 'Normal', '': 'Normal'})
train_df, validate_df = train_test_split(dfd, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

test_datagen = ImageDataGenerator(
    rotation_range=15,
    height_shift_range=0.1,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
)
EPOCHS = 101
BS = 32
train_generator = test_datagen.flow_from_dataframe(
    train_df,
    "C:\\Users\Korisnik\Desktop\ori\\xray_chest\chest_xray_data_set",
    x_col='X_ray_image_name',
    y_col='Label_1_Virus_category',
    target_size=(128,128),
    class_mode='categorical',
    batch_size=BS,
)

val_set = test_datagen.flow_from_dataframe(
    validate_df,
    "C:\\Users\Korisnik\Desktop\ori\\xray_chest\chest_xray_data_set",
    x_col='X_ray_image_name',
    y_col='Label_1_Virus_category',
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=BS,
)

H= model.fit(
        train_generator,
        steps_per_epoch=4000 // BS,
        epochs = EPOCHS,
        validation_data = val_set,
        validation_steps=1000 // BS) 

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