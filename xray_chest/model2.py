import glob
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model

# data preprocessing
img_dir = "C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set\*.jpeg"
imglist = glob.glob(img_dir)
for img_path in imglist:
    img = Image.open(img_path)
    img.resize((224, 224)).save(img_path)

# architecture
model = Sequential()
model.add(InputLayer(input_shape=(224, 224, 3)))

model.add(ZeroPadding2D(3))
model.add(Conv2D(32, (7, 7), padding="valid", strides=(2, 2)))
model.add(BatchNormalization(axis=2))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(ZeroPadding2D(1))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization(axis=2))
model.add(Dropout(0.25))

model.add(ZeroPadding2D(1))
model.add(Conv2D(64, (3, 3), padding="valid", strides=(2, 2)))
model.add(BatchNormalization(axis=2))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same", strides=(2, 2)))
model.add(BatchNormalization(axis=2))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding="same", strides=(2, 2)))
model.add(BatchNormalization(axis=2))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

data = pd.read_csv("C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set\metadata\chest_xray_metadata.csv")
dfd = pd.DataFrame(data=data)
dfd["Label_1_Virus_category"] = dfd["Label_1_Virus_category"].replace({None: 'Normal', '': 'Normal'})

# 80% slika za obucavanje, 20% slika za validaciju
train_df, validate_df = train_test_split(dfd, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

datagen = ImageDataGenerator(
    height_shift_range=0.1,
    rescale=1./255,
    # rotation_range = 40,
    # width_shift_range=0.1,
    # shear_range = 0.2,
    # zoom_range = 0.2,
    # horizontal_flip = True,
    # brightness_range = (0.5, 1.5)
)

train_generator = datagen.flow_from_dataframe(
    train_df,
    "C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set",
    x_col='X_ray_image_name',
    y_col='Label_1_Virus_category',
    #rescale=1./255,
    #width_shift_range=0.1,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=16
)

val_set = datagen.flow_from_dataframe(
    validate_df,
    "C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set",
    x_col='X_ray_image_name',
    y_col='Label_1_Virus_category',
    #rescale=1. / 255,
    #shear_range=0.1,
    #zoom_range=0.2,
    #horizontal_flip=True,
    #width_shift_range=0.1,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32)

history = model.fit(train_generator, batch_size=64, epochs=100, validation_data=val_set)


# plot metrics
N = 100 # epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Chest X-ray Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

# testiranje

# ucitavanje test podataka
test_data = pd.read_csv("C:\\Users\\teodo\Desktop\ori\\xray_chest\chest-xray-dataset-test\chest_xray_test_dataset.csv", nrows=624)
test = pd.DataFrame(data=test_data)

test["Label_1_Virus_category"] = test["Label_1_Virus_category"].replace({None: 'Normal', '': 'Normal'})
del test['Label_2_Virus_category']
test['X_ray_image_name'] = test['X_ray_image_name'].astype(str)

test_idg = ImageDataGenerator(
    rescale=1./255,
    # height_shift_range=0.001,
)

test_generator = test_idg.flow_from_dataframe(
    test,
    "C:\\Users\\teodo\Desktop\ori\\xray_chest\chest-xray-dataset-test\test",
    x_col='X_ray_image_name',
    y_col='Label_1_Virus_category',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=16
)

test_results = model.evaluate(test_generator, batch_size=8)




