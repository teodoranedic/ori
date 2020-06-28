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
'''
results:
331/331 [==============================] - 154s 466ms/step - loss: 0.1064 - accuracy: 0.9368
265/265 [==============================] - 204s 768ms/step - loss: 0.1172 - accuracy: 0.9267 - val_loss: 0.4141 - val_accuracy: 0.8202

93% accuracy on training set
82% accuracy on validation set
?? prediction on one image
'''

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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

data = pd.read_csv("C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set\metadata\chest_xray_metadata.csv")
dfd = pd.DataFrame(data=data)
dfd["Label_1_Virus_category"] = dfd["Label_1_Virus_category"].replace({None: 'Normal', '': 'Normal'})

# 80% slika za obucavanje, 20% slika za validaciju
train_df, validate_df = train_test_split(dfd, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

test_datagen = ImageDataGenerator(
    height_shift_range=0.1
)

train_generator = test_datagen.flow_from_dataframe(
    train_df,
    "C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set",
    x_col='X_ray_image_name',
    y_col='Label_1_Virus_category',
    rescale=1./255,
    width_shift_range=0.1,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=16
)

val_set = test_datagen.flow_from_dataframe(
    validate_df,
    "C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set",
    x_col='X_ray_image_name',
    y_col='Label_1_Virus_category',
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32)

history = model.fit(train_generator, batch_size=64, epochs=100, validation_data=val_set)



