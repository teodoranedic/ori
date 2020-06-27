#import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense,BatchNormalization,SpatialDropout2D
from sklearn.model_selection import train_test_split

'''
results

67% on training set
'''

model=Sequential()

model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))
model.add(SpatialDropout2D(0.1))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
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

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.5))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

adam=keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

from keras.preprocessing.image import ImageDataGenerator


import pandas as pd
data = pd.read_csv("C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set\metadata\chest_xray_metadata.csv")

dfd = pd.DataFrame(data=data)
dfd["Label_1_Virus_category"] = dfd["Label_1_Virus_category"].replace({None: 'Normal', '': 'Normal'})
#train_df, validate_df = train_test_split(dfd, test_size=0.00, random_state=42)
#train_df = train_df.reset_index(drop=True)
#validate_df = validate_df.reset_index(drop=True)

test_datagen = ImageDataGenerator(
    #rotation_range=15,
    height_shift_range=0.1
)

train_generator = test_datagen.flow_from_dataframe(
    dfd,
    "C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set",
    x_col='X_ray_image_name',
    y_col='Label_1_Virus_category',
    rescale=1./255,
   # shear_range=0.1,
   # zoom_range=0.2,
    #horizontal_flip=True,
    width_shift_range=0.1,
    target_size=(128,128),
    class_mode='categorical',
    batch_size=16
)

# val_set = test_datagen.flow_from_dataframe(
#     validate_df,
#     "C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set",
#     x_col='X_ray_image_name',
#     y_col='Label_1_Virus_category',
#     rescale=1. / 255,
#     shear_range=0.1,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     width_shift_range=0.1,
#     target_size=(128, 128),
#     class_mode='categorical',
#     batch_size=32)

model.fit(
        train_generator,
        steps_per_epoch=99,
        epochs=20)
        # validation_data=val_set,
        # validation_steps=12)  # formule u slikama



