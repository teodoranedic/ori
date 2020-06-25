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
'''
deep learning model using VGG16 network
result 25 epochs, 500 steps per epoch, batch size 8

500/500 [==============================] - 582s 1s/step - 
loss: 0.5160 - accuracy: 0.7515 - val_loss: 0.3889 - val_accuracy: 0.8160
'''

# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(128, 128, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("[INFO] compiling model...")

INIT_LR = 0.001
EPOCHS = 25
BS = 8

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# load data
print("[INFO] loading data...")
data = pd.read_csv("C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set\metadata\chest_xray_metadata.csv")
dfd = pd.DataFrame(data=data)
dfd["Label_1_Virus_category"] = dfd["Label_1_Virus_category"].replace({None: 'Normal', '': 'Normal'})
train_df, validate_df = train_test_split(dfd, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=15,
    fill_mode="nearest")

train_generator = trainAug.flow_from_dataframe(
    train_df,
    "C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set",
    x_col='X_ray_image_name',
    y_col='Label_1_Virus_category',
    rescale=1./255,
    width_shift_range=0.1,
    target_size=(128,128),
    class_mode='categorical',
    batch_size=BS,
)

val_set = trainAug.flow_from_dataframe(
    validate_df,
    "C:\\Users\\teodo\Desktop\ori\\xray_chest\chest_xray_data_set",
    x_col='X_ray_image_name',
    y_col='Label_1_Virus_category',
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
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



