from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet201
from keras.applications.resnet_v2 import ResNet50V2
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from keras.layers import MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


# ======================================================================
# BASELINE ANN MODEL
# ======================================================================

def Baseline_ANN():
    ANN = Sequential(
        [
            Flatten(),
            BatchNormalization(),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),

            Dense(1, activation='sigmoid'),
        ],
        name='Baseline_ANN'
    )

    ANN.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics='accuracy')
    return ANN


# ======================================================================
# CNN MODEL
# ======================================================================

def Regularized_CNN(input_shape: tuple = (128, 128, 3)):
    CNN = Sequential(
        [
            Input(shape=input_shape),
            BatchNormalization(momentum=0.7),
            Conv2D(16, kernel_size=3, strides=(1, 1), activation='relu'),
            Conv2D(16, kernel_size=3, strides=(1, 1), activation='relu'),
            MaxPooling2D(),
            Conv2D(64, kernel_size=3, strides=(1, 1), activation='relu'),
            Conv2D(64, kernel_size=3, strides=(1, 1), activation='relu'),
            MaxPooling2D(),
            Conv2D(128, kernel_size=3, strides=(1, 1), activation='relu'),
            Conv2D(128, kernel_size=3, strides=(1, 1), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(momentum=0.7),
            Dropout(0.20),
            Dense(128, activation='relu'),
            BatchNormalization(momentum=0.7),
            Dense(1, activation='sigmoid')
        ],
        name='Regularized_CNN'
    )

    CNN.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics='accuracy')

    CNN.summary()

    return CNN


# ======================================================================
# SOTA MODEL - RESNET50 V2
# ======================================================================

def ResNet50_V2(input_shape: tuple = (128, 128, 3)):
    resnet_base = ResNet50V2(input_shape=input_shape,
                             weights='imagenet',
                             include_top=False)

    resnet_base.trainable = False

    resnet = Sequential(
        [
            resnet_base,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ],
        name='ResNet50_V2'
    )

    resnet.compile(loss='binary_crossentropy',
                   optimizer=Adam(learning_rate=0.001),
                   metrics=['accuracy'])

    resnet.summary()

    return resnet


# ======================================================================
# SOTA MODEL - DENSENET201
# ======================================================================
def DenseNet_201(input_shape: tuple = (128, 128, 3)):
    densenet_base = DenseNet201(input_shape=input_shape,
                                weights='imagenet',
                                include_top=False)
    densenet_base.trainable = False

    densenet201 = Sequential(
        [
            densenet_base,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ],
        name='DenseNet201'
    )

    densenet201.compile(loss='binary_crossentropy',
                        optimizer=Adam(learning_rate=0.001),
                        metrics=['accuracy'])

    densenet201.summary()

    return densenet201
