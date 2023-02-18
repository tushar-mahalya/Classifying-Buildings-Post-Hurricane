from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_dataset import image_dataset_from_directory


def Simple_Data_Generator(data_dirs: dict, image_dim: tuple = (128, 128), batch_size: int = 64):
    train_generator = image_dataset_from_directory(data_dirs['train_dir'],
                                                   batch_size=batch_size,
                                                   image_size=image_dim,
                                                   label_mode='binary')
    val_generator = image_dataset_from_directory(data_dirs['val_dir'],
                                                 batch_size=batch_size,
                                                 image_size=image_dim,
                                                 label_mode='binary')
    balanced_test_generator = image_dataset_from_directory(data_dirs['balanced_test_dir'],
                                                           batch_size=batch_size,
                                                           image_size=image_dim,
                                                           label_mode='binary',
                                                           shuffle=False)
    imbalanced_test_generator = image_dataset_from_directory(data_dirs['imbalanced_test_dir'],
                                                             batch_size=batch_size,
                                                             image_size=image_dim,
                                                             label_mode='binary',
                                                             shuffle=False)

    return train_generator, val_generator, balanced_test_generator, imbalanced_test_generator


def SotA_Data_Generator(data_dirs: dict, model_preprocessor, batch_size: int = 64, target_dim: tuple = (224, 224)):
    # Initializing the ImageDataGenerator() Object for Augmenting Training, Validation and Testing Data

    train_datagen = ImageDataGenerator(rotation_range=20,
                                       width_shift_range=0.3,
                                       height_shift_range=0.3,
                                       shear_range=0.2,
                                       preprocessing_function=model_preprocessor)

    test_datagen = ImageDataGenerator(preprocessing_function=model_preprocessor)

    # Generating Training Data from 'train_another'
    train_generator = train_datagen.flow_from_directory(data_dirs['train_dir'],
                                                        batch_size=batch_size,
                                                        target_size=target_dim,
                                                        class_mode='binary',
                                                        color_mode='rgb')
    # Generating Validation Data from 'validation_another'
    val_generator = test_datagen.flow_from_directory(data_dirs['val_dir'],
                                                     batch_size=batch_size,
                                                     target_size=target_dim,
                                                     class_mode='binary',
                                                     color_mode='rgb')
    # Generating Test_1 Data from 'test'
    balanced_test_generator = test_datagen.flow_from_directory(data_dirs['balanced_test_dir'],
                                                               batch_size=batch_size,
                                                               target_size=target_dim,
                                                               class_mode='binary',
                                                               color_mode='rgb',
                                                               shuffle=False)
    # Generating Test_2 Data from 'test_another'
    imbalanced_test_generator = test_datagen.flow_from_directory(data_dirs['imbalanced_test_dir'],
                                                                 batch_size=batch_size,
                                                                 target_size=target_dim,
                                                                 class_mode='binary',
                                                                 color_mode='rgb',
                                                                 shuffle=False)

    return train_generator, val_generator, balanced_test_generator, imbalanced_test_generator
