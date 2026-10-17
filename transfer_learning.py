import numpy as np
import pandas
import time
import random
import os

import argparse
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from training_plot import TrainingPlot

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import massageData
import utils
from sklearn.metrics import confusion_matrix

#TODO: consider scaling image to 299x299.

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.efficientnet import EfficientNetB0


def create_parser():
    parser = argparse.ArgumentParser(description='Train a transfer-learning model.')
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_folder', default='data/numpy_bitmap_3/')
    parser.add_argument('--results_folder', default='results/')
    parser.add_argument('--experiment_name')
    parser.add_argument('--model_version', type=int, default=1)
    parser.add_argument('--using_binarization', action='store_true')
    parser.add_argument('--transfer_model', default='InceptionV3',
                        choices=['MobileNet', 'InceptionV3', 'ResNet50', 'VGG19', 'EfficientNetB0'])
    parser.add_argument('--tune_source_model', action='store_true')
    parser.add_argument('--fine_tune_layers', type=int, default=20)
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    return parser


FLAGS = create_parser().parse_args([])


model_filename = 'transfer_learning_model'
model_weights_filename = 'transfer_learning_model_weights'
class_file_name = 'class_names_transfer_learning'
confusion_file_name = 'confusion_matrix_transfer_learning'

available_models = ['MobileNet', 'InceptionV3', 'ResNet50', 'VGG19', 'EfficientNetB0']

def get_data_folder():
    return FLAGS.data_folder

def get_num_classes():
    return len(os.listdir(get_data_folder()))

def get_suffix_name():
    return "_" + FLAGS.experiment_name if FLAGS.experiment_name else ""

def get_class_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format(class_file_name, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_confusion_matrix_name():
    return "{}{}".format(confusion_file_name, get_suffix_name())

def get_confusion_matrix_filename(split=None):
    suffix_name = get_suffix_name()
    filename  = "{}{}".format(confusion_file_name, suffix_name)
    if split:
        filename += "_" + split
    return os.path.join(FLAGS.results_folder, filename)

def get_model_filename():
    suffix_name = get_suffix_name()
    filename  = "{}{}".format(model_filename, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_model_weights_filename():
    suffix_name = get_suffix_name()
    filename  = "{}{}".format(model_weights_filename, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)


def get_experiment_report_filename():
    suffix_name = get_suffix_name()
    filename =  "{}{}".format("transfer_learning_keras_results", suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_training_plots_directory():
    root_folder = os.path.join(FLAGS.results_folder, "training_plots/")
    os.makedirs(root_folder, exist_ok=True)

    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_filename, suffix_name)
    dirpath = os.path.join(root_folder, filename) + "/"

    os.makedirs(dirpath, exist_ok=True)
    return dirpath

def get_model_name_only():
    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_filename, suffix_name)
    return filename

def get_training_plot_filename():
    directory = get_training_plots_directory()
    return os.path.join(directory, "{}_training_plot".format(get_model_name_only()))


def compile_transfer_model(model, learning_rate=0.001):
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=FLAGS.label_smoothing
        ),
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(
            k=3, name='top_3_accuracy'
        )]
    )
    return model


def encode_values(encoder, Y, forConfusionMatrix=False):
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    if not forConfusionMatrix:
        # Want 1-hot for training
        dummy_y = to_categorical(encoded_Y)
        return dummy_y
    else:
        # Want the class labels (numbers) for confusion.
        return encoded_Y

    # return encoded_Y

def decode_values(encoder, dummy_y):
    # encoded_Y = np.argmax(dummy_y)
    # return encoder.inverse_transform(encoded_Y)
    return encoder.inverse_transform(dummy_y)

def model():
    # Based on code snippets from https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8

    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Adding a second convolutional layer
    classifier.add(Conv2D(14, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(get_num_classes(), activation='softmax'))
    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

def transfer_learning(train_dataset, dev_dataset, source_model=None,
                      tune_source_model=True):
    # Based on code snippets from:https://keras.io/applications/
    print("Building transfer learning model...")
    # create the base pre-trained model
    base_model = source_model or InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    # Let's reduce to 128 as that's what gave us good perf in the simple model.
    x = Dense(128, activation='relu')(x)
    #x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(get_num_classes(), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    compile_transfer_model(model)

    # train the model on the new data for a few epochs
    print("Tuning our last custom layer...")
    plot_losses = TrainingPlot(get_training_plot_filename() + "_tune_our_last_layer")
    head_callbacks = [
        plot_losses,
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    head_history = model.fit(
        train_dataset, epochs=FLAGS.max_iter, verbose=1,
        validation_data=dev_dataset, callbacks=head_callbacks
    )

    if not tune_source_model:
        return model, head_history, None

    # Unfreeze only the configurable tail of the selected source model. Batch
    # normalization statistics remain frozen during fine-tuning.
    for layer in base_model.layers:
        layer.trainable = False
    layers_to_unfreeze = max(0, min(FLAGS.fine_tune_layers, len(base_model.layers)))
    for layer in base_model.layers[-layers_to_unfreeze:] if layers_to_unfreeze else []:
        layer.trainable = not isinstance(layer, keras.layers.BatchNormalization)

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from tensorflow.keras.optimizers import SGD
    model.compile(
        optimizer=SGD(learning_rate=0.0001, momentum=0.9, clipnorm=1),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=FLAGS.label_smoothing
        ),
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(
            k=3, name='top_3_accuracy'
        )]
    )

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    print("Tuning the last 2 inceptions layers ...")
    fine_tune_plot = TrainingPlot(
        get_training_plot_filename() + "_tune_transfer_model_last_layer"
    )
    fine_tune_callbacks = [
        fine_tune_plot,
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    fine_tune_history = model.fit(
        train_dataset, epochs=FLAGS.max_iter, verbose=1,
        validation_data=dev_dataset, callbacks=fine_tune_callbacks
    )
    return model, head_history, fine_tune_history

def get_onehot_vector(index, num_classes):
    result = np.zeros(num_classes)
    result[index] = 1
    return result

def convert_predictions_to_onehot(predictions, num_classes):
    num_examples = predictions.shape[0]
    result = np.zeros((num_examples, num_classes))
    for i in range(num_examples):
        result[i] = get_onehot_vector(predictions[i], num_classes)
    return result

def create_transfer_dataset(x, y, new_image_size, preprocess_input,
                            batch_size, shuffle=False):
        """Resize and preprocess transfer images in float32 batches."""
        dataset = tf.data.Dataset.from_tensor_slices((
            np.asarray(x, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
        ))
        if shuffle:
                dataset = dataset.shuffle(
                    buffer_size=len(x), seed=FLAGS.seed,
                    reshuffle_each_iteration=True
                )
        dataset = dataset.batch(batch_size)

        def preprocess_batch(images, labels):
                images = tf.image.resize(images, new_image_size)
                images = tf.image.grayscale_to_rgb(images)
                images = tf.cast(images, tf.float32) * 255.0
                images = preprocess_input(images)
                return tf.cast(images, tf.float32), labels

        return dataset.map(
            preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

# for inline mutate x instead of 

def assertXIsNotNan(X):
    if np.isnan(X).any():
        raise ValueError("Oh no input is nan!!!")


def get_source_model(source_model_name):
    print ('Using base model: {}'.format(source_model_name))

    input_size = (FLAGS.input_size, FLAGS.input_size)
    if source_model_name == 'MobileNet':
        new_image_size = input_size
        source_model = MobileNet(input_shape=(*input_size, 3), weights='imagenet', include_top=False)
        preprocess_input = keras.applications.mobilenet.preprocess_input
    elif source_model_name == 'InceptionV3':
        new_image_size = input_size
        source_model = InceptionV3(input_shape=(*input_size, 3), weights='imagenet', include_top=False)
        preprocess_input = keras.applications.inception_v3.preprocess_input
    elif source_model_name == 'ResNet50':
        new_image_size = input_size
        source_model = ResNet50(input_shape=(*input_size, 3), weights='imagenet', include_top=False)
        preprocess_input = keras.applications.resnet50.preprocess_input
    elif source_model_name == 'VGG19':
        new_image_size = input_size
        source_model = VGG19(input_shape=(*input_size, 3), weights='imagenet', include_top=False)
        preprocess_input = keras.applications.vgg19.preprocess_input
    elif source_model_name == 'EfficientNetB0':
        new_image_size = input_size
        source_model = EfficientNetB0(
            input_shape=(*input_size, 3), weights='imagenet', include_top=False
        )
        preprocess_input = keras.applications.efficientnet.preprocess_input
    else:
        raise AssertionError("Please use one of the available models only: {}".format(
            ", ".join(available_models)
        ))

    return new_image_size, source_model, preprocess_input

def main():
    global FLAGS
    FLAGS = create_parser().parse_args()
    os.makedirs(FLAGS.results_folder, exist_ok=True)
    # Check if correct model flag is used. Warn the user early if not.
    if FLAGS.transfer_model not in available_models:
        print ("Please rerun with one of the models: {}".format(", ".join(available_models)))
        return


    print("folder = ", FLAGS.data_folder)
    t0 = time.time()
    # fix random seed for reproducibility
    seed = FLAGS.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # load dataset
    # TODO: make this configurable.
    data = massageData.massageData(folder=FLAGS.data_folder, binarize=FLAGS.using_binarization)
    X, Y = data.getTrain()
    X_dev, Y_dev = data.getDev()
    X_test, Y_test = data.getTest()

    class_names = utils.get_label(Y_dev)
    # Reshape to CNN format (M, 28, 28, 1) from (M, 784)
    print("X shape before:", X.shape)
    X = X.reshape((-1, 28, 28, 1))
    print("X shape after: ", X.shape)

    print("X dev shape before:", X_dev.shape)
    X_dev = X_dev.reshape((-1, 28, 28, 1))
    print("X dev shape after: ", X_dev.shape)

    X_test = X_test.reshape((-1, 28, 28, 1))

    print("Converting to 3 channnels...")
    
    print ("Done load dataset")

    # do some more preprocessing    # encode class values as integers
    encoder = LabelEncoder()
    # dummy_y will be one-hot encoding of classes
    encoder.fit(Y)
    dummy_y = encode_values(encoder, Y)
    dummy_y_dev = encode_values(encoder, Y_dev)
    dummy_y_test = encode_values(encoder, Y_test)
    dummy_y_test_confusion_matrix = encode_values(encoder, Y_test, forConfusionMatrix=True)
    dummy_y_dev_confusion_matrix = encode_values(encoder, Y_dev, forConfusionMatrix=True)

    print ('Dummy_y (should be one vector if class numbers):', dummy_y)

    # build the model
    tensorboard = TensorBoard()

    new_image_size, source_model, preprocess_input = get_source_model(FLAGS.transfer_model)

    print("Creating transfer-learning input pipeline...")
    train_dataset = create_transfer_dataset(
        X, dummy_y, new_image_size, preprocess_input, FLAGS.batch_size, shuffle=True
    )
    dev_dataset = create_transfer_dataset(
        X_dev, dummy_y_dev, new_image_size, preprocess_input, FLAGS.batch_size
    )
    test_dataset = create_transfer_dataset(
        X_test, dummy_y_test, new_image_size, preprocess_input, FLAGS.batch_size
    )
    print ("Done preprocessing dataset")

    model, head_history, fine_tune_history = transfer_learning(
        train_dataset, dev_dataset, source_model, FLAGS.tune_source_model
    )

    t1 = time.time()
    training_duration_secs = t1 - t0
    experiment_result_string = "-------------------\n"
    experiment_result_string += "Using model: {}\n".format(FLAGS.transfer_model)
    experiment_result_string += "\nTraining time(secs): {}".format(training_duration_secs)
    experiment_result_string += "\nMax training iterations: {}".format(FLAGS.max_iter)
    experiment_result_string += "\nTraining time / Max training iterations: {}".format( 1.0 * training_duration_secs / FLAGS.max_iter)
    head_best_epoch = int(np.argmin(head_history.history["val_loss"]))
    experiment_result_string += "\nBest-epoch head train/dev accuracy gap: {:.6f}".format(
        head_history.history["accuracy"][head_best_epoch] -
        head_history.history["val_accuracy"][head_best_epoch]
    )
    if FLAGS.tune_source_model:
        fine_tune_best_epoch = int(np.argmin(fine_tune_history.history["val_loss"]))
        experiment_result_string += "\nBest-epoch fine-tune train/dev accuracy gap: {:.6f}".format(
            fine_tune_history.history["accuracy"][fine_tune_best_epoch] -
            fine_tune_history.history["val_accuracy"][fine_tune_best_epoch]
        )

    dummy_y_pred_dev = model.predict(dev_dataset)
    dummy_y_pred_test = model.predict(test_dataset)

    # Need to take argmax to find most likely class.
    dummy_y_pred_dev_class = dummy_y_pred_dev.argmax(axis=-1)
    dummy_y_pred_test_class = dummy_y_pred_test.argmax(axis=-1)


    # evaluate the model
    scores = model.evaluate(dev_dataset, verbose=0, return_dict=True)
    print("Model metric names: ", model.metrics_names)
    experiment_result_string += "\nTransfer learning model DEV top-1 accuracy: %.2f%%" % (
        scores["accuracy"] * 100
    )
    if dummy_y_dev.shape[-1] >= 10:
        experiment_result_string += "\nTransfer learning model DEV top-3 accuracy: %.2f%%" % (
            scores["top_3_accuracy"] * 100
        )

    scores = model.evaluate(test_dataset, verbose=0, return_dict=True)
    experiment_result_string += "\nTransfer learning model TEST top-1 accuracy: %.2f%%" % (
        scores["accuracy"] * 100
    )
    if dummy_y_test.shape[-1] >= 10:
        experiment_result_string += "\nTransfer learning model TEST top-3 accuracy: %.2f%%" % (
            scores["top_3_accuracy"] * 100
        )

    print(experiment_result_string)
    utils.write_contents_to_file(get_experiment_report_filename(), experiment_result_string)

    dev_conf_matrix = confusion_matrix(dummy_y_dev_confusion_matrix, dummy_y_pred_dev_class)
    test_conf_matrix = confusion_matrix(dummy_y_test_confusion_matrix, dummy_y_pred_test_class)
    print("Dev confusion matrix", dev_conf_matrix)
    print("Test confusion matrix", test_conf_matrix)
    pickle.dump(class_names, open(get_class_filename(), 'wb'))
    pickle.dump(dev_conf_matrix, open(get_confusion_matrix_filename("dev"), 'wb'))
    pickle.dump(test_conf_matrix, open(get_confusion_matrix_filename("test"), 'wb'))

def generate_confusion_matrix():
    class_names = pickle.load(open(get_class_filename(), 'rb'))[:10]
    for split in ("dev", "test"):
        confusion = pickle.load(open(get_confusion_matrix_filename(split), 'rb'))[:10,:10]
        utils.create_confusion_matrices(
            class_names, confusion, get_confusion_matrix_name() + "_" + split)

if __name__ == '__main__':
    main()
    #generate_confusion_matrix()
