import numpy as np
import pandas
import time
import random
import os
import argparse
import datetime

import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization,
    GlobalAveragePooling2D, Dropout, ReLU, Input, RandomTranslation,
    RandomZoom, RandomRotation
)
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

def create_parser():
    parser = argparse.ArgumentParser(description='Train a Quick Draw CNN.')
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--data_folder', default='data/numpy_bitmap/')
    parser.add_argument('--results_folder', default='results/')
    parser.add_argument('--experiment_name')
    parser.add_argument('--model_version', type=int, default=1)
    parser.add_argument('--using_binarization', action='store_true')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    return parser


FLAGS = create_parser().parse_args([])


model_filename = 'simple_cnn_keras_model'
model_weights_filename = 'simple_cnn_keras_model_weights'
class_file_name = 'class_names_simple_cnn'
confusion_file_name = 'confusion_matrix_simple_cnn'

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
    filename = "{}{}".format(model_filename, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_tensorboard_directory():
    tensorboard_root_folder = os.path.join(FLAGS.results_folder, "tensorboard/")
    os.makedirs(tensorboard_root_folder, exist_ok=True)

    suffix_name = get_suffix_name()
    filename = "{}{}".format(model_filename, suffix_name)
    dirpath = os.path.join(tensorboard_root_folder, filename) + "/"

    os.makedirs(dirpath, exist_ok=True)

    now = datetime.datetime.now()
    now_timestring = now.strftime("%Y%m%d_%H%M%S")

    timestamped_dirpath = os.path.join(dirpath, now_timestring)
    os.makedirs(timestamped_dirpath, exist_ok=True)

    return timestamped_dirpath

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

def get_tensorboard_callback(frequency=2):
    logdir = get_tensorboard_directory()
    print("Tensorboard logdir: ", logdir)
    return keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=frequency,
    write_graph=True, write_images=True)

def get_model_weights_filename():
    suffix_name = get_suffix_name()
    filename  = "{}{}".format(model_weights_filename, suffix_name)
    return os.path.join(FLAGS.results_folder, filename + ".weights.h5")


def get_experiment_report_filename():
    suffix_name = get_suffix_name()
    filename =  "{}{}".format("simple_cnn_keras_results", suffix_name)
    return os.path.join(FLAGS.results_folder, filename)


def compile_classifier(classifier):
    classifier.compile(
        optimizer='adam',
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=FLAGS.label_smoothing
        ),
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(
            k=3, name='top_3_accuracy'
        )]
    )
    return classifier


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
    return compile_classifier(classifier)

def model_v2():
    # model_v2() only has 1 conv layer compared to 2 conv layers in model()
    # Based on code snippets from https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
    print ('Using model version 2')

    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # # Adding a second convolutional layer
    # classifier.add(Conv2D(14, (3, 3), activation='relu'))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(get_num_classes(), activation='softmax'))
    # Compiling the CNN
    return compile_classifier(classifier)

def model_v3():
    # model_v3() only has 1 conv layer compared to 2 conv layers in model() and removed max pool
    # Based on code snippets from https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
    print ('Using model version 3')

    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(get_num_classes(), activation='softmax'))
    # Compiling the CNN
    return compile_classifier(classifier)

def model_v4():
    # model_v2() only has 1 conv layer compared to 2 conv layers in model()
    # Based on code snippets from https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
    print ('Using model version 2 dense 256')

    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # # Adding a second convolutional layer
    # classifier.add(Conv2D(14, (3, 3), activation='relu'))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dense(get_num_classes(), activation='softmax'))
    # Compiling the CNN
    return compile_classifier(classifier)

def model_v5():
    # Like model_v4() but onyl has 64 units in the last layer
    # Based on code snippets from https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
    print ('Using model version 5 dense 64 units...')

    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # # Adding a second convolutional layer
    # classifier.add(Conv2D(14, (3, 3), activation='relu'))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dense(get_num_classes(), activation='softmax'))
    # Compiling the CNN
    return compile_classifier(classifier)


def model_v6():
    """Three convolutional blocks with normalization and global pooling."""
    inputs = Input(shape=(28, 28, 1))
    x = inputs
    if FLAGS.augmentation:
        x = RandomTranslation(0.08, 0.08)(x)
        x = RandomZoom(0.1)(x)
        x = RandomRotation(0.05)(x)
    for filters in (32, 64, 128):
        x = Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(get_num_classes(), activation='softmax')(x)
    return compile_classifier(keras.Model(inputs, outputs, name='quickdraw_cnn_v6'))

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


def main():
    global FLAGS
    FLAGS = create_parser().parse_args()
    os.makedirs(FLAGS.results_folder, exist_ok=True)
    print("folder = ", FLAGS.data_folder)
    t0 = time.time()
    # fix random seed for reproducibility
    seed = FLAGS.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)



    # load dataset
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
    
    print ("Done load dataset")

    # do some more preprocessing
    # encode class values as integers
    encoder = LabelEncoder()
    # dummy_y will be one-hot encoding of classes
    encoder.fit(Y)
    dummy_y = encode_values(encoder, Y)
    dummy_y_dev = encode_values(encoder, Y_dev)
    dummy_y_test = encode_values(encoder, Y_test)
    dummy_y_dev_confusion_matrix = encode_values(encoder, Y_dev, forConfusionMatrix=True)
    dummy_y_test_confusion_matrix = encode_values(encoder, Y_test, forConfusionMatrix=True)

    print ('Dummy_y (should be one vector if class numbers):', dummy_y)

    print ("Done preprocessing dataset")

    # build the model
    # add flags to switch model
    if FLAGS.model_version == 1:
        cnn_model = model()
    elif FLAGS.model_version == 2:
        cnn_model = model_v2()
    elif FLAGS.model_version == 3:
        cnn_model = model_v3()
    elif FLAGS.model_version == 4:
        cnn_model = model_v4()
    elif FLAGS.model_version == 5:
        cnn_model = model_v5()
    elif FLAGS.model_version == 6:
        cnn_model = model_v6()
    else:
        raise ValueError("Unknown model_version {}; expected one of 1, 2, 3, 4, 5, or 6".format(
            FLAGS.model_version))

    plot_losses = TrainingPlot(get_training_plot_filename())
    callbacks = [
        get_tensorboard_callback(), plot_losses,
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    history = cnn_model.fit(
        X, dummy_y, epochs=FLAGS.max_iter, batch_size=FLAGS.batch_size,
        verbose=1, callbacks=callbacks,
        validation_data=(X_dev, dummy_y_dev)
    )

    t1 = time.time()
    training_duration_secs = t1 - t0
    experiment_result_string = "-------------------\n"
    experiment_result_string += "\nTraining time(secs): {}".format(training_duration_secs)
    experiment_result_string += "\nMax training iterations: {}".format(FLAGS.max_iter)
    experiment_result_string += "\nTraining time / Max training iterations: {}".format( 1.0 * training_duration_secs / FLAGS.max_iter)
    experiment_result_string += "\nFinal train/dev accuracy gap: {:.6f}".format(
        history.history["accuracy"][-1] -
        history.history["val_accuracy"][-1]
    )

    dummy_y_pred_dev = cnn_model.predict(X_dev)
    dummy_y_pred_test = cnn_model.predict(X_test)


    # Need to take argmax to find most likely class.
    dummy_y_pred_dev_class = dummy_y_pred_dev.argmax(axis=-1)
    dummy_y_pred_test_class = dummy_y_pred_test.argmax(axis=-1)


    # evaluate the model
    scores = cnn_model.evaluate(X_dev, dummy_y_dev, verbose=0, return_dict=True)
    experiment_result_string += "\nSimple CNN model DEV top-1 accuracy: %.2f%%" % (
        scores["accuracy"] * 100
    )
    experiment_result_string += "\nSimple CNN model DEV top-3 accuracy: %.2f%%" % (
        scores["top_3_accuracy"] * 100
    )

    scores = cnn_model.evaluate(X_test, dummy_y_test, verbose=0, return_dict=True)
    experiment_result_string += "\nSimple CNN model TEST top-1 accuracy: %.2f%%" % (
        scores["accuracy"] * 100
    )
    experiment_result_string += "\nSimple CNN model TEST top-3 accuracy: %.2f%%" % (
        scores["top_3_accuracy"] * 100
    )
    

    print(experiment_result_string)
    utils.write_contents_to_file(get_experiment_report_filename(), experiment_result_string)

    # serialize model to JSON
    # TODO: make this configurable.
    model_json = cnn_model.to_json()
    with open(get_model_filename(), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    cnn_model.save_weights(get_model_weights_filename())
    print("Saved model to disk")



    # print("predictions ", estimator.predict(X_dev))
    # print("actual output ", Y_dev)

    print ("Done building estimator")

    # kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

    # results = cross_val_score(estimator, X, dummy_y, cv=kfold, verbose=1, fit_params={'callbacks': [tensorboard]})
    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


    print("Dummy y pred dev class", dummy_y_pred_dev_class[0])
    dev_conf_matrix = confusion_matrix(dummy_y_dev_confusion_matrix, dummy_y_pred_dev_class)
    test_conf_matrix = confusion_matrix(dummy_y_test_confusion_matrix, dummy_y_pred_test_class)
    print ("Dev confusion matrix", dev_conf_matrix)
    print ("Test confusion matrix", test_conf_matrix)
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
    generate_confusion_matrix()
