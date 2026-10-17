import numpy
import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline




def baseline_model():
    # create model
    model = Sequential()
    # TODO: change 1st dense to 1, change activation to sigmoid
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    print ("Done loading the libraries")
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)



    # load dataset
    dataframe = pandas.read_csv("iris.csv", header=None)
    dataset = dataframe.values
    X = dataset[:,0:4].astype(float)
    Y = dataset[:,4]

    print ("Done load dataset")

    #TODO: from 3 files, create the label and combine to 1 training data file

    # do some more preprocessing
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = to_categorical(encoded_Y)

    print ("Done preprocessing dataset")

    # build the model
    print ("Done building estimator")

    kfold = KFold(n_splits=2, shuffle=True, random_state=seed)
    results = []
    for train_indices, test_indices in kfold.split(X):
        estimator = baseline_model()
        estimator.fit(X[train_indices], dummy_y[train_indices],
                      epochs=200, batch_size=5, verbose=0)
        predictions = numpy.argmax(estimator.predict(X[test_indices], verbose=0), axis=-1)
        labels = numpy.argmax(dummy_y[test_indices], axis=-1)
        results.append(numpy.mean(predictions == labels))
    print("Baseline: %.2f%% (%.2f%%)" % (numpy.mean(results) * 100, numpy.std(results) * 100))

if __name__ == '__main__':
    main()