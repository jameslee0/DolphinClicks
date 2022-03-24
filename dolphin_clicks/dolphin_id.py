# James Lee
# RedID: 820655947
# Class: CS550-01
# Due Date: 11/23/2021
# I promise that the attached assignment is my own work. I recognize that should this not be the case,
# I will be subject to penalties as outlined in the course syllabus. James Lee


# Might make your life easier for appending to lists
from collections import defaultdict
# We the undersigned promise that we have in good faith attempted to follow the principles of pair programming. Although
# we were free to discuss ideas with others, the implementation is our own. We have shared a common workspace (possibly
# virtually) and taken turns at the keyboard for the majority of the work that we are submitting. Furthermore, any non
# programming portions of the assignment were done independently. We recognize that should this not be the case, we will
# be subject to penalties as outlined in the course syllabus. Emily Pham and Taylor Nguyen
# Third party libraries
import numpy as np

# Only needed if you plot your confusion matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow import keras
from tensorflow.keras.models import *
from lib.partition import split_by_day
import lib.file_utilities as util


# Any other modules you create

def group_testing_data(testing_data):
    left_range = 0
    right_range = 100
    grouped = []
    iterationsAmount = int(len(testing_data) / 100)

    for iteration in range(iterationsAmount):
        grouped_clicks = []
        for x in range(left_range, right_range):
            grouped_clicks.append(testing_data[x])
        grouped.append(grouped_clicks)
        left_range += 100
        right_range += 100
    return grouped


def dolphin_classifier(data_directory):
    """
    Neural net classification of dolphin echolocation clicks to species
    :param data_directory:  root directory of data
    :return:  None
    """

    plt.ion()  # enable interactive plotting

    # use_onlyN = np.Inf  # debug, only read this many files for each species

    # This will get the files that are in the directory within the folder stated
    Ggfiles = util.get_files(data_directory + "/" + "Gg")
    Lofiles = util.get_files(data_directory + "/" + "Lo")
    Ggparse = util.parse_files(Ggfiles)
    Loparse = util.parse_files(Lofiles)

    # Here we are splitting data by the day
    Gg_split_data = split_by_day(Ggparse)
    Lo_split_data = split_by_day(Loparse)

    # Here we are splitting the testing files into training files and testing files for each species.
    GgTrainDates, GgTestDates = train_test_split(list(Gg_split_data.keys()))
    LoTrainDates, LoTestDates = train_test_split(list(Lo_split_data.keys()))


    GgTrainingData = [Gg_split_data[date] for date in GgTrainDates]

    GgTestingData = [Gg_split_data[date] for date in GgTestDates]

    LoTrainingData = [Lo_split_data[date] for date in LoTrainDates]

    LoTestingData = [Lo_split_data[date] for date in LoTestDates]

    # We are creating the label for Gg's Testing Data
    label_Gg_test = np.array([0])
    for feature in range(0, len(GgTestingData[0][0][-1]) - 1):
        currentLabel = [0]
        label_Gg_test = np.concatenate((label_Gg_test, currentLabel))

    # We are creating the label for Gg's Training Data
    label_Gg_train = np.array([0])
    for feature in range(0, len(GgTrainingData[0][0][-1]) - 1):
        currentLabel = [0]
        label_Gg_train = np.concatenate((label_Gg_train, currentLabel))

    # We are creating the label for Lo's Testing Data
    label_Lo_test = np.array([1])
    for feature in range(0, len(LoTestingData[0][0][-1]) - 1):
        currentLabel = [1]
        label_Lo_test = np.concatenate((label_Lo_test, currentLabel))

    # We are creating the label for Lo's Training Data
    label_Lo_train = np.array([1])
    for feature in range(0, len(LoTrainingData[0][0][-1]) - 1):
        currentLabel = [1]
        label_Lo_train = np.concatenate((label_Lo_train, currentLabel))

    # We will turn the testing features into arrays and assign them a variable
    GgTestFeatures = np.asarray(GgTestingData[0][0][-1])
    LoTestFeatures = np.asarray(LoTestingData[0][0][-1])
    GgTrainFeatures = np.asarray(GgTrainingData[0][0][-1])
    LoTrainFeatures = np.asarray(LoTrainingData[0][0][-1])

    # Here we are concatenating the testing and training features depending on if they are testing or training,
    # and if they are labels or features.
    dataTest = np.concatenate((GgTestFeatures, LoTestFeatures))
    labelTest = np.concatenate((label_Gg_test, label_Lo_test))
    dataTrain = np.concatenate((GgTrainFeatures, LoTrainFeatures))
    labelTrain = np.concatenate((label_Gg_train, label_Lo_train))

    # Calculating the class weight for model.
    class_weight = compute_class_weight('balanced', np.unique(labelTrain), labelTrain)
    class_weight = dict(enumerate(class_weight))

    # Creation of model
    model = Sequential()

    # We are adding hidden layers here
    model.add(Input(shape=(20)))
    model.add(Dense(100, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dense(100, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dense(100, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dense(2, activation='softmax'))

    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    model.fit(dataTrain, labelTrain, class_weight=class_weight, batch_size=100, epochs=10)

    prediction_Data = []

    # This will create the prediction sum
    prediction = model.predict(dataTest)
    for predict in prediction:
        prediction_Data.append(predict)

    # Find the log of each of the predictions
    log_prediction = np.log(prediction_Data)

    # Group the predictions together into groups of 100
    predictions_grouped = group_testing_data(prediction)

    # Initializing for for loop
    predictions_Gg_Sum = 0
    predictions_Lo_Sum = 0
    labelCounter = 0
    counter = 0
    correctPredictions = 0
    incorrectPredictions = 0

    # This is supposed to add up the predictions and see what we predict the species to be, and will
    # count how many correct or incorrect predictions we get.
    for i in predictions_grouped:
        for count in range(0, len(predictions_grouped)):
            if labelCounter > 100:
                continue
            predictions_Gg_Sum += predictions_grouped[counter][labelCounter][0]
            predictions_Lo_Sum += predictions_grouped[counter][labelCounter][1]
            # print("gg sum", predictions_Gg_Sum)
            # print("lo sum", predictions_Lo_Sum)
            if predictions_Gg_Sum > predictions_Lo_Sum:
                # print(labelTest[labelCounter])
                if labelTest[labelCounter] == 0:
                    correctPredictions += 1
                    # print(correctPredictions)
                else:
                    incorrectPredictions += 1
            else:
                if labelTest[labelCounter] == 1:
                    correctPredictions += 1
                else:
                    incorrectPredictions += 1
            labelCounter += 1
        # print("labeL: ", labelCounter)
        # print("counter: ", counter)
        counter += 1
        labelCounter = 0
        if counter > len(predictions_grouped):
            break

    accuracy = correctPredictions / (correctPredictions + incorrectPredictions)

    print("The amount of correct predictions is: ", correctPredictions)
    print("The accuracy is: ", accuracy * 100, "%")

    # This is the creation of the confusion matrix.
    confused_Matrix = confusion_matrix(y_true=labelTest, y_pred=np.argmax(log_prediction, axis=-1))
    cm_plot_labels = ['0', '1']
    matrixDisplay = ConfusionMatrixDisplay(confusion_matrix=confused_Matrix, display_labels=cm_plot_labels)
    matrixDisplay.plot()
    plt.savefig("confusion_matrix")
    plt.show()

    model.evaluate(dataTest, labelTest)


if __name__ == "__main__":
    data_directory = "/Users/james/CS550/A4/features"  # root directory of data
    dolphin_classifier(data_directory)
