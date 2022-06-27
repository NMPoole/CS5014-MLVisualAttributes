#######################################################################################################################
# CS5014 Machine Learning: Practical 2 - Learning Visual Attributes:
#
# Auxiliary functions required for the P1main script.
#
# Author: 170004680
#######################################################################################################################

# Imports:

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample

#######################################################################################################################
# Functions:
#######################################################################################################################


# Pre-process data by applying feature selection, data imputation, re-sampling, and scaling.
#
# When pre-processing the training set, the dataImputer and scaler will be none and will be created. Also, isResample
# should be true to balance the multi-classes in the data file.
#
# Params: inputData - Input data to pre-process.
#         dataImputer - Imputer to handle null values. None when pre-processing called on the training set.
#         isResample - Whether the input data should be class balanced according to some output data.
#         outputData - Output data used for re-sampling, when specified.
#         randomSeed - Seed to use for reproducibility.
#         scaler - Scaler to use to scale input data. None when pre-processing called on the training set.
#
# Return: Processed input data, output data (possibly resampled), as well as imputer and scaler used.
def preprocessData(inputData, dataImputer, isResample, outputData, randomSeed, scaler, featureSelector):

    # Data Imputation: Replace missing numerical values in a feature with the feature mean.
    if dataImputer is None:
        dataImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        dataImputer.fit(inputData)

    inputData = pd.DataFrame(dataImputer.transform(inputData), columns=inputData.columns, index=inputData.index)

    # Feature Selection: Use 80th percentile for texture and color, scoring with ANOVA F-measure.
    # Attempted classification of colour by removing histogram of gradients and attempted classification of texture by
    # removing colour histogram but got worse performance. Naturally, texture and colour are not so separate.
    if featureSelector is None:
        featureSelector = SelectPercentile(percentile=80)
        featureSelector.fit(inputData, outputData.values.ravel())

    inputData = pd.DataFrame(featureSelector.transform(inputData),
                             columns=inputData.columns[featureSelector.get_support(indices=True)],
                             index=inputData.index)

    # Resampling of the classes for balancing.
    if isResample:
        # Aside: Printing out the class distributions for each output column - highly imbalanced!
        print("Showing Output Class Frequencies To Demonstrate Imbalanced Distribution:")
        print(outputData.value_counts())

        inputData, outputData = resampleData(inputData, outputData, randomSeed)

        # Demonstrate balanced classes via resampling (random over-sampling).
        print("Showing Output Class Frequencies After Balancing (Using Random Over-Sampling):")
        print(outputData.value_counts())

    # Scaling of the input features.
    inputData, scaler = scaleData(inputData, scaler)

    return inputData, dataImputer, outputData, scaler, featureSelector


# Given input and output data, randomly over-sample the multiple output classes to balance their distribution.
#
# Under-sampling was not suitable as there is not enough data for each class.
# More advanced class balancing techniques should be investigated, such as SMOTE.
#
# Params: inputData - Input data to be resampled according to output classes.
#         outputData - Output data with classes to be rebalanced.
#         randomSeed - Random seed for reproducibility when resampling.
#
# Return: Input and output data sets having been oversampled randomly to balance the output classes.
def resampleData(inputData, outputData, randomSeed):
    outputColumnName = outputData.columns[0]
    data = pd.concat([inputData, outputData], axis=1)  # Concat input and output for ease.

    outputValueCounts = outputData.value_counts()  # Get frequencies of output classes.

    maxValueCount = outputValueCounts[0]  # Get value with highest frequency to match all other classes with.

    # For all other output classes other than the most frequent...
    for currOutputValueIndex in range(1, len(outputValueCounts)):
        currOutputValueName = outputValueCounts.index[currOutputValueIndex][0]
        classDifference = maxValueCount - outputValueCounts[currOutputValueIndex]  # Difference in frequencies.

        dataTemp = data.loc[data[outputColumnName] == currOutputValueName]
        # Randomly over-sample the current output class to match the most frequent output class.
        addedOversampledData = resample(dataTemp, replace=True, n_samples=classDifference, random_state=randomSeed)
        data = pd.concat([data, addedOversampledData], axis=0)

    inputDataOversampled = data.drop(outputColumnName, axis=1)
    outputDataOversampled = data.loc[:, [outputColumnName]]

    # Return data as input and output as given, but oversampled randomly to balance the classes.
    return inputDataOversampled, outputDataOversampled


# Scale the given data features, either by fitting to the data with a new scaler, or using a given pre-fit scaler.
#
# Params: data - The data to be scaled.
#         scaler - Scaler used to scale the data, 'None' to create and fit a new scaler.
#
# Return: Scaled data and the scaler used for scaling.
def scaleData(data, scaler):
    if scaler is None:
        scaler = RobustScaler().fit(data)  # Lots of outliers by the Z-score > 3 definition.

    trainingData = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
    return trainingData, scaler


# Train the classifier. Uses grid search to search the given parameter space to find the optimal model
# configuration. Cross validation is also used to ensure the optimal configuration corresponds to a model which has the
# best balanced accuracy on unseen data.
#
# Params: outputClassName - Output class the classifier is associated with classifying.
#         classifier - Classifier to train - e.g., neural network.
#         parameterSpace - Parameter space to search through for optimal configuration.
#         inputTrain - Input training data used for training.
#         outputTrain - Output data used as the expected outputs when training.
#
# Return: The trained model (trained with the best configuration found by grid search).
def trainNN(outputClassName, classifier, parameterSpace, inputTrain, outputTrain):
    print(outputClassName + " - Training Model (Using Grid Search And Cross Validation).")

    # Training using grid search and cross validation to get neural network trained with optimal parameters.
    model = GridSearchCV(classifier, parameterSpace, scoring='balanced_accuracy', cv=5, n_jobs=-1)
    # Model returned by training uses the best parameters from the grid search.
    model.fit(inputTrain, outputTrain.values.ravel())

    print(outputClassName + ' - Best Parameters Found:  ', model.best_params_)  # Show best parameter set found.
    print(outputClassName + ' - Best Cross Validation Balanced Accuracy Score: ', model.best_score_)  # Show best score.

    return model


# Given a model and pre-processed data, get the model's predictions on the data and compare to a set of actual outcomes
# using relevant metrics.
#
# Params: model - Classifier used to make predictions.
#         inputData - Pre-processed data to get predictions of.
#         outputData - Actual outputs of the input data to compare with predictions.
def evaluate(model, inputData, outputData):
    # Predict the outputs for the given input data - to be compared with the actual outputs (outputData).
    outputDataPredicted = model.predict(inputData)

    # Show/Visualise relevant metrics:

    precisionScore = precision_score(outputData, outputDataPredicted, average='macro', zero_division=0)
    print("Macro Average Precision Score: ", precisionScore)
    recallScore = recall_score(outputData, outputDataPredicted, average='macro', zero_division=0)
    print("Macro Average Recall Score: ", recallScore)
    f1Score = f1_score(outputData, outputDataPredicted, average='macro', zero_division=0)
    print("Macro Average F1 Score: ", f1Score)

    # Shows metrics for specific classes as a more in-depth breakdown.
    print(classification_report(outputData, outputDataPredicted, zero_division=0, digits=4))
