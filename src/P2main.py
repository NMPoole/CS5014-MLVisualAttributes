#######################################################################################################################
# CS5014 Machine Learning: Practical 2 - Learning Visual Attributes:
#
# Main Python script for the practical.
#
# Author: 170004680
#######################################################################################################################

# Imports:
from P2functions import preprocessData, trainNN, evaluate
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Constants:

TRAIN_DATA_FILE = "../data/data_train.csv"  # Where to get the input training data from.
TEST_DATA_FILE = "../data/data_test.csv"  # Where to get the testing data from.
OUTPUT_FEATURES = ["color", "texture"]  # List of output features.
TRAIN_SET_PROPORTION = 0.8  # Use 80% of the input training data for training, 20% for evaluation.
RANDOM_SEED = 1010  # Seed ensures same randomness for reproducibility.

#######################################################################################################################
# Part 1: Loading Data.
#######################################################################################################################

# Load data from CSV file and parse missing values to NaN (null).
dataFrameTrainFile = pd.read_csv(TRAIN_DATA_FILE, na_values="?")
print("Read Data From File: '" + TRAIN_DATA_FILE + "'.")

#######################################################################################################################
# Part 2: Pre-Processing.
#######################################################################################################################

# Get input and output data sets from the training CSV file:
dataFrameInput = dataFrameTrainFile.drop(OUTPUT_FEATURES, axis=1)  # Input data.
dataFrameOutputCol = dataFrameTrainFile.loc[:, [OUTPUT_FEATURES[0]]]  # Color output data.
dataFrameOutputTex = dataFrameTrainFile.loc[:, [OUTPUT_FEATURES[1]]]  # Texture output data.

# Splitting the input and output data into training and validation sets for texture and colour.
print("Training Data Split Into Training And Evaluation (i.e., Testing) Partitions.")

inputTrainTex, inputEvalTex, outputTrainTex, outputEvalTex = \
    train_test_split(dataFrameInput, dataFrameOutputTex, train_size=TRAIN_SET_PROPORTION, random_state=RANDOM_SEED,
                     stratify=dataFrameOutputTex)

inputTrainCol, inputEvalCol, outputTrainCol, outputEvalCol = \
    train_test_split(dataFrameInput, dataFrameOutputCol, train_size=TRAIN_SET_PROPORTION, random_state=RANDOM_SEED,
                     stratify=dataFrameOutputCol)

# Pre-processing steps carried out after splitting: feature selection, data imputation, class balancing, scaling.
print("Starting Pre-Processing: Data Imputation, Data Leakage Prevention, Class Balancing, Scaling, etc...")

inputTrainTex, dataImputerTex, outputTrainTex, scalerTex, featureSelectorTex = \
    preprocessData(inputTrainTex, None, True, outputTrainTex, RANDOM_SEED, None, None)

inputTrainCol, dataImputerCol, outputTrainCol, scalerCol, featureSelectorCol = \
    preprocessData(inputTrainCol, None, True, outputTrainCol, RANDOM_SEED, None, None)

# Data Leakage:
#   Prevented as no data dependant actions are completed before the data set is split into a training and evaluation set.
#   Also, duplicate rows are removed, if they exist, before splitting so the same data could not accidentally end up in
#   both the training and evaluation sets. Thus, none of the data in the training set has 'knowledge' of the data in the
#   evaluation set.

#######################################################################################################################
# Part 3.1: Training - Logistic Regression (Baseline Models).
#######################################################################################################################

# Training simple multi-class logistic regression models as a baseline comparison for other models.

print("Training Baseline Logistic Regression Model For Predicting Texture...")
logRegTex = LogisticRegression(penalty='l2', class_weight=None, max_iter=1000)
logRegTex.fit(inputTrainTex, outputTrainTex.values.ravel())

print("Training Baseline Logistic Regression Model For Predicting Color...")
logRegCol = LogisticRegression(penalty='l2', class_weight=None, max_iter=10000)
logRegCol.fit(inputTrainCol, outputTrainCol.values.ravel())

#######################################################################################################################
# Part 3.2: Evaluation - Logistic Regression (Baseline Models).
#######################################################################################################################

# Ensure that evaluation/testing data is pre-processed equivalently to the training set.
inputEvalTex, dataImputerTex, outputEvalTex, scalerTex, featureSelectorTex = \
    preprocessData(inputEvalTex, dataImputerTex, False, outputEvalTex, RANDOM_SEED, scalerTex, featureSelectorTex)

inputEvalCol, dataImputerCol, outputEvalCol, scalerCol, featureSelectorCol = \
    preprocessData(inputEvalCol, dataImputerCol, False, outputEvalCol, RANDOM_SEED, scalerCol, featureSelectorCol)

# Evaluation on the base-line logistic regression model for both texture and color:

print("Evaluating Logistic Regression Model On Evaluation Set For Texture...")
evaluate(logRegTex, inputEvalTex, outputEvalTex.values.ravel())

print("Evaluating Logistic Regression Model On Evaluation Set For Color...")
evaluate(logRegCol, inputEvalCol, outputEvalCol.values.ravel())

#######################################################################################################################
# Part 4.1: Training - MLPClassifier (i.e., Neural Networks).
#######################################################################################################################

mlp = MLPClassifier(random_state=RANDOM_SEED, early_stopping=True, validation_fraction=0.1, max_iter=1000, n_iter_no_change=100)

# Create parameter space to search through to find the best neural network configuration parameters.
parameterSpace = {
    # Number of hidden layers and neurons.
    'hidden_layer_sizes': [(50,), (100,), (200,), (400,), (50, 50), (100, 100), (200, 200), (400, 400)],
    # Find best solver:
    'solver': ('sgd', 'adam', 'lbfgs'),
    # Activation function by neurons.
    'activation': ('logistic', 'tanh', 'relu'),
    # L2 penalty (regularization term) parameter.
    'alpha': [0.001, 0.01, 0.1],
    # The initial learning rate used. It controls the step-size in updating the weights.
    'learning_rate_init': [0.001, 0.01, 0.1],
    # Mini-batch size used by optimizer algorithms.
    'batch_size': [50, 200, 500, 1000],
}
# Note: Limitation of the grid search - SGD investigated but not SHD with momentum, a competitor to Adam in the literature!
# SGD with Momentum may be better for this application and should be investigated in future work.

# Train neural networks: uses grid search to find the best parameters, whilst using cross validation.
#modelTex = trainNN("Texture", mlp, parameterSpace, inputTrainTex, outputTrainTex)
#modelCol = trainNN("Color", mlp, parameterSpace, inputTrainCol, outputTrainCol)

# Model created with parameters set to the best found from grid search - no need to execute grid search every execution.
print("Training Neural Network Model For Predicting Texture...")
modelTex = MLPClassifier(random_state=RANDOM_SEED, early_stopping=True, validation_fraction=0.1, max_iter=1000, n_iter_no_change=100,
                         activation='relu', alpha=0.01, batch_size=50, hidden_layer_sizes=(100,), learning_rate_init=0.01)
modelTex.fit(inputTrainTex, outputTrainTex.values.ravel())

print("Training Neural Network Model For Predicting Color...")
modelCol = MLPClassifier(random_state=RANDOM_SEED, early_stopping=True, validation_fraction=0.1, max_iter=1000, n_iter_no_change=100,
                         activation='relu', alpha=0.001, batch_size=500, hidden_layer_sizes=(100, 100), learning_rate_init=0.01)
modelCol.fit(inputTrainCol, outputTrainCol.values.ravel())

#######################################################################################################################
# Part 4.2: Evaluation - MLPClassifier (i.e., Neural Networks).
#######################################################################################################################

# Ensure that evaluation/testing data is pre-processed equivalently to the training set.
# Required pre-processing for the evaluation set already carried out when evaluating the LogReg baseline models.

# Evaluation using the developed neural network model for both texture and color:

print("Evaluating Neural Network Model On Evaluation Set For Texture...")
evaluate(modelTex, inputEvalTex, outputEvalTex.values.ravel())

print("Evaluating Neural Network Model On Evaluation Set For Color...")
evaluate(modelCol, inputEvalCol, outputEvalCol.values.ravel())

#######################################################################################################################
# Part 5: Outputting Model Predictions To File (For Test Data File Input).
#######################################################################################################################

# Read in the test CSV data from file.
dataFrameTestFile = pd.read_csv(TEST_DATA_FILE, na_values="?")
print("Read Data From File: '" + TEST_DATA_FILE + "'.")

# Same pre-processing of data as the training set (note that resampling is not required).
inputTestTex, dataImputerTex, outputTestTex, scalerTex, featureSelectorTex = \
    preprocessData(dataFrameTestFile, dataImputerTex, False, None, RANDOM_SEED, scalerTex, featureSelectorTex)

inputTestCol, dataImputerCol, outputTestCol, scalerCol, featureSelectorCol = \
    preprocessData(dataFrameTestFile, dataImputerCol, False, None, RANDOM_SEED, scalerCol, featureSelectorCol)

# Get predictions of model for the test data.
outputDataPredictedTex = modelTex.predict(inputTestTex)
outputDataPredictedCol = modelCol.predict(inputTestCol)

# Save predictions to output CSV files.
pd.DataFrame(outputDataPredictedTex).to_csv("texture_test.csv", header=False, index=False)
print("Saved Texture Predictions To File: 'texture_test.csv'.")
pd.DataFrame(outputDataPredictedCol).to_csv("colour_test.csv", header=False, index=False)
print("Saved Color Predictions To File: 'colour_test.csv'.")

#######################################################################################################################
# Part 4: Advanced Tasks.
#######################################################################################################################

# Advanced Task: Comparison Of Neural Networks To Random Forests:

rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)

# Create parameter space to search through to find the best random forest configuration parameters.
rfParameterSpace = {
    'n_estimators': [100, 250, 500, 1000, 2500, 5000],  # The number of trees in the forest.
    'max_depth': [10, 50, 100, None],  # The maximum depth of a tree.
    'max_features': ['log2', 'sqrt'],  # The number of features to consider when looking for the best split.
}

# Train random forest models: uses grid search to find the best parameters, whilst using cross validation.
#rfModelTex = trainNN("Texture", rf, rfParameterSpace, inputTrainTex, outputTrainTex)
#rfModelCol = trainNN("Color", rf, rfParameterSpace, inputTrainCol, outputTrainCol)

print("Training Random Forest Model For Predicting Texture...")
rfModelTex = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, max_depth=50, max_features='log2', n_estimators=5000)
rfModelTex.fit(inputTrainTex, outputTrainTex.values.ravel())
print("Training Random Forest Model For Predicting Color...")
rfModelCol = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, max_depth=50, max_features='log2', n_estimators=2500)
rfModelCol.fit(inputTrainCol, outputTrainCol.values.ravel())

# Evaluation using the developed random forest model for both texture and color:
print("Evaluating Random Forest Model On Evaluation Set For Texture...")
evaluate(rfModelTex, inputEvalTex, outputEvalTex.values.ravel())
print("Evaluating Random Forest Model On Evaluation Set For Color...")
evaluate(rfModelCol, inputEvalCol, outputEvalCol.values.ravel())

