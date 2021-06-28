#%% LSTM model predictions without the use of Traffic Signals in the intersection as Model features
import pandas as pd
import numpy as np

np.random.seed(0)
numLagsPoints = 30
numHorizon = 30
framePerSecond = 10
df = pd.read_csv("dataFiles/vehicles_main_06-11_GLC.csv", index_col=0)
df = df[(df["type"] != "bus")]
df = df[(df["type"] != "passenger5")]
df.reset_index(drop=True, inplace=True)
df[["lat", "long"]] = df[["lat", "long"]].round(6)
df[["x", "y"]] = df[["x", "y"]].round(2)
# %% remove vehicles  with number of datapoints less than the sum of past observation and horizon time steps
groupData = df[["veh_id", "time"]].groupby("veh_id").count()
removeVehIdx = groupData[groupData["time"] < numLagsPoints + numHorizon].index
# %% filtered data
df_Filtered = df.drop(df[df.veh_id.isin(removeVehIdx.values)].index)
# %% vehicles used the intersection
validVehicles = []
groupData = df_Filtered.groupby("veh_id")
for idx, df in groupData:
    df.reset_index(drop=True, inplace=True)
    directions = df["edgeID"].unique()
    if "-12408" in directions:
        validVehicles.append(idx)

df_Filtered = df_Filtered[df_Filtered["veh_id"].isin(validVehicles)]
uniqueVehicles = df_Filtered.veh_id.unique()

# %% implementing rolling window for each vehicle
def prepareData(data, n_lags, n_seq):
    X = []
    y = []
    numHorizon = round(n_seq * framePerSecond)
    numLagsPoints = round(n_lags * framePerSecond)

    totalLen = data.shape[0] - numLagsPoints - numHorizon + 1
    data.loc[:, ["x", "y", "speed", "angle"]] = \
        np.divide(np.subtract(data.loc[:, ["x", "y", "speed", "angle"]], np.asarray(allDataMean)),
                  np.asarray(allDataStd))
    for i in range(int(totalLen)):
        X.append(data.iloc[i:numLagsPoints + i].values)
        y.append(data.loc[numLagsPoints + i:numLagsPoints + i + numHorizon - 1,
                 ["x", "y"]].values)
    return X, y
# %% dividing the vehicles section by turned and non-turned at intersection
direction = {"--31272#7": "--31272#6", "-31272#6": "-31272#7", "--30892#17": "--30892#16", "-30892#16": "-30892#17"}
b = {'passenger3': {"cnt": 0, "straight": [], "turned": []}, 'passenger2b': {"cnt": 0, "straight": [], "turned": []},
     'bus': {"cnt": 0, "straight": [], "turned": []},
     'passenger5': {"cnt": 0, "straight": [], "turned": []}, 'passenger1': {"cnt": 0, "straight": [], "turned": []},
     'passenger2a': {"cnt": 0, "straight": [], "turned": []}, 'passenger4': {"cnt": 0, "straight": [], "turned": []}}
a = df_Filtered.groupby("veh_id")
turnedVehicles = []
straightVehicles = []
for i, df in a:
    directions = df["edgeID"].unique()
    if "-12408" in directions:
        validVehicles.append(i)
        for k in directions:
            if k in direction.keys():
                start = k
                break
        if df["type"].unique() == "bus":
            b["bus"]["cnt"] += 1
            if direction[start] in directions:
                b["bus"]["straight"].append(list(df["veh_id"].unique())[0])
            else:
                b["bus"]["turned"].append(list(df["veh_id"].unique())[0])
        elif df["type"].unique() == "passenger3":
            b["passenger3"]["cnt"] += 1
            if direction[start] in directions:
                b["passenger3"]["straight"].append(list(df["veh_id"].unique())[0])
            else:
                b["passenger3"]["turned"].append(list(df["veh_id"].unique())[0])
        elif df["type"].unique() == "passenger2b":
            b["passenger2b"]["cnt"] += 1
            if direction[start] in directions:
                b["passenger2b"]["straight"].append(list(df["veh_id"].unique())[0])
            else:
                b["passenger2b"]["turned"].append(list(df["veh_id"].unique())[0])
        elif df["type"].unique() == "passenger5":
            b["passenger5"]["cnt"] += 1
            if direction[start] in directions:
                b["passenger5"]["straight"].append(list(df["veh_id"].unique())[0])
            else:
                b["passenger5"]["turned"].append(list(df["veh_id"].unique())[0])
        elif df["type"].unique() == "passenger1":
            b["passenger1"]["cnt"] += 1
            if direction[start] in directions:
                b["passenger1"]["straight"].append(list(df["veh_id"].unique())[0])
            else:
                b["passenger1"]["turned"].append(list(df["veh_id"].unique())[0])
        elif df["type"].unique() == "passenger2a":
            b["passenger2a"]["cnt"] += 1
            if direction[start] in directions:
                b["passenger2a"]["straight"].append(list(df["veh_id"].unique())[0])
            else:
                b["passenger2a"]["turned"].append(list(df["veh_id"].unique())[0])
        elif df["type"].unique() == "passenger4":
            b["passenger4"]["cnt"] += 1
            if direction[start] in directions:
                b["passenger4"]["straight"].append(list(df["veh_id"].unique())[0])
            else:
                b["passenger4"]["turned"].append(list(df["veh_id"].unique())[0])
for i in b:    
    straightVehicles.extend(b[i]["straight"])

for i in b:
    turnedVehicles.extend(b[i]["turned"])
# %% spliting data 0.65:0.15:0.20 for training:validation:testing set
training_X = []
training_Y = []
testing_X = []
testing_Y = []
validation_X = []
validation_Y = []
straight_testing_X = []
straight_testing_Y = []
turning_testing_X = []
turning_testing_Y = []
allData = []
trainSplit = int(len(uniqueVehicles) * 0.65)
valSplit = int(len(uniqueVehicles) * 0.15)
testSplit = int(len(uniqueVehicles) * 0.2)
trainingVehicles = uniqueVehicles[:trainSplit]
validationVehicles = uniqueVehicles[trainSplit:trainSplit + valSplit]
testingVehicles = uniqueVehicles[trainSplit + valSplit:]
#%%
allData = df_Filtered[df_Filtered["veh_id"].isin(trainingVehicles)][["x", "y", "speed", "angle"]]
# standardisation
allDataMean = allData.mean()
allDataStd = allData.std()
del allData
# %% data preparation (rolling window) for all the valid vehicles
groupData = df_Filtered.groupby("veh_id")
vehCnt = 0
for vehID, df in groupData:
    vehCnt = vehCnt + 1
    inputData = []
    outputData = []
    curVehData = df.loc[:, ["time", "x", "y", "speed", "angle"]]
    curVehData.sort_values("time", inplace=True)
    curVehData.reset_index(drop=True, inplace=True)
    curVehData.drop("time", axis=1, inplace=True)
    x, y = prepareData(curVehData, (numLagsPoints / framePerSecond), (numHorizon / framePerSecond))

    if len(y):
        inputData.append(x)
        outputData.append(y)
    if len(inputData):
        if vehID in trainingVehicles:
            training_X.append(np.concatenate(inputData[0]))
            training_Y.append(np.concatenate(outputData[0]))
        elif vehID in validationVehicles:
            validation_X.append(np.concatenate(inputData[0]))
            validation_Y.append(np.concatenate(outputData[0]))
        elif vehID in testingVehicles:
            testing_X.append(np.concatenate(inputData[0]))
            testing_Y.append(np.concatenate(outputData[0]))
            if vehID in straightVehicles:
                straight_testing_X.append(np.concatenate(inputData[0]))
                straight_testing_Y.append(np.concatenate(outputData[0]))
            elif vehID in turnedVehicles:
                turning_testing_X.append(np.concatenate(inputData[0]))
                turning_testing_Y.append(np.concatenate(outputData[0]))

# reshaping the data to support TF LSTM Model
trainingData_X = []
trainingData_Y = []
for i in range(len(training_X)):
    trainingData_X.append(training_X[i].reshape(int(len(training_X[i]) / numLagsPoints), numLagsPoints, 4))
    trainingData_Y.append(training_Y[i].reshape(int(len(training_Y[i]) / numHorizon), numHorizon, 2))

validationData_X = []
validationData_Y = []
for i in range(len(validation_X)):
    validationData_X.append(validation_X[i].reshape(int(len(validation_X[i]) / numLagsPoints), numLagsPoints, 4))
    validationData_Y.append(validation_Y[i].reshape(int(len(validation_Y[i]) / numHorizon), numHorizon, 2))

testingData_X = []
testingData_Y = []
for i in range(len(testing_X)):
    testingData_X.append(testing_X[i].reshape(int(len(testing_X[i]) / numLagsPoints), numLagsPoints, 4))
    testingData_Y.append(testing_Y[i].reshape(int(len(testing_Y[i]) / numHorizon), numHorizon, 2))

straight_testingData_X = []
straight_testingData_Y = []
for i in range(len(straight_testing_X)):
    straight_testingData_X.append(
        straight_testing_X[i].reshape(int(len(straight_testing_X[i]) / numLagsPoints), numLagsPoints, 4))
    straight_testingData_Y.append(
        straight_testing_Y[i].reshape(int(len(straight_testing_Y[i]) / numHorizon), numHorizon, 2))

turning_testingData_X = []
turning_testingData_Y = []
for i in range(len(turning_testing_X)):
    turning_testingData_X.append(
        turning_testing_X[i].reshape(int(len(turning_testing_X[i]) / numLagsPoints), numLagsPoints, 4))
    turning_testingData_Y.append(
        turning_testing_Y[i].reshape(int(len(turning_testing_Y[i]) / numHorizon), numHorizon, 2))
    
    
trainingData_X = np.concatenate(trainingData_X)
trainingData_Y = np.concatenate(trainingData_Y)
validationData_X = np.concatenate(validationData_X)
validationData_Y = np.concatenate(validationData_Y)
testingData_X = np.concatenate(testingData_X)
testingData_Y = np.concatenate(testingData_Y)
straight_testingData_X = np.concatenate(straight_testingData_X)
straight_testingData_Y = np.concatenate(straight_testingData_Y)
turning_testingData_X = np.concatenate(turning_testingData_X)
turning_testingData_Y = np.concatenate(turning_testingData_Y)
print(len(trainingData_X), len(validationData_X), len(testingData_X), flush=True)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from numpy import sqrt
import time
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as callback

K.clear_session()
#%%
class timecallback(callback.Callback):
    def __init__(self):
        self.times = []
        self.totalTime = time.perf_counter()
        self.timetaken = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.6f} "
            ",val_loss is {:7.6f}.".format(
                epoch, logs["loss"], logs["val_loss"]
            ), flush=True
        )
        print("Time Taken for Epoch:{} is {}. TotalTime Consumed: {}".format(epoch, time.perf_counter() - self.timetaken
                                                                             , time.perf_counter() - self.totalTime),
              flush=True)
        self.timetaken = time.perf_counter()

# %% Model Definition
class lstmModel():
    def __init__(self):
        pass

    def build_EncoderModel(self, train_x, train_y, val_X, val_Y, learningRate, hiddenLayers, batchSize):
        timetaken = timecallback()
        self.keras_callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=0, ),
            timetaken
        ]
        verbose, epochs, batch_size = 0, 10, batchSize
        n_timesteps, n_features, n_outputs, n_features_out = train_x.shape[1], train_x.shape[2], train_y.shape[1], \
                                                             train_y.shape[2]
        # define model
        model = Sequential()
        model.add(LSTM(hiddenLayers[0], activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(RepeatVector(n_outputs))
        model.add(LSTM(hiddenLayers[0], activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(hiddenLayers[1])))
        model.add(Dense(n_features_out, activation="linear"))
        opt = Adam(learning_rate=learningRate)
        model.compile(optimizer=opt, loss='mse')
        # fit network
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose
                  , validation_data=(val_X, val_Y)
                  , callbacks=self.keras_callbacks)
        return model

# %%
import tensorflow as tf
tf.compat.v1.set_random_seed(0)
lstm = lstmModel()
learningRate = 0.0001
batchSize = 64
hiddenLayer = [128, 64]
encoderModels = {}
encoderModels["{}-{}-{}".format(hiddenLayer, learningRate, batchSize)] = lstm.build_EncoderModel(trainingData_X,
                                                                                trainingData_Y,
                                                                                validationData_X,
                                                                                validationData_Y,
                                                                                learningRate,
                                                                                hiddenLayer, batchSize,
                                                                                )
encoderModels["{}-{}-{}".format(hiddenLayer, learningRate, batchSize)].save(
    "./savedModels/encoder_SUMOData_3-3_{}-{}-{}.h5".format(hiddenLayer, learningRate, batchSize))
# %%
predictedModelsOutputEncoder = {}
predictedModelsStraightEncoder = {}
predictedModelsTurnedEncoder = {}
for i in list(encoderModels.keys()):
    print("Predicting for Encoder model {}: ".format(i))
    pred = encoderModels[i].predict(testingData_X)
    predictedModelsOutputEncoder[i] = pred
    pred = encoderModels[i].predict(straight_testingData_X)
    predictedModelsStraightEncoder[i] = pred
    pred = encoderModels[i].predict(turning_testingData_X)
    predictedModelsTurnedEncoder[i] = pred
# %% calculating the euclidean distance between predicted and actual output
def euclDistCalc(actual, pred, actuaCol=["act_x", "act_y"], predCol=["pred_x", "pred_y"]):
    return np.linalg.norm(actual[actuaCol].values - pred[predCol].values,
                          axis=1)
# %% calculating the mean euclidean distance for all testing vehicles
eucledianDist = {}
for mdl in list(encoderModels.keys()):
    eucledianDist[mdl] = {}
    predictedOutput = predictedModelsOutputEncoder[mdl]
    actualOutput = [{i + 1: pd.DataFrame([]) for i in range(int(numHorizon / framePerSecond))}]
    predOutput = [{i + 1: pd.DataFrame([]) for i in range(int(numHorizon / framePerSecond))}]

    for j in range(int(numHorizon / framePerSecond)):
        predData = predictedOutput[:, j * framePerSecond:(j + 1) * framePerSecond, :]
        predDataDF = pd.DataFrame(predData.reshape(int(len(predData) * framePerSecond), 2))
        predOutput[0][j + 1] = np.add(np.multiply(predDataDF, np.asarray(allDataStd[:2])),
                                      np.asarray(allDataMean[:2]))
        predOutput[0][j + 1] = predOutput[0][j + 1].round(2)
        actualData = testingData_Y[:, j * framePerSecond:(j + 1) * framePerSecond, :]
        actualDataDF = pd.DataFrame(actualData.reshape(int(len(actualData) * framePerSecond), 2))
        actualOutput[0][j + 1] = np.add(np.multiply(actualDataDF, np.asarray(allDataStd[:2])),
                                        np.asarray(allDataMean[:2]))
        actualOutput[0][j + 1] = actualOutput[0][j + 1].round(2)
    del predData, predDataDF, actualDataDF, actualData

    for j in range(int(numHorizon / framePerSecond)):
        eucledianDist[mdl][j + 1] = {}
        a = actualOutput[0][j + 1].copy()
        a.columns = ["act_x", "act_y"]
        b = predOutput[0][j + 1].copy()
        b.columns = ["pred_x", "pred_y"]
        c = euclDistCalc(a, b)
        eucledianDist[mdl][j + 1]["dist"] = c.mean()
        del a, b, c
    pd.DataFrame.from_dict(eucledianDist, orient="index").to_csv(
        "euclDistOutput_3-3_{}.csv".format(mdl))

# %% calculating the mean euclidean distance for non-turning testing vehicles
eucledianDist_straight = {}
for mdl in list(predictedModelsStraightEncoder.keys()):
    eucledianDist_straight[mdl] = {}
    predictedOutput = predictedModelsStraightEncoder[mdl]
    actualOutput = [{i + 1: pd.DataFrame([]) for i in range(int(numHorizon / framePerSecond))}]
    predOutput = [{i + 1: pd.DataFrame([]) for i in range(int(numHorizon / framePerSecond))}]

    for j in range(int(numHorizon / framePerSecond)):
        predData = predictedOutput[:, j * framePerSecond:(j + 1) * framePerSecond, :]
        predDataDF = pd.DataFrame(predData.reshape(int(len(predData) * framePerSecond), 2))
        predOutput[0][j + 1] = np.add(np.multiply(predDataDF, np.asarray(allDataStd[:2])),
                                      np.asarray(allDataMean[:2]))
        predOutput[0][j + 1] = predOutput[0][j + 1].round(2)
        actualData = straight_testingData_Y[:, j * framePerSecond:(j + 1) * framePerSecond, :]
        actualDataDF = pd.DataFrame(actualData.reshape(int(len(actualData) * framePerSecond), 2))
        actualOutput[0][j + 1] = np.add(np.multiply(actualDataDF, np.asarray(allDataStd[:2])),
                                        np.asarray(allDataMean[:2]))
        actualOutput[0][j + 1] = actualOutput[0][j + 1].round(6)
    del predData, predDataDF, actualDataDF, actualData

    for j in range(int(numHorizon / framePerSecond)):
        eucledianDist_straight[mdl][j + 1] = {}
        a = actualOutput[0][j + 1].copy()
        a.columns = ["act_x", "act_y"]
        b = predOutput[0][j + 1].copy()
        b.columns = ["pred_x", "pred_y"]
        c = euclDistCalc(a, b)
        eucledianDist_straight[mdl][j + 1]["dist"] = c.mean()
        del a, b
    del c, actualOutput, predOutput, predictedOutput
    pd.DataFrame.from_dict(eucledianDist_straight, orient="index").\
        to_csv("euclDistOutputStraight_3-3_{}.csv".format(mdl))

# %% calculating the mean euclidean distance for turning testing vehicles
eucledianDist_turned = {}
for mdl in list(predictedModelsTurnedEncoder.keys()):
    eucledianDist_turned[mdl] = {}
    predictedOutput = predictedModelsTurnedEncoder[mdl]
    actualOutput = [{i + 1: pd.DataFrame([]) for i in range(int(numHorizon / framePerSecond))}]
    predOutput = [{i + 1: pd.DataFrame([]) for i in range(int(numHorizon / framePerSecond))}]

    for j in range(int(numHorizon / framePerSecond)):
        predData = predictedOutput[:, j * framePerSecond:(j + 1) * framePerSecond, :]
        predDataDF = pd.DataFrame(predData.reshape(int(len(predData) * framePerSecond), 2))
        predOutput[0][j + 1] = np.add(np.multiply(predDataDF, np.asarray(allDataStd[:2])),
                                      np.asarray(allDataMean[:2]))
        predOutput[0][j + 1] = predOutput[0][j + 1].round(2)
        actualData = turning_testingData_Y[:, j * framePerSecond:(j + 1) * framePerSecond, :]
        actualDataDF = pd.DataFrame(actualData.reshape(int(len(actualData) * framePerSecond), 2))
        actualOutput[0][j + 1] = np.add(np.multiply(actualDataDF, np.asarray(allDataStd[:2])),
                                        np.asarray(allDataMean[:2]))
        actualOutput[0][j + 1] = actualOutput[0][j + 1].round(6)
    del predData, predDataDF, actualDataDF, actualData

    for j in range(int(numHorizon / framePerSecond)):
        eucledianDist_turned[mdl][j + 1] = {}
        a = actualOutput[0][j + 1].copy()
        a.columns = ["act_x", "act_y"]
        b = predOutput[0][j + 1].copy()
        b.columns = ["pred_x", "pred_y"]
        c = euclDistCalc(a, b)
        eucledianDist_turned[mdl][j + 1]["dist"] = c.mean()
        del a, b
    del c, actualOutput, predOutput, predictedOutput
    pd.DataFrame.from_dict(eucledianDist_turned, orient="index").to_csv(
        "euclDistOutputTurned_3-3_{}.csv".format(mdl))
