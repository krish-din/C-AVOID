# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:51:22 2021

@author: cyril
"""
import pandas as pd
import numpy as np
import os
import math
import random

random.seed(0)
np.random.seed(0)
frames = 10
newFrames = 10
numLagsPoints = 30
numHorizon = 30
totalHorizon = 30
dfSizeReduction = math.floor(frames / newFrames)
framePerSecond = frames // dfSizeReduction
df = pd.read_csv("vehicles_MCInter_07-12_noTLS_PadV.csv")
junctionOfInterest = "137525"
c = df[df.leadVehicleDistance < -1]
df.loc[c.index, ['leadVehicleID', 'leadVehicleDistance', 'leadVehicleSpeed', 'leadVehicleX', 'leadVehicleY',
                 'leadVehicle2ID', 'leadVehicle2Distance', 'leadVehicle2Speed', 'leadVehicle2X', 'leadVehicle2Y']] = \
    [np.nan, -1, -1, -1, -1, np.nan, -1, -1, -1, -1]
c = df[df.leadVehicleDistance > 51]
df.loc[c.index, ['leadVehicleID',
                 'leadVehicleDistance', 'leadVehicleSpeed', 'leadVehicleX',
                 'leadVehicleY', 'leadVehicle2ID',
                 'leadVehicle2Distance', 'leadVehicle2Speed', 'leadVehicle2X',
                 'leadVehicle2Y']] = [np.nan, -1, -1, -1, -1, np.nan, -1, -1, -1, -1]
df.loc[df[df.edgeID == junctionOfInterest].index, "LaneIndex"] = 3
typeOfVehicles = ["passenger1", "passenger2a", "passenger2b", "passenger3", "passenger4", "taxi", "uber"]
df = df[df.type.isin(typeOfVehicles)]
df.reset_index(drop=True, inplace=True)
df[["lat", "long"]] = df[["lat", "long"]].round(6)
df[["x", "y"]] = df[["x", "y"]].round(2)
df['leadVehicleX'].replace(-1, 4000, inplace=True)
df['leadVehicleY'].replace(-1, 4000, inplace=True)
df['leadVehicleSpeed'].replace(-1, 50, inplace=True)
dff = df["LaneIndex"].copy()
dumm = pd.get_dummies(dff)
dumm.rename(columns={1: 'Lane1', 2: 'Lane2', 3: 'Lane3'}, inplace=True)
df = df.loc[:, ['time', 'veh_id', 'lat', 'long', 'x', 'y', 'angle', 'type', 'speed', 'acc',
                'turnSignal', 'edgeID', 'leadVehicleID',
                'leadVehicleDistance', 'leadVehicleSpeed', 'leadVehicleX',
                'leadVehicleY', 'collision', 'collisionVehicleID', 'closeCall',
                'closeCallVehicleID', ]].copy()
df = pd.concat([df, dumm], axis=1)
# %% remove data with number of datapoints less than num_data_points
groupData = df[["veh_id", "time"]].groupby("veh_id").count()
removeVehIdx = groupData[groupData["time"] < numLagsPoints + totalHorizon].index
# %% filtered data
df_Filtered = df.drop(df[df.veh_id.isin(removeVehIdx.values)].index)
# %% vehicles used the intersection
validVehicles = []
groupData = df_Filtered.groupby("veh_id")
for idx, df in groupData:
    df.reset_index(drop=True, inplace=True)
    directions = df["edgeID"].unique()
    if junctionOfInterest in directions:
        validVehicles.append(idx)

df_Filtered = df_Filtered[df_Filtered["veh_id"].isin(validVehicles)]
uniqueVehicles = df_Filtered.veh_id.unique()
collidingVehicles = df_Filtered[df_Filtered["collision"] == 1][["veh_id", "collisionVehicleID"]]
collisionPairs = {}
from ast import literal_eval as lt
for i in collidingVehicles.itertuples():
    if i.veh_id not in collisionPairs.keys():
        collisionPairs[i.veh_id] = []
    collisionVehicles = lt(i.collisionVehicleID)
    for k in collisionVehicles:
        if k not in collisionPairs[i.veh_id] and k in validVehicles:
            collisionPairs[i.veh_id].append(k)

for i in list(collisionPairs.keys()):
    if not len(collisionPairs[i]):
        collisionPairs.pop(i, None)

collidedVeh = list(collisionPairs.keys())
collidedVeh.extend([item for sublist in list(collisionPairs.values()) for item in sublist])
collidedVeh = list(set(collidedVeh))
uniqueVehicles = np.setdiff1d(uniqueVehicles, collidedVeh)


# %%
def prepareData(data, n_lags, n_seq, initial):
    X = []
    y = []
    numHorizon = round(n_seq * framePerSecond)
    numLagsPoints = round(n_lags * framePerSecond)

    totalLen = data.shape[0] - numLagsPoints - numHorizon + 1
    # data = np.divide(np.subtract(data, np.asarray(allDataMin)), np.asarray(allDataMax - allDataMin))
    data.loc[:, ["x", "y", "speed", 'acc', "angle", "leadVehicleSpeed", "leadVehicleX", "leadVehicleY"]] = \
        np.divide(np.subtract(
            data.loc[:, ["x", "y", "speed", 'acc', "angle", "leadVehicleSpeed", "leadVehicleX", "leadVehicleY"]],
            np.asarray(allDataMean)), np.asarray(allDataStd))
    for i in range(int(totalLen)):
        X.append(data.iloc[i:numLagsPoints + i].values)
        y.append(data.loc[numLagsPoints + i:numLagsPoints + i + numHorizon - 1,
                 ["x", "y"]].values)
    return X, y


# %%
direction = {"152330#0": "152330#1", "153479#11": "152404#0", "-152330#1": "-152330#0", "-152404#0": "-153479#11"}
a = df_Filtered.groupby("veh_id")
turnedVehicles = []
straightVehicles = []
for i, df in a:
    directions = df["edgeID"].unique()
    for k in directions:
        if k in direction.keys():
            start = k
            break
    if direction[start] in directions:
        straightVehicles.append(list(df["veh_id"].unique())[0])
    else:
        turnedVehicles.append(list(df["veh_id"].unique())[0])

# %%
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
# %%
trainSplit = int(len(uniqueVehicles) * 0.65)
valSplit = int(len(uniqueVehicles) * 0.15)
testSplit = int(len(uniqueVehicles) * 0.2)
trainingVehicles = uniqueVehicles[:trainSplit]
validationVehicles = uniqueVehicles[trainSplit:trainSplit + valSplit]
testingVehicles = uniqueVehicles[trainSplit + valSplit:]
## collided vehicles split
vehsPair = {}
for i in collisionPairs:
    for k in collisionPairs[i]:
        if k in vehsPair.keys():
            if k in vehsPair.keys():
                if i in vehsPair[k]:
                    continue
        if i in vehsPair.keys():
            vehsPair[i].append(k)
        else:
            vehsPair[i] = []
            vehsPair[i].append(k)

collidedVehicles = list(vehsPair.keys())
trainSplit = int((len(collidedVehicles) * 0.65))
valSplit = int((len(collidedVehicles) * 0.15))
testSplit = int(len(collidedVehicles)) - (valSplit + trainSplit)

x = list(vehsPair.keys())[:trainSplit]
collisionTrainingVehicles = []
for i in x:
    collisionTrainingVehicles.extend([i])
    collisionTrainingVehicles.extend(vehsPair[i])
collisionTrainingVehicles = list(set(collisionTrainingVehicles))

x = list(vehsPair.keys())[trainSplit: valSplit + trainSplit]
collisionValidationVehicles = []
for i in x:
    if i not in collisionTrainingVehicles:
        collisionValidationVehicles.extend([i])
        collisionValidationVehicles.extend(vehsPair[i])
collisionValidationVehicles = list(set(collisionValidationVehicles))

x = list(vehsPair.keys())[valSplit + trainSplit: valSplit + trainSplit + testSplit]
collisionTestingVehicles = []
for i in x:
    if i not in collisionTrainingVehicles and i not in collisionValidationVehicles:
        collisionTestingVehicles.extend([i])
        collisionTestingVehicles.extend(vehsPair[i])
collisionTestingVehicles = list(set(collisionTestingVehicles))

trainingVehicles = np.append(trainingVehicles, collisionTrainingVehicles)
validationVehicles = np.append(validationVehicles, collisionValidationVehicles)
testingVehicles = np.append(testingVehicles, collisionTestingVehicles)
allData = df_Filtered[df_Filtered["veh_id"].isin(trainingVehicles)][["x", "y", "speed", 'acc', "angle",
                                                                     "leadVehicleSpeed", "leadVehicleX",
                                                                     "leadVehicleY"]]
# standardisation
allDataMean = allData.mean()
allDataStd = allData.std()
del allData
print((len(trainingVehicles) + len(validationVehicles) + len(testingVehicles), len(df_Filtered.veh_id.unique())),
      flush=True)
print((allDataStd, allDataMean), flush=True)
# %%
saved = False
if not saved:
    groupData = df_Filtered.groupby("veh_id")
    vehCnt = 0
    for vehID, df in groupData:
        vehCnt = vehCnt + 1
        inputData = []
        outputData = []
        curVehData = df.loc[:, ["time", "x", "y", "speed", 'acc', "angle",
                                "leadVehicleSpeed", "leadVehicleX", "leadVehicleY"
                                , 'Lane1', 'Lane2', 'Lane3']]
        curVehData.sort_values("time", inplace=True)
        curVehData.reset_index(drop=True, inplace=True)
        curVehData.drop("time", axis=1, inplace=True)
        x, y = prepareData(curVehData, (numLagsPoints / framePerSecond), (numHorizon / framePerSecond), False)

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

    trainingData_X = []
    trainingData_Y = []
    for i in range(len(training_X)):
        trainingData_X.append(training_X[i].reshape(int(len(training_X[i]) / numLagsPoints), numLagsPoints, 11))
        trainingData_Y.append(training_Y[i].reshape(int(len(training_Y[i]) / numHorizon), numHorizon, 2))

    validationData_X = []
    validationData_Y = []
    for i in range(len(validation_X)):
        validationData_X.append(validation_X[i].reshape(int(len(validation_X[i]) / numLagsPoints), numLagsPoints, 11))
        validationData_Y.append(validation_Y[i].reshape(int(len(validation_Y[i]) / numHorizon), numHorizon, 2))

    testingData_X = []
    testingData_Y = []
    for i in range(len(testing_X)):
        testingData_X.append(testing_X[i].reshape(int(len(testing_X[i]) / numLagsPoints), numLagsPoints, 11))
        testingData_Y.append(testing_Y[i].reshape(int(len(testing_Y[i]) / numHorizon), numHorizon, 2))

    straight_testingData_X = []
    straight_testingData_Y = []
    for i in range(len(straight_testing_X)):
        straight_testingData_X.append(
            straight_testing_X[i].reshape(int(len(straight_testing_X[i]) / numLagsPoints), numLagsPoints, 11))
        straight_testingData_Y.append(
            straight_testing_Y[i].reshape(int(len(straight_testing_Y[i]) / numHorizon), numHorizon, 2))

    turning_testingData_X = []
    turning_testingData_Y = []
    for i in range(len(turning_testing_X)):
        turning_testingData_X.append(
            turning_testing_X[i].reshape(int(len(turning_testing_X[i]) / numLagsPoints), numLagsPoints, 11))
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
    print("Data Saving Started", flush=True)
    df_train_x = pd.DataFrame(trainingData_X.reshape(int(len(trainingData_X) * numLagsPoints), 11))
    df_train_x.to_csv("trainingDataX_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    del df_train_x
    df_train_y = pd.DataFrame(trainingData_Y.reshape(int(len(trainingData_Y) * numHorizon), 2))
    df_train_y.to_csv("trainingDataY_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    del df_train_y
    df_val_x = pd.DataFrame(validationData_X.reshape(int(len(validationData_X) * numLagsPoints), 11))
    df_val_x.to_csv("valDataX_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    del df_val_x
    df_val_y = pd.DataFrame(validationData_Y.reshape(int(len(validationData_Y) * numHorizon), 2))
    df_val_y.to_csv("valDataY_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    del df_val_y
    df_test_x = pd.DataFrame(testingData_X.reshape(int(len(testingData_X) * numLagsPoints), 11))
    df_test_x.to_csv("testingDataX_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    del df_test_x
    df_test_y = pd.DataFrame(testingData_Y.reshape(int(len(testingData_Y) * numHorizon), 2))
    df_test_y.to_csv("testingDataY_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    del df_test_y

    df_test_x_stgt = pd.DataFrame(straight_testingData_X.reshape(int(len(straight_testingData_X) * numLagsPoints), 11))
    df_test_x_stgt.to_csv("testingDataXSTGHT_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    del df_test_x_stgt
    df_test_y_stght = pd.DataFrame(straight_testingData_Y.reshape(int(len(straight_testingData_Y) * numHorizon), 2))
    df_test_y_stght.to_csv("testingDataYSTGHT_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    del df_test_y_stght

    df_test_x_turn = pd.DataFrame(turning_testingData_X.reshape(int(len(turning_testingData_X) * numLagsPoints), 11))
    df_test_x_turn.to_csv("testingDataXTURN_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    del df_test_x_turn
    df_test_y_turn = pd.DataFrame(turning_testingData_Y.reshape(int(len(turning_testingData_Y) * numHorizon), 2))
    df_test_y_turn.to_csv("testingDataYTURN_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    del df_test_y_turn
    del curVehData, df_Filtered, inputData, outputData, groupData, testing_X, testing_Y, training_Y, \
        training_X, x, y, validation_X, validation_Y
    print("Data Saving Completed", flush=True)
else:
    df = pd.read_csv("trainingDataX_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    df.drop("Unnamed: 0", inplace=True, axis=1)
    trainingData = np.array(df)
    trainingData_X = trainingData.reshape(int(len(trainingData) / numLagsPoints), 30, 11)

    df = pd.read_csv("trainingDataY_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    df.drop("Unnamed: 0", inplace=True, axis=1)
    trainingData = np.array(df)
    trainingData_Y = trainingData.reshape(int(len(trainingData) / numHorizon), 30, 2)

    df = pd.read_csv("valDataX_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    df.drop("Unnamed: 0", inplace=True, axis=1)
    validationData = np.array(df)
    validationData_X = validationData.reshape(int(len(validationData) / numLagsPoints), 30, 11)

    df = pd.read_csv("valDataY_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    df.drop("Unnamed: 0", inplace=True, axis=1)
    validationData = np.array(df)
    validationData_Y = validationData.reshape(int(len(validationData) / numHorizon), 30, 2)

    df = pd.read_csv("testingDataX_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    df.drop("Unnamed: 0", inplace=True, axis=1)
    testingData = np.array(df)
    testingData_X = testingData.reshape(int(len(testingData) / numLagsPoints), 30, 11)

    df = pd.read_csv("testingDataY_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    df.drop("Unnamed: 0", inplace=True, axis=1)
    testingData = np.array(df)
    testingData_Y = testingData.reshape(int(len(testingData) / numHorizon), 30, 2)

    df = pd.read_csv("testingDataXSTGHT_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    df.drop("Unnamed: 0", inplace=True, axis=1)
    testingData = np.array(df)
    straight_testingData_X = testingData.reshape(int(len(testingData) / numLagsPoints), 30, 11)

    df = pd.read_csv("testingDataYSTGHT_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    df.drop("Unnamed: 0", inplace=True, axis=1)
    testingData = np.array(df)
    straight_testingData_Y = testingData.reshape(int(len(testingData) / numHorizon), 30, 2)

    df = pd.read_csv("testingDataXTURN_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    df.drop("Unnamed: 0", inplace=True, axis=1)
    testingData = np.array(df)
    turning_testingData_X = testingData.reshape(int(len(testingData) / numLagsPoints), 30, 11)

    df = pd.read_csv("testingDataYTURN_Unregulated_wULI_MCI_wACC_PadV_3-3.csv")
    df.drop("Unnamed: 0", inplace=True, axis=1)
    testingData = np.array(df)
    turning_testingData_Y = testingData.reshape(int(len(testingData) / numHorizon), 30, 2)

    df = None
    print(len(trainingData_X), len(validationData_X), len(testingData_X), flush=True)
    del df_Filtered, groupData, testing_X, testing_Y, training_Y, training_X, validation_X, validation_Y
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


# %%
class timecallback(callback.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.totalTime = time.perf_counter()
        self.timetaken = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.6f} "
            ",accuracy is {:7.6f}"
            ",val_loss is {:7.6f}"
            ",val_accuracy is {:7.6f}.".format(
                epoch, logs["loss"], logs["acc"], logs["val_loss"], logs["val_acc"]
            ), flush=True
        )
        print("Time Taken for Epoch:{} is {}. TotalTime Consumed: {}".format(epoch, time.perf_counter() - self.timetaken
                                                                             , time.perf_counter() - self.totalTime),
              flush=True)
        self.timetaken = time.perf_counter()


# %% ds
class lstmModel():
    def __init__(self):
        pass

    def build_EncoderModel(self, train_x, train_y, val_x, val_y, learningRate, hiddenLayers, batchSize):
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
        model.compile(optimizer=opt, loss='mse', metrics=["accuracy"])
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose
                  , validation_data=(val_x, val_y)
                  , callbacks=self.keras_callbacks)
        return model


# %%
import tensorflow as tf

tf.compat.v1.set_random_seed(0)
lstm = lstmModel()
encoderModels = {}
learningRate = [0.0001]
batchSize = [64, 48, 32]
hiddenLayers = [[256, 192]]
for hiddenLayer in hiddenLayers:
    for lr in learningRate:
        for bs in batchSize:
            if (hiddenLayer == [256, 256] and bs in [64, 48]) or \
                    (hiddenLayer == [128, 128] and bs in [32]):
                continue
            print("current_setupMultipleSUMO_Unregulated_wULI_MCI_wACC_PadV_{}-{}-{}".format(hiddenLayer, lr, bs),
                  flush=True)
            encoderModels["{}-{}-{}".format(hiddenLayer, lr, bs)] = lstm.build_EncoderModel(trainingData_X,
                                                                                            trainingData_Y,
                                                                                            validationData_X,
                                                                                            validationData_Y,
                                                                                            lr, hiddenLayer, bs)
            encoderModels["{}-{}-{}".format(hiddenLayer, lr, bs)].save_weights \
                ('./checkpoints/my_checkpoint_encoder_SUMOData_Unregulated_wULI_MCI_wACC_PadV_3-3_{}-{}-{}'.format(
                    hiddenLayer, lr, bs))
            encoderModels["{}-{}-{}".format(hiddenLayer, lr, bs)].save(
                "./savedModels/encoder_SUMOData_Unregulated_wULI_MCI_wACC_PadV_3-3_{}-{}-{}.h5".format(hiddenLayer, lr,
                                                                                                    bs))
# %%
predictedModelsOutputEncoder = {}
predictedModelsStraightEncoder = {}
predictedModelsTurnedEncoder = {}
for i in list(encoderModels.keys()):
    print("Predicting for Encoder model {}: ".format(i), flush=True)
    pred = encoderModels[i].predict(testingData_X)
    predictedModelsOutputEncoder[i] = pred
    pred = encoderModels[i].predict(straight_testingData_X)
    predictedModelsStraightEncoder[i] = pred
    pred = encoderModels[i].predict(turning_testingData_X)
    predictedModelsTurnedEncoder[i] = pred


# %%
def euclDistCalc(actual, pred, actuaCol=["act_x", "act_y"], predCol=["pred_x", "pred_y"]):
    return np.linalg.norm(actual[actuaCol].values - pred[predCol].values,
                          axis=1)


# %%
rmseEncoder = {}
eucledianDist = {}
for mdl in list(encoderModels.keys()):
    rmseEncoder[mdl] = {}
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
        mseHP = mean_squared_error(actualOutput[0][j + 1].iloc[:, 0], predOutput[0][j + 1].iloc[:, 0])
        mseLP = mean_squared_error(actualOutput[0][j + 1].iloc[:, 1], predOutput[0][j + 1].iloc[:, 1])
        rmseEncoder[mdl][j + 1] = {}
        rmseEncoder[mdl][j + 1]["rmse"] = sqrt(
            mean_squared_error(actualOutput[0][j + 1], predOutput[0][j + 1]))
        rmseEncoder[mdl][j + 1]["mse"] = (
            mean_squared_error(actualOutput[0][j + 1], predOutput[0][j + 1]))
        rmseEncoder[mdl][j + 1]["HP"] = sqrt(mseHP)
        rmseEncoder[mdl][j + 1]["LP"] = sqrt(mseLP)
        eucledianDist[mdl][j + 1] = {}
        a = actualOutput[0][j + 1].copy()
        a.columns = ["act_x", "act_y"]
        b = predOutput[0][j + 1].copy()
        b.columns = ["pred_x", "pred_y"]
        c = euclDistCalc(a, b)
        eucledianDist[mdl][j + 1]["dist"] = c.mean()
        del a, b
        # c["dist"] = c.apply(lambda l: np.linalg.norm(np.array([l[0], l[1]]) - np.array([l[2], l[3]])), axis=1)
        # eucledianDist[mdl][j + 1]["dist"] = c["dist"].mean()
    pd.DataFrame.from_dict(rmseEncoder, orient="index").to_csv(
        "rmseOutput_Unregulated_wULI_MCI_wACC_PadV_3-3_{}.csv".format(mdl))
    pd.DataFrame.from_dict(eucledianDist, orient="index").to_csv(
        "euclDistOutput_Unregulated_wULI_MCI_wACC_PadV_3-3_{}.csv".format(mdl))

# %%
rmseEncoder_straight = {}
eucledianDist_straight = {}
for mdl in list(predictedModelsStraightEncoder.keys()):
    rmseEncoder_straight[mdl] = {}
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
        mseHP = mean_squared_error(actualOutput[0][j + 1].iloc[:, 0], predOutput[0][j + 1].iloc[:, 0])
        mseLP = mean_squared_error(actualOutput[0][j + 1].iloc[:, 1], predOutput[0][j + 1].iloc[:, 1])
        rmseEncoder_straight[mdl][j + 1] = {}
        rmseEncoder_straight[mdl][j + 1]["rmse"] = sqrt(
            mean_squared_error(actualOutput[0][j + 1], predOutput[0][j + 1]))
        rmseEncoder_straight[mdl][j + 1]["mse"] = (
            mean_squared_error(actualOutput[0][j + 1], predOutput[0][j + 1]))
        rmseEncoder_straight[mdl][j + 1]["HP"] = sqrt(mseHP)
        rmseEncoder_straight[mdl][j + 1]["LP"] = sqrt(mseLP)
        eucledianDist_straight[mdl][j + 1] = {}
        a = actualOutput[0][j + 1].copy()
        a.columns = ["act_x", "act_y"]
        b = predOutput[0][j + 1].copy()
        b.columns = ["pred_x", "pred_y"]
        c = euclDistCalc(a, b)
        eucledianDist_straight[mdl][j + 1]["dist"] = c.mean()
        del a, b
        # c["dist"] = c.apply(lambda l: np.linalg.norm(np.array([l[0], l[1]]) - np.array([l[2], l[3]])), axis=1)
        # eucledianDist[mdl][j + 1]["dist"] = c["dist"].mean()
    del c, actualOutput, predOutput, predictedOutput
    pd.DataFrame.from_dict(rmseEncoder_straight, orient="index").to_csv(
        "rmseOutputStraight_Unregulated_wULI_MCI_wACC_PadV_3-3_{}.csv".format(mdl))
    pd.DataFrame.from_dict(eucledianDist_straight, orient="index").to_csv(
        "euclDistOutputStraight_Unregulated_wULI_MCI_wACC_PadV_3-3_{}.csv".format(mdl))

# %%
rmseEncoder_turned = {}
eucledianDist_turned = {}
for mdl in list(predictedModelsTurnedEncoder.keys()):
    rmseEncoder_turned[mdl] = {}
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
        mseHP = mean_squared_error(actualOutput[0][j + 1].iloc[:, 0], predOutput[0][j + 1].iloc[:, 0])
        mseLP = mean_squared_error(actualOutput[0][j + 1].iloc[:, 1], predOutput[0][j + 1].iloc[:, 1])
        rmseEncoder_turned[mdl][j + 1] = {}
        rmseEncoder_turned[mdl][j + 1]["rmse"] = sqrt(
            mean_squared_error(actualOutput[0][j + 1], predOutput[0][j + 1]))
        rmseEncoder_turned[mdl][j + 1]["mse"] = (
            mean_squared_error(actualOutput[0][j + 1], predOutput[0][j + 1]))
        rmseEncoder_turned[mdl][j + 1]["HP"] = sqrt(mseHP)
        rmseEncoder_turned[mdl][j + 1]["LP"] = sqrt(mseLP)
        eucledianDist_turned[mdl][j + 1] = {}
        a = actualOutput[0][j + 1].copy()
        a.columns = ["act_x", "act_y"]
        b = predOutput[0][j + 1].copy()
        b.columns = ["pred_x", "pred_y"]
        c = euclDistCalc(a, b)
        eucledianDist_turned[mdl][j + 1]["dist"] = c.mean()
        del a, b
        # c["dist"] = c.apply(lambda l: np.linalg.norm(np.array([l[0], l[1]]) - np.array([l[2], l[3]])), axis=1)
        # eucledianDist[mdl][j + 1]["dist"] = c["dist"].mean()
    del c, actualOutput, predOutput, predictedOutput
    pd.DataFrame.from_dict(rmseEncoder_turned, orient="index").to_csv(
        "rmseOutputTurned_Unregulated_wULI_MCI_wACC_PadV_3-3_{}.csv".format(mdl))
    pd.DataFrame.from_dict(eucledianDist_turned, orient="index").to_csv(
        "euclDistOutputTurned_Unregulated_wULI_MCI_wACC_PadV_3-3_{}.csv".format(mdl))
