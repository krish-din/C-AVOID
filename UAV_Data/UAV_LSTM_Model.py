import pandas as pd
import numpy as np

numLagsPoints = 30
numHorizon = 30
framePerSecond = 10
# combining all the data from the UAV drone dataset
df1 = pd.read_csv("dataFiles/orph_b1_tracksTLS_reduced.csv")
maxVeh = df1.veh_id.max() + 1
df2 = pd.read_csv("dataFiles/orph_b2_tracksTLS_reduced.csv")
df2["veh_id"] = df2["veh_id"] + maxVeh
maxVeh = df2.veh_id.max() + 1
df3 = pd.read_csv("dataFiles/orph_b3_tracksTLS_reduced.csv")
df3["veh_id"] = df3["veh_id"] + maxVeh
maxVeh = df3.veh_id.max() + 1
df4 = pd.read_csv("dataFiles/orph_b4_tracksTLS_reduced.csv")
df4["veh_id"] = df4["veh_id"] + maxVeh
df = pd.concat([df1, df2, df3, df4])
del df1, df2, df3, df4
# %% remove vehicles  with number of datapoints less than the sum of past observation and horizon time steps
groupData = df[["veh_id", "frame"]].groupby("veh_id").count()
removeVehIdx = groupData[groupData["frame"] < numLagsPoints + numHorizon].index
# %% filtered data
df_Filtered = df.drop(df[df.veh_id.isin(removeVehIdx.values)].index)
uniqueVehicles = df_Filtered.veh_id.unique()


# %% implementing rolling window for each vehicle
def prepareData(data, n_lags, n_seq, initial):
    X = []
    y = []
    numHorizon = round(n_seq * framePerSecond)
    numLagsPoints = round(n_lags * framePerSecond)

    totalLen = data.shape[0] - numLagsPoints - numHorizon + 1
    data = np.divide(np.subtract(data, np.asarray(allDataMean)), np.asarray(allDataStd))
    for i in range(int(totalLen)):
        X.append(data.iloc[i:numLagsPoints + i].values)
        y.append(data.loc[numLagsPoints + i:numLagsPoints + i + numHorizon - 1,
                 ["CenterPointX", "CenterPointY"]].values)
    return X, y


# %% filtering the vehicles that didnt use the intersection and
# dividing the valid vehicles by turned and non-turned at intersection
validVehicles = []
straightVehicles = []
turnedVehicles = []
invalidVehicles = []
middleVehicles = []
for vehId in uniqueVehicles:
    curVehData = df_Filtered[df_Filtered["veh_id"] == vehId][
        ["CenterPointX", "CenterPointY"]]
    curVehData = curVehData.interpolate(method='linear', axis=0, limit_direction="backward")
    curVehData.reset_index(drop=True, inplace=True)
    crossed = (curVehData["CenterPointX"] > 454) & (curVehData["CenterPointX"] < 1077) & \
              (curVehData["CenterPointY"] > 240) & (curVehData["CenterPointY"] < 631)
    if crossed.any():
        validVehicles.append(vehId)
        straight_a, straight_b = ((curVehData["CenterPointX"] > 872) & (curVehData["CenterPointX"] < 1077)), \
                                 ((curVehData["CenterPointX"] < 676) & (curVehData["CenterPointX"] > 454))
        turning_a, turning_b = ((curVehData["CenterPointY"] > 518) & (curVehData["CenterPointY"] < 631)), \
                               ((curVehData["CenterPointY"] < 336) & (curVehData["CenterPointY"] > 240))
        straight_A, straight_B = ((curVehData["CenterPointY"] > 518) & (curVehData["CenterPointY"] < 631)), \
                                 ((curVehData["CenterPointY"] < 336) & (curVehData["CenterPointY"] > 240))
        turning_A, turning_B = ((curVehData["CenterPointX"] > 872) & (curVehData["CenterPointX"] < 1077)), \
                               ((curVehData["CenterPointX"] < 676) & (curVehData["CenterPointX"] > 454))

        crossedTurning = np.array([
            ((curVehData["CenterPointX"] > 426) & (curVehData["CenterPointX"] < 574) &
             (curVehData["CenterPointY"] > 582) & (curVehData["CenterPointY"] < 631)).any(),
            ((curVehData["CenterPointX"] > 789) & (curVehData["CenterPointX"] < 1000) &
             (curVehData["CenterPointY"] > 539) & (curVehData["CenterPointY"] < 662)).any(),
            ((curVehData["CenterPointX"] > 935) & (curVehData["CenterPointX"] < 1120) &
             (curVehData["CenterPointY"] > 230) & (curVehData["CenterPointY"] < 290)).any(),
            ((curVehData["CenterPointX"] > 652) & (curVehData["CenterPointX"] < 784) &
             (curVehData["CenterPointY"] > 263) & (curVehData["CenterPointY"] < 294)).any()]).any()
        if ((straight_a.any() or straight_b.any()) and (turning_a.any() or turning_b.any())) or crossedTurning:
            turnedVehicles.append(vehId)
        elif ((straight_A.any() or straight_B.any()) and (turning_A.any() or turning_B.any())) or crossedTurning:
            turnedVehicles.append(vehId)
        elif (straight_a.any() and straight_b.any()) or (straight_A.any() and straight_B.any()):
            straightVehicles.append(vehId)
        else:
            straightVehicles.append(vehId)
            middleVehicles.append(vehId)
    else:
        invalidVehicles.append(vehId)
# %%
df_Filtered = df_Filtered[df_Filtered["veh_id"].isin(validVehicles)]
uniqueVehicles = df_Filtered.veh_id.unique()
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
trainSplit = int(len(uniqueVehicles) * 0.65)
valSplit = int(len(uniqueVehicles) * 0.15)
testSplit = int(len(uniqueVehicles) * 0.2)
trainingVehicles = uniqueVehicles[:trainSplit]
validationVehicles = uniqueVehicles[trainSplit:trainSplit + valSplit]
testingVehicles = uniqueVehicles[trainSplit + valSplit:]
# %%
allData = df_Filtered[df_Filtered["veh_id"].isin(trainingVehicles)][['CenterPointX', 'CenterPointY', 'Velocity']]
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
    curVehData = df.loc[:, ["frame", 'CenterPointX', 'CenterPointY', 'Velocity']]
    curVehData.sort_values("frame", inplace=True)
    curVehData.reset_index(drop=True, inplace=True)
    curVehData.drop("frame", axis=1, inplace=True)
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
# reshaping the data to support TF LSTM Model
trainingData_X = []
trainingData_Y = []
for i in range(len(training_X)):
    trainingData_X.append(training_X[i].reshape(int(len(training_X[i]) / numLagsPoints), numLagsPoints, 3))
    trainingData_Y.append(training_Y[i].reshape(int(len(training_Y[i]) / numHorizon), numHorizon, 2))

validationData_X = []
validationData_Y = []
for i in range(len(validation_X)):
    validationData_X.append(validation_X[i].reshape(int(len(validation_X[i]) / numLagsPoints), numLagsPoints, 3))
    validationData_Y.append(validation_Y[i].reshape(int(len(validation_Y[i]) / numHorizon), numHorizon, 2))

testingData_X = []
testingData_Y = []
for i in range(len(testing_X)):
    testingData_X.append(testing_X[i].reshape(int(len(testing_X[i]) / numLagsPoints), numLagsPoints, 3))
    testingData_Y.append(testing_Y[i].reshape(int(len(testing_Y[i]) / numHorizon), numHorizon, 2))

straight_testingData_X = []
straight_testingData_Y = []
for i in range(len(straight_testing_X)):
    straight_testingData_X.append(
        straight_testing_X[i].reshape(int(len(straight_testing_X[i]) / numLagsPoints), numLagsPoints, 3))
    straight_testingData_Y.append(
        straight_testing_Y[i].reshape(int(len(straight_testing_Y[i]) / numHorizon), numHorizon, 2))

turning_testingData_X = []
turning_testingData_Y = []
for i in range(len(turning_testing_X)):
    turning_testingData_X.append(
        turning_testing_X[i].reshape(int(len(turning_testing_X[i]) / numLagsPoints), numLagsPoints, 3))
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
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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
            ",val_loss is {:7.6f}.".format(
                epoch, logs["loss"], logs["val_loss"]
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

    def build_EncoderModel(self, train_x, train_y, val_X, val_Y, learningRate, hiddenLayers, batchSize):
        timetaken = timecallback()
        self.keras_callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=0, ),
            # tensorboard,
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
    "./savedModels/encoder_UAVData_NWSplit4_3-3_{}-{}-{}.h5".format(hiddenLayer, learningRate, batchSize))
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
        "euclDistOutput_UAVData_3-3_{}.csv".format(mdl))

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
    pd.DataFrame.from_dict(eucledianDist_straight, orient="index").to_csv(
        "euclDistOutputStraight_UAVData_3-3_{}.csv".format(mdl))
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
        "euclDistOutputTurned_UAVData_3-3_{}.csv".format(mdl))
