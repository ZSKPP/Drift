# Python interpreter 3.8
# -----------------------
# Pandas       -> ver. 1.5.3
# Numpy        -> ver. 1.23.5
# Scipy        -> ver. 1.10.1
# Scikit-learn -> ver. 1.2.2
# MatplotLib   -> ver. 3.7.5
# Xgboost      -> ver. 2.0.3
# Skmultiflow  -> ver. 0.5.3
# Frouros      -> ver. 0.6.1

import warnings

warnings.filterwarnings("ignore")
import time
import os
import csv
import math
import numpy as np
import pandas as pd

# from xgboost import XGBClassifier
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
)

from skmultiflow.trees import HoeffdingTreeClassifier
#####      frouros #ver. 0.6.1
# Concept drift / Streaming / Change detection
from frouros.detectors.concept_drift import BOCD, BOCDConfig
from frouros.detectors.concept_drift import CUSUM, CUSUMConfig
from frouros.detectors.concept_drift import (
    GeometricMovingAverage,
    GeometricMovingAverageConfig,
)
from frouros.detectors.concept_drift import PageHinkley, PageHinkleyConfig

# Concept drift / Streaming / Statistical process control
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.detectors.concept_drift import ECDDWT, ECDDWTConfig
from frouros.detectors.concept_drift import EDDM, EDDMConfig
from frouros.detectors.concept_drift import HDDMA, HDDMAConfig
from frouros.detectors.concept_drift import HDDMW, HDDMWConfig
from frouros.detectors.concept_drift import RDDM, RDDMConfig

# Concept drift / Streaming / Window based
from frouros.detectors.concept_drift import ADWIN, ADWINConfig
from frouros.detectors.concept_drift import KSWIN, KSWINConfig
from frouros.detectors.concept_drift import STEPD, STEPDConfig
from frouros.metrics import PrequentialError

import matplotlib.pyplot as plt
from matplotlib import rc
rc("pdf", fonttype=42)


class myHoeffdingTree:
    def __init__(self):
        self.classifier = HoeffdingTreeClassifier()

    def fit(self, X, y):
        self.classifier.fit(X.to_numpy(), y.to_numpy().astype("int"))

    def predict(self, X):
        return self.classifier.predict(X.to_numpy())


def loadData(fileName):
    # Load data from .arff or .csv file to DataFrame.
    if os.path.splitext(fileName)[1] == ".arff":
        dataFrame = pd.DataFrame(loadarff(fileName)[0])
    if os.path.splitext(fileName)[1] == ".csv":
        if not csv.Sniffer().has_header(open(fileName, "r").read(8192)):
            dataFrame = pd.read_table(
                fileName,
                delimiter=str(
                    csv.Sniffer().sniff(open(fileName, "r").read()).delimiter
                ),
                header=None,
            )
        else:
            dataFrame = pd.read_table(
                fileName,
                delimiter=str(
                    csv.Sniffer().sniff(open(fileName, "r").read()).delimiter
                ),
            )
    # Encoding to numeric type.
    classLabelEncoder = LabelEncoder()
    for column in dataFrame.columns:
        if not pd.api.types.is_numeric_dtype(dataFrame[column]):
            dataFrame[column] = classLabelEncoder.fit_transform(dataFrame[column])

    return dataFrame


def trainAndTest(classifier, dataFrame, position, trainingSamples):
    training = dataFrame.iloc[position : position + trainingSamples, :]
    X_train = training.drop(training.columns[len(training.columns) - 1], axis=1)
    y_train = training[training.columns[len(training.columns) - 1]].astype("int")

    classifier.fit(X_train, y_train)

    tests = dataFrame.iloc[position + trainingSamples :, :]
    X = tests.drop(tests.columns[len(tests.columns) - 1], axis=1)
    y = tests[tests.columns[len(tests.columns) - 1]].to_numpy().astype("int")

    y_predict = classifier.predict(X)

    return y, y_predict


def generateAccuracyWithoutDrifts(classifier, dataFrame, trainingSamples, classCount):
    metric = PrequentialError(alpha=0.999)
    accuracyWithoutDrifts = []
    (y, y_predict) = trainAndTest(classifier, dataFrame, 0, trainingSamples)
    # Assume: len(y) = len(y_predict)
    for i in range(0, len(y)):
        if y[i : (i + 1)] == y_predict[i : (i + 1)]:
            error = 0
        else:
            error = 1
        metric_error = metric(error_value=error)
        #accuracyWithoutDrifts.append(metric_error) #ERROR Line 221
        accuracyWithoutDrifts.append(1 - metric_error)  #ACCURACY line 222

    acc = accuracy_score(y, y_predict)
    mcc = matthews_corrcoef(y, y_predict)

    if classCount > 2:
        prec = precision_score(y, y_predict, average="macro")
        recall = recall_score(y, y_predict, average="macro")
        f1 = f1_score(y, y_predict, average="macro")
    else:
        prec = precision_score(y, y_predict)
        recall = recall_score(y, y_predict)
        f1 = f1_score(y, y_predict)

    return accuracyWithoutDrifts, acc, mcc, prec, recall, f1


def findDrift(y, y_predict, y_all, y_predict_all, driftDetector):
    # Assume: len(y) = len(y_predict)
    # metric = PrequentialError(alpha = 0.999)
    i = 0
    while i < len(y):
        y_all.append(y[i])
        y_predict_all.append(y_predict[i])
        if y[i : (i + 1)] == y_predict[i : (i + 1)]:
            error = 0
        else:
            error = 1
        # metric_error = metric(error_value=error)

        _ = driftDetector.update(value=error)
        if driftDetector.drift:
            driftDetector.reset()
            return i
        i = i + 1

    return -1


def completeAccuracyArray(y, y_predict, y_all, y_predict_all, i, trainingSamples):
    k = 0
    while (k < trainingSamples) and (i < len(y)):
        y_all.append(y[i])
        y_predict_all.append(y_predict[i])
        i = i + 1
        k = k + 1


def generateAccuracyWithDrifts(
    classifier, driftDetector, dataFrame, trainingSamples, classCount
):
    j = 0
    t = 0
    drft = 0
    finish = False
    drifts = []
    y_all = []
    y_predict_all = []
    while not finish:
        if j + trainingSamples < len(dataFrame):
            (y, y_predict) = trainAndTest(classifier, dataFrame, j, trainingSamples)
            i = findDrift(y, y_predict, y_all, y_predict_all, driftDetector)
            if i == -1:
                finish = True
            else:
                drft = drft + i + t
                drifts.append(drft)
                j = drft + trainingSamples
                completeAccuracyArray(
                    y, y_predict, y_all, y_predict_all, i, trainingSamples
                )
            t = trainingSamples
        else:
            finish = True

    accuracy = []
    metric = PrequentialError(alpha=0.999)
    # Assume: len(y_all) = len(y_predict_all)
    for i in range(0, len(y_all)):
        if y_all[i : (i + 1)] == y_predict_all[i : (i + 1)]:
            error = 0
        else:
            error = 1
        metric_error = metric(error_value=error)
        #accuracy.append(metric_error)  #ERROR  line 136
        accuracy.append(1 - metric_error) #ACCURACY line 137

    acc = accuracy_score(y_all, y_predict_all)
    mcc = matthews_corrcoef(y_all, y_predict_all)

    if classCount > 2:
        prec = precision_score(y_all, y_predict_all, average="macro")
        recall = recall_score(y_all, y_predict_all, average="macro")
        f1 = f1_score(y_all, y_predict_all, average="macro")
    else:
        prec = precision_score(y_all, y_predict_all)
        recall = recall_score(y_all, y_predict_all)
        f1 = f1_score(y_all, y_predict_all)

    return accuracy, acc, mcc, prec, recall, f1, drifts


def DrawChart(accuracyWithoutDrifts, accuracyWithDrifts, drifts):
    plt.subplot(2, 1, 1)
    plt.ylim([0.0, 1.01])
    plt.plot(range(0, len(accuracyWithoutDrifts)), accuracyWithoutDrifts, "r")
    plt.xlabel("Record", fontsize=12)
    #plt.ylabel("Error rate")
    plt.ylabel("Accuracy before")
    plt.tick_params(axis="both", which="major")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    if len(drifts) > 0:
        plt.axvline(x=(drifts[0] - 1), color="b", linewidth=2)
        if (drifts[0] + RECORDS_IN_CHUNK - 1) < len(accuracyWithDrifts):
            plt.axvline(x=(drifts[0] + RECORDS_IN_CHUNK - 1), color="g", linewidth=2)
        for i in range(1, len(drifts)):
            plt.axvline(x=(drifts[i] - 1), color="b", linewidth=2)
            if (drifts[i] + RECORDS_IN_CHUNK - 1) < len(accuracyWithDrifts):
                plt.axvline(
                    x=(drifts[i] + RECORDS_IN_CHUNK - 1), color="g", linewidth=2
                )
    plt.ylim([0.0, 1.01])
    plt.plot(range(0, len(accuracyWithDrifts)), accuracyWithDrifts, "r")
    plt.xlabel("Record", fontsize=12)
    #plt.ylabel("Error rate")
    plt.ylabel("Accuracy after")
    plt.tick_params(axis="both", which="major")
    plt.grid(True)
    plt.show()


##################################################################################
# Main()
##################################################################################
a=36
np.random.seed(a)
random_state = a
#classifier = RandomForestClassifier(random_state=random_state)
classifier = GaussianNB()
# classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=random_state)
# classifier = DecisionTreeClassifier(random_state=random_state)
# classifier = SVC(random_state = random_state,kernel="linear", C=0.025)   # kernel='poly', kernel='sigmoid',   kernel='rbf'  NO!! kernel='precomputed'
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier = MLPClassifier(random_state=random_state, hidden_layer_sizes=(5,), max_iter=900)
# classifier = AdaBoostClassifier(random_state=random_state)
#classifier = myHoeffdingTree()
#classifier = HoeffdingTreeClassifier()      #NO!!!!
# classifier = XGBClassifier(random_state=random_state)
########################################################

##### Concept drift / Streaming / Change detection

# driftDetector = BOCD(config=BOCDConfig())
#driftDetector = CUSUM(config=CUSUMConfig())
# driftDetectorConfig = GeometricMovingAverageConfig()
# driftDetector = GeometricMovingAverage(config=GeometricMovingAverageConfig())
#driftDetector = PageHinkley(config=PageHinkleyConfig())

##### Concept drift / Streaming / Statistical process control
#driftDetector = DDM(config=DDMConfig())
#driftDetector = ECDDWT(config=ECDDWTConfig())
#driftDetector = #EDDM(config=EDDMConfig())
driftDetector = HDDMA(config=HDDMAConfig())
#driftDetector = HDDMW(config=HDDMWConfig())
#driftDetector = RDDM(config=RDDMConfig())

##### Concept drift / Streaming / Window based
#driftDetector = ADWIN(config=ADWINConfig())
#driftDetector = KSWIN(config=KSWINConfig())
#driftDetector = STEPD(config=STEPDConfig())

########################################################

#inputPath = "Benchmarks/"
inputPath = ""
#inputPath = "Nie_wysłany/"
#inputPath = "Gauss/"
#inputPath = "EAAI/"
#inputPath = "Benjamin1/"
#inputPath = "EAAI/"
#fileName = "Abrupt_HP_10_1.arff"
#fileName = "agrawal_drift_1000_525_1_0.0.arff"
#fileName = "Agrawal_0.csv"
#fileName: str = "Abrupt_HP_1_10.arff"
#fileName = "Agrawal_0.2.csv"
#fileName = "agrawal_drift_1000_525_1_0.1.arff"
#fileName = "agrawal_drift_1000_525_1_0.15.arff"
#fileName = "agrawal_drift_1000_525_1_0.2.arff"

#fileName = "Agrawal_drift_1000_525_100_0.0.arff"
#fileName = "Agrawal_drift_1000_525_100_0.1.arff"
#fileName = "Agrawal_drift_1000_525_100_0.15.arff"
#fileName = "Agrawal_drift_1000_525_100_0.2.arff"

#fileName = "agrawal_drift_1000_525_100_0.0.arff"
#fileName = "agrawal_drift_1000_525_100_0.1.arff"
#fileName = "agrawal_drift_1000_525_100_0.15.arff"
#fileName = "agrawal_drift_1000_525_100_0.2.arff"

#fileName = "LED_drift_1000_525_1_0.0.arff"
#fileName = "LED_drift_1000_525_1_0.1.arff"
#fileName = "LED_drift_1000_525_1_0.15.arff"
#fileName = "LED_drift_1000_525_1_0.2.arff"

#fileName = "LED_drift_1000_525_100_0.0.arff"
#fileName = "LED_drift_1000_525_100_0.1.arff"
#fileName = "LED_drift_1000_525_100_0.15.arff"
#fileName = "LED_drift_1000_525_100_0.2.arff"

#fileName = "Sine_drift_1000_525_100_0.0_T.arff"
#fileName = "Sine_drift_1000_525_100_0.1_T.arff"
#fileName = "Sine_drift_1000_525_100_0.15_T.arff"
#fileName = "Sine_drift_1000_525_100_0.2_T.arff"

#fileName = "Sine_drift_1000_525_1_0.0_T.arff"
#fileName = "Sine_drift_1000_525_1_0.1_T.arff"
#fileName = "Sine_drift_1000_525_1_0.15_T.arff"
#fileName = "Sine_drift_1000_525_1_0.2_T.arff"

#fileName ="INSECTS abrupt_balanced.csv"
#fileName = "SEA_drift.arff"
#fileName = "Abrupt_HP_10_1.arff"
#fileName = "Abrupt_HP_10_1_01_noisy.arff"
#fileName = "abrupt_HP_15_3.arff"
#fileName = "Gradual_HP_15_5.arff"
#fileName = "Recurring_HP_15_5.arff"
#fileName = "Incremental_HP_10_5.arff"

#fileName ="phishing.arff"
#fileName ="electricity.arff"
#fileName ="ozone.arff"
#fileName ="arrhythmia.arff"
#fileName ="outdoor.arff"
#fileName ="rialto.arff"
#fileName =("poker-hand.arff")
#fileName = "incremental_HP_20_5.arff"
#fileName = "Recurring_HP_10_1.arff"
#fileName = "Incremental_HP_10_1.arff"
#fileName = "gradual_HP_10_1.arff"
#fileName ="INSECTS abrupt_balanced.csv"
fileName = STREAM_FILE = "Agrawal_gradual_10000_seed17.arff"
#fileName ="INSECTS abrupt_imbalanced.csv"
#fileName ="INSECTS gradual_balanced.csv"
#fileName ="INSECTS gradual_imbalanced.csv"
#fileName ="INSECTS incremental_balanced.csv"
#fileName ="INSECTS incremental_imbalanced.csv"
#fileName = "INSECTS reoccurring_balanced.csv"
#fileName = "INSECTS reoccurring_imbalanced.csv"

#dataFrame = loadData(inputPath + fileName)
dataFrame = loadData(fileName)
RECORDS_IN_CHUNK = math.ceil(len(dataFrame) * 0.05)
RECORDS_IN_CHUNK = 200


NUMBER_OF_CLASSES = len(
    dataFrame[dataFrame.columns[len(dataFrame.columns) - 1]].unique()
)

st = time.time()
(
    accuracyWithoutDrifts,
    accWithoutDrifts,
    mccWithoutDrifts,
    precWithoutDrifts,
    recallWithoutDrifts,
    f1WithoutDrifts,
) = generateAccuracyWithoutDrifts(
    classifier, dataFrame, RECORDS_IN_CHUNK, NUMBER_OF_CLASSES
)
en = time.time()
timeWithoutDrifts = en - st

st = time.time()
(
    accuracyWithDrifts,
    accWithDrifts,
    mccWithDrifts,
    precWithDrifts,
    recallWithDrifts,
    f1WithDrifts,
    drifts,
) = generateAccuracyWithDrifts(
    classifier, driftDetector, dataFrame, RECORDS_IN_CHUNK, NUMBER_OF_CLASSES
)
en = time.time()
timeWithDrifts = en - st

print("-----------------------------------------------------------")
print("Benchmark                 : ", fileName)
print("Number of class           : ", NUMBER_OF_CLASSES)
print("Records in chunk          : ", RECORDS_IN_CHUNK)
print("Classifier name           : ", type(classifier).__name__)
print("Drift detector name       : ", type(driftDetector).__name__)
print("Number of drifts detected : ", len(drifts))
print("Drifts position           :", drifts)
print("-----------------------------------------------------------")
print("     ACC without drifts: ", round(accWithoutDrifts, 3))
print("     MCC without drifts: ", round(mccWithoutDrifts, 3))
print("     PREC without drifts: ", round(precWithoutDrifts, 3))
print("     SENSITIVITY without drifts: ", round(recallWithoutDrifts, 3))
print("     F1 without drifts: ", round(f1WithoutDrifts, 3))
print("     Time:", round(timeWithoutDrifts, 3))
print("-----------------------------------------------------------")
print("     ACC with drifts and retraining classifier: ", round(accWithDrifts, 3))
print("     MCC with drifts and retraining classifier: ", round(mccWithDrifts, 3))
print("     PREC with drifts: ", round(precWithDrifts, 3))
print("     SENSITIVITY with drifts: ", round(recallWithDrifts, 3))
print("     F1 with drifts: ", round(f1WithDrifts, 3))
print("     Time:", round(timeWithDrifts, 3))
print("-----------------------------------------------------------")

DrawChart(accuracyWithoutDrifts, accuracyWithDrifts, drifts)

