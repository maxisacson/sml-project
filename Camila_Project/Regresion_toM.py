import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.mlab as mlab

from sklearn import linear_model

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import cross_val_score

from sklearn import gaussian_process

import matplotlib.pyplot as plt
import math

pd.set_option('display.max_columns', 500)
titanic = pd.read_csv("mg5pythia8_hp200.root.test2.csv")
selected_sample = titanic

selected_sample["mass_truth"] = np.sqrt(2 * (selected_sample["pt(mc nuH)"]) * (selected_sample["pt(mc tau)"])
                                        * (np.cosh(
    selected_sample["eta(mc nuH)"] - selected_sample["eta(mc tau)"]) - np.cos(
    selected_sample["phi(mc nuH)"] - selected_sample["phi(mc tau)"])))

##some variables are empy, this avoids pandas complaining
selected_sample[~np.isfinite(selected_sample)] = -999.000000  # Set non-finite (nan, inf, -inf) to zero

predictors_final = ["et(met)", " phi(met)", "pt(reco tau1)", "eta(reco tau1)", "phi(reco tau1)", "pt(reco bjet1)",
                    "pt(reco bjet1)", "eta(reco bjet1)", "phi(reco bjet1)", "m(reco bjet1)", "phi(reco bjet3)",
                    "m(reco bjet3)", "pt(reco bjet4)", "eta(reco bjet4)", "phi(reco bjet4)", "m(reco bjet4)",
                    "pt(reco jet1)", "eta(reco jet1)", "phi(reco jet1)", "m(reco jet1)", "pt(reco jet2)",
                    "eta(reco jet2)", "phi(reco jet2)", "m(reco jet2)", "nbjet"]

# Initialize our algorithm class
alg = linear_model.BayesianRidge()
# alg = linear_model.Ridge(alpha = 10)# Generate cross validation folds for the  dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(selected_sample.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (selected_sample[predictors_final].iloc[train, :])
    # The target we're using to train the algorithm.
    train_target = selected_sample["mass_truth"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(selected_sample[predictors_final].iloc[test, :])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
target = list(selected_sample["mass_truth"])
pred = list(predictions)
met = list(selected_sample["et(met)"])
met_phi = list(selected_sample[" phi(met)"])
taupt = list(selected_sample["pt(reco tau1)"])
tauphi = list(selected_sample["phi(reco tau1)"])

## I want to check the resolution of the prediction wrt the truth value resol = (pt_pred - pt_true)/pt_true*100. I also want to compare it with the resolution from the MET (default).

result_predict = []

mT_predict = []
mT_default = []

pt_predict = []
pt_true = []

count_fail = 0
for i in range(len(target)):
    porcentage_predict = (target[i] - pred[i]) / target[i] * 100

    result_predict.append(porcentage_predict)

    if taupt[i] > 0 and pred[i] > 0:
        mt_new = target[i]
        mt_old = math.sqrt(2 * met[i] * taupt[i] * (1 - math.cos(tauphi[i] - met_phi[i])))

        mT_predict.append(mt_new)
        mT_default.append(mt_old)
    if pred[i] < 0:
        count_fail += 1

print("% times prediction is <0")
print(count_fail)

prediction_list = np.array(list(result_predict))
default_list = np.array(list(target))

mt_prediction_list = np.array(list(mT_predict))
mt_defaul_list = np.array(list(mT_default))

print("M resolution")
print(stats.describe(prediction_list))

bins = 200
plt.hist(prediction_list, bins)
plt.title("Preddicted result")
plt.xlabel("resolution (%)")
plt.ylabel("Frequency")
plt.axis([-10, 10, 0, 8000])
plt.show()

print("M and mT distribution")
print(stats.describe(mt_prediction_list))
print(stats.describe(default_list))

bins = 100
plt.hist(mt_prediction_list, bins)
plt.title("Preddicted result")
plt.xlabel("M [GeV]")
plt.ylabel("Frequency")
plt.axis([190, 210, 0, 3000])
plt.show()

plt.hist(default_list, bins=100)
plt.title("default result")
plt.xlabel("m truth [GeV]")
plt.ylabel("Frequency")
plt.axis([190, 210, 0, 8000])
plt.show()
