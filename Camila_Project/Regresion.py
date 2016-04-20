import pandas as pd
import numpy as np
from scipy import stats

from scipy.stats import norm
import matplotlib.mlab as mlab

import scipy as sp

from sklearn import linear_model
from sklearn import neighbors

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import cross_val_score

from sklearn import gaussian_process

import matplotlib.pyplot as plt
import math

pd.set_option('display.max_columns', 500)
titanic1 = pd.read_csv("../test/mg5pythia8_hp200.root.test3.csv")
titanic2 = pd.read_csv("../test/mg5pythia8_hp300.root.test.csv")
titanic3 = pd.read_csv("../test/mg5pythia8_hp400.root.test.csv")

selected_sample = titanic1#@+titanic2+titanic3

##some variables are empy, this avoids pandas complaining
selected_sample[~np.isfinite(selected_sample)] = -999.000000  # Set non-finite (nan, inf, -inf) to zero

selected_sample["truth_met"] = np.sqrt(selected_sample['pt(mc nuH)'] ** 2
                                       + selected_sample['pt(mc nuTau)'] ** 2
                                       + 2 * selected_sample['pt(mc nuH)'] * selected_sample['pt(mc nuTau)']
                                       * np.cos(selected_sample['phi(mc nuH)']
                                                - selected_sample['phi(mc nuTau)']))

##just to print some info about out dataset
print(selected_sample.head(10))
print(selected_sample.describe())

##Sklearn has a function that will help us with feature selection, SelectKBest. This selects the best features from the data, and allows us to specify how many it selects.

predictors = ["et(met)", "phi(met)", "ntau", "nbjet", "njet", "pt(reco tau1)", "eta(reco tau1)", "phi(reco tau1)",
              "m(reco tau1)", "pt(reco bjet1)", "eta(reco bjet1)", "phi(reco bjet1)", "m(reco bjet1)", "pt(reco bjet2)",
              "eta(reco bjet2)", "phi(reco bjet2)", "m(reco bjet2)", "pt(reco bjet3)", "eta(reco bjet3)",
              "phi(reco bjet3)", "m(reco bjet3)", "pt(reco bjet4)", "eta(reco bjet4)", "phi(reco bjet4)",
              "m(reco bjet4)", "pt(reco jet1)", "eta(reco jet1)", "phi(reco jet1)", "m(reco jet1)", "pt(reco jet2)",
              "eta(reco jet2)", "phi(reco jet2)", "m(reco jet2)", "pt(reco jet3)", "eta(reco jet3)", "phi(reco jet3)",
              "m(reco jet3)", "pt(reco jet4)", "eta(reco jet4)", "phi(reco jet4)", "m(reco jet4)"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(selected_sample[predictors], selected_sample["pt(mc nuH)"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.
# plt.bar(range(len(predictors)), scores)
# plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()

# picked the best features according to SelectKBest
predictors_final = ["et(met)", "phi(met)", "pt(reco tau1)", "eta(reco tau1)", "phi(reco tau1)", "pt(reco bjet1)",
                     "eta(reco bjet1)", "phi(reco bjet1)", "m(reco bjet1)"]#, "pt(reco jet1)",
                    #"eta(reco jet1)", "phi(reco jet1)", "m(reco jet1)", "pt(reco jet2)", "eta(reco jet2)",
                    #"phi(reco jet2)", "m(reco jet2)", "nbjet"]

# Initialize our algorithm class
#alg = linear_model.BayesianRidge(n_iter=2000)#, tol=10.1, alpha_1=1e-02, alpha_2=1e-02, lambda_1=1e-02, lambda_2=1e-02, compute_score=True, fit_intercept=True, normalize=True, copy_X=True, verbose=False)
alg = neighbors.KNeighborsRegressor(n_neighbors=1, leaf_size=60)

kf = KFold(selected_sample.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (selected_sample[predictors_final].iloc[train, :])
    # The target we're using to train the algorithm.
    train_target = selected_sample["pt(mc nuH)"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(selected_sample[predictors_final].iloc[test, :])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
target = list(selected_sample["pt(mc nuH)"])
pred = list(predictions)
met = list(selected_sample["et(met)"])
met_phi = list(selected_sample["phi(met)"])
taupt = list(selected_sample["pt(reco tau1)"])
tauphi = list(selected_sample["phi(reco tau1)"])
met_truth = list(selected_sample["truth_met"])

## I want to check the resolution of the prediction wrt the truth value resol = (pt_pred - pt_true)/pt_true*100. I also want to compare it with the resolution from the MET (default).

result_predict = []
result_default = []
result_truthmet = []

mT_predict = []
mT_default = []

pt_predict = []
pt_true = []

ratio = []

count_fail = 0
for i in range(len(target)):
    porcentage_predict = (target[i] - pred[i]) / target[i] * 100
    porcentage_met = (target[i] - met[i]) / target[i] * 100
    resol_truthmet = (met_truth[i] - met[i]) / met_truth[i] * 100


    result_predict.append(porcentage_predict)
    result_default.append(porcentage_met)
    result_truthmet.append(resol_truthmet)

    pt_predict.append(pred[i])
    pt_true.append(target[i])

    if taupt[i] > 0 and pred[i] > 0:
        mt_new = math.sqrt(2 * pred[i] * taupt[i] * (1 - math.cos(tauphi[i] - met_phi[i])))
        mt_old = math.sqrt(2 * met[i] * taupt[i] * (1 - math.cos(tauphi[i] - met_phi[i])))

        mT_predict.append(mt_new)
        mT_default.append(mt_old)
    if pred[i] < 0:
        count_fail = count_fail + 1

print("% times prediction is <0")
print(count_fail)

prediction_list = np.array(list(result_predict))
defaul_list = np.array(list(result_default))
met_truth_list = np.array(list(result_truthmet))

mt_prediction_list = np.array(list(mT_predict))
mt_defaul_list = np.array(list(mT_default))

pt_prediction_list = np.array(list(pt_predict))
pt_true_list = np.array(list(pt_true))

selected_sample["predicted_pTnu"] = predictions

selected_sample["resolution_pred_pT"] = prediction_list
selected_sample["resolution_default_pT"] = defaul_list


print(selected_sample.head(10))

selected_sample.to_csv('output_KNeighbours_200GeV.csv')


print("pT resolution")
print(stats.describe(prediction_list))
print(stats.describe(defaul_list))

print("pT ")
print(stats.describe(pt_prediction_list))
print(stats.describe(pt_true_list))

print("mT distribution")
print(stats.describe(mt_prediction_list))
print(stats.describe(mt_defaul_list))

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

params = {'figure.facecolor': 'white',
          'figure.subplot.bottom': 0.0,
          'font.size': 16,
          'legend.fontsize': 16,
          'legend.borderpad': 0.2,
          'legend.labelspacing': 0.2,
          'legend.handlelength': 1.5,
          'legend.handletextpad': 0.4,
          'legend.borderaxespad': 0.2,
          'lines.markeredgewidth': 2.0,
          'lines.linewidth': 2.0,
          'axes.prop_cycle': plt.cycler('color', colors)}
plt.rcParams.update(params)


fig, ax = plt.subplots(3)
ax[0].hist(predictions, 50, range=(0.0, 400.0), label='Pred', histtype='step')
ax[0].hist(target, 50, range=(0.0, 400.0), label='Truth', histtype='step')
ax[0].set_xlim([0, 400])
ax[0].set_xlabel("pT (GeV)")
ax[0].legend(loc='upper right')
ax[1].hist(result_predict, 50, range=(-100.0, 100.0), label='Pred', histtype='step')
ax[1].hist(result_default, 50, range=(-100.0, 100.0), label='Default', histtype='step')
ax[1].hist(met_truth_list, 50, range=(-100.0, 100.0), label='truth_met - met/truth_met', histtype='step')
ax[1].set_xlim([-100, 100])
ax[1].set_xlabel("Resolution (%)")
ax[1].legend(loc='upper left')
ax[2].hist(mT_predict, 50, range=(0.0, 400.0), label='Pred', histtype='step')
ax[2].hist(mT_default, 50, range=(0.0, 400.0), label='Default', histtype='step')
ax[2].set_xlim([0, 400])
ax[2].set_xlabel("mT [GeV]")
ax[2].legend(loc='upper left')
#fig.tight_layout(pad=1.3)
fig.savefig('multi_{}_{}_{}.pdf'.format("1",
                                        "2",
                                        "3"))
plt.show()
