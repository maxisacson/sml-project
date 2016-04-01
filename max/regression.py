#!/usr/bin/env python3

import sys
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcess
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, f_regression

pd.set_option('display.max_columns', 500)


def getData(datafile):
    ds = pd.read_csv(datafile)
    return ds


def main(argv):
    datafile = "../test/mg5pythia8_hp200.root.test3.csv"
    try:
        datafile = argv[1]
    except IndexError:
        pass

    ds = getData(datafile)
    ds.fillna(-999.)

    predictors = sp.array([
        "et(met)", "phi(met)",
        #"nbjet", "njet",
        "pt(reco tau1)", "eta(reco tau1)", "phi(reco tau1)", "m(reco tau1)",
        "pt(reco bjet1)", "eta(reco bjet1)", "phi(reco bjet1)", "m(reco bjet1)",
        "pt(reco jet1)", "eta(reco jet1)", "phi(reco jet1)", "m(reco jet1)",
        "pt(reco jet2)", "eta(reco jet2)", "phi(reco jet2)", "m(reco jet2)",
    ], dtype=str)

    target = "pt(mc nuH)"
    #target = "m(true h+)"

    ds["m(true h+)"] = sp.sqrt(
        2*(ds["pt(mc nuH)"])
         *(ds["pt(mc tau)"])
         *(sp.cosh(ds["eta(mc nuH)"] - ds["eta(mc tau)"])
           - sp.cos(ds["phi(mc nuH)"] - ds["phi(mc tau)"])
           )
        )

    selector = SelectKBest(score_func=f_regression, k=5)
    selector.fit(ds[predictors], ds[target])
    ind = selector.get_support(indices=True)
    final_predictors = predictors[ind]
    print(final_predictors)

    ds = ds[:1000]
    folds = KFold(ds.shape[0], n_folds=3, random_state=123)

    models = {}
    models["gp"] = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, nugget=1e-1)
    models["krr"] = KernelRidge(kernel='linear')
    models["svr"] = SVR(kernel='rbf', gamma=0.001, C=1e5)
    models["nusvr"] = NuSVR(kernel='linear')
    models["linearsvr"] = LinearSVR(C=1e1, loss='epsilon_insensitive',
                                    max_iter=1e4, verbose=True, tol=1e-1)

    model = models["gp"]

    predictions = []
    try:
        for train, test in folds:
            training_sample = ds[final_predictors].iloc[train, :]

            target_sample = ds[target].iloc[train]

            testing_sample = ds[final_predictors].iloc[test, :]

            model.fit(training_sample, target_sample)
            pred = model.predict(testing_sample)
            predictions.append(pred)

    except MemoryError as e:
        print("MemoryError")
        type_, value_, traceback_ = sys.exc_info()
        traceback.print_tb(traceback_)
        sys.exit(1)
    except Exception as e:
        print("Unexpected exception")
        print(e)
        type_, value_, traceback_ = sys.exc_info()
        traceback.print_tb(traceback_)
        sys.exit(2)

    predictions = sp.concatenate(predictions, axis=0)
    t = sp.array(ds[target], dtype=float)
    met = sp.array(ds["et(met)"], dtype=float)
    resolution = (predictions - t)*100/t
    resolution_met = (met - t)*100/t

    print("")
    print("Prediction resolution: mean (sigma): {} ({})"
          .format(resolution.mean(),
                  resolution.std()))

    print("MET resolution: mean (sigma): {} ({})"
          .format(resolution_met.mean(),
                  resolution_met.std()))

    bins = sp.linspace(0, 400, 50)

    plt.subplot(2, 3, 1)
    plt.hist(t, bins, facecolor='blue', label='Obs', alpha=0.5, normed=1,
             histtype='stepfilled')
    plt.hist(predictions, bins, facecolor='orange', label='Pred',
        alpha=0.5, normed=1, histtype='stepfilled')
    plt.hist(met, bins, edgecolor='red', label='MET', alpha=0.5, normed=1,
             histtype='step', linewidth=2)
    plt.xlabel(target)
    plt.legend(loc='best')

    plt.subplot(2, 3, 2)
    plt.hist(resolution, sp.linspace(-100, 100, 50),
             facecolor='green', label='Res', alpha=0.5, histtype='stepfilled')
    plt.xlabel('Resolution')
    plt.legend(loc='best')

    plt.subplot(2, 3, 4)
    hist2d_pred, x_edges, y_edges = sp.histogram2d(t, predictions, bins=bins)
    plt.pcolor(hist2d_pred)
    plt.xlabel(target)
    plt.ylabel('Prediction')

    plt.subplot(2, 3, 5)
    plt.scatter(t, predictions)
    plt.xlabel(target)
    plt.ylabel('Prediction')

    plt.subplot(2, 3, 3)
    plt.hist(resolution_met, sp.linspace(-100, 100, 50),
             facecolor='green', label='Res (MET)', alpha=0.5, histtype='stepfilled')
    plt.xlabel('Resolution')
    plt.legend(loc='best')

    plt.subplot(2, 3, 6)
    plt.scatter(t, met)
    plt.xlabel(target)
    plt.ylabel('MET')

    plt.show()

if __name__ == "__main__":
    sys.exit(main(sys.argv))
