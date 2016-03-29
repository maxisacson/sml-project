import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import gaussian_process

from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


def main(argv):

    nhidden = 10
    max_train_epochs = 100

    pd.set_option('display.max_columns', 500)
    dataset = pd.read_csv("../test/mg5pythia8_hp200.root.test2.csv",)

    # Some variables are empy, this avoids pandas complaining
    # Set non-finite (nan, inf, -inf) to zero
    dataset[ ~np.isfinite(dataset) ] = -999.000000
    # Set to NaN
    dataset.replace(-999.000000, np.nan)

    # Normalize dataset to mean 0 var 1 column wize. NaNs are skipped.
    datasetmean = dataset.mean()
    datasetstd = dataset.std()
    selected_sample = ( dataset - datasetmean ) / datasetstd
    
    # Just to print some info about out dataset
    print(selected_sample.head(10))
    # print(selected_sample.describe())

    # Sklearn has a function that will help us with feature selection, SelectKBest.
    # This selects the best features from the data, and allows us to specify how
    # many it selects.

    predictors = ["et(met)"," phi(met)","ntau","nbjet","njet","pt(reco tau1)","eta(reco tau1)","phi(reco tau1)","m(reco tau1)","pt(reco bjet1)","eta(reco bjet1)","phi(reco bjet1)","m(reco bjet1)","pt(reco bjet2)","eta(reco bjet2)","phi(reco bjet2)","m(reco bjet2)","pt(reco bjet3)","eta(reco bjet3)","phi(reco bjet3)","m(reco bjet3)","pt(reco bjet4)","eta(reco bjet4)","phi(reco bjet4)","m(reco bjet4)","pt(reco jet1)","eta(reco jet1)","phi(reco jet1)","m(reco jet1)","pt(reco jet2)","eta(reco jet2)","phi(reco jet2)","m(reco jet2)","pt(reco jet3)","eta(reco jet3)","phi(reco jet3)","m(reco jet3)","pt(reco jet4)","eta(reco jet4)","phi(reco jet4)","m(reco jet4)"]

    # Perform feature selection
    selector = SelectKBest(f_classif, k=5)
    selector.fit(selected_sample[predictors], selected_sample["pt(mc nuH)"])

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)

    train, test = train_test_split( selected_sample,
                                    test_size=0.3,
                                    random_state=1 )

    #picked the best features according to SelectKBest
    predictors_final = ["et(met)",
                        " phi(met)", "pt(reco tau1)", "eta(reco tau1)",
                        "phi(reco tau1)", "pt(reco bjet1)", "pt(reco bjet1)",
                        "eta(reco bjet1)", "phi(reco bjet1)", "m(reco bjet1)",
                        "phi(reco bjet3)", "m(reco bjet3)", "pt(reco bjet4)",
                        "eta(reco bjet4)", "phi(reco bjet4)", "m(reco bjet4)",
                        "pt(reco jet1)", "eta(reco jet1)", "phi(reco jet1)",
                        "m(reco jet1)", "pt(reco jet2)", "eta(reco jet2)",
                        "phi(reco jet2)", "m(reco jet2)", "nbjet"]

    npredictors = len(predictors_final)
    predictions = []
    print( "Using {} predictors, {} hidden layers, training for max {} epochs"
           .format(npredictors, nhidden, max_train_epochs) )


    # The predictors we're using the train the algorithm.
    train_predictors = train[predictors_final]
    # The target we're using to train the algorithm.
    train_target = train["pt(mc nuH)"]

    # reformat the data to use with pybrain
    supervised_data = SupervisedDataSet(npredictors, 1)
    for pred, targ in zip( train_predictors.as_matrix(),
                           train_target.as_matrix()      ):
        supervised_data.addSample(pred, targ)

    # Training the algorithm using the predictors and target.
    net = buildNetwork( npredictors, nhidden, 1, bias=True,
                        hiddenclass=SigmoidLayer, outclass=LinearLayer )
    trainer = BackpropTrainer(net, supervised_data, verbose=True)
    trainer.trainUntilConvergence(maxEpochs=max_train_epochs)

    # We can now make predictions on the test fold
    for x in test[predictors_final].as_matrix():
        predictions.append(net.activate(x))


    # re-normalize to physical data
    selected_sample = datasetstd*selected_sample + datasetmean
    predictions = datasetstd['pt(mc nuH)']*predictions + datasetmean['pt(mc nuH)']
    target = list(datasetstd['pt(mc nuH)']*test["pt(mc nuH)"] + datasetmean['pt(mc nuH)'])

    pred=list(predictions)
    met=list(selected_sample["et(met)"])
    met_phi=list(selected_sample[" phi(met)"])
    taupt=list(selected_sample["pt(reco tau1)"])
    tauphi=list(selected_sample["phi(reco tau1)"])


    # I want to check the resolution of the prediction wrt the truth value resol
    # = (pt_pred - pt_true)/pt_true*100. I also want to compare it with the
    # resolution from the MET (default).

    result_predict=[]
    result_default=[]

    mT_predict=[]
    mT_default=[]

    count_fail =0
    print(len(target))
    print(len(pred))
    for i in range(len(target)):
        porcentage_predict= (target[i] - pred[i])/target[i]*100
        porcentage_met= (target[i] - met[i])/target[i]*100

        result_predict.append(porcentage_predict)
        result_default.append(porcentage_met)

        if taupt[i]>0 and pred[i]>0:
            mt_new=np.sqrt(2*pred[i]*taupt[i]*(1-np.cos(tauphi[i]-met_phi[i])))
            mt_old=np.sqrt(2*met[i]*taupt[i]*(1-np.cos(tauphi[i]-met_phi[i])))

            mT_predict.append(mt_new)
            mT_default.append(mt_old)
        if pred[i]<0:
            count_fail=count_fail+1

    print("% times prediction is <0")
    print(count_fail)

    prediction_list = np.array(list(result_predict))
    defaul_list = np.array(list(result_default))

    mt_prediction_list = np.array(list(mT_predict))
    mt_defaul_list = np.array(list(mT_default))


    print("pT resolution")
    print(stats.describe(prediction_list))
    print(stats.describe(defaul_list))

    fig, axres = plt.subplots(2, sharex=True)
    bins=2000
    axres[0].hist(prediction_list, 2000)
    axres[0].set_title("Preddicted result")
    # axres[0].set_xlabel("resolution (%)")
    axres[0].set_ylabel("Frequency")
    axres[0].axis([-200,200,0,250])
    axres[0].autoscale(axis='y')

    axres[1].hist(defaul_list, 2000)
    axres[1].set_title("default result")
    axres[1].set_xlabel("resolution (%)")
    axres[1].set_ylabel("Frequency")
    axres[1].axis([-200,200,0,250])
    axres[1].autoscale(axis='y')

    fig.savefig("res_{}class".format(nhidden))

    print("mT distribution")
    print(stats.describe(mt_prediction_list))
    print(stats.describe(mt_defaul_list))

    fig, axmet = plt.subplots(2, sharex=True)
    bins=100
    axmet[0].hist(mt_prediction_list, 100)
    axmet[0].set_title("Preddicted result")
    # axmet[0].set_xlabel("mT [GeV]")
    axmet[0].set_ylabel("Frequency")
    axmet[0].axis([0,400,0,50])
    axmet[0].autoscale(axis='y')

    axmet[1].hist(mt_defaul_list, 100)
    axmet[1].set_title("default result")
    axmet[1].set_xlabel("mT [GeV]")
    axmet[1].set_ylabel("Frequency")
    axmet[1].axis([0,400,0,50])
    axmet[1].autoscale(axis='y')
    
    fig.savefig("met_{}class".format(nhidden))

    plt.show()



if __name__=="__main__":
    main(sys.argv[1:])
