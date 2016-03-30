from keras.models import Sequential
from keras.layers.core import Dense, Activation

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool

from scipy import stats
from scipy.stats import norm

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import gaussian_process


def main(argv):
    hidden = [[20,5],[35,10],[50,20]]
    
    multipool = Pool(processes=3)
    result = multipool.map(train_net, hidden)

    figres, axres = plt.subplots( len(result)+1, sharex=True )
    figmet, axmet = plt.subplots( len(result)+1, sharex=True )

    # print(result['prediction_list'])
    for i in range(len(result)):
        axres[i].hist(result[i]['prediction_list'], 2000)
        axres[i].set_title( "Preddicted result {}"
                            .format(result[i]['layer_conf']) )
        # axres[i].set_xlabel("resolution (%)")
        axres[i].set_ylabel("Frequency")
        axres[i].axis([-200,200,0,250])
        axres[i].autoscale(axis='y')

        axmet[i].hist(result[i]['mt_prediction_list'], 100)
        axmet[i].set_title( "Preddicted result {}"
                            .format(result[i]['layer_conf']) )
        # axmet[i].set_xlabel("mT [GeV]")
        axmet[i].set_ylabel("Frequency")
        axmet[i].axis([0,400,0,50])
        axmet[i].autoscale(axis='y')
        
    axres[-1].hist(result[0]['defaul_list'], 2000)
    axres[-1].set_title("default result")
    axres[-1].set_xlabel("resolution (%)")
    axres[-1].set_ylabel("Frequency")
    axres[-1].axis([-200,200,0,250])
    axres[-1].autoscale(axis='y')

    axmet[-1].hist(result[0]['mt_defaul_list'], 100)
    axmet[-1].set_title("default result")
    axmet[-1].set_xlabel("mT [GeV]")
    axmet[-1].set_ylabel("Frequency")
    axmet[-1].axis([0,400,0,50])
    axmet[-1].autoscale(axis='y')

    figres.savefig("res.pdf")
    figmet.savefig("met.pdf")
    plt.show()
    
def train_net(nn_layers, max_train_epochs=200):

    nhidden = 100
    # train_var = "mass_truth"
    train_var = "pt(mc nuH)"
    
    pd.set_option('display.max_columns', 500)
    dataset = pd.read_csv("../test/mg5pythia8_hp200.root.test2.csv",)
    dataset["mass_truth"] = ( np.sqrt(2*(dataset["pt(mc nuH)"])
                                      *(dataset["pt(mc tau)"])
                                      * ( np.cosh(dataset["eta(mc nuH)"]
                                                  - dataset["eta(mc tau)"])
                                          - np.cos(dataset["phi(mc nuH)"]
                                                   - dataset["phi(mc tau)"]))) )

    # Some variables are empy, this avoids pandas complaining
    # Set non-finite (nan, inf, -inf) to zero
    dataset[ ~np.isfinite(dataset) ] = -999.000000
    # Set to NaN
    dataset.replace(-999.000000, np.nan)

    # Normalize dataset to mean 0 var 1 column wize. NaNs are skipped.
    datasetmean = dataset.mean()
    datasetstd = dataset.std()
    selected_sample = ( dataset - datasetmean ) / datasetstd + 0.5
    
    # Just to print some info about out dataset
    # print(selected_sample.head(10))
    # print(selected_sample.describe())

    # Sklearn has a function that will help us with feature selection, SelectKBest.
    # This selects the best features from the data, and allows us to specify how
    # many it selects.

    predictors = ["et(met)"," phi(met)","ntau","nbjet","njet","pt(reco tau1)","eta(reco tau1)","phi(reco tau1)","m(reco tau1)","pt(reco bjet1)","eta(reco bjet1)","phi(reco bjet1)","m(reco bjet1)","pt(reco bjet2)","eta(reco bjet2)","phi(reco bjet2)","m(reco bjet2)","pt(reco bjet3)","eta(reco bjet3)","phi(reco bjet3)","m(reco bjet3)","pt(reco bjet4)","eta(reco bjet4)","phi(reco bjet4)","m(reco bjet4)","pt(reco jet1)","eta(reco jet1)","phi(reco jet1)","m(reco jet1)","pt(reco jet2)","eta(reco jet2)","phi(reco jet2)","m(reco jet2)","pt(reco jet3)","eta(reco jet3)","phi(reco jet3)","m(reco jet3)","pt(reco jet4)","eta(reco jet4)","phi(reco jet4)","m(reco jet4)"]

    # Perform feature selection
    selector = SelectKBest(f_classif, k=5)
    selector.fit(selected_sample[predictors], selected_sample[train_var])

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

    # The predictors we're using the train the algorithm.
    train_predictors = train[predictors_final]
    # The target we're using to train the algorithm.
    train_target = train[train_var]

    # The predictors we're using the test the algorithm.
    test_predictors = test[predictors_final]
    # The target we're using to test the algorithm.
    test_target = test[train_var]

    model = Sequential()
    model.add( Dense( output_dim=nn_layers[0],
                      input_dim=npredictors,
                      init='glorot_uniform'  ) )
    model.add( Activation("sigmoid") )
    for i in range(len(nn_layers) - 1):
        model.add( Dense( output_dim=nn_layers[i+1],
                          input_dim=nn_layers[i],
                          init='glorot_uniform'  ) )
        model.add( Activation("sigmoid") )
        # print("Added hidden {} {}".format(nn_layers[i+1],nn_layers[i]))
    model.add( Dense(output_dim=1, init='glorot_uniform') )
    model.add( Activation("linear") )
    model.compile(loss='mse', optimizer='sgd')
    model.fit( train_predictors.as_matrix(), train_target.as_matrix(),
               nb_epoch=max_train_epochs, batch_size=32 )

    objective_score = model.evaluate( test_predictors.as_matrix(),
                                      test_target.as_matrix(),
                                      batch_size=32 )
    print(objective_score)
    
    predictions = model.predict_proba(test_predictors.as_matrix(),
                                      batch_size=32)
    
    # # We can now make predictions on the test fold
    # for x in test[predictors_final].as_matrix():
    #     predictions.append(net.activate(x))

    # re-normalize to physical data
    selected_sample = datasetstd*(selected_sample - 0.5) + datasetmean
    predictions = ( datasetstd[train_var]*(predictions - 0.5)
                    + datasetmean[train_var] )
    target = list( datasetstd[train_var]*(test[train_var] - 0.5)
                   + datasetmean[train_var] )

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

    count_fail = 0

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

    prediction_list = np.array(list(result_predict))
    defaul_list = np.array(list(result_default))

    mt_prediction_list = np.array(list(mT_predict))
    mt_defaul_list = np.array(list(mT_default))

    return { 'layer_conf'         : [npredictors] + nn_layers + [1],
             'train_errs'         : 0,
             'val_errs'           : 0,
             'prediction_list'    : prediction_list,
             'defaul_list'        : defaul_list,
             'mt_prediction_list' : mt_prediction_list,
             'mt_defaul_list'     : mt_defaul_list }

if __name__=="__main__":
    main(sys.argv[1:])
