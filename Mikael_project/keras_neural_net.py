import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool

from scipy import stats
from scipy.stats import norm

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import History

# To run on GPU, prepend set the THEANO_FLAGS:
# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python keras_neural_net.py

# To-do
# Train on m truth


class NeuralNet:

    def __init__( self, traindata, testdata, targets, predictors, hidden_nodes,
                  hidden_activations ):
        self.train=pd.DataFrame(traindata)
        self.test=pd.DataFrame(testdata)
        if not all( t in traindata for t in targets):
            raise ValueError( "Some training targets variable in '{}'"
                              " does not exist.".format(targets) )
        if not all( t in testdata for t in targets):
            raise ValueError( "Some test targets variable in '{}'"
                              " does not exist.".format(targets) )
        self.targets=targets
        self.ntargets=len(targets)
        if not all( p in traindata for p in predictors ):
            raise ValueError( "Some training predictor variable in '{}'"
                              " does not exist.".format(predictors) )
        if not all( p in testdata for p in predictors ):
            raise ValueError( "Some test predictor variable in '{}'"
                              " does not exist.".format(predictors) )
        self.predictors=predictors
        self.npredictors=len(predictors)
        if not len(hidden_nodes)==len(hidden_activations):
            raise ValueError( "You must specify 1 activation function per hidden"
                              " layer" )
        self.layers=hidden_nodes
        self.activations=hidden_activations
        self.model = Sequential()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def compile(self):
        self.model.add( Dense( output_dim = self.layers[0],
                               input_dim  = self.npredictors,
                               init       = 'glorot_uniform'  ) )
        self.model.add( Activation(self.activations[0]) )
        for i in range(len(self.layers) - 1):
            self.model.add( Dense( output_dim = self.layers[i+1],
                                   input_dim  = self.layers[i],
                                   init       = 'glorot_uniform'  ) )
            self.model.add( Activation(self.activations[i+1]) )
        self.model.add( Dense(output_dim=self.ntargets, init='glorot_uniform') )
        self.model.add( Activation("linear") )
        self.model.compile(loss='mse', optimizer='sgd')

    def scale(self, method='mean_std', fillnan='nan'):
        self.scale_method=method
        self.trainmean = self.train.mean()
        self.trainstd = self.train.std()
        self.trainmin = self.train.min()
        self.trainmax = self.train.max()
        self.train1q = self.train.quantile(q=0.01)
        self.train99q = self.train.quantile(q=0.99)
        if method=='min_max':
            # Normalize dataset to min 0 and max 1. NaNs are skipped
            self.train = ( 2*( self.train - self.trainmin )
                               / ( self.trainmax - self.trainmin) - 1 )
            self.test = ( 2*( self.test - self.trainmin )
                               / ( self.trainmax - self.trainmin) - 1 )
        elif method=='mean_std':
            # Normalize dataset to mean 0 var 1 column wize. NaNs are skipped.
            self.train = ( self.train - self.trainmean ) / self.trainstd
            self.test = ( self.test - self.trainmean ) / self.trainstd
        elif method=='quantile':
            # Normalize dataset so that q(1%)=-1, q(99%)=1. NaNs are skipped.
            self.train = ( 2*( self.train - self.train1q )
                               / ( self.train99q - self.train1q) - 1 )
            self.test = ( 2*( self.test - self.train1q )
                               / ( self.train99q - self.train1q) - 1 )
        else:
            raise ValueError("Unrecognized scaling option '{}'".format(method))
        if fillnan=='zero':
            # replace NaN with 0.0
            self.train = self.train.fillna(0.0)
            self.test = self.test.fillna(0.0)
        elif fillnan=='nan':
            pass
        else:
            raise ValueError( "Unrecognized NaN fill option '{}'"
                              .format(fillnan) )
        
    def unscale(self):
        if self.scale_method=='mean_std':
            self.train = self.train*self.trainstd + self.trainmean
            self.test = self.test*self.trainstd + self.trainmean
            # self.predictions = ( self.predictions
            #                      * np.std(self.predictions)
            #                      + np.mean(self.predictions) )
            self.predictions = ( self.predictions
                                 * self.trainstd[self.targets].as_matrix()
                                 + self.trainmean[self.targets].as_matrix() )
        elif self.scale_method=='min_max':
            self.train = ( 0.5*( self.train + 1 )
                           * ( self.trainmax - self.trainmin )
                           + self.trainmin )
            self.test = ( 0.5*( self.test + 1 )
                          * ( self.trainmax - self.trainmin )
                          + self.trainmin )
            self.predictions = ( 0.5*( self.predictions + 1 )
                                 * ( self.trainmax[self.targets].as_matrix()
                                     - self.trainmin[self.targets].as_matrix() )
                                 + self.trainmin[self.targets].as_matrix() )
        elif self.scale_method=='quantile':
            self.train = ( 0.5*( self.train + 1 )
                           * ( self.train99q - self.train1q )
                           + self.train1q )
            self.test = ( 0.5*( self.test + 1 )
                          * ( self.train99q - self.train1q )
                          + self.train1q )
            self.predictions = ( 0.5*( self.predictions + 1 )
                                 * ( self.train99q[self.targets].as_matrix()
                                     - self.train1q[self.targets].as_matrix() )
                                 + self.train1q[self.targets].as_matrix() )

    def fit(self, n_epochs=100):
        history = self.model.fit( self.train[self.predictors].as_matrix(),
                                  self.train[self.targets].as_matrix(),
                                  nb_epoch   = n_epochs,
                                  batch_size = 32,
                                  verbose    = 0 )
        return history.history
        
    def predict(self):
        predictions = self.model.predict_proba( self.test[self.predictors]
                                                .as_matrix(),  batch_size=32)
        self.predictions = predictions
        
    # def predict(self, data, scale_data=True):
    #     predictions = self.model.predict_proba( data[self.predictors]
    #                                             .as_matrix(),  batch_size=32)
    #     return predictions
            
def main(argv):

    train_args = [ {'layers':[25,15], 'activation':['sigmoid','sigmoid']}]
    # train_args = [ {'layers':[100,70,50], 'activation':['sigmoid','sigmoid','sigmoid']}]
    # train_args = [ {'layers':[100,50,25],
    #                 'activation':['sigmoid','sigmoid','sigmoid']}]
    # train_args = [ {'layers':[25,20], 'activation':['sigmoid','sigmoid']},
    #                {'layers':[15,10], 'activation':['sigmoid','sigmoid']} ]

    train_var = ["pt(mc nuH)"]
    scale_method = 'min_max' # 'min_max', 'mean_std', or 'quantile'
    n_epochs = 5000

    model_file = "nn-weights-{}-{}-{}-{}.h5".format( train_args[0]['layers'],
                                                     train_args[0]['activation'],
                                                     scale_method,
                                                     n_epochs)
    plot_file = "nn-plot-{}-{}-{}-{}.pdf".format( train_args[0]['layers'],
                                                  train_args[0]['activation'],
                                                  scale_method,
                                                  n_epochs)
    
    pd.set_option('display.max_columns', 500)
    dataset = pd.read_csv("../test/mg5pythia8_hp200.root.test3.csv", nrows=10000)
    # Add truth mass of H+ to dataset
    dataset["mass_truth"] = ( np.sqrt(2*(dataset["pt(mc nuH)"])
                                      *(dataset["pt(mc tau)"])
                                      * ( np.cosh(dataset["eta(mc nuH)"]
                                                  - dataset["eta(mc tau)"])
                                          - np.cos(dataset["phi(mc nuH)"]
                                                   - dataset["phi(mc tau)"]))) )

    # Replace invalid values with NaN
    dataset = dataset.where(dataset > -998.0, other=np.nan)
    # Predictor variables as determined by SciKit Learn's selectKBest
    predictors_final = ["et(met)",
                        "phi(met)", "pt(reco tau1)", "eta(reco tau1)",
                        "phi(reco tau1)", "pt(reco bjet1)", "pt(reco bjet1)",
                        "eta(reco bjet1)", "phi(reco bjet1)", "m(reco bjet1)",
                        "phi(reco bjet3)", "m(reco bjet3)", "pt(reco bjet4)",
                        "eta(reco bjet4)", "phi(reco bjet4)", "m(reco bjet4)",
                        "pt(reco jet1)", "eta(reco jet1)", "phi(reco jet1)",
                        "m(reco jet1)", "pt(reco jet2)", "eta(reco jet2)",
                        "phi(reco jet2)", "m(reco jet2)", "nbjet"]
    # Split dataset into a training and a testing (validation) set
    train, test = train_test_split( dataset,
                                    test_size=0.3,
                                    random_state=1 )
    # Initialize the neural net
    nn = NeuralNet( traindata          = train,
                    testdata           = test,
                    targets            = train_var,
                    predictors         = predictors_final,
                    hidden_nodes       = train_args[0]['layers'],
                    hidden_activations = train_args[0]['activation'] )
    nn.scale(method=scale_method,fillnan='zero')
    nn.compile()
    # Load weights from file if the training already has been done
    if os.path.isfile(model_file):
        print("Loading model weights from file {}".format(model_file))
        nn.load_weights(model_file)
    else:
        history = nn.fit(n_epochs)
        print(history['loss'][-1])
        print("Saving model weights to file {}".format(model_file))
        nn.save_weights(model_file)
    # Predict 'train_var' and rescale to its physical value.
    nn.predict() 
    nn.unscale()
    # multipool = Pool(processes=3)
    # result = multipool.map(train_net, train_args)

    pred    = np.array(nn.predictions)
    target  = np.array(nn.test[train_var].as_matrix())
    met     = np.array(nn.test["et(met)"].as_matrix().reshape(-1,1))
    met_phi = np.array(nn.test["phi(met)"].as_matrix())
    # taupt   = np.array(nn.test["pt(reco tau1)"].as_matrix())
    # tauphi  = np.array(nn.test["phi(reco tau1)"].as_matrix())
    resolution_predict_tmp = (target - pred)/target*100
    resolution_predict = [ item for sublist in resolution_predict_tmp
                           for item in sublist if abs(item) < 200 ]
    resolution_default_tmp = (target - met)/target*100
    resolution_default = [ item for sublist in resolution_default_tmp
                           for item in sublist if abs(item) < 200]
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    params = { 'figure.facecolor': 'white',
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
               'axes.prop_cycle': plt.cycler('color',colors)}
    plt.rcParams.update(params)
    
    fig, ax = plt.subplots(2)
    ax[0].hist(pred, 50, label='Pred', histtype='step')
    ax[0].hist(met, 50, label='Obs', histtype='step')
    ax[0].hist(nn.test['pt(mc nuH)'], 50, label='Truth', histtype='step')
    ax[0].set_xlim([0,400])
    ax[0].set_xlabel("pt (GeV)")
    ax[0].legend(loc='upper right')

    ax[1].hist(resolution_predict, 100, label='Pred', histtype='step')
    ax[1].hist(resolution_default, 100, label='Obs', histtype='step')
    ax[1].set_xlim([-100,100])
    ax[1].set_xlabel("Resolution (%)")
    ax[1].legend(loc='upper left')

    fig.tight_layout(pad=0.3)
    fig.savefig(plot_file)
    plt.show()



    
# def main(argv):
#     # train_args = [ {'layers':[25,20], 'activation':['sigmoid','sigmoid']},
#     #                {'layers':[25,20], 'activation':['tanh','tanh']},
#     #                {'layers':[25,20], 'activation':['softmax','softmax']},
#     #                {'layers':[25,20], 'activation':['relu','relu']} ]
#     # train_args = [ {'layers':[25,10], 'activation':['sigmoid','sigmoid']},
#     #                {'layers':[25,10], 'activation':['tanh','tanh']},
#     #                {'layers':[25,10], 'activation':['softmax','softmax']},
#     #                {'layers':[25,10], 'activation':['relu','relu']},
#     #                {'layers':[25,5], 'activation':['sigmoid','sigmoid']},
#     #                {'layers':[25,5], 'activation':['tanh','tanh']},
#     #                {'layers':[25,5], 'activation':['softmax','softmax']},
#     #                {'layers':[25,5], 'activation':['relu','relu']} ]
#     # train_args = [ {'layers':[15,20], 'activation':['sigmoid','sigmoid']},
#     #                {'layers':[15,10], 'activation':['sigmoid','sigmoid']},
#     #                {'layers':[15,5], 'activation':['sigmoid','sigmoid']},
#     #                {'layers':[15,5], 'activation':['sigmoid','sigmoid']},
#     #                {'layers':[10,15], 'activation':['sigmoid','sigmoid']},
#     #                {'layers':[10,10], 'activation':['sigmoid','sigmoid']},
#     #                {'layers':[10,5], 'activation':['sigmoid','sigmoid']},
#     #                {'layers':[5,5], 'activation':['sigmoid','sigmoid']}]
#     train_args = [ {'layers':[15,10], 'activation':['sigmoid','sigmoid']}]
#     # train_args = [ {'layers':[25,20], 'activation':['sigmoid','sigmoid']},
#     #                {'layers':[15,10], 'activation':['sigmoid','sigmoid']} ]


    

#     # re-normalize to physical data
#     dataset.unscale()
#     predictions = ( dataset.dstd[train_var]*(predictions - 0.0)
#                     + dataset.dmean[train_var] )
#     target  = list( dataset.dstd[train_var]*(test[train_var] - 0.0)
#                     + dataset.dmean[train_var] )
#     pred    = list(predictions)
#     met     = list( dataset.dstd["et(met)"]*( test["et(met)"] - 0.0 )
#                     + dataset.dmean["et(met)"] )
#     met_phi = list( dataset.dstd["phi(met)"]*( test["phi(met)"] - 0.0 )
#                     + dataset.dmean["phi(met)"] )
#     taupt   = list( dataset.dstd["pt(reco tau1)"]*( test["pt(reco tau1)"] - 0.0 )
#                    + dataset.dmean["pt(reco tau1)"] )
#     tauphi  = list( dataset.dstd["phi(reco tau1)"]*( test["phi(reco tau1)"]
#                                                    - 0.0 )
#                     + dataset.dmean["phi(reco tau1)"] )
    
#     # I want to check the resolution of the prediction wrt the truth value resol
#     # = (pt_pred - pt_true)/pt_true*100. I also want to compare it with the
#     # resolution from the MET (default).

#     result_predict=[]
#     result_default=[]

#     mT_predict=[]
#     mT_default=[]

#     count_fail = 0

#     for i in range(len(target)):
#         porcentage_predict= (target[i] - pred[i])/target[i]*100
#         porcentage_met= (target[i] - met[i])/target[i]*100

#         result_predict.append(porcentage_predict)
#         result_default.append(porcentage_met)

#         if taupt[i]>0 and pred[i]>0:
#             mt_new=np.sqrt(2*pred[i]*taupt[i]*(1-np.cos(tauphi[i]-met_phi[i])))
#             mt_old=np.sqrt(2*met[i]*taupt[i]*(1-np.cos(tauphi[i]-met_phi[i])))

#             mT_predict.append(mt_new)
#             mT_default.append(mt_old)
#         if pred[i]<0:
#             count_fail=count_fail+1

#     prediction_list = np.array(list(result_predict))
#     defaul_list = np.array(list(result_default))

#     mt_prediction_list = np.array(list(mT_predict))
#     mt_defaul_list = np.array(list(mT_default))

#     return { 'layer_conf'         : [npredictors] + args['layers'] + [1],
#              'prediction_list'    : prediction_list,
#              'defaul_list'        : defaul_list,
#              'mt_prediction_list' : mt_prediction_list,
#              'mt_defaul_list'     : mt_defaul_list,
#              'objective_score'    : objective_score }

if __name__=="__main__":
    main(sys.argv[1:])
