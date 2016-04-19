import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool

from scipy import stats
from scipy.stats import norm,t

from sklearn.cross_validation import train_test_split,StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import History

# To run on GPU, prepend set the THEANO_FLAGS:
# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python keras_neural_net.py

# To-do
# Train on mass_truth (H+)
# EarlyStopping?

class NeuralNet:

    def __init__( self, traindata, testdata, targets, predictors, hidden_nodes,
                  hidden_activations ):
        self.train=pd.DataFrame(traindata)
        self.test=pd.DataFrame(testdata)
        for t in targets:
            if not t in traindata:
                raise ValueError( "Training set target '{}' does not exist."
                                  .format(t) )
            if not t in testdata:
                raise ValueError( "Test set target '{}' does not exist."
                                  .format(t) )
        self.targets=targets
        self.ntargets=len(targets)
        for p in predictors:
            if not p in traindata:
                raise ValueError( "Training set predictor '{}' does not exist."
                                  .format(p) )
            if not p in testdata:
                raise ValueError( "Test set predictor '{}' does not exist."
                                  .format(p) )
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

    def scale(self, method='min_max', fillnan='nan'):
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
            raise ValueError("Unrecognized NaN fill option '{}'".format(fillnan))
        
    def unscale(self):
        if self.scale_method=='mean_std':
            self.train = self.train*self.trainstd + self.trainmean
            self.test = self.test*self.trainstd + self.trainmean
            for t in self.targets:
                self.predictions = ( self.predictions * self.trainstd[t]
                                     + self.trainmean[t] )

        elif self.scale_method=='min_max':
            self.train = ( 0.5*( self.train + 1 )
                           * ( self.trainmax - self.trainmin )
                           + self.trainmin )
            self.test = ( 0.5*( self.test + 1 )
                          * ( self.trainmax - self.trainmin )
                          + self.trainmin )
            for t in self.targets:
                self.predictions[t] = ( 0.5*( self.predictions[t] + 1 )
                                        * ( self.trainmax[t] - self.trainmin[t] )
                                        + self.trainmin[t] )
                
        elif self.scale_method=='quantile':
            self.train = ( 0.5*( self.train + 1 )
                           * ( self.train99q - self.train1q )
                           + self.train1q )
            self.test = ( 0.5*( self.test + 1 )
                          * ( self.train99q - self.train1q )
                          + self.train1q )
            for t in self.targets:
                self.predictions = ( 0.5*( self.predictions + 1 )
                                     * ( self.train99q[t] - self.train1q[t] )
                                     + self.train1q[t] )


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
        self.predictions = pd.DataFrame(predictions)
        self.predictions.columns = self.targets
        
            
def main(argv):

    # User configuration
    # train_args = [ {'layers':[25,10], 'activation':['sigmoid','sigmoid']}]
    train_args = [ {'layers':[25,15,25], 'activation':['sigmoid','sigmoid','sigmoid']}]
    # train_var = ["pt(mc nuH)","eta(mc nuH)","phi(mc nuH)","e(mc nuH)"]
    train_var = ["pt(mc nuH)"]
    scale_method = 'min_max' # 'min_max', 'mean_std', or 'quantile'
    n_epochs = 1000
    npts = 100000
    kfolds = 3

    # Setup matplotlib
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


    # Define output names
    layernames = '-'.join([str(x) for x in train_args[0]['layers']])
    activationnames = '-'.join([x[0:3] for x in train_args[0]['activation']])
    # Read dataset 
    pd.set_option('display.max_columns', 100)
    datafiles = [ "../test/mg5pythia8_hp200.root.test3.csv"]
    # datafiles = [ "../test/mg5pythia8_hp200.root.test3.csv",
    #               "../test/mg5pythia8_hp300.root.test.csv",
    #               "../test/mg5pythia8_hp400.root.test.csv" ]
    tmp_frames = []
    for d in datafiles:
        tmp_frames.append(pd.read_csv(d))
    dataset_tmp = pd.concat(tmp_frames)
    if len(dataset_tmp) < npts:
        npts = len(dataset_tmp)
        print("WARNING: Number of requested data points more than available")
        print("         Loaded all {} datapoints".format(npts))
    dataset = dataset_tmp.sample(npts, random_state=1)
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
    predictors_final = [ "et(met)", "phi(met)", "nbjet",
                         "pt(reco tau1)", "eta(reco tau1)",
                         "phi(reco tau1)", "m(reco tau1)",
                         "pt(reco bjet1)", "eta(reco bjet1)",
                         "phi(reco bjet1)", "m(reco bjet1)",
                         # "pt(reco bjet2)", "eta(reco bjet2)",
                         # "phi(reco bjet2)", "m(reco bjet2)",
                         "pt(reco jet1)", "eta(reco jet1)",
                         "phi(reco jet1)", "m(reco jet1)",
                         "pt(reco jet2)", "eta(reco jet2)",
                         "phi(reco jet2)", "m(reco jet2)" ]

    ntest = int(npts/kfolds)
    pred_all = [None for i in range(kfolds)]
    truth_all = [None for i in range(kfolds)]
    met_all = [None for i in range(kfolds)]
    history_all = [None for i in range(kfolds)]
    res_pred_all = [None for i in range(kfolds)]
    res_def_all = [None for i in range(kfolds)]

    def run_fold(i):
    # for i in range(kfolds):
        print("Running fold {}/{}".format(i+1,kfolds))
        base_ofile = "{}.{}.{}.{}.{}p.{}e.{}o{}".format( layernames,
                                                         activationnames,
                                                         scale_method,
                                                         '-'.join(train_var),
                                                         npts,
                                                         n_epochs, i+1, kfolds )
        testindex = [x for x in range(ntest*i,ntest*(i+1))]
        trainindex = [x for x in range(npts) if x not in testindex]
        train = dataset.irow(trainindex)
        test = dataset.irow(testindex)
        nn = None
        nn = NeuralNet( traindata          = train,
                        testdata           = test,
                        targets            = train_var,
                        predictors         = predictors_final,
                        hidden_nodes       = train_args[0]['layers'],
                        hidden_activations = train_args[0]['activation'] )
        nn.scale(method=scale_method,fillnan='zero')
        nn.compile()
        # Load weights and history from file if already trained with this setup
        history = None
        if os.path.isfile(base_ofile + '.h5'):
            print("Loading model weights from file {}".format(base_ofile + '.h5'))
            nn.load_weights(base_ofile + '.h5')
            tmp_hist = []
            print("Loading model history from file {}".format(base_ofile + '.loss'))
            with open(base_ofile + '.his','r') as f:
                for l in f.readlines():
                    tmp_hist.append(float(l.replace('\n','')))
            history = {'loss':tmp_hist}
        else:
            history = nn.fit(n_epochs)
            # print(history['loss'][-1])
            print("Saving model weights to file {}".format(base_ofile + '.h5'))
            nn.save_weights(base_ofile + '.h5')
            print("Saving model history to file {}".format(base_ofile + '.loss'))
            with open(base_ofile + '.his','w') as f:
                for l in history['loss']:
                    f.write(str(l)+'\n')
        # Predict 'train_var' and rescale to its physical value.
        nn.predict() 
        nn.unscale()
        # multipool = Pool(processes=3)
        # result = multipool.map(train_net, train_args)

        pred    = np.array(nn.predictions)
        target  = np.array(nn.test[train_var].as_matrix())
        met     = np.array(nn.test["et(met)"].as_matrix().reshape(-1,1))
        met_phi = np.array(nn.test["phi(met)"].as_matrix())
        resolution_predict_tmp = (target - pred)/target*100
        resolution_predict = [ item for sublist in resolution_predict_tmp
                               for item in sublist if abs(item) < 200 ]
        resolution_default_tmp = (target - met)/target*100
        resolution_default = [ item for sublist in resolution_default_tmp
                               for item in sublist if abs(item) < 200]
        pred_all[i] = [float(x) for x in pred]
        truth_all[i] = [float(x) for x in target]
        met_all[i] = [float(x) for x in met]
        history_all[i] = history['loss']
        res_pred_all[i] = [float(x) for x in resolution_predict]
        res_def_all[i] = [float(x) for x in resolution_default]

        # if one training target
        binranges = np.linspace(0, 300, 50)
        fig1, ax1 = plt.subplots(1)
        ax1.hist( nn.test[train_var[0]], bins=binranges,
                  label='Truth', histtype='step' )
        ax1.hist( nn.predictions[train_var[0]], bins=binranges,
                  label='Pred', histtype='step' )
        # ax1.set_xlim( [ min([i]),
        #                    max(nn.test[train_var])  ] )
        ax1.set_xlabel(train_var[0])
        ax1.legend()
        fig1.tight_layout(pad=0.3)
        fig1.savefig(base_ofile + '.pt.pdf')
        plt.close(fig1)

        # Plot training error ("loss") as function of training epoch
        fig2, ax2 = plt.subplots(1)
        ax2.plot(history['loss'], color='k')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_ylim([0.0, 0.012])
        # ax2.set_ylim([0.9*min(history['loss']), history['loss'][1]])
        fig2.tight_layout(pad=0.3)
        fig2.savefig(base_ofile + '.his.pdf')
        plt.close(fig2)
        
        fig3, ax3 = plt.subplots(1)
        ax3.hist( resolution_default, bins=50, label='Default',
                  histtype='step' )
        ax3.hist( resolution_predict, bins=50, label='Pred',
                  histtype='step' )
        ax3.set_xlabel('Resolution (%)')
        ax3.legend()
        fig3.tight_layout(pad=0.3)
        fig3.savefig(base_ofile + '.res.pdf')
        plt.close(fig3)


        # plt.show()

    # foldrange = [i for i in range(kfolds)]
    # multipool = Pool(processes=3)
    # result = multipool.map(run_fold, foldrange)

        
    for i in range(kfolds):
        run_fold(i)
        
    # Plot combined results
    base_ofile = "{}.{}.{}.{}.{}p.{}e.{}".format( layernames, activationnames,
                                                  scale_method,
                                                  '-'.join(train_var),
                                                  npts,
                                                  n_epochs, 'all' )
    # if one training target
    binranges = np.linspace(0, 300, 50)
    fig1, ax1 = plt.subplots(1)
    predplot = [x for x in [y for y in pred_all]][0]
    truthplot = [x for x in [y for y in truth_all]][0]
    metplot = [x for x in [y for y in met_all]][0]
    ax1.hist( truthplot, bins=binranges,
              label='Truth', histtype='step' )
    ax1.hist( metplot, bins=binranges,
              label='MET', histtype='step' )
    ax1.hist( predplot, bins=binranges,
              label='Pred', histtype='step' )
    # ax1.set_xlim( [ min([i]),
    #                    max(nn.test[train_var])  ] )
    ax1.set_xlabel(train_var[0])
    ax1.legend()
    fig1.tight_layout(pad=0.3)
    fig1.savefig(base_ofile + '.pt.pdf')
    plt.close(fig1)

    # Plot training error ("loss") as function of training epoch
    fig2, ax2 = plt.subplots(1)
    for i, his in enumerate(history_all):
        ax2.plot(his, label='Fold {}'.format(i+1))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.set_ylim([0.0, 0.02])
    fig2.tight_layout(pad=0.3)
    fig2.savefig(base_ofile + '.his.pdf')
    plt.close(fig2)

    respredplot = [x for x in [y for y in res_pred_all]][0]
    resdefplot = [x for x in [y for y in res_def_all]][0]
    fig3, ax3 = plt.subplots(1)
    ax3.hist( resdefplot, bins=50, label='Default',
              histtype='step' )
    ax3.hist( respredplot, bins=50, label='Pred',
              histtype='step' )
    ax3.legend()
    ax3.set_xlabel('Resolution (%)')
    fig3.tight_layout(pad=0.3)
    fig3.savefig(base_ofile + '.res.pdf')
    plt.close(fig3)


if __name__=="__main__":
    main(sys.argv[1:])
