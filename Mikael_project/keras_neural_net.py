import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from multiprocessing import Pool

from scipy import stats
from scipy.stats import norm,t

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation
from keras.callbacks import History

import ROOT

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
        # Use mean-square-error loss function and Nesteroc gradient descent
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mse', optimizer=sgd)

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
    train_args = [ {'layers':[25,10], 'activation':['sigmoid','sigmoid']}]
    # train_args = [ {'layers':[25,15], 'activation':['sigmoid','sigmoid']}]
    # train_args = [ {'layers':[19,35,25,15], 'activation':['sigmoid','sigmoid','sigmoid','sigmoid']}]
    # train_var = ["pt(mc nuH)","eta(mc nuH)","phi(mc nuH)","e(mc nuH)"]
    train_var = ["pt(mc nuH)"]
    scale_method = 'min_max' # 'min_max', 'mean_std', or 'quantile'
    n_epochs = 500
    npts = 200000
    kfolds = 3

    # Define output names
    layernames = '-'.join([str(x) for x in train_args[0]['layers']])
    activationnames = '-'.join([x[0:3] for x in train_args[0]['activation']])
    base_ofile = "{}.{}.{}.{}.{}p.{}e.{}".format( layernames, activationnames,
                                                  scale_method,
                                                  '-'.join(train_var),
                                                  npts,
                                                  n_epochs, 'all' )
    # Open rootfile
    ofile = ROOT.TFile(base_ofile + '.root', 'recreate')
    # Read dataset 
    pd.set_option('display.max_columns', 100)
    # datafiles = [ "../test/mg5pythia8_hp200.root.test3.csv"]
    datafiles = [ "../test/mg5pythia8_hp200.root.test3.csv",
                  "../test/mg5pythia8_hp300.root.test.csv",
                  "../test/mg5pythia8_hp400.root.test.csv" ]
    tmp_frames = []
    for d in datafiles:
        print("INFO: Loading data file {}".format(d))
        tmp_frames.append(pd.read_csv(d))
    dataset_tmp = pd.concat(tmp_frames)
    if len(dataset_tmp) < npts:
        npts = len(dataset_tmp)
        print("WARNING: Number of requested data points more than available")
        print("         Loaded all {} datapoints".format(npts))
    else:
        print("INFO: Using {} of {} data points".format(npts,len(dataset_tmp)))
    dataset = dataset_tmp.sample(npts, random_state=1)
    # Add truth mass of H+ to dataset
    dataset["mass_truth"] = ( np.sqrt(2*(dataset["pt(mc nuH)"])
                                      *(dataset["pt(mc tau)"])
                                      * ( np.cosh(dataset["eta(mc nuH)"]
                                                  - dataset["eta(mc tau)"])
                                          - np.cos(dataset["phi(mc nuH)"]
                                                   - dataset["phi(mc tau)"]))) )
    dataset['met_truth'] = ( np.sqrt(  dataset['pt(mc nuH)']**2
                                       + dataset['pt(mc nuTau)']**2
                                       + 2*dataset['pt(mc nuH)']
                                       *dataset['pt(mc nuTau)']
                                       * np.cos(  dataset['phi(mc nuH)']
                                                  - dataset['phi(mc nuTau)'])) )
    
    # Replace invalid values with NaN
    dataset = dataset.where(dataset > -998.0, other=np.nan)
    # Predictor variables 
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
    pt_pred_all = [None for i in range(kfolds)]
    pt_truth_all = [None for i in range(kfolds)]
    # pt_def_all = [None for i in range(kfolds)]
    met_all = [None for i in range(kfolds)]
    history_all = [None for i in range(kfolds)]
    res_pred_all = [None for i in range(kfolds)]
    res_def_all = [None for i in range(kfolds)]
    mt_pred_all = [None for i in range(kfolds)]
    mt_def_all = [None for i in range(kfolds)]

    def run_fold(i):
    # for i in range(kfolds):
        print("INFO: Running fold {}/{}".format(i+1,kfolds))
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
        # Otherwise train with this setup.
        history = None
        if os.path.isfile(base_ofile + '.h5'):
            print( "INFO: Loading model weights from file {}"
                   .format(base_ofile + '.h5') )
            nn.load_weights(base_ofile + '.h5')
            tmp_hist = []
            print( "INFO: Loading model history from file {}"
                   .format(base_ofile + '.loss') )
            with open(base_ofile + '.his','r') as f:
                for l in f.readlines():
                    tmp_hist.append(float(l.replace('\n','')))
            history = {'loss':tmp_hist}
        else:
            history = nn.fit(n_epochs)
            # print(history['loss'][-1])
            print( "INFO: Saving model weights to file {}"
                   .format(base_ofile + '.h5') )
            nn.save_weights(base_ofile + '.h5')
            print( "INFO: Saving model history to file {}"
                   .format(base_ofile + '.loss') )
            with open(base_ofile + '.his','w') as f:
                for l in history['loss']:
                    f.write(str(l)+'\n')
        # Predict 'train_var' and rescale to its physical value.
        nn.predict() 
        nn.unscale()

        pt_pred  = np.array(nn.predictions)
        pt_truth = np.array(nn.test[train_var].as_matrix())
        # pt_def   = np.array(nn.test["pt(reco tau1)"].as_matrix())
        met      = np.array(nn.test["et(met)"].as_matrix().reshape(-1,1))
        met_phi  = np.array(nn.test["phi(met)"].as_matrix())
        taupt    = np.array(nn.test["pt(reco tau1)"].as_matrix())
        tauphi   = np.array(nn.test["phi(reco tau1)"].as_matrix())
        mt_pred  = []
        mt_def   = []

        for j in range(len(taupt)):
            if taupt[j]>0 and pt_pred[j]>0:
                mt_pred.append( np.sqrt( 2*pt_pred[j]*taupt[j]*
                                         (1 - np.cos(tauphi[j]-met_phi[j])) ) )
                mt_def.append( np.sqrt( 2*met[j]*taupt[j]*
                                        (1 - np.cos(tauphi[j]-met_phi[j])) ) )

        res_predict     = [ item for sublist in (pt_truth - pt_pred)/pt_truth*100
                            for item in sublist if abs(item) < 200 ]
        res_met     = [ item for sublist in (pt_truth - met)/pt_truth*100
                            for item in sublist if abs(item) < 200]
        
        pt_pred_all[i]  = [float(x) for x in pt_pred]
        pt_truth_all[i] = [float(x) for x in pt_truth]
        # pt_def_all[i] = [float(x) for x in pt_def]
        met_all[i]      = [float(x) for x in met]
        history_all[i]  = history['loss']
        res_pred_all[i] = [float(x) for x in res_predict]
        res_def_all[i]  = [float(x) for x in res_met]
        mt_pred_all[i]  = [float(x) for x in mt_pred]
        mt_def_all[i]   = [float(x) for x in mt_def]


        
        pt_pred_hist = ROOT.TH1F( "pt_prediction", "", 100, 0, 500 )
        pt_truth_hist = ROOT.TH1F( "pt_default", "", 100, 0, 500 )
        for x in nn.test[train_var[0]]:
            pt_truth_hist.Fill(float(x))
        for x in nn.predictions[train_var[0]]:
            pt_pred_hist.Fill(float(x))
        pt_pred_hist.SetLineColor(1)
        pt_truth_hist.SetLineColor(2)
        c1 = ROOT.TCanvas("c1")
        pt_pred_hist.Draw()
        pt_truth_hist.Draw("same")
        l1 = ROOT.TLegend(0.7, 0.7, 0.9, 0.85)
        l1.AddEntry(pt_pred_hist, "Neural Network", "l")
        l1.AddEntry(pt_truth_hist, "pt(mc nuH)", "l")
        l1.Draw()
        c1.Print(base_ofile + '.pt.pdf')
  
        mt_pred_hist = ROOT.TH1F( "mt_prediction", "", 100, 0, 500 )
        mt_def_hist = ROOT.TH1F( "mt_default", "", 100, 0, 500 )
        for x in mt_pred:
            mt_pred_hist.Fill(float(x))
        for x in mt_def:
            mt_def_hist.Fill(float(x))
        mt_pred_hist.SetLineColor(1)
        mt_def_hist.SetLineColor(2)
        c4 = ROOT.TCanvas("c4")
        mt_pred_hist.Draw()
        mt_def_hist.Draw("same")
        l4 = ROOT.TLegend(0.7, 0.7, 0.9, 0.85)
        l4.AddEntry(mt_pred_hist, "Neural Network", "l")
        l4.AddEntry(mt_def_hist, "Default", "l")
        l4.Draw()
        c4.Print(base_ofile + '.mt.pdf')

        hist_graph = ROOT.TGraph( len(history['loss']),
                                  np.array([ float(x) for x in
                                             range(len(history['loss'])) ] ),
                                  np.array(history['loss']) )
        hist_graph.SetNameTitle("hist_graph", "Training history;Epoch;Loss")
        c2 = ROOT.TCanvas("c2")
        hist_graph.Draw()
        c2.Print(base_ofile + '.his.pdf')
        
        res_pred_hist = ROOT.TH1F( "res_pred_hist", "", 100, -200, 200 )
        res_def_hist = ROOT.TH1F( "res_def_hist", "", 100, -200, 200 )
        for x in res_predict:
            res_pred_hist.Fill(float(x))
        for x in res_met:
            res_def_hist.Fill(float(x))
        res_pred_hist.SetLineColor(1)
        res_def_hist.SetLineColor(2)
        c3 = ROOT.TCanvas("c3")
        res_pred_hist.Draw()
        res_def_hist.Draw("same")
        l3 = ROOT.TLegend(0.7, 0.7, 0.9, 0.85)
        l3.AddEntry(res_pred_hist, "Neural Network", "l")
        l3.AddEntry(res_def_hist, "Default", "l")
        l3.Draw()
        c3.Print(base_ofile + '.res.pdf')

        
    for i in range(kfolds):
        run_fold(i)
        
    # Plot combined results
    base_ofile = "{}.{}.{}.{}.{}p.{}e.{}".format( layernames, activationnames,
                                                  scale_method,
                                                  '-'.join(train_var),
                                                  npts,
                                                  n_epochs, 'all' )

    ptpredplot = [x for x in [y for y in pt_pred_all]][0]
    pttruthplot = [x for x in [y for y in pt_truth_all]][0]
    metplot = [x for x in [y for y in met_all]][0]

    pt_pred_hist = ROOT.TH1F( "pt_prediction", "", 100, 0, 500 )
    pt_truth_hist = ROOT.TH1F( "pt_truth", "", 100, 0, 500 )
    met_hist = ROOT.TH1F( "met", "", 100, 0, 500 )
    for x in ptpredplot:
        pt_pred_hist.Fill(float(x))
    for x in pttruthplot:
        pt_truth_hist.Fill(float(x))
    for x in metplot:
        met_hist.Fill(float(x))
    pt_pred_hist.SetLineColor(1)
    pt_truth_hist.SetLineColor(2)
    met_hist.SetLineColor(3)
    c1 = ROOT.TCanvas("c1")
    pt_pred_hist.Draw()
    pt_truth_hist.Draw("same")
    met_hist.Draw("same")
    l1 = ROOT.TLegend(0.7, 0.7, 0.9, 0.85)
    l1.AddEntry(pt_pred_hist, "Neural Network", "l")
    l1.AddEntry(pt_truth_hist, "pt(mc nuH)", "l")
    l1.AddEntry(met_hist, "met", "l")
    l1.Draw()
    c1.Print(base_ofile + '.pt.pdf')
    
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

    res_pred_hist = ROOT.TH1F( "res_prediction", "", 100, -200, 200 )
    res_def_hist = ROOT.TH1F( "res_met", "", 100, -200, 200 )
    for x in respredplot:
        res_pred_hist.Fill(float(x))
    for x in resdefplot:
        res_def_hist.Fill(float(x))
    res_pred_hist.SetLineColor(1)
    res_def_hist.SetLineColor(2)
    c3 = ROOT.TCanvas("c3")
    res_pred_hist.Draw()
    res_def_hist.Draw("same")
    l3 = ROOT.TLegend(0.7, 0.7, 0.9, 0.85)
    l3.AddEntry(res_pred_hist, "Neural Network", "l")
    l3.AddEntry(res_def_hist, "met", "l")
    l3.Draw()
    c3.Print(base_ofile + '.res.pdf')


    mtpredplot = [x for x in [y for y in mt_pred_all]][0]
    mtdefplot = [x for x in [y for y in mt_def_all]][0]

    mt_pred_hist = ROOT.TH1F( "mt_prediction", "", 100, 0, 500 )
    mt_def_hist = ROOT.TH1F( "mt_met", "", 100, 0, 500 )
    for x in mtpredplot:
        mt_pred_hist.Fill(float(x))
    for x in mtdefplot:
        mt_def_hist.Fill(float(x))
    mt_pred_hist.SetLineColor(1)
    mt_def_hist.SetLineColor(2)
    c4 = ROOT.TCanvas("c4")
    mt_pred_hist.Draw()
    mt_def_hist.Draw("same")
    l4 = ROOT.TLegend(0.7, 0.7, 0.9, 0.85)
    l4.AddEntry(mt_pred_hist, "Neural Network", "l")
    l4.AddEntry(mt_def_hist, "met", "l")
    l4.Draw()
    c4.Print(base_ofile + '.mt.pdf')

    prof_pred_hist = ROOT.TProfile( "profile_prediction", "",
                                    100, 0, 500, 0, 3 )
    prof_def_hist = ROOT.TProfile( "profile_met", "",
                                   100, 0, 500, 0, 3 )
    prof_truth_hist = ROOT.TProfile( "profile_truthmet", "",
                                     100, 0, 500, 0, 3 )
    for x,t in zip(ptpredplot, pttruthplot):
        prof_pred_hist.Fill(float(t), float(x)/float(t))
    for x,t in zip(metplot, pttruthplot):
        prof_def_hist.Fill(float(t), float(x)/float(t))
    for x,t in zip(dataset['met_truth'].as_matrix(), pttruthplot):
        prof_truth_hist.Fill(float(t), float(x)/float(t))
    prof_pred_hist.SetLineColor(1)
    prof_def_hist.SetLineColor(2)
    prof_truth_hist.SetLineColor(3)
    prof_pred_hist.SetMarkerColor(1)
    prof_def_hist.SetMarkerColor(2)
    prof_truth_hist.SetMarkerColor(3)
    c5 = ROOT.TCanvas("c5")
    prof_pred_hist.Draw()
    prof_def_hist.Draw("same")
    prof_truth_hist.Draw("same")
    l5 = ROOT.TLegend(0.7, 0.7, 0.9, 0.85)
    l5.AddEntry(prof_pred_hist, "Neural Network", "l")
    l5.AddEntry(prof_def_hist, "met", "l")
    l5.AddEntry(prof_truth_hist, "Truth met", "l")
    l5.Draw()
    c5.Print(base_ofile + '.prof.pdf')

    pt_truth_hist.Write()
    pt_pred_hist.Write()
    met_hist.Write()
    res_pred_hist.Write()
    res_def_hist.Write()
    mt_pred_hist.Write()
    mt_def_hist.Write()
    prof_pred_hist.Write()
    prof_def_hist.Write()
    prof_truth_hist.Write()

    ofile.Close()


if __name__=="__main__":
    main(sys.argv[1:])
