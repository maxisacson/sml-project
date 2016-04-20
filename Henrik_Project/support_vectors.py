import pandas as pd
import numpy as np

from six import iteritems

from scipy import stats

#from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from ROOT import TH1F, TFile, TProfile

from rutil import init_atlas_style, show_hists

parser = ArgumentParser(description = 'Regression using ensemble methods')
parser.add_argument('-r',
                    '--regressor',
                    required=True,
                    help='the regressor to use',
                    metavar='<regressor>')
parser.add_argument('-C',
                    '--cost',
                    default=1.0,
                    type=float,
                    help='the cost parameter',
                    metavar='<cost>')
parser.add_argument('-g',
                    '--gamma',
                    default=0.1,
                    type=float,
                    help='the gamma parameter for the RBF kernel',
                    metavar='<gamma>')
parser.add_argument('-e',
                    '--epsilon',
                    default=0.1,
                    type=float,
                    help='the epsilon parameter',
                    metavar='<gamma>')
parser.add_argument('-d',
                    '--degree',
                    default=3,
                    type=int,
                    help='the degree parameter for the polynomial kernel',
                    metavar='<degree>')
parser.add_argument('-i',
                    '--iterations',
                    default=1000,
                    type=int,
                    help='maximum number of iterations',
                    metavar='<maxiter>')
parser.add_argument('-m',
                    '--matplotlib',
                    action='store_true',
                    help='plot using matplotlib')
arguments = parser.parse_args()

def plot_matplotlib(sample):
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
    ax[0].hist(sample[met_pred_parameter], 50, range=(0.0, 400.0), label='Pred', histtype='step')
    ax[0].hist(sample[met_reco_parameter], 50, range=(0.0, 400.0), label='Obs', histtype='step')
    ax[0].hist(sample[met_target_parameter], 50, range=(0.0, 400.0), label='Target', histtype='step')
    ax[0].set_xlim([0,400])
    ax[0].set_xlabel("pT (GeV)")
    ax[0].legend(loc='upper right')

    ax[1].hist(sample[met_pred_resolution], 50, range=(-100.0, 100.0), label='Pred', histtype='step')
    ax[1].hist(sample[met_reco_resolution], 50, range=(-100.0, 100.0), label='Obs', histtype='step')
    ax[1].set_xlim([-100,100])
    ax[1].set_xlabel("Resolution (%)")
    ax[1].legend(loc='upper left')

    fig.tight_layout(pad=0.3)
    fig.savefig('{}_C{}_g{}_e{}_d{}_i{}.pdf'.format(arguments.regressor,
                                                    arguments.cost,
                                                    arguments.gamma,
                                                    arguments.epsilon,
                                                    arguments.degree,
                                                    arguments.iterations))
    plt.show()



def plot_root(sample):
    base_name = '{}_C{}_g{}_e{}_d{}_i{}'.format(arguments.regressor,
                                                    arguments.cost,
                                                    arguments.gamma,
                                                    arguments.epsilon,
                                                    arguments.degree,
                                                    arguments.iterations)

    met_pred, met_truth, met_reco, mt_pred, mt_reco = [
        TH1F(n, '', 100, 0, 500)
        for n
        in ['pt_{}'.format(arguments.regressor), 'pt_truth', 'met_reco',
            'mt_{}'.format(arguments.regressor), 'mt_reco']]
    res_pred, res_reco = [
        TH1F(n, '', 100, -100, 100)
        for n
        in ['res_{}'.format(arguments.regressor), 'res_reco']]
    profile = TProfile('profile_pred_{}'.format(arguments.regressor),
                       '', 100, 0, 500)

    map(met_pred.Fill, sample[met_pred_parameter])
    map(met_truth.Fill, sample[met_truth_parameter])
    map(met_reco.Fill, sample[met_reco_parameter])
    map(mt_pred.Fill, sample[mt_pred_parameter])
    map(mt_reco.Fill, sample[mt_reco_parameter])
    map(res_pred.Fill, sample[met_pred_resolution])
    map(res_reco.Fill, sample[met_reco_resolution])
    map(profile.Fill, sample['pt(reco tau1)'], sample[met_pred_ratio])

    root_file = TFile.Open('{}.root'.format(base_name), 'RECREATE')
    root_file.cd()
    met_pred.Write()
    mt_pred.Write()
    res_pred.Write()
    profile.Write()
    root_file.ls()
    root_file.Close()

    init_atlas_style()

    met_pred.SetName('Pred')
    met_reco.SetName('Reco')
    met_truth.SetName('Target')
    o1 = show_hists((met_pred, met_truth, met_reco),
                    'Missing ET',
                    '{}_met.pdf'.format(base_name))

    res_pred.SetName('Pred')
    res_reco.SetName('Reco')
    o2 = show_hists((res_pred, res_reco),
                    'Missing ET resolution',
                    '{}_res.pdf'.format(base_name))

    mt_pred.SetName('Pred')
    mt_reco.SetName('Reco')
    o3 = show_hists((mt_pred, mt_reco),
                    'Transverse mass',
                    '{}_mt.pdf'.format(base_name))

    profile.SetName('Profile')
    o3 = show_hists((profile,),
                    'Profile',
                    '{}_profile.pdf'.format(base_name))
    raw_input('Press any key...')


# All available variables in the dataset
all_predictors = [
    'et(met)', 'phi(met)', 'ntau', 'nbjet', 'njet', 'pt(reco tau1)',
    'eta(reco tau1)', 'phi(reco tau1)', 'm(reco tau1)', 'pt(reco bjet1)',
    'eta(reco bjet1)', 'phi(reco bjet1)', 'm(reco bjet1)', 'pt(reco bjet2)',
    'eta(reco bjet2)', 'phi(reco bjet2)', 'm(reco bjet2)', 'pt(reco bjet3)',
    'eta(reco bjet3)', 'phi(reco bjet3)', 'm(reco bjet3)', 'pt(reco bjet4)',
    'eta(reco bjet4)', 'phi(reco bjet4)', 'm(reco bjet4)', 'pt(reco jet1)',
    'eta(reco jet1)', 'phi(reco jet1)', 'm(reco jet1)', 'pt(reco jet2)',
    'eta(reco jet2)', 'phi(reco jet2)', 'm(reco jet2)', 'pt(reco jet3)',
    'eta(reco jet3)', 'phi(reco jet3)', 'm(reco jet3)', 'pt(reco jet4)',
    'eta(reco jet4)', 'phi(reco jet4)', 'm(reco jet4)'
]
# Predictors as selected by SelektKBest
selected_predictors = [
    'et(met)', 'phi(met)',
    'pt(reco tau1)', 'eta(reco tau1)', 'phi(reco tau1)',
    'pt(reco bjet1)', 'eta(reco bjet1)', 'phi(reco bjet1)', 'm(reco bjet1)',
    #'phi(reco bjet3)', 'm(reco bjet3)',
    #'pt(reco bjet4)', 'eta(reco bjet4)', 'phi(reco bjet4)', 'm(reco bjet4)',
    'pt(reco jet1)', 'eta(reco jet1)', 'phi(reco jet1)', 'm(reco jet1)',
    #'pt(reco jet2)', 'eta(reco jet2)', 'phi(reco jet2)', 'm(reco jet2)',
    'nbjet'
]
selected_displays = [
    'et(met)',
    'pt(mc nuH)',
    'pt(reco tau1)',
    'pt(mc tau)',
    'mass_truth'
]

nupt_truth_parameter = 'pt(mc nuH)'
met_truth_parameter = 'et(mc met)'
met_reco_parameter = 'et(met)'
met_pred_parameter = 'pt(pred nuH)'
met_reco_resolution = 'et(res met)'
met_pred_resolution = 'pt(res pred nuH)'
met_pred_ratio = 'pt(ratio nuH)'
#mt_truth_parameter = 'mt(mc)'
mt_reco_parameter = 'mt(net)'
mt_pred_parameter = 'mt(pred)'

pd.set_option('display.max_columns', 500)
sample_200 = pd.read_csv('../test/mg5pythia8_hp200.root.test3.csv')
sample_300 = pd.read_csv('../test/mg5pythia8_hp300.root.test.csv')
sample_400 = pd.read_csv('../test/mg5pythia8_hp400.root.test.csv')


# Make a combined sample
combined_sample = pd.concat((sample_200, sample_300, sample_400))
dataset = combined_sample.sample(100000, random_state=1)

# Replace invalid values with NaN
dataset = dataset.where(dataset > -998.0, other=np.nan)

# Compute the H+ truth mass
dataset['mass_truth'] = (
    np.sqrt(  2
            * (dataset['pt(mc nuH)'])
            * (dataset['pt(mc tau)'])
            * (  np.cosh(dataset['eta(mc nuH)'] - dataset['eta(mc tau)'])
               - np.cos(dataset['phi(mc nuH)'] - dataset['phi(mc tau)'])))
)

dataset[met_truth_parameter] = \
    np.sqrt(  dataset['pt(mc nuH)']**2
            + dataset['pt(mc nuTau)']**2
            + 2*dataset['pt(mc nuH)']*dataset['pt(mc nuTau)']
            * np.cos(  dataset['phi(mc nuH)']
                     - dataset['phi(mc nuTau)']))

#dataset[mt_truth_parameter] = \
    #np.sqrt(  2*dataset[met_truth_parameter]*dataset['pt(mc tau)']
            #* (1 - np.cos(dataset['phi(mc tau)'] -
                          #* dataset[metphi_truth_parameter])))
dataset[mt_reco_parameter] = \
    np.sqrt(  2*dataset[met_reco_parameter]*dataset['pt(mc tau)']
            * (1 - np.cos(dataset['phi(mc tau)'] - dataset['phi(met)'])))


#print(dataset[selected_displays].head(10))
#print(dataset.head(10)['mass_truth'])
#print(dataset.describe())

# Prepare the training and test datasets
train, test = train_test_split(dataset,
                               test_size = 0.3,
                               random_state = 0)
# Select predictors
train_predictors = train[selected_predictors]
test_predictors = test[selected_predictors]
# Drop rows with NaN values
train_predictors.dropna()
test_predictors.dropna()

# Approximate MET truth with neutrino pT or use real MET truth
met_target_parameter = nupt_truth_parameter
#met_target_parameter = met_truth_parameter

targets = train[met_target_parameter]

sc = StandardScaler()
sc.fit(train_predictors)
train_predictors_std = sc.transform(train_predictors)

available_regressors = {
    'svr_linear': (
        'SVR, Linear kernel',
        SVR(kernel='linear',
            C=arguments.cost,
            max_iter=arguments.iterations,
            epsilon=arguments.epsilon),
    ),
    'svr_poly': (
        'SVR, Polynomial kernel',
        SVR(kernel='poly',
            C=arguments.cost,
            max_iter=arguments.iterations,
            epsilon=arguments.epsilon,
            degree=arguments.degree,
            gamma=arguments.gamma),
    ),
    'svr_rbf': (
        'SVR, RBF kernel',
        SVR(kernel='rbf',
            C=arguments.cost,
            max_iter=arguments.iterations,
            epsilon=arguments.epsilon,
            gamma=arguments.gamma),
    ),
    'svr_sigmoid': (
        'SVR, Sigmoid kernel',
        SVR(kernel='sigmoid',
            C=arguments.cost,
            max_iter=arguments.iterations,
            epsilon=arguments.epsilon,
            gamma=arguments.gamma),
    )
}

name,regressor = available_regressors[arguments.regressor]
print('Regression with {}'.format(name))

regressor.fit(train_predictors_std, targets)

test_predictors_std = sc.transform(test_predictors)

test.loc[:,met_pred_parameter] = regressor.predict(test_predictors_std)
test.loc[:,met_reco_resolution] = (test[met_target_parameter] - test[met_reco_parameter]) \
    / test[met_target_parameter] * 100.0
test.loc[:,met_pred_resolution] = (test[met_target_parameter] - test[met_pred_parameter]) \
    / test[met_target_parameter] * 100.0
test.loc[:,met_pred_ratio] = test[met_pred_parameter]/test[met_target_parameter]

test.loc[:,mt_pred_parameter] = \
    np.sqrt(  2*test[met_pred_parameter]*test['pt(mc tau)']
            * (1 - np.cos(test['phi(mc tau)'] - test['phi(met)'])))

print(test[[met_target_parameter, met_pred_parameter, met_reco_parameter]].head(10))
print(test[[mt_pred_parameter, mt_reco_parameter]].head(10))

if arguments.matplotlib:
    plot_matplotlib(test)
else:
    plot_root(test)
