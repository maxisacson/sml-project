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

parser = ArgumentParser(description = 'Regression using ensemble methods')
parser.add_argument('-r',
                    '--regressors',
                    nargs = '+',
                    help='the regressors to use',
                    metavar='<regressors>')
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
arguments = parser.parse_args()

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

target_parameter = 'pt(mc nuH)'
reco_parameter = 'et(met)'
pred_parameter = 'pt(pred nuH)'
reco_resolution = 'et(res met)'
pred_resolution = 'pt(res pred nuH)'

pd.set_option('display.max_columns', 500)
sample_200 = pd.read_csv('../test/mg5pythia8_hp200.root.test3.csv')
#sample_300 = pd.read_csv('../test/mg5pythia8_hp300.root.test.csv')


# Make a combined sample
#dataset = sample_200 + sample_300
dataset = sample_200

# Replace invalid values with NaN
dataset = dataset.where(dataset > -998.0, other=0)

# Compute the H+ truth mass
dataset['mass_truth'] = (
    np.sqrt(  2
            * (dataset['pt(mc nuH)'])
            * (dataset['pt(mc tau)'])
            * (  np.cosh(dataset['eta(mc nuH)'] - dataset['eta(mc tau)'])
               - np.cos(dataset['phi(mc nuH)'] - dataset['phi(mc tau)'])))
)

print(dataset[selected_displays].head(10))
#print(dataset.head(10)['mass_truth'])
#print(dataset.describe())

train, global_test = train_test_split(dataset,
                                      test_size = 0.3,
                                      random_state = 0)
train_predictors = train[selected_predictors]
targets = train[target_parameter]

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

regressors = [
    (k, v)
    for k, v
    in iteritems(available_regressors)
    if k in arguments.regressors
]

for key,(name,regressor) in regressors:
    print('Regression with {}'.format(name))

    regressor.fit(train_predictors_std, targets)

    test = global_test.copy()
    test_predictors = test[selected_predictors]
    test_predictors_std = sc.transform(test_predictors)

    #test.loc[:,pred_parameter] = regressor.predict(test[selected_predictors])
    test.loc[:,pred_parameter] = regressor.predict(test_predictors_std)
    test.loc[:,reco_resolution] = (test[target_parameter] - test[reco_parameter]) \
        / test[target_parameter] * 100.0
    test.loc[:,pred_resolution] = (test[target_parameter] - test[pred_parameter]) \
        / test[target_parameter] * 100.0

    print(test[[target_parameter, pred_parameter, reco_parameter]].head(10))

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
    ax[0].hist(test[pred_parameter], 50, range=(0.0, 400.0), label='Pred', histtype='step')
    ax[0].hist(test[reco_parameter], 50, range=(0.0, 400.0), label='Obs', histtype='step')
    ax[0].hist(test[target_parameter], 50, range=(0.0, 400.0), label='Truth', histtype='step')
    ax[0].set_xlim([0,400])
    ax[0].set_xlabel("pT (GeV)")
    ax[0].legend(loc='upper right')

    ax[1].hist(test[pred_resolution], 50, range=(-100.0, 100.0), label='Pred', histtype='step')
    ax[1].hist(test[reco_resolution], 50, range=(-100.0, 100.0), label='Obs', histtype='step')
    ax[1].set_xlim([-100,100])
    ax[1].set_xlabel("Resolution (%)")
    ax[1].legend(loc='upper left')

    fig.tight_layout(pad=0.3)
    fig.savefig('{}_C{}_g{}_e{}_d{}_i{}.pdf'.format(key,
                                                    arguments.cost,
                                                    arguments.gamma,
                                                    arguments.epsilon,
                                                    arguments.degree,
                                                    arguments.iterations))
    plt.show()
