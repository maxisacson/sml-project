import pandas as pd
import numpy as np

from six import iteritems

from scipy import stats

#from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import \
    RandomForestRegressor, \
    ExtraTreesRegressor, \
    GradientBoostingRegressor

from sklearn.tree import \
    DecisionTreeRegressor

import matplotlib.pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser(description = 'Regression using ensemble methods')
parser.add_argument('-r',
                    '--regressor',
                    required=True,
                    help='the regressor to use',
                    metavar='<regressor>')
parser.add_argument('-d',
                    '--depth',
                    default=3,
                    type=int,
                    help='the tree depth',
                    metavar='<depth>')
parser.add_argument('-t',
                    '--trees',
                    default=100,
                    type=int,
                    help='the number of trees',
                    metavar='<trees>')
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

target_parameters = ['pt(mc nuH)', 'phi(mc nuH)', 'pt(mc nuTau)', 'phi(mc nuTau)']
pred_parameters = ['pt(pred nuH)', 'phi(pred nuH)', 'pt(pred nuTau)', 'phi(pred nuTau)']
reco_met = 'et(met)'
pred_met = 'et(pred_met)'
truth_met = 'et(mc met)'
reco_met_res = 'et(res met)'
pred_met_res = 'pt(res pred met)'

pd.set_option('display.max_columns', 500)
available_regressors = {
    'decision_tree': (
        'Decision Tree',
        DecisionTreeRegressor(max_features = 'auto',
                              max_depth = arguments.depth,
                              random_state = 0),
    ),
    'random_forest': (
        'Random Forest',
        RandomForestRegressor(n_estimators = arguments.trees,
                              max_features = 'auto',
                              max_depth = arguments.depth,
                              n_jobs = 4,
                              random_state = 0),
    ),
    # NOTE: Gradient boost does not seem to support multiple targets
    #'gradient_boosting': (
        #'Gradient Boosting',
        #GradientBoostingRegressor(n_estimators = arguments.trees,
                                  #max_features = 'auto',
                                  #max_depth = arguments.depth,
                                  #random_state = 0),
    #),
    'extra_trees': (
        'Extra Trees',
        ExtraTreesRegressor(n_estimators = arguments.trees,
                            max_features = 'auto',
                            max_depth = arguments.depth,
                            n_jobs = 4,
                            random_state = 0),
    )
}

try:
    name,regressor = available_regressors[arguments.regressor]
except KeyError:
    print('Error: No regressor with name {}'.format(arguments.regressor))
    exit(1)

print('Regression with {}'.format(name))

# Load sample(s)
sample_200 = pd.read_csv('../test/mg5pythia8_hp200.root.test3.csv')
#sample_300 = pd.read_csv('../test/mg5pythia8_hp300.root.test.csv')

# Combine sample(s)
#dataset = sample_200 + sample_300
dataset = sample_200

# Replace invalid values with 0 (NaN?)
dataset = dataset.where(dataset > -998.0, other=0)

dataset[truth_met] = np.sqrt(  dataset['pt(mc nuH)']**2
                               + dataset['pt(mc nuTau)']**2
                               + 2*dataset['pt(mc nuH)']*dataset['pt(mc nuTau)']
                                 * np.cos(  dataset['phi(mc nuH)']
                                          - dataset['phi(mc nuTau)']))

# Split 70/30 into training and test datasets
train, test = train_test_split(dataset,
                               test_size = 0.3,
                               random_state = 0)
predictors = train[selected_predictors]
targets = train[target_parameters]

# Fit to the training sample
regressor.fit(predictors, targets)

# Predict test sample
temp = regressor.predict(test[selected_predictors])

# Convert to DataFrame and concatenate with test
pred = pd.DataFrame(temp, columns=pred_parameters, index=test.index)
test = pd.concat([test, pred], axis=1)

# Compute the predicted MET
test.loc[:,pred_met] = np.sqrt(  test['pt(pred nuH)']**2
                               + test['pt(pred nuTau)']**2
                               + 2*test['pt(pred nuH)']*test['pt(pred nuTau)']
                                 * np.cos(  test['phi(pred nuH)']
                                          - test['phi(pred nuTau)']))

test.loc[:,reco_met_res] = (test[truth_met] - test[reco_met]) \
    / test[truth_met] * 100.0
test.loc[:,pred_met_res] = (test[truth_met] - test[pred_met]) \
    / test[truth_met] * 100.0

print(test[[p for l in zip(target_parameters, pred_parameters) for p in l]].head(10))
print(test[[truth_met, reco_met, pred_met]].head(10))
print(test[[reco_met_res, pred_met_res]].head(10))

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
ax[0].hist(test[pred_met], 50, range=(0.0, 400.0), label='Pred', histtype='step')
ax[0].hist(test[reco_met], 50, range=(0.0, 400.0), label='Obs', histtype='step')
ax[0].hist(test[truth_met], 50, range=(0.0, 400.0), label='Truth', histtype='step')
ax[0].set_xlim([0,400])
ax[0].set_xlabel("pT (GeV)")
ax[0].legend(loc='upper right')

ax[1].hist(test[pred_met_res], 50, range=(-100.0, 100.0), label='Pred', histtype='step')
ax[1].hist(test[reco_met_res], 50, range=(-100.0, 100.0), label='Obs', histtype='step')
ax[1].set_xlim([-100,100])
ax[1].set_xlabel("Resolution (%)")
ax[1].legend(loc='upper left')

fig.tight_layout(pad=0.3)
fig.savefig('multi_{}_{}_{}.pdf'.format(arguments.regressor,
                                        arguments.trees,
                                        arguments.depth))
plt.show()
