import pandas as pd
import numpy as np
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
titanic = pd.read_csv("mg5pythia8_hp200.root.test2.csv")
selected_sample=titanic

##some variables are empy, this avoids pandas complaining
selected_sample[ ~np.isfinite(selected_sample) ] = -999.000000  # Set non-finite (nan, inf, -inf) to zero

##just to print some info about out dataset
print(selected_sample.head(10))
print(selected_sample.describe())

##Sklearn has a function that will help us with feature selection, SelectKBest. This selects the best features from the data, and allows us to specify how many it selects.

predictors = ["et(met)"," phi(met)","ntau","nbjet","njet","pt(reco tau1)","eta(reco tau1)","phi(reco tau1)","m(reco tau1)","pt(reco bjet1)","eta(reco bjet1)","phi(reco bjet1)","m(reco bjet1)","pt(reco bjet2)","eta(reco bjet2)","phi(reco bjet2)","m(reco bjet2)","pt(reco bjet3)","eta(reco bjet3)","phi(reco bjet3)","m(reco bjet3)","pt(reco bjet4)","eta(reco bjet4)","phi(reco bjet4)","m(reco bjet4)","pt(reco jet1)","eta(reco jet1)","phi(reco jet1)","m(reco jet1)","pt(reco jet2)","eta(reco jet2)","phi(reco jet2)","m(reco jet2)","pt(reco jet3)","eta(reco jet3)","phi(reco jet3)","m(reco jet3)","pt(reco jet4)","eta(reco jet4)","phi(reco jet4)","m(reco jet4)"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(selected_sample[predictors], selected_sample["pt(mc nuH)"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

#picked the best features according to SelectKBest
predictors_final = ["et(met)", " phi(met)","pt(reco tau1)", "eta(reco tau1)", "phi(reco tau1)","pt(reco bjet1)","pt(reco bjet1)","eta(reco bjet1)","phi(reco bjet1)","m(reco bjet1)","phi(reco bjet3)","m(reco bjet3)","pt(reco bjet4)","eta(reco bjet4)","phi(reco bjet4)","m(reco bjet4)","pt(reco jet1)","eta(reco jet1)","phi(reco jet1)","m(reco jet1)","pt(reco jet2)","eta(reco jet2)","phi(reco jet2)","m(reco jet2)"]


# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the  dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(selected_sample.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (selected_sample[predictors_final].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = selected_sample["pt(mc nuH)"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(selected_sample[predictors_final].iloc[test,:])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
target=list(selected_sample["pt(mc nuH)"])
pred=list(predictions)
met=list(selected_sample["et(met)"])


## I want to check the resolution of the prediction wrt the truth value resol = (pt_pred - pt_true)/pt_true*100. I also want to compare it with the resolution from the MET (default).

result_predict=[]
result_default=[]

for i in range(len(target)):
    porcentage_predict= (target[i] - pred[i])/target[i]*100
    porcentage_met= (target[i] - met[i])/target[i]*100
    result_predict.append(porcentage_predict)
    result_default.append(porcentage_met)

result=list(result_predict)
default=list(result_default)

prediction_list = np.array(list(result_predict))
defaul_list = np.array(list(result_default))

print(stats.describe(prediction_list))
print(stats.describe(defaul_list))


plt.hist(result, bins=2000)
plt.title("Preddicted result")
plt.xlabel("resolution (%)")
plt.ylabel("Frequency")
plt.axis([-200,200,0,600])
plt.show()

plt.hist(default, bins=2000)
plt.title("default result")
plt.xlabel("resolution (%)")
plt.ylabel("Frequency")
plt.axis([-200,200,0,600])
plt.show()



