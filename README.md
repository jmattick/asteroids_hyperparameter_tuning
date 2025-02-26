# asteroids_hyperparameter_tuning

Data was downloaded from: https://www.kaggle.com/shrutimehta/nasa-asteroids-classification

Data can be found in nasa.csv

The dataset contains information about 4687 asteroids and a hazardous classification of True or False.

The dataset was read using pandas. Features most likely to impact Hazard level of an asteroid were selected. 
The samples in the dataset were shuffled using numpy to 
allow the data to be split for training and testing. 20% of the dataset was reserved for the final 
test of the model. 

```python
import pandas as pd
import numpy as np

# read data using pandas
data = pd.read_csv("nasa.csv")

# select features
features = ['Absolute Magnitude', 'Est Dia in KM(min)',
            'Est Dia in KM(max)',
            'Epoch Date Close Approach', 'Relative Velocity km per sec',
            'Miss Dist.(kilometers)',
            'Orbit Uncertainity',
            'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant',
            'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
            'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance',
            'Perihelion Arg', 'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly',
            'Mean Motion']

# shuffle indices
np.random.permutation(data)

# get length of data and determine index of 80%
size = len(data)
split_idx = int(size * 0.8)

# split data
train, test = data[:split_idx], data[split_idx:]

# set X and y for train dataset
X_train = train[features]
y_train = train.Hazardous

# set X and y for test dataset
X_test = test[features]
y_test = test.Hazardous
```

The data was scaled using sklearn's StandardScaler(). The scaler was fit on all of the 
training data.

```python
from sklearn.preprocessing import StandardScaler

# scale data using sklearn StandardScaler
sc = StandardScaler()

# fit scaler on entire training data
sc.fit(X_train)

# transform training and validation data
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
```

A function `test_model` was created to score the models on a given dataset. 
The function returns the F1 score of the model. The F1 score was used instead of accuracy since the distribution of the 
hazard class is not uniform with the majority of the samples not being hazardous.
The F1 score was calculated by using sklearn's f1_score method.

```python
from sklearn.metrics import f1_score

def test_model(train_X, val_X, train_y, val_y, model):
    """Function to return the F1 score of a given model and dataset"""
    # Fit model
    model.fit(train_X, train_y)

    # get predicted values
    val_predict = model.predict(val_X)

    # return accuracy
    return f1_score(val_y, val_predict)
``` 

An n-fold cross validation method was implemented. This method splits a dataset into 
n chunks and evaluates the model iteratively over n-1 parts. 
The function returns a list of scores.

```python
def n_fold_cross_validation(dataset, features, target, model, sc, n=5):
    # get length of data and determine index of 80%
    size = len(dataset)
    fold_size = math.floor(size / n)
    extra = size % n
    scores = []  # holds each fold
    start = 0
    # subset data n times
    for i in range(n):
        idx = fold_size
        if extra > 0:
            idx += 1
            extra -= 1
        end = start + idx
        # get "test" set
        n_test = dataset[start:end]
        n_X_test = n_test[features]
        n_y_test = n_test[target]
        # get k-1 folds for training
        b = dataset[end:]
        if start != 0:
            a = dataset[:start]
            n_train = pd.concat([a, b])
        else:
            n_train = b
        n_X_train = n_train[features]
        n_y_train = n_train[target]
        # apply scaler
        n_X_train = sc.transform(n_X_train)
        n_X_test = sc.transform(n_X_test)
        # test model
        scores.append(test_model(n_X_train, n_X_test, n_y_train, n_y_test, model))
        # set new start
        start = end

    return scores
```

Next, grid search was implemented. The `grid_search` method 
evaluates a model on a given dataset for every parameter combination 
given to the method. For each parameter combination, the n_fold_cross_validation 
is called to get average score of the model. The function returns 
3 lists. The first contains the parameters used in the model, the second contains the 
average score for each model, and the third contains the standard deviation of the scores 
of each model. 

```python 

def grid_search(model, params, dataset, features, target, sc, n=5):
    # Format parameters to use in model
    param_names = [k for k in params]
    combos = list(itertools.product(*[params[k] for k in params]))
    combos_dict = list()  # list of dict)
    # print(combos)
    for i in combos:
        temp = dict()
        for j in range(len(param_names)):
            pn = param_names[j]
            pv = i[j]
            temp[pn] = pv
        combos_dict.append(temp)

    # print(combos_dict)
    param_info = []
    scores = []
    std = []
    for p in combos_dict:
        curr_model = model(**p)
        curr_scores = n_fold_cross_validation(dataset, features, target, curr_model, sc, n)
        scores.append(np.average(curr_scores))
        param_info.append(p)
        std.append(np.std(curr_scores))

    return param_info, scores, std
```

The SVC model from sklearn was used on this dataset. The hyper-parameters 
tuned using grid_search were the C and kernel. The parameter options were 
given in the form of a dictionary containing a list of potential parameter values.

```python
params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'rbf']}

# use grid search
grid_p, grid_s, grid_e = grid_search(SVC, params, train, features, "Hazardous", sc, 5)

grid_p_str = [str(p) for p in grid_p]

# plot F1 score for each model
plt.bar(grid_p_str, grid_s)
plt.xlabel("Model parameters")
plt.xticks(rotation=90)
plt.ylabel("Average F1 score")
plt.title("F1 score of model")
plt.tight_layout()
plt.savefig("grid_search.png")
plt.close()

```

The results of the grid_search using the SVC classifier to tune
the C and kernel are shown in the plot below. There was a variety of success between combinations of parameters.

![grid_plot](grid_search.png)

The model with the highest average F1 score was selected to be tested using the test dataset. 

best model: {'C': 1, 'kernel': 'linear'}

F1 score average: 0.8565524132689802

F1 score std: 0.0170101986512874

```python
# Find index of max score
max_idx = np.argmax(grid_s)
max_score = grid_s[max_idx]
max_std = grid_e[max_idx]
max_params = grid_p[max_idx]
print('best model: ' + str(max_params))
print('F1 score average: ' + str(max_score))
print('F1 score std: ' + str(max_std))

# Run best model on test set using entire training set no CV
score_nocv = test_model(X_train, X_test, y_train, y_test, SVC(**max_params))

# plot the cv score vs test score
bar_plot_scores = [max_score, score_nocv]
bar_plot_labels = ['Cross Validation Training Set', 'Test Set']
plt.bar(bar_plot_labels, bar_plot_scores)
plt.xlabel("")
plt.xticks(rotation=90)
plt.ylabel("Average F1 score")
plt.title("F1 score of Training and Test Datasets")
plt.tight_layout()
plt.savefig("train_test_compare.png")
plt.close()
```

The F1 score of the test dataset was 0.8634. This value was within the standard deviation of the n-fold cross validated training scores
for the same model. The small differences are due to the different samples present in the test and training datasets. 

![test_res](train_test_compare.png)