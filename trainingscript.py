# os
import os

# data
import pandas as pd

# math
import numpy as np

# sklearn
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Prepare data for training and testing
# This assumes the data is placed in a subfolder 'data'
path = "./data" if os.path.exists("./data") else "."
data = pd.read_csv(os.path.join(path, "transactions.csv"))
X = data.drop("Class", axis=1)
y = data["Class"]
X_data = X.values
y_data = y.values

# oversample unbalanced data
X_oversampled, y_oversampled = resample(X_data[y_data == 1], y_data[y_data == 1], replace=True, n_samples=X_data[y_data == 0].shape[0], random_state=42)

X_balanced = np.vstack((X_data[y_data == 0], X_oversampled))
y_balanced = np.hstack((y_data[y_data == 0], y_oversampled))

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2,
                                                    random_state=1, stratify=y_balanced)


adaboost = AdaBoostClassifier()
ada_params = {'algorithm': 'SAMME.R', 'estimator': DecisionTreeClassifier(max_depth=4), 'learning_rate': 1.0, 'n_estimators': 100, 'random_state': 0}
adaboost.set_params(**ada_params)

adaboost.fit(X_train, y_train)

# Testing and scoring

def leader_board_predict_fn(values):
    v = values
    try: 
        decision_function_values = adaboost.predict(v)
        return decision_function_values
    except Exception as e:
        print(str(e))
        raise e

decision_function_values = leader_board_predict_fn(X_test)
dataset_score = roc_auc_score(y_test, decision_function_values)

print("Score: ", dataset_score)
