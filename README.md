# SMOTE-oversampling-with-Imbalance-Data

# How to deal with Imbalanced Data?

1. Under-sample the majority class(exclude rows)
- lose out on a lot of data that could be used to train our model thus **improving its accuracy (e.g. higher bias)**

2. **Over-sample** the minority class
- leads to **overfitting** because the model learns from the same examples

# SMOTE (Synthetic Minority Oversampling Technique)
<img width="727" alt="Screenshot 2024-03-06 at 3 28 53 PM" src="https://github.com/ColleenJung/SMOTE-oversampling-with-Imbalance-Data/assets/119357849/25d29f0b-1955-4bb8-addc-b1955e98ef12">



- Take difference between a sample and its nearest neighbour
- Multiply the difference by a random number between 0 and 1
- Add this difference to the sample to generate a new synthetic example in feature space
- Continue on with next nearest neighbour up to user-defined number

# SMOTE using library - 'imblearn.over_sampling'

- The Python implementation of SMOTE actually comes in its own library (outside Scikit-Learn) which can be installed as follows:

```
# Import necessary libraries
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Create a sample dataset
X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.1, 0.9], n_informative=3, n_redundant=1, 
                           flip_y=0, n_features=20, n_clusters_per_class=1, 
                           n_samples=1000, random_state=10)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# Now X_smote and y_smote are your oversampled data.
```

```
# Initialize and train a RandomForestClassifier on the balanced dataset
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_smote, y_train_smote)

# Make predictions on the test set and compute the confusion matrix
y_pred = clf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print(conf_matrix)

```

