from sklearn.datasets import load_breast_cancer
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

class KNN:
    def __init__(self, k):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        # the X in this function is the data containing the test split
        # X has test split -> multiple rows having all 4 features
        
        # we need to make a prediction for each row of test data
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        # x is one test row
        
        # compute distances to all points from the training split
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        
        # indices of the k nearest data points
        k_indices = np.argsort(distances)[:self.k]
        
        # take their labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        
        most_common = Counter(k_nearest_labels).most_common(1)
        print("most common: ", most_common)
        return most_common[0][0]
    
    
iris = load_breast_cancer()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print("class names len =", len(class_names))

knn = KNN(k = 3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("accuracy: ", np.mean(y_pred == y_test))
print("predictions: ", class_names[y_pred])


from sklearn.metrics import confusion_matrix, classification_report

print("\nConfusion matrix: ", confusion_matrix(y_test, y_pred))
print("\nClassification report: ", classification_report(y_test, y_pred))


