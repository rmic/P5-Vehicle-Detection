from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.externals import joblib

class Classifier:

    svr = None
    clf = None
    scaler = None
    def __init__(self, classifier=svm.SVC()):
        self.svr = classifier


    def train(self, X, y, parameters={'kernel':['rbf'], 'C':[10]}):
        #print(X.shape)
        self.scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = self.scaler.transform(X)

        # print("scaled X "+str(scaled_X.shape))
        # print("y "+str(y.shape))
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
        # print('Using spatial binning of:',spatial, 'and', histbin,'histogram bins')
        # print('Feature vector length:', len(X_train[0]))
        #parameters = {'kernel':['linear', 'rbf'], 'C':[.1, 1, 2, 5, 10]}


        # svr = svm.SVC()
        print("Training ...")
        self.clf = GridSearchCV(self.svr, parameters)
        self.clf.fit(X_train, y_train)
        print("Training finished")
        print(str(self.clf.cv_results_))

    def save(self, filename):
        joblib.dump(self, filename)

    def load(filename):
        return joblib.load(filename)

    def predict(self, features):
        scaled_feats = self.scaler.transform(X=features)
        return self.clf.predict(scaled_feats)
