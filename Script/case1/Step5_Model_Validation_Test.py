import os
import time
import pickle
import numpy as np
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

def gb_model():
    
    clf = GradientBoostingRegressor(n_estimators = 100, max_depth = 3, learning_rate = 0.05)

    for i in range(10):

        train_feature = []
        train_labels = []
        val_feature = []
        val_labels = []
        
        with open(os.path.join("Fold"+str(i),"train.pkl"), "rb") as f:
            train = pickle.load(f)
        for a in train:
            # print(a)
            train_feature.append(a[1])
            train_labels.append(a[2])
        with open(os.path.join("Fold"+str(i),"valid.pkl"), "rb") as f:
            val = pickle.load(f)
        for a in val:
            val_feature.append(a[1])
            val_labels.append(a[2])

        train_feature = np.matrix(train_feature)
        # print(train_feature)
        # train_labels = np.matrix(train_labels)
        # print(train_labels)
        # print(train_labels.ravel())
        val_feature = np.matrix(val_feature)
        # val_labels = np.matrix(val_labels)

        rf = clf.fit(train_feature, train_labels)
        pred_train = rf.predict(train_feature)
        print(pred_train)
        with open(os.path.join("Fold"+str(i),"train_pre_labels.csv"), "w") as f:
            f.write("predict,labels\n")
            for a in range(len(train_labels)):
                f.write(str(pred_train[a])+","+str(train_labels[a])+"\n")
        mse_train = mean_squared_error(train_labels, pred_train)
        pearson_train = pearsonr(train_labels, pred_train)
        ci_train = concordance_index(train_labels, pred_train)
        with open(os.path.join("Fold"+str(i),"train_matrix.txt"), "w") as f:
            f.write("mse,pearson,ci\n")
            f.write(str(mse_train)+","+str(pearson_train)+","+str(ci_train))
        
        pred_val = rf.predict(val_feature)
        with open(os.path.join("Fold"+str(i),"val_pre_labels.csv"), "w") as f:
            f.write("predict,labels\n")
            for a in range(len(val_labels)):
                f.write(str(pred_val[a])+","+str(val_labels[a])+"\n")
        mse_val = mean_squared_error(val_labels, pred_val)
        pearson_val = pearsonr(val_labels, pred_val)
        ci_val = concordance_index(val_labels, pred_val)
        with open(os.path.join("Fold"+str(i),"val_matrix.txt"), "w") as f:
            f.write("mse,pearson,ci\n")
            f.write(str(mse_val)+","+str(pearson_val)+","+str(ci_val))

def run():

    start = time.time()

    gb_model()

    end = time.time()
    runtime_h = (end - start) / 3600
    print("run " + str(runtime_h) + " h.")

def main():
    run()
    
if __name__=="__main__":
    main() 