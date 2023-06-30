import os
import time
import random
import pickle

class config:

    def __init__(self):

        self.train_val = ["I3A","I3C","I3D","I3E","I3F","I3G","I3H","I3K","I3L","I3M","I3N","I3Q","I3R","I3S","I3T","I3V","I3W","I3Y","Z4A","Z4C","Z4D","Z4E","Z4F","Z4G","Z4H","Z4I","Z4K","Z4L","Z4M","Z4N","Z4Q","Z4R","Z4S","Z4T","Z4V","Z4W","Z4Y","V5A","V5C","V5D","V5E","V5F","V5G","V5H","V5I","V5K","V5L","V5M","V5N","V5Q","V5R","V5S","V5T","V5W","V5Y","E8A","E8C","E8D","E8F","E8G","E8H","E8I","E8K","E8L","E8M","E8N","E8Q","E8R","E8S","E8T","E8V","E8W","E8Y","E13A","E13C","E13D","E13F","E13G","E13H","E13I","E13K","E13L","E13M","E13N","E13Q","E13R","E13S","E13T","E13V","E13W","E13Y",]
        self.test = []

def Cross_validation(idx, num):
    
    random.shuffle(idx)
    fold_rt = []
    num_individual = len(idx) // num + 1
    for i in range(num):
        start_idx = i*num_individual 
        end_idx = min((i+1)*num_individual, len(idx)) 
        test = idx[start_idx:end_idx]
        train = [ii for ii in idx if ii not in test]
        fold_rt.append([train, test])

    return fold_rt

def train_val(lst):

    all_train_val = []
    for i in lst:
        with open("input_vectors_" + i + ".pkl", "rb") as f:
            graphs_dict, labels = pickle.load(f)
        for j in graphs_dict.keys():
            all_train_val.append([j, graphs_dict[j], labels[j]])
    
    return all_train_val

def run():

    start = time.time()

    settings = config()

    all_train_val = train_val(settings.train_val)
    fold = Cross_validation(list(range(len(all_train_val))), 10)
    for ii in range(len(fold)):
        if not os.path.exists(os.path.join(".", "Fold"+str(ii))):
            os.mkdir(os.path.join(".", "Fold"+str(ii)))
        train_pkl = open(os.path.join(".", "Fold"+str(ii), "train.pkl"), "wb")
        valid_pkl = open(os.path.join(".", "Fold"+str(ii), "valid.pkl"), "wb")
        train_data = [all_train_val[i] for i in fold[ii][0]] 
        valid_data = [all_train_val[i] for i in fold[ii][1]] 
        pickle.dump(train_data, train_pkl) 
        pickle.dump(valid_data, valid_pkl)

    end = time.time()
    runtime_h = (end - start) / 3600
    print("run " + str(runtime_h) + " h.")

def main():
    run()
    
if __name__=="__main__":
    main() 