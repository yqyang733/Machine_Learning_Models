import os
import time
import pickle
import numpy as np

class config:

    def __init__(self):

        self.features_lst = ["I3A","I3C","I3D","I3E","I3F","I3G","I3H","I3K","I3L","I3M","I3N","I3Q","I3R","I3S","I3T","I3V","I3W","I3Y","Z4A","Z4C","Z4D","Z4E","Z4F","Z4G","Z4H","Z4I","Z4K","Z4L","Z4M","Z4N","Z4Q","Z4R","Z4S","Z4T","Z4V","Z4W","Z4Y","V5A","V5C","V5D","V5E","V5F","V5G","V5H","V5I","V5K","V5L","V5M","V5N","V5Q","V5R","V5S","V5T","V5W","V5Y","E8A","E8C","E8D","E8F","E8G","E8H","E8I","E8K","E8L","E8M","E8N","E8Q","E8R","E8S","E8T","E8V","E8W","E8Y","E13A","E13C","E13D","E13F","E13G","E13H","E13I","E13K","E13L","E13M","E13N","E13Q","E13R","E13S","E13T","E13V","E13W","E13Y",]
        # self.features_lst = ["I3A",]
        self.features_sele = -1

def get_all_features(lst, sele):

    all_features = dict()
    for i in lst:
        with open("Descriptors_" + i + ".pkl", "rb") as f:
            descriptors = pickle.load(f)
        for j in descriptors[0].keys():
            for z in descriptors[0][j].keys():
                for x in descriptors[0][j][z]:
                    tmp = list()
                    for y in x.keys():
                        tmp.append((y, x[y]))
                    tmp = tuple(sorted(tmp))
                    if tmp in all_features:
                        all_features[tmp] += 1
                    else:
                        all_features[tmp] = 1
    feature_sorted = sorted(zip(all_features.values(), all_features.keys()), reverse=True)
    all_fea_sort = dict()
    for i in feature_sorted:
        all_fea_sort[i[1]] = i[0]

    with open("Descriptors_frequency.pkl", "wb") as f:
        pickle.dump(all_fea_sort, f)
    f.close() 

    fea_sele = list()
    if sele == -1:
        for i in range(len(feature_sorted)):
            fea_sele.append(feature_sorted[i][1])
    else:
        for i in range(sele):
            fea_sele.append(feature_sorted[i][1])

    fea_sele = np.array(fea_sele, dtype=object)
    
    with open("Descriptors_sele.pkl", "wb") as f:
        pickle.dump(fea_sele, f)
    f.close()

def run():

    start = time.time()

    settings = config()

    get_all_features(settings.features_lst, settings.features_sele)

    end = time.time()
    runtime_h = (end - start) / 3600
    print("run " + str(runtime_h) + " h.")

def main():
    run()
    
if __name__=="__main__":
    main() 