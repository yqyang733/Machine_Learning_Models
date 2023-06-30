import time
import pickle
import numpy as np

class config:

    def __init__(self):

        self.features_lst = ["I3A","I3C","I3D","I3E","I3F","I3G","I3H","I3K","I3L","I3M","I3N","I3Q","I3R","I3S","I3T","I3V","I3W","I3Y","Z4A","Z4C","Z4D","Z4E","Z4F","Z4G","Z4H","Z4I","Z4K","Z4L","Z4M","Z4N","Z4Q","Z4R","Z4S","Z4T","Z4V","Z4W","Z4Y","V5A","V5C","V5D","V5E","V5F","V5G","V5H","V5I","V5K","V5L","V5M","V5N","V5Q","V5R","V5S","V5T","V5W","V5Y","E8A","E8C","E8D","E8F","E8G","E8H","E8I","E8K","E8L","E8M","E8N","E8Q","E8R","E8S","E8T","E8V","E8W","E8Y","E13A","E13C","E13D","E13F","E13G","E13H","E13I","E13K","E13L","E13M","E13N","E13Q","E13R","E13S","E13T","E13V","E13W","E13Y",]
        # self.features_lst = ["I3A",]

def make_data(graphs_dict):    
    
    with open("Descriptors_sele.pkl", "rb") as f:
        selected_descriptors = pickle.load(f)
    
    data = dict()
    
    ''' For each complex '''
    for name in graphs_dict.keys():
        whole_descriptors = dict()
        
        for type in graphs_dict[name].keys():
            
            ''' 
            one descriptor check  
            e.g. (16, 16):[{(1, 16, 6, '1'): 2, (0, 16, 6, '1'): 1}, ...]    
            '''
            for descriptor in graphs_dict[name][type]:
                if tuple(sorted(descriptor.items())) in whole_descriptors:
                    whole_descriptors[tuple(sorted(descriptor.items()))] += 1
                else:
                    whole_descriptors[tuple(sorted(descriptor.items()))] = 1  
        
        ''' Create a row vector of size 2,500 for each complex. '''
        row_vetor = list()
        for selected_descriptor in selected_descriptors:
            row_vetor.append(whole_descriptors[selected_descriptor]) if selected_descriptor in whole_descriptors else row_vetor.append(0)
                
        data[name] = np.array(row_vetor, dtype = np.float32)    
        
    return data

def get_input(lst):

    for i in lst:
        with open("Descriptors_" + i + ".pkl", "rb") as f:
            graphs_dict, labels = pickle.load(f)
        data = make_data(graphs_dict)
        with open("input_vectors_" + i + ".pkl", "wb") as f:
            pickle.dump((data, labels), f) 

def run():

    start = time.time()

    settings = config()

    get_input(settings.features_lst)

    end = time.time()
    runtime_h = (end - start) / 3600
    print("run " + str(runtime_h) + " h.")

def main():
    run()
    
if __name__=="__main__":
    main() 