# Machine_Learning_Models
Scripts for some machine learning methods.  

## Case1
Descriptions:   
 - Purpose: Obtaining features of protein-peptide interaction interface and use them for machine learning modeling and for predicting ddg.
 - System: Protein peptide FEP results     
 - Installation dependence: pymol; openbabel
    ```python
    conda install -y -c schrodinger pymol
    conda install -c openbabel openbabel
    ```

Path:  
 - [Case1](./Script/case1/)   

Usage: 
 - python Step1_get_descriptors_labels.py  
 - python Step2_get_all_unique_features_frequency.py  
 - python Step3_get_input.py  
 - python Step4_data_split.py  
 - python Step5_Model_Validation_Test.py  