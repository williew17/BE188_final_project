# BE188_final_project
Analysis of GI lesions from Mesejo et al.

### Data is formatted as follows:
1. Each row is a different lesion, light combination (3rd column) (white light or narrow band).
2. There are three types of lesions and the type is in the name as well as in the second column. 1->hyperplasic, 2->serrated, 3->adenoma.
3. The 4th column until the end are the different extracted features of the lesion as from Mesejo et al.

###### Attribute Info
1. First 422 attributes: 2D TEXTURAL FEATURES 
  - 166 features: AHT: Autocorrelation Homogeneous Texture (Invariant Gabor Texture) 
  - Next 256: Rotational Invariant LBP 
2. Next 76 attributes: 2D COLOR FEATURES 
  - 16 Color Naming 
  - 13 Discriminative Color 
  - 7 Hue 
  - 7 Opponent 
  - 33 color gray-level co-occurrence matrix 
3. Last 200 attributes: 3D SHAPE FEATURES 
  - 100 shapeDNA 
  - 100 KPCA
### How to run
Instructions should be foud within comments in the code, but each classifer has its own python file, either PLSR.py or random_forest.py
running either of those files with python3 should run the default config for each classifier, If different modes are wanted, refer to the help section '--help' for the different types of modes available and how to use them.

