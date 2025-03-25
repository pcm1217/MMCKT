# MMCKT
Multi-Modal Cross-domain Knowledge Transfer for Next POI Recommendation  
Our datasets are collected from the following links. https://sites.google.com/site/yangdingqi/home/foursquare-dataset.   
1.model_ST.py is the Python code that stores the model.  
2.main_ST.py A is the code for training the model, saving the model weight file and obtaining the result;
3.datasets_ST.py is the code for loading training data. This code can choose whether to load all region data for the first stage training or fine-tune the target region data for the second stage based on the region marker bit. This code can increase the offset for POI numbers in other fields to ensure the uniqueness of POI numbers.
4.utils.py is the test function code for testing recall and ndcg indicators.
# Requirements
python 3.8+
pytorch 1.11.0
