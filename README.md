# graph2route
Graph2Route: A Dynamic Spatial-Temporal Graph Neural Network for Pick-up and Delivery Route Prediction.

# Usage
Use the following command to run the code: `python run.py`
 
# Repo Structure
The structure of our code and description of important files are given as follows:  
│    ├────algorithm/  
│    │    ├────graph2route_logistics/:  code of graph2route for pick-up route prediction in logistics  
│    │    └────graph2route_food/:  code of graph2route for food delivery route prediction  
├────data/dataset/  
│    ├────logistics_p/:  Logistics-P data.  
│    └────food_pd/: Food_PD data.  
├────my_utils/  
│    ├────eval.py: Implemention of the evaluation metrics.  
│    └────utils.py  
└────run.py: Main function of this project, mainly used for  training and evaluating different models.  
