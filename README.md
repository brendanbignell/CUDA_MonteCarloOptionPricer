# CUDA Barrier Option Pricing & Machine Learning Model Training

This a demo of how Barrier Options can be priced using Monte Carlo Simulations running on a cluster of GPUs.

The Demo constists of two parts:

# 1. CUDA Monte Carlo Simulation for pricing Barrier Options 	 
   
   Support the following barrier types:

	- Up-and-Out, Down-and-Out, Up-and-In &Down-and-In

	- European & American Exercise (American options use the Longstaff-Schwartz algorithm)

	- Call &Put Options

	The main() function in CUDA_MonteCarloOptionPricer.cu is used to drive batch pricing of random portfolios of Barrier Options. 


   The option parameter and pricing results are written to a CSV file. Sample 100K and 1M output files are included in the repo.

   ![alt text](https://github.com/brendanbignell/CUDA_MonteCarloOptionPricer/blob/master/images/CreateBarriers.png)

## 2. Python Jupyter Notebooks 

### 2.1.  barrier_quick_ml_model_eval.ipynb
Demonstrates how to train a machine learning models to predict the price of a Barrier Option. For this example Support Vector Machine (SVM), Random Forest and XGBoost GBDT models were initiially evaluated. The XGBoost model was selected for further training and testing. 

   ![alt text](https://github.com/brendanbignell/CUDA_MonteCarloOptionPricer/blob/master/images/QuickModelEvals.png)

### 2.2.  barrier_xgboost.ipynb

XGBoost Gradient Boosted Decision Tree Models were trained using a file of 100K to 10 million barrier option. The XGBoost model is then used to predict the price of a Barrier Option using the option parameters as input features.  Some statistics, Mathpoltlib and Ploty charts are then used to visualize the model performance.
 
