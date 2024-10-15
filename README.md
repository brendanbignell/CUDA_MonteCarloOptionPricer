# CUDA Barrier Option Pricing & ML Model Training

This a demo of how Barrier Options can be priced using Monte Carlo Simulations running on a cluster of NVidia GPUs.  The option pricing results are then used to train a machine learning models to predict the price of the Barrier Options. The idea being that a trained ML model could be used to price options more quickly than running a Monte Carlo simulation.  Test show an XGBoost Gradient Boosted Decision Tree model can predict more than 400K barrier option prices per second without given much though to optimization.

The Demo constists of two parts:

## 1. CUDA Monte Carlo Simulation for pricing Barrier Options 	 
   
   The following barrier option types are supported:

	- Up-and-Out, Down-and-Out, Up-and-In & Down-and-In

	- European & American Exercise (American options use the Longstaff-Schwartz algorithm)

	- Call & Put Options

	The main() function in CUDA_MonteCarloOptionPricer.cu is used to drive batch pricing of random portfolios of Barrier Options.


   The option parameters and pricing results are written to a CSV file. Sample 100K and 1M output files are included in the repo.

   ![Create Barriers](https://github.com/brendanbignell/CUDA_MonteCarloOptionPricer/blob/master/images/CreateBarriers.png)

## 2. Python Jupyter Notebooks 

### 2.1.  barrier_quick_ml_model_eval.ipynb
Demonstrates how to train a machine learning models to predict the price of Barrier Options. For this example Support Vector Machine (SVM), Random Forest and XGBoost GBDT models were initiially evaluated. The XGBoost model was selected for further training and testing as the initial results seemed the most promising. 

   ![Model Eval](https://github.com/brendanbignell/CUDA_MonteCarloOptionPricer/blob/master/images/QuickModelEvals.png)

### 2.2.  barrier_xgboost.ipynb

XGBoost Gradient Boosted Decision Tree Models were trained using a file of 100K to 10 million barrier options. The XGBoost model is then used to predict the price.  Model accuracy statistics are calculated, Mathpoltlib and Ploty charts are then used to visualize the model performance.
 
 ![European Barriers](https://github.com/brendanbignell/CUDA_MonteCarloOptionPricer/blob/master/images/EuropeanBarriers.png)

 ![American Barriers](https://github.com/brendanbignell/CUDA_MonteCarloOptionPricer/blob/master/images/AmericanBarriers.png)

 ## 3. Further Work

 #### 3.1. Option Paramters
 The random option parameters could be chosen more wisely to better represent real world options.

 #### 3.2. Results Analysis
 Further analysis of ML model prediction results.  E.g. Outlier, why are American options predicted better than European options?  There is surely alot that can be done to improve the model accuracy.

 #### 3.3. Hyperparameter tuning
 E.g. Using a grid of heuristic driven search of the XGBoost model hyperparameters to improve accuracy.

 ### 3.4. Alternate Models
 More advanced ML models could be evaluated. E.g. Neural Networks

 ### 3.5. Code refinement and optimization
 The code was quickly hacked together as a proof of concept and pretty much all of it could be improved. E.g. The CUDA code could be optimized to run faster, the Python code could be refactored to be more readable and efficient.



