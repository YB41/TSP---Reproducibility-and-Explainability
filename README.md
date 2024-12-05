# Bachelorarbeit-2024
Bachelorarbeit Automatic Algorithmic Trading
<br><br>

The goal of this project is to create a reproducible Framework for time series prediction. 
The data is downloaded using yfinance - in which the data range can be modified. Our approach looks back as far as 2014 - depending on the asset. Our main focus in the project was the analysis and reproducability of our results. 
We chose to only consider the "Close" of each individual asset.

Due to this, all models were taken 'out-of-box' and not modified. Further, all models are initialized with a corresponding seed - ensuring that using the same data, model configuration and seed lead to the same results. 

The models are all trained on log-diff data. The diff function returns the difference between day x and day x+1. 

Results can be found in #insert paper#.

For examples, see folder Example_Performances


Install are the required libraries with the help of the requirements.txt 
```python
pip install -r requirements.txt
```
Once all the libraries are installed we can continue with the next step.

Decide which asset you want to look and download it with the help of the notebook
**Download_Data.ipynb** 

If you've done the above points, you can get started in running our different models.
For this just open the corresponding notebook in models, there you can decide if you want to run a SVR, RF or NN.
**NN.ipynb, SVM.ipynb, RandomForest.ipynb**

The results are created automatically in the results folder.
