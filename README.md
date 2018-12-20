# Compare 3 learning methods to recognize fake notes, based on a dataset composed of 4 feture
This is a part of a project done at the University of Lisbon - FCT - UNL (Portougal), class of Machine Learning 2018/19.
We compare three different classificatory (logistic regression, K Nearest Neighbors, Naïve Bayes) to classify real and fake banknotes using features (Variance, Skewness, Curtosis and Entropy).
The data was taken from the Machine Learning Repository website (Lohweg, 2012). The dataset was extracted from genuine and forges banknotes and consists of images of the banknotes. The features are: Variance, Skewness, Curtosis, and Entropy.

### Prerequisites

Python 3 

Packages used: matplotlib, sklearn, numpy

## Deployment

The logisitc regression and k nearest neighbours used are from scikit-learn python package. We apply over them only a cross-validation method to retrieve the best values for our purpose.
Then we implemented the Naive Bayse clissifier without using any external plugin. We did that using kernel density estimators to evaluate each feature and the apply to it the naive bayse theory.

##  Important steps during implementation

We first look at the feature distribution using th KDE's function furnished from scikit-learn. 
With the graph below we see a cool visualization of the data to understaing the distribution of out dataset.
![Aims](https://github.com/barloccia/Recognize-fake-notes-comparing-3-machine-learning-based-methods/blob/master/Screen%20Shot%202018-12-20%20at%2020.12.24.png)
For each method we then implemented the cross-validation to find best paramaters. Below an example of the graph retrieved after validation.
![Aims](https://github.com/barloccia/Recognize-fake-notes-comparing-3-machine-learning-based-methods/blob/master/Screen%20Shot%202018-12-20%20at%2020.12.31.png)
Finally, we compute testing over the test data set and we compare the different methods using many scores, like the Mc'nemar test as show in the table below.


![Aims](https://github.com/barloccia/Recognize-fake-notes-comparing-3-machine-learning-based-methods/blob/master/Screen%20Shot%202018-12-20%20at%2020.30.52.png)

## Running the tests
Download the package and run tp1.py like: python3 tp1.py

Graphs are saved in the properly folder, results are directly printed.

## Built With

* [Python 3](https://www.python.org/download/releases/3.0/) 

# Authors

* **Gabriele Barlacchi** - *Initial work* 
* **Leo torchia** - *Initial work* 
* **Sara Silva** - *Initial work* 


