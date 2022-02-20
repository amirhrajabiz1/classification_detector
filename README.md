# classification_detector
a program for find the best algorithm between classification algorithms(KNN, Decision tree, logistic regression, SVM)
------------------------------------------------------------------------------------------
in this program you can give your data and see which of this algorithms have better score.
------------------------------------------------------------------------------------------
this is an example:
classification(data=df, features=['age', 'sex','trtbps', 'chol','cp','fbs','restecg', 'thalachh','oldpeak', 'slp','exng', 'caa','thall'], y_ouput='output', wanna_normalize=False,goal='negFal', number_of_repeats=5)
----------------
classification: this is the main function.
----------------
data: this gives the data as dataFrame.
---------------
features: this gives the list of Xs.
---------------
y_ouput: this gives the name of y column. Note: the values of this columns should be in binary(0, 1) format.
---------------
goal: this is the score type we want to evaluate the model by it. the goal can be these:
F1-score: the highest f1-score.(default)
negFal: the lowest negative False percent.
posFal: the lowest positive False percent.
negtru: the highest negative True percent.
postru: the highest positive True percent.
jac   : jaccard-score.
----------------
wanna_normalize: this show we want to normalize our data or not(default:True).
----------------
number_of_repeats: this show the number of test we want to do. for example number_of_repeats=3 indicates that first we randomize Xs and calculate the score of algorithms then again randomize the Xs and calculate scores and so on for 3 times. this is for average score.(default=1)
-------------------------------------------------------------------------------------------------------
output of the function:
the best F1-score score for 1 states is: 0.802214, for state 1, with algorithm KNN with k: 9.
------------
the best average of F1-score scores for 1 states is: 0.802214 with algorithm KNN.
---------------------------------------------------------------------------------


