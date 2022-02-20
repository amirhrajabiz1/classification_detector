import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def classification(data:pd, features:list, y_ouput:str, goal="F1-score", wanna_normalize=True, number_of_repeats=1)->str:
    '''in this def we give the data in pandas dataframe, and
       we give the goal includes:
           F1-score: the highest f1-score.
           negFal: the lowest negative False percent.
           posFal: the lowest positive False percent.
           negtru: the highest negative True percent.
           postru: the highest positive True percent.
           jac   : jaccard-score.
           NOTICE: this function is ony for binary classification.'''
    
    #imports
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split

    ##########
    
    #define dictionaris for algorithms to compare scores##########################
    F1_score = {"KNN":{}, "DecisionTree":{}, "LogisticRegression":{}, "SVM":{}}
    negFal   = {"KNN":{}, "DecisionTree":{}, "LogisticRegression":{}, "SVM":{}}
    posFal   = {"KNN":{}, "DecisionTree":{}, "LogisticRegression":{}, "SVM":{}}
    negtru   = {"KNN":{}, "DecisionTree":{}, "LogisticRegression":{}, "SVM":{}}
    postru   = {"KNN":{}, "DecisionTree":{}, "LogisticRegression":{}, "SVM":{}}
    jac      = {"KNN":{}, "DecisionTree":{}, "LogisticRegression":{}, "SVM":{}}
    ##############################################################################

    #######Normalize check
    if wanna_normalize:
        X = preprocessing.StandardScaler().fit(data[features]).transform(data[features].astype(float))
        y=data[y_ouput]
    else:
        X = data[features]
        y = data[y_ouput]
    #######################
    
    #features of the algorithms
    features_of_algorithms = {'KNN':{'k':{}}, 'Tree':{"max_depth":{}},
                              'Logistic': {'C':{}, 'solver':{}}, 'SVM': {'kernel':{}}}

    
    for i in range(1, number_of_repeats+1):
    
        #split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        ###########

        #calculate KNN
        KNN(X_train, y_train, X_test, y_test, F1_score, negFal, posFal, negtru, postru, jac, features_of_algorithms, goal=goal, state=i)
        ##############

        #calculate Tree
        Tree(X_train, y_train, X_test, y_test, F1_score, negFal, posFal, negtru, postru, jac, features_of_algorithms, goal=goal, state=i)
        ##############

        #calculate logisticRegression
        Logistic(X_train, y_train, X_test, y_test, F1_score, negFal, posFal, negtru, postru, jac, features_of_algorithms, goal=goal, state=i)
        #############################

        #calculate SVM
        SVM(X_train, y_train, X_test, y_test, F1_score, negFal, posFal, negtru, postru, jac, features_of_algorithms, goal=goal, state=i)
        ##############
    
    ######calculate the best score of the all goal scores
    best_alg, best_state, best_score, features_output_list = pick_best_classification_algorithm(F1_score, jac, negFal, negtru, posFal,
                                       postru,state = number_of_repeats, goal = goal,
                                       features_of_algorithms = features_of_algorithms)
    print("the best %s score for %i states is: %f, for state %i, with algorithm %s with "
          %(goal, number_of_repeats, best_score, best_state, best_alg), end='')
    for i in range(len(features_output_list)):
        print(features_output_list[i], end='')
        if(i%2 == 0):
            print(": ", end='')
    print('.')
    print('------------')
    ####################################
    
    ####calculate the best algorithm base on the average of the goal score
    if(goal == 'F1-score'):
        dic_score = F1_score
    elif(goal == 'jac'):
        dic_score = jac
    elif(goal == 'negFal'):
        dic_score = negFal
    elif(goal == 'negtru'):
        dic_score = negtru
    elif(goal == 'posFal'):
        dic_score = posFal
    elif(goal == 'postru'):
        dic_score = postru
    
    avg_best_score, avg_best_alg = best_alg_base_average_scores(dic_score, goal, number_of_repeats)
    print("the best average of %s scores for %i states is: %f with algorithm %s." %(goal, number_of_repeats, avg_best_score, avg_best_alg))
    #######################################################################


def best_alg_base_average_scores(dic_score, goal, state):
    from sys import maxsize
    avg_best_score_max=0.0
    avg_best_score_min = maxsize
    avg_best_alg=None
    for(i, v) in dic_score.items():
        average = 0
        for(j, w) in v.items():
            average += w
        average /= state
        if(goal=='F1-score' or goal=='jac' or goal=='negtru' or goal=='postru'):
            if(average >= avg_best_score_max):
                avg_best_score_max = average
                avg_best_alg = i
        elif(goal=='negFal' or goal=='posFal'):
            if(average <= avg_best_score_min):
                avg_best_score_min = average
                avg_best_alg = i
    
    if(avg_best_score_max==0.0):
        avg_best_score= avg_best_score_min
    else:
        avg_best_score = avg_best_score_max
        
    return(avg_best_score, avg_best_alg)


def pick_best_classification_algorithm(F1_score, jac, negFal, negtru, posFal, postru, state, goal, features_of_algorithms):
    
    if(goal=='F1-score'):
        alg, state, score = max_pick(F1_score)
    if(goal=='jac'):
        alg, state, score = max_pick(jac)
    if(goal=='negtru'):
        alg, state, score = max_pick(negtru)
    if(goal=='postru'):
        alg, state, score = max_pick(postru)
    if(goal=='negFal'):
        alg, state, score = min_pick(negFal)
    if(goal=='posFal'):
        alg, state, score = min_pick(posFal)
    output_list=[]
    
    output_list=[]
    for(i, v) in features_of_algorithms.items():
        for(j, w) in v.items():
            for(k, x) in w.items():
                if(i==alg) and (k==state):
                    output_list.append(j)
                    output_list.append(x)
    
    return(alg, state, score, output_list)
        

def max_pick(dic_score):
    max_state=0
    max_score=0
    max_alg  =None
    for(i, v) in dic_score.items():
        for(j, w) in v.items():
            if(w>=max_score):
                max_score=w
                max_state=j
                max_alg  =i
    return(max_alg, max_state, max_score)


def min_pick(dic_score):
    from sys import maxsize
    min_state=0
    min_score=maxsize
    min_alg  =None
    for(i, v) in dic_score.items():
        for(j, w) in v.items():
            if(w<=min_score):
                min_score=w
                min_state=j
                min_alg  =i
    return(min_alg, min_state, min_score)


def SVM(X_train, y_train, X_test, y_test, F1_score, negFal, posFal, negtru, postru, jac, features_of_algorithms, goal, state):
    from sklearn import svm
    
    kernels={'linear', 'poly', 'rbf', 'sigmoid'}
    kernels_scores = {'linear': None, 'poly': None, 'rbf': None, 'sigmoid': None}
    for kernel in kernels:
        clf = svm.SVC(kernel=kernel)
        clf.fit(X_train, y_train)
        yhat = clf.predict(X_test)
        kernels_scores[kernel] = goal_cal_score(y_test, yhat, goal)
        
    best_kernel = max(kernels_scores, key=kernels_scores.get)
    clf = svm.SVC(kernel=best_kernel)
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    
    features_of_algorithms['SVM']['kernel'][state] = best_kernel
    
    F1_score["SVM"][state], jac["SVM"][state], negFal["SVM"][state], negtru["SVM"][state], posFal["SVM"][state], postru["SVM"][state] = cal_scores(y_test, yhat)
    
    
def Logistic(X_train, y_train, X_test, y_test, F1_score, negFal, posFal, negtru, postru, jac, features_of_algorithms, goal, state):
    from sklearn.linear_model import LogisticRegression
    
    algs = {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}
    scores = {'newton-cg':{}, 'lbfgs':{}, 'liblinear':{}, 'sag':{}, 'saga':{}}
    for alg in algs:
        for i in np.arange(0.1, 3, 0.1):
            LR = LogisticRegression(C=i, solver=alg)
            LR.fit(X_train, y_train)
            yhat = LR.predict(X_test)
            scores[alg][i] = goal_cal_score(y_test, yhat, goal)
    
    ################
    max_score = 0
    best_alg=None
    best_C = 0
    for (i, v) in scores.items():
        for(j, w) in v.items():
            score = w
            if(score >= max_score):
                max_score = score
                best_alg = i
                best_C = j
    ##################
    
    LR = LogisticRegression(C=best_C, solver=best_alg)
    LR.fit(X_train, y_train)
    yhat = LR.predict(X_test)
    
    features_of_algorithms['Logistic']['C'][state] = best_C
    features_of_algorithms['Logistic']['solver'][state] = best_alg
    
    F1_score["LogisticRegression"][state], jac["LogisticRegression"][state], negFal["LogisticRegression"][state], negtru["LogisticRegression"][state], posFal["LogisticRegression"][state], postru["LogisticRegression"][state] = cal_scores(y_test, yhat)
    
    
def KNN(X_train, y_train, X_test, y_test, F1_score, negFal, posFal, negtru, postru, jac, features_of_algorithms, goal, state):
    
    from sklearn.neighbors import KNeighborsClassifier

    scores = {}
    for k in range(2, 20):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh = neigh.fit(X_train, y_train)
        yhat=neigh.predict(X_test)
        #scores[k] = metrics.accuracy_score(y_test, yhat)
        scores[k] = goal_cal_score(y_test, yhat, goal)
        
    max_k = max(scores, key=scores.get)
    neigh = KNeighborsClassifier(n_neighbors=max_k)
    neigh = neigh.fit(X_train, y_train)
    yhat=neigh.predict(X_test)
    
    features_of_algorithms['KNN']['k'][state] = max_k
    
    F1_score["KNN"][state], jac["KNN"][state], negFal["KNN"][state], negtru["KNN"][state], posFal["KNN"][state], postru["KNN"][state] = cal_scores(y_test, yhat)
    

def Tree(X_train, y_train, X_test, y_test, F1_score, negFal, posFal, negtru, postru, jac, features_of_algorithms, goal, state):
    
    from sklearn.tree import DecisionTreeClassifier

    scores = {}
    for max_d in range(2, len(X_train)+1):
        outputTree = DecisionTreeClassifier(criterion='entropy', max_depth=max_d)
        outputTree = outputTree.fit(X_train, y_train)
        yhat=outputTree.predict(X_test)
        scores[max_d] = goal_cal_score(y_test, yhat, goal)
        
    max_d_best_score = max(scores, key=scores.get)
    outputTree = DecisionTreeClassifier(criterion='entropy', max_depth=max_d_best_score)
    outputTree = outputTree.fit(X_train, y_train)
    yhat=outputTree.predict(X_test)
    
    features_of_algorithms['Tree']['max_depth'][state] = max_d_best_score
    
    F1_score["DecisionTree"][state], jac["DecisionTree"][state], negFal["DecisionTree"][state], negtru["DecisionTree"][state], posFal["DecisionTree"][state], postru["DecisionTree"][state] = cal_scores(y_test, yhat)


def cal_scores(y_test, yhat):
    
    from sklearn import metrics

    #F1-score
    F1_score = metrics.f1_score(y_test, yhat, average='weighted')
    #########
    
    #jaccard-score
    jac = metrics.jaccard_score(y_test, yhat, pos_label=0)
    #########
    
    #negative-False
    negFal = metrics.confusion_matrix(y_test, yhat, labels=[1,0])[1][0]
    ###############
    
    #negative-True
    negtru = metrics.confusion_matrix(y_test, yhat, labels=[1, 0])[1][1]
    ##############
    
    #positive-False
    posFal = metrics.confusion_matrix(y_test, yhat, labels=[1, 0])[0][1]
    ##############
    
    #positive_True
    postru = metrics.confusion_matrix(y_test, yhat, labels=[1, 0])[0][0]
    ##############
    
    return(F1_score, jac, negFal, negtru, posFal, postru)


def goal_cal_score(y_test, yhat, goal):
    
    from sklearn import metrics

    #F1-score
    F1_score = metrics.f1_score(y_test, yhat, average='weighted')
    #########
    
    #jaccard-score
    jac = metrics.jaccard_score(y_test, yhat, pos_label=0)
    #########
    
    #negative-False
    negFal = metrics.confusion_matrix(y_test, yhat, labels=[1,0])[1][0]
    if(negFal>0):
        negFal = 1/negFal
    elif(negFal==0):
        negFal = 1
    ###############
    
    #negative-True
    negtru = metrics.confusion_matrix(y_test, yhat, labels=[1, 0])[1][1]
    ##############
    
    #positive-False
    posFal =metrics.confusion_matrix(y_test, yhat, labels=[1, 0])[0][1]
    if(posFal > 0):
        posFal = 1/posFal
    elif(posFal == 0):
        posFal = 1
    ##############
    
    #positive_True
    postru = metrics.confusion_matrix(y_test, yhat, labels=[1, 0])[0][0]
    ##############
    
    if(goal == 'F1-score'):
        return F1_score
    elif(goal == 'jac'):
        return jac
    elif(goal == 'negFal'):
        return negFal
    elif(goal == 'negtru'):
        return negtru
    elif(goal == 'posFal'):
        return posFal
    elif(goal == 'postru'):
        return postru
    