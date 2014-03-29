This project is implemented in Matlab
The algorithms are from Classification toolbox for Matlab.
The X_train.txt y_train.txt X_test_L.txt y_test_L.txt have to be placed in the same folder as the programs.

The bayes_baseline.m file executes Bayes linear and quadratic classification on the training data as well 
as test data and also performs 5-fold cross validation of the bayes classifier. It generates confusion matrix for 
each case of linear and quadratic classifier.

The classification5fold.m file executes 5 fold cross validation of different classifers.

The compareClassifiers.m file compares varios classiifers

The fisherItr.m file iteratively determines classification error for various reduced features using fisher mapping.

The klmItr.m file iteratively determines classification error for various reduced features using klm mapping.

The nflfisherItr.m file iteratively determines classification error for various reduced features using nonlinear mapping. and also works on X_test_U.txt file

The pcaAnalysi.m file iteratively determines classification error for various reduced features using pca mapping.

The perceptron_ets.m file iteratively determines classification error for various eta using perceptron.

The test5fold.m file iteratively determines classification error various reduced features in different classifiers.


OUTPUT:

The mapping of the X_test_U.txt is given by predictedMapping.mat
The predicted labels are given by PredLabel.mat
