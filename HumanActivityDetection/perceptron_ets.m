clc;
close all;
clear all;


feat_set = load('X_train.txt');
class_set = load('y_train.txt');

feat_test_set = load('X_test_L.txt');
class_test_set = load('y_test_L.txt'); 

a = dataset(feat_set, class_set);

testSet = dataset(feat_test_set, class_test_set);

w_init = 0;
N = 1;
for eta = 0.1:0.1:5
    V = perlc(a,100,eta);
    [err(N,:), cnum(N,:)]  = testc(testSet,V);
    N=N+1;
end
% lab1 = getlabels(testSet);
% lab2 = labeld(testSet,V);
% fid = fopen('perceptroncnf.txt','w+');
% confMatLdc = confmat(lab1,lab2, 'count',fid);
% fclose(fid);
stem(err);
title('Perceptron classification error vs learning rate ')
xlabel('eta, Learning rate');
ylabel('Error')
[minVal,idx] = min(err)