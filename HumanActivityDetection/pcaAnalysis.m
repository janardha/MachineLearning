clc;
close all;
clear all;


feat_set = load('X_train.txt');
class_set = load('y_train.txt');

feat_test_set = load('X_test_L.txt');
class_test_set = load('y_test_L.txt'); 

a = dataset(feat_set, class_set);

testSet = dataset(feat_test_set, class_test_set);

w2 = qdc([],0,0);

for N = 1:size(a,2)-1
    
    pcaMapp = pca(a,N);
    redTrainSet = a*pcaMapp;
    V1 = qdc(redTrainSet,0,0);

    V2 = perlc(redTrainSet);
    V = {V1,V2};
    
    redTestSet = testSet*pcaMapp;
    [err(N,:), cnum(N,:)]  = testc(redTestSet,V);
    
end

errorVal = cell2mat(err);
plot(errorVal);

[m1 , idx1] = min(errorVal(:,1));
[m2 , idx2] = min(errorVal(:,2));
