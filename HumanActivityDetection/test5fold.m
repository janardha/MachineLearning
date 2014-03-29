clc;
close all;
clear all;
 
 
feat_set = load('X_train.txt');
class_set = load('y_train.txt');
 
feat_test_set = load('X_test_L.txt');
class_test_set = load('y_test_L.txt'); 
 
a = dataset(feat_set, class_set);
 
testSet = dataset(feat_test_set, class_test_set);

V1 = fisherc(a);
V3 = perlc(a);%perceptron linear
 
% kernelc(); %kernel based
V = {V1, V3};
[err, cnum] = testc(testSet,V);
err1 = cell2mat(err);
for N = 1:size(a,2)-1
    
    V4 = klldc(a, N);
    V5 = pcldc(a, N);
    [V6, K, E] = knnc(a,N);
    V = {V4, V5, V6};
    [errItr(N,:), cnumItr(N,:)] = testc(testSet,V);
    
end
figure(1);
e1 = cell2mat(errItr);
plot(e1);
title('KL,PCA and KNN error vs reduceed feature set');
xlabel('N');
ylabel('Error');
legend('KLLDC', 'PCLDC', 'KNNC');
[minVal1(1),idx1(1)] = min(e1(:,1));
[minVal1(2),idx1(2)] = min(e1(:,2));
[minVal1(3),idx1(3)] = min(e1(:,3));