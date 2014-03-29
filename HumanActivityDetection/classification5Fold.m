clc;
close all;
clear all;
 
 
feat_set = load('X_train.txt');
class_set = load('y_train.txt');
 
% feat_test_set = load('X_test_L.txt');
% class_test_set = load('y_test_L.txt'); 
 
a = dataset(feat_set, class_set);
 
% testSet = dataset(feat_test_set, class_test_set);
 
divRatio = floor(0.2 * size(a,1));
 
trainSeg1 = a(1:divRatio,:);
trainSeg2 = a(divRatio:2*divRatio-1,:);
trainSeg3 = a(2*divRatio:3*divRatio-1,:);
trainSeg4 = a(3*divRatio:4*divRatio-1,:);
trainSeg5 = a(4*divRatio:5*divRatio-1,:);
 
train1 = [trainSeg2;trainSeg3;trainSeg4;trainSeg5];
train2 = [trainSeg1;trainSeg3;trainSeg4;trainSeg5];
train3 = [trainSeg1;trainSeg2;trainSeg4;trainSeg5];
train4 = [trainSeg1;trainSeg2;trainSeg3;trainSeg5];
train5 = [trainSeg1;trainSeg2;trainSeg3;trainSeg4];
% %% training 1
% 
% % for N = 1:size(a,2)-1
% % V1 = fisherc(a);
% N = 2;
% 
%     V2 = klldc(a, N);
%     V3 = pcldc(a, N);
%     [V4, K, E] = knnc(a, N);
%     % [V5, H] = parzenc(train1);
% 
%     % testp();%test parzen
%     V6 = pca(a,N)*qdc;
%     %perceptron linear
%     V6 = a * V6;
% %     V6 = w1'*w2;
%     % kernelc(); %kernel based
%     V = { V2, V3, V6};
%     [err(N,:), cnum(N,:)]  = testc(testSet,V);
% % end
%% seg 1
 
V1 = fisherc(train1);
[V2, H] = parzenc(train1);
V3 = perlc(train1);%perceptron linear
 
% kernelc(); %kernel based
V = {V1, V2, V3};
[err, cnum] = testc(trainSeg1,V);
err1 = cell2mat(err);
for N = 1:size(a,2)-1
    
    V4 = klldc(train1, N);
    V5 = pcldc(train1, N);
    [V6, K, E] = knnc(train1,N);
    V = {V4, V5, V6};
    [errItr(N,:), cnumItr(N,:)] = testc(trainSeg1,V);
    
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
%% Seg2
V1 = fisherc(train2);
[V2, H] = parzenc(train2);
V3 = perlc(train2);%perceptron linear
 
% kernelc(); %kernel based
V = {V1, V2, V3};
[err, cnum] = testc(trainSeg2,V);
 err2 = cell2mat(err);
for N = 1:2:size(a,2)-1
    
    V4 = klldc(train2, N);
    V5 = pcldc(train2, N);
    [V6, K, E] = knnc(train2,N);
    V = {V4, V5, V6};
    [errItr(N,:), cnumItr(N,:)] = testc(trainSeg2,V);
    
end
figure(2);
e2 = cell2mat(errItr);
plot(e2);
title('KL,PCA and KNN error vs reduceed feature set');
xlabel('N');
ylabel('Error');
legend('KLLDC', 'PCLDC', 'KNNC');
[minVal2(1),idx2(1)] = min(e2(:,1));
[minVal2(2),idx2(2)] = min(e2(:,2));
[minVal2(3),idx2(3)] = min(e2(:,3));
%% Seg3
V1 = fisherc(train3);
[V2, H] = parzenc(train3);
V3 = perlc(train3);%perceptron linear
 
% kernelc(); %kernel based
V = {V1, V2, V3};
[err, cnum] = testc(trainSeg3,V);
err3 = cell2mat(err);
for N = 1:2:size(a,2)-1
    
    V4 = klldc(train3, N);
    V5 = pcldc(train3, N);
    [V6, K, E] = knnc(train3,N);
    V = {V4, V5, V6};
    [errItr(N,:), cnumItr(N,:)] = testc(trainSeg3,V);
    
end
figure(3);
e3 = cell2mat(errItr);
plot(e3);
title('KL,PCA and KNN error vs reduceed feature set');
xlabel('N');
ylabel('Error');
legend('KLLDC', 'PCLDC', 'KNNC');
[minVal3(1),idx3(1)] = min(e3(:,1));
[minVal3(2),idx3(2)] = min(e3(:,2));
[minVal3(3),idx3(3)] = min(e3(:,3));
%% Seg4
V1 = fisherc(train4);
[V2, H] = parzenc(train4);
V3 = perlc(train4);%perceptron linear
 
% kernelc(); %kernel based
V = {V1, V2, V3};
[err, cnum] = testc(trainSeg4,V);
 err4 = cell2mat(err);
for N = 1:2:size(a,2)-1
    
    V4 = klldc(train4, N);
    V5 = pcldc(train4, N);
    [V6, K, E] = knnc(train4,N);
    V = {V4, V5, V6};
    [errItr(N,:), cnumItr(N,:)] = testc(trainSeg4,V);
    
end
figure(4);
e4 = cell2mat(errItr);
plot(e4);
title('KL,PCA and KNN error vs reduceed feature set');
xlabel('N');
ylabel('Error');
legend('KLLDC', 'PCLDC', 'KNNC');
[minVal4(1),idx4(1)] = min(e4(:,1));
[minVal4(2),idx4(2)] = min(e4(:,2));
[minVal4(3),idx4(3)] = min(e4(:,3));
%% Seg5
V1 = fisherc(train5);
[V2, H] = parzenc(train5);
V3 = perlc(train5);%perceptron linear
 
% kernelc(); %kernel based
V = {V1, V2, V3};
[err, cnum] = testc(trainSeg5,V);
 err5 = cell2mat(err);
for N = 1:2:size(a,2)-1
    
    V4 = klldc(train5, N);
    V5 = pcldc(train5, N);
    [V6, K, E] = knnc(train5,N);
    V = {V4, V5, V6};
    [errItr(N,:), cnumItr(N,:)] = testc(trainSeg5,V);
    
end
figure(5);
e5 = cell2mat(errItr);
plot(e5);
title('KL,PCA and KNN error vs reduceed feature set');
xlabel('N');
ylabel('Error');
legend('KLLDC', 'PCLDC', 'KNNC');
[minVal5(1),idx5(1)] = min(e5(:,1));
[minVal5(2),idx5(2)] = min(e5(:,2));
[minVal5(3),idx5(3)] = min(e5(:,3));