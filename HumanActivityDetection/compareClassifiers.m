clc;
close all;
clear all;


feat_set = load('X_train.txt');
class_set = load('y_train.txt');

feat_test_set = load('X_test_L.txt');
class_test_set = load('y_test_L.txt'); 

a = dataset(feat_set, class_set);

testSet = dataset(feat_test_set, class_test_set);

% divRatio = floor(0.2 * size(a,1));

% trainSeg1 = a(1:divRatio,:);
% trainSeg2 = a(divRatio:2*divRatio-1,:);
% trainSeg3 = a(2*divRatio:3*divRatio-1,:);
% trainSeg4 = a(3*divRatio:4*divRatio-1,:);
% trainSeg5 = a(4*divRatio:5*divRatio-1,:);
% 
% train1 = [trainSeg2;trainSeg3;trainSeg4;trainSeg5];
% train2 = [trainSeg1;trainSeg3;trainSeg4;trainSeg5];
% train3 = [trainSeg1;trainSeg2;trainSeg4;trainSeg5];
% train4 = [trainSeg1;trainSeg2;trainSeg3;trainSeg5];
% train5 = [trainSeg1;trainSeg2;trainSeg3;trainSeg4];
%% training 1

% for N = 1:size(a,2)-1
% V1 = fisherc(a);
N = 2;

    V2 = klldc(a, N);
    V3 = pcldc(a, N);
    % [V4, K, E] = knnc(a);
    % [V5, H] = parzenc(train1);

    % testp();%test parzen
    V6 = pca(a,N)*qdc;
    %perceptron linear
    V6 = a * V6;
%     V6 = w1'*w2;
    % kernelc(); %kernel based
    V = { V2, V3, V6};
    [err(N,:), cnum(N,:)]  = testc(testSet,V);
% end
% %% seg 1
% 
% V1 = fisherc(train1);
% V2 = klldc(train1, 0.95);
% V3 = pcldc(train1, 0.95);
% [V4, K, E] = knnc(train1);
% [V5, H] = parzenc(train1);
% 
% % testp();%test parzen
% 
% V6 = perlc(train1);%perceptron linear
% 
% % kernelc(); %kernel based
% V = {V1, V2, V3, V4, V5, V6};
% testc(trainSeg1,V);
% testc(testSet,V);
% 
% %% Seg2
% V1 = fisherc(train2);
% V2 = klldc(train2, 0.95);
% V3 = pcldc(train2, 0.95);
% [V4, K, E] = knnc(train2);
% [V5, H] = parzenc(train2);
% 
% % testp();%test parzen
% 
% V6 = perlc(train2);%perceptron linear
% 
% % kernelc(); %kernel based
% V = {V1, V2, V3, V4, V5, V6};
% testc(trainSeg2,V);
% testc(testSet,V);
% 
% %% seg 3
% 
% V1 = fisherc(train3);
% V2 = klldc(train3, 0.95);
% V3 = pcldc(train3, 0.95);
% [V4, K, E] = knnc(train3);
% [V5, H] = parzenc(train3);
% 
% % testp();%test parzen
% 
% V6 = perlc(train3);%perceptron linear
% 
% % kernelc(); %kernel based
% V = {V1, V2, V3, V4, V5, V6};
% testc(trainSeg3,V);
% testc(testSet,V);
% 
% %% seg4
% 
% V1 = fisherc(train4);
% V2 = klldc(train4, 0.95);
% V3 = pcldc(train4, 0.95);
% [V4, K, E] = knnc(train4);
% [V5, H] = parzenc(train4);
% 
% % testp();%test parzen
% 
% V6 = perlc(train4);%perceptron linear
% 
% % kernelc(); %kernel based
% V = {V1, V2, V3, V4, V5, V6};
% testc(trainSeg4,V);
% testc(testSet,V);
% 
% %% seg 5
% 
% V1 = fisherc(train5);
% V2 = klldc(train5, 0.95);
% V3 = pcldc(train5, 0.95);
% [V4, K, E] = knnc(train5);
% [V5, H] = parzenc(train5);
% 
% % testp();%test parzen
% 
% V6 = perlc(train5);%perceptron linear
% 
% % kernelc(); %kernel based
% V = {V1, V2, V3, V4, V5, V6};
% testc(trainSeg5,V);
% testc(testSet,V);