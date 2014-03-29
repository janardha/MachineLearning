clc;
close all;
clear all;


feat_set = load('X_train.txt');
class_set = load('y_train.txt');
 
a = dataset(feat_set, class_set);

feat_set = load('X_test_L.txt');
class_test_set = load('y_test_L.txt');
 
testSet = dataset(feat_set, class_test_set);
%% bayes baseline
% meanErr = cell(0,0);
%  for i = 1:5
%     [b,c] = gendat(a,0.8);
%     
%     w1 = ldc(b);
%     w1 = setname(w1,'ldc');
% 
%     w2 = b*qdc([],0,0); %[w,r,s,m] = qdc(A,R,S,M)/ w = A*qdc([],r,s) R>=0,S<=1
%     w2 = setname(w2,'qdc');
% 
%     V = {w1,w2};
% 
%     disp([newline 'Errors for individual classifiers']);
%     [err(i,:), cnum(i,:)] = testc(c, V)
% %     meanErr = meanErr + err(i,:);
%  end
 
%  meanErr = meanErr/5;
%  
%  sequential data
%  
% meanSeqErr = cell(0,0);
divRatio = floor(0.2 * size(a,1));

trainSeg1 = a(1:divRatio,:);
trainSeg2 = a(divRatio:2*divRatio-1,:);
trainSeg3 = a(2*divRatio:3*divRatio-1,:);
trainSeg4 = a(3*divRatio:4*divRatio-1,:);
trainSeg5 = a(4*divRatio:5*divRatio-1,:);

trainldc1 = ldc([trainSeg2;trainSeg3;trainSeg4;trainSeg5]);
trainldc2 = ldc([trainSeg1;trainSeg3;trainSeg4;trainSeg5]);
trainldc3 = ldc([trainSeg1;trainSeg2;trainSeg4;trainSeg5]);
trainldc4 = ldc([trainSeg1;trainSeg2;trainSeg3;trainSeg5]);
trainldc5 = ldc([trainSeg1;trainSeg2;trainSeg3;trainSeg4]);

trainqdc1 = qdc([trainSeg2;trainSeg3;trainSeg4;trainSeg5],0,0);
trainqdc2 = qdc([trainSeg1;trainSeg3;trainSeg4;trainSeg5],0,0);
trainqdc3 = qdc([trainSeg1;trainSeg2;trainSeg4;trainSeg5],0,0);
trainqdc4 = qdc([trainSeg1;trainSeg2;trainSeg3;trainSeg5],0,0);
trainqdc5 = qdc([trainSeg1;trainSeg2;trainSeg3;trainSeg4],0,0);

V1 = {trainldc1,trainqdc1};
V2 = {trainldc2,trainqdc2};
V3 = {trainldc3,trainqdc3};
V4 = {trainldc4,trainqdc4};
V5 = {trainldc5,trainqdc5};

disp([newline 'Errors for individual classifiers']);
[err_seq(1,:), cnum(1,:)] = testc(trainSeg1, V1)
% meanSeqErr = meanSeqErr + err(1,:);
[err_seq(2,:), cnum(2,:)] = testc(trainSeg2, V2)
% meanSeqErr = meanSeqErr + err(2,:);
[err_seq(3,:), cnum(3,:)] = testc(trainSeg3, V3)
% meanSeqErr = meanSeqErr + err(3,:);
[err_seq(4,:), cnum(4,:)] = testc(trainSeg4, V4)
% meanSeqErr = meanSeqErr + err(4,:);
[err_seq(5,:), cnum(5,:)] = testc(trainSeg5, V5)
% meanSeqErr = meanSeqErr + err(5,:);

% meanSeqErr = meanSeqErr/5;
trainldc = ldc(a);
trainqdc = a*qdc;
v = {trainldc,trainqdc};
[e, c] = testc(testSet, v);

lab1 = getlabels(testSet);
lab2 = labeld(testSet,trainldc);
lab3 = labeld(testSet,trainqdc);
fid = fopen('linearconfmat.txt','w+');
confMatLdc = confmat(lab1,lab2, 'count',fid);
fclose(fid);

fid = fopen('quadconfmat.txt','w+');
confMatQdc = confmat(lab1,lab3, 'count',fid);
fclose(fid);