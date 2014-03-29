clc;
close all;
clear all;


feat_set = load('X_train.txt');
class_set = load('y_train.txt');

feat_test_set = load('X_test_L.txt');
class_test_set = load('y_test_L.txt'); 

a = dataset(feat_set, class_set);

testSet = dataset(feat_test_set, class_test_set);

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


N = 5;

nlfMapp = nlfisherm(train1,N);
redTrainSet = train1*nlfMapp;
V = qdc(redTrainSet,0,0);

redTestSet = trainSeg1*nlfMapp;
[err1, cnum]  = testc(redTestSet,V);

nlfMapp = nlfisherm(train2,N);
redTrainSet = train2*nlfMapp;
V = qdc(redTrainSet,0,0);

redTestSet = trainSeg2*nlfMapp;
[err2, cnum]  = testc(redTestSet,V);

nlfMapp = nlfisherm(train3,N);
redTrainSet = train3*nlfMapp;
V = qdc(redTrainSet,0,0);

redTestSet = trainSeg3*nlfMapp;
[err3, cnum]  = testc(redTestSet,V);

nlfMapp = nlfisherm(train4,N);
redTrainSet = train4*nlfMapp;
V = qdc(redTrainSet,0,0);

redTestSet = trainSeg4*nlfMapp;
[err4, cnum]  = testc(redTestSet,V);

nlfMapp = nlfisherm(train5,N);
redTrainSet = train5*nlfMapp;
V = qdc(redTrainSet,0,0);

redTestSet = trainSeg5*nlfMapp;
[err5, cnum]  = testc(redTestSet,V);

%% test and conf matrix
nlfMapp = nlfisherm(a,N);
redTrainSet = a*nlfMapp;
V = qdc(redTrainSet,0,0);

redTestSet = testSet*nlfMapp;
[err(N,:), cnum(N,:)]  = testc(redTestSet,V);
    
lab1 = getlabels(testSet);
lab2 = labeld(redTestSet,V);
fid = fopen('nlfisherQdc.txt','w+');
confMatLdc = confmat(lab1,lab2, 'count',fid);
fclose(fid);
