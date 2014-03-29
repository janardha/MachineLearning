clc;
close all;
clear all;

feat_set = load('X_train.txt');
class_set = load('y_train.txt');

feat_test_set = load('X_test_L.txt');
class_test_set = load('y_test_L.txt'); 

a = dataset(feat_set, class_set);

testSet = dataset(feat_test_set, class_test_set);

% [b,c] = gendat(a,0.7);
%% bayes baseline
% divRatio = floor(0.2 * size(a,1));
% 
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
% 
%     w2 = qdc([],0,0); %[w,r,s,m] = qdc(A,R,S,M)/ w = A*qdc([],r,s) R>=0,S<=1
%     w2 = setname(w2,'qdc');

 for N = 1:5
    
%     w3 = klm([],N)*ldc;
%     w3 = setname(w3,'klm-ldc');
    
%     w4 = klm([],N)*w2;
%     w4 = setname(w4,'klm-qdc');
%     
%     w5 = affline([],N)*w2;
%     w5 = setname(w5,'affine-qdc');
    
%     w6 = pca([],N)*w2;
%     w6 = setname(w6,'pca-qdc');
%     
%     w7 = knnm(train5,N);
%     w7 = setname(w7,'knnm-qdc');
%     
        w8 = fisherm(a,N, 0.95);
        wfld = a*w8;
        w8 = setname(w8,'fisherm');
        cfld_percep = perlc(wfld);
        cfld_qdc = qdc(wfld,0,0);
        cfld_ldc = ldc(wfld);
        
        w9 = nlfisherm(a,N);
        wnfld = a*w9;
        w9 = setname(w9,'nlfisherm');
        
        cnfld_percep = perlc(wnfld);
        cnfld_qdc = qdc(wnfld,0,0);
        cnfld_ldc = ldc(wnfld);
        
    V1 = {cfld_percep, cfld_qdc, cfld_ldc};
    V2 = {cnfld_percep, cnfld_qdc, cnfld_ldc};
   
%     disp([newline 'Errors for individual classifiers']);
    testFld = testSet*w8;
    testnFld = testSet*w9;
    [err(N,:), cnum(N,:)]  = testc(testFld, V1);
    [err1(N,:), cnum1(N,:)]  = testc(testnFld, V2);
%     meanErr = meanErr + err(i,:);
 end
 
 %n=142, accuracy = 92%
%  plot(err);
% 
% e1 = cell2mat(err(:,1));
% 
% maxVal = zeros(5,1);
% 
% for idx = 1:5
%     for i = 1:size(e1,1)
%         if maxVal(idx) < e1(i,idx)
%             maxVal(idx) = e1(i,idx);
%             maxidx(idx) = i;
%         end
%     end
% end
% 
% W_pca = pca([],2)*w2;
% v_pca = train1*W_pca;
% 
% testc(trainSeg5, V);
% 
% scatterd(v_pca);
% plotc(W_pca');
