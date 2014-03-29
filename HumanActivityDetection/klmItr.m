clc;
close all;
clear all;

feat_set = load('X_train.txt');
class_set = load('y_train.txt');

feat_test_set = load('X_test_L.txt');
class_test_set = load('y_test_L.txt'); 

a = dataset(feat_set, class_set);

testSet = dataset(feat_test_set, class_test_set);

for N = 1:size(a,2)-1
    
    [w,fr] = klm(a,N);
    wfld = a*w;
    w = setname(w,'fisherm');
    cklm_percep = perlc(wfld);
    cklm_qdc = qdc(wfld,0,0);
    cklm_ldc = ldc(wfld);
    
    V1 = {cklm_percep, cklm_qdc, cklm_ldc};
   
    testklm = testSet*w;
    [err(N,:), cnum(N,:)]  = testc(testklm, V1);
end
figure(1);
e1 = cell2mat(err);
plot(e1)
title('Fisher liner mapping error vs reduced number of features');
xlabel('N');
ylabel('Error');

[min_val(1), indx(1)] = min(e1(:,1));
[min_val(2), indx(2)] = min(e1(:,2));
[min_val(3), indx(3)] = min(e1(:,3));