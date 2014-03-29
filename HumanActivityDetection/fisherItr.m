clc;
close all;
clear all;

feat_set = load('X_train.txt');
class_set = load('y_train.txt');

feat_test_set = load('X_test_L.txt');
class_test_set = load('y_test_L.txt'); 

a = dataset(feat_set, class_set);

testSet = dataset(feat_test_set, class_test_set);

for N = 1:5
    
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
   
    testFld = testSet*w8;
    testnFld = testSet*w9;
    [err(N,:), cnum(N,:)]  = testc(testFld, V1);
    [err1(N,:), cnum1(N,:)]  = testc(testnFld, V2);
end
figure(1);
e1 = cell2mat(err);
plot(e1)
title('Fisher liner mapping error vs reduced number of features');
xlabel('N');
ylabel('Error');
figure(2);
e2 = cell2mat(err1);
plot(e2)
title('Fisher non-linear mapping error vs reduced number of features');
xlabel('N');
ylabel('Error');

[min_val(1), indx(1)] = min(e1(:,1));
[min_val(2), indx(2)] = min(e1(:,2));
[min_val(3), indx(3)] = min(e1(:,3));
[min_val1(1), indx1(1)] = min(e2(:,1));
[min_val1(2), indx1(2)] = min(e2(:,2));
[min_val1(3), indx1(3)] = min(e2(:,3));
