function [net,errTrain,errVal,errTest] = entrenar(net,X_train,Y_train,X_test,Y_test,X_val,Y_val)
%ENTRENAR Summary of this function goes here
%   Detailed explanation goes here
    net = train(net,X_train',Y_train');
    salidas_pred = sim(net,X_train')';
    errTrain = immse(salidas_pred,Y_train);
    salidas_pred = sim(net,X_test')';
    errTest = immse(salidas_pred,Y_test);
    salidas_pred = sim(net,X_val')';
    errVal = immse(salidas_pred,Y_val);
end

