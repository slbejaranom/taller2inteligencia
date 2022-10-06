clear,
close all,
clc,
rng(5);                                                                                                                                           
%% Generación del dataset
load("precio_dolar.mat");
dataset = zeros(length(preciodolarcolombia.Precio),5);
for i=1:1:length(preciodolarcolombia.Precio)
    if(i-1 < 1)
        valor_1 = 0;
    else
        valor_1 = preciodolarcolombia.Precio(i-1);
    end
    if(i-2 < 1)
        valor_2 = 0;
    else
        valor_2 = preciodolarcolombia.Precio(i-2);
    end
    if(i-3 < 1)
        valor_3 = 0;
    else
        valor_3 = preciodolarcolombia.Precio(i-3);
    end
    if(i-4 < 1)
        valor_4 = 0;
    else
        valor_4 = preciodolarcolombia.Precio(i-4);
    end
    salida = preciodolarcolombia.Precio(i);
    dataset(i,:) = [valor_4 valor_3 valor_2 valor_1 salida];
end
%% Preparación del dataset
[filas, columnas] = size(dataset);
porcentaje_validacion = 0.2;
porcentaje_test = 0.1;

% Le hacemos shuffle al dataset
shuffled_dataset = shuffle_dataset(dataset);

[X_train,Y_train,X_test,Y_test,X_val,Y_val] = split_dataset(porcentaje_test,porcentaje_validacion,dataset);


%Generación de la red
capas = 1;
neuronas = 30;
netarch = neuronas*ones(1,capas);
net = feedforwardnet(netarch);

%Configuraciones de la red
%Usamos todo el dataset provisto a la red para entrenar dado que los de
%validación y test los tenemos en otras variables y matlab no permite
%proveerlos a través de este toolbox
net.divideParam.trainRatio = 1;
%Configuraciones de las capas
% Coloque en la consola help nntransfer para ver las funciones que hay
funcion_capas_ocultas = "poslin";
for i = 1:1:capas
    net.layers{i}.transferFcn = funcion_capas_ocultas;
end
%LA ULTIMA CAPA SIEMPRE ES LA DE SALIDA
net.layers{capas+1}.transferFcn = "purelin";
%FIN Configuraciones de la red

%Entrenamiento
[net,errTrain,errVal,errTest] = entrenar(net,X_train,Y_train,X_test,Y_test,X_val,Y_val);
%ESPECIFICO DE ESTE PROBLEMA PARA VISUALIZACIÓN
Fechas_train = preciodolarcolombia.Fechas(1:round(filas*(1-porcentaje_validacion-porcentaje_test)));
Fechas_test = preciodolarcolombia.Fechas(round(filas*(1-porcentaje_validacion-porcentaje_test))-1:round(filas*(1-porcentaje_test)));
Fechas_val = preciodolarcolombia.Fechas(round(filas*(1-porcentaje_test))-1:end);
salidas_pred = sim(net,X_train')';
view(net);
figure,
plot(preciodolarcolombia.Fechas,preciodolarcolombia.Precio);
hold on;
plot(Fechas_train,salidas_pred);
salidas_pred = sim(net,X_test')';
NASH = NSE(Y_test',salidas_pred');
plot(Fechas_test,salidas_pred);
salidas_pred = sim(net,X_val');
plot(Fechas_val,salidas_pred)
legend("Precio del dolar real", "Precio del dolar predicho en entrenamiento", "Precio del dolar predicho en test", "Precio del dolar predicho en validación");
%FIN VISUALIZACIÓN