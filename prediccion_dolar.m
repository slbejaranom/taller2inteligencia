clear,
close all,
clc,
rng("shuffle");                                                                                                                                           
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
porcentaje_validacion = 0.2;
[filas, ~] = size(dataset);

X_train = dataset(1:round(filas*(1-porcentaje_validacion)),1:4);
Y_train = dataset(1:round(filas*(1-porcentaje_validacion)),5);

X_test = dataset(round(filas*(1-porcentaje_validacion))-1:end,1:4);
Y_test = dataset(round(filas*(1-porcentaje_validacion))-1:end,5);

Fechas_train = preciodolarcolombia.Fechas(1:round(filas*(1-porcentaje_validacion)));
Fechas_test = preciodolarcolombia.Fechas(round(filas*(1-porcentaje_validacion))-1:end);

net = feedforwardnet([10 10]);
net = train(net,X_train',Y_train');
salidas_pred = sim(net,X_train')';
view(net);
figure,
plot(preciodolarcolombia.Fechas,preciodolarcolombia.Precio);
hold on;
plot(Fechas_train,salidas_pred);
salidas_pred = sim(net,X_test')';
plot(Fechas_test,salidas_pred);
legend("Precio del dolar real", "Precio del dolar predicho en entrenamiento", "Precio del dolar predicho en test");