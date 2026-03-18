%% PINN Ej (a)  y' = y,  y(0)=1
clear; clc; close all;
%% 1. Definición del problema
% EDO: y' = y, y(0) = 1, x ∈ [0,2]
x_span = [0 2];
y0 = 1;
%% 2. Arquitectura PINN
% Red simple suficiente para función suave exponencial
layers = [
   featureInputLayer(1,'Name','input')
   fullyConnectedLayer(10,'Name','fc1')
   tanhLayer('Name','tanh1')
   fullyConnectedLayer(1,'Name','output')
];
pinn = dlnetwork(layers);
%% 3. Parámetros de entrenamiento
numEpochs = 2500;
learningRate = 0.01;
gifFilename = 'PINN_exponencial.gif';
x_physics = dlarray(linspace(x_span(1),x_span(2),200),'CB');  % Puntos para PINN
x0_dl = dlarray(0,'CB');    % Condición inicial
y0_target = 1;
% Solución de referencia con Runge-Kutta (ode45)
[x_rk, y_rk] = ode45(@(x,y) y, x_span, y0);
averageGrad = [];
averageSqGrad = [];
%% 4. Preparación de la figura para GIF
fig = figure('Color','w','Position',[100 100 1000 450]);
% Subplot pérdida
subplot(1,2,1)
hLoss = animatedline('Color','b','LineWidth',1.5);
set(gca,'YScale','log')
title('Evolución de la pérdida')
xlabel('Época'); ylabel('Loss'); grid on;
% Subplot comparación soluciones
subplot(1,2,2)
plot(x_rk, y_rk, 'r-','LineWidth',1.5,'DisplayName','RK45');
hold on;
hPinn = plot(NaN,NaN,'b--','LineWidth',2,'DisplayName','PINN');
legend('Location','northwest');
title('Solución aproximada')
xlabel('x'); ylabel('y(x)');
grid on;
%% 5. Entrenamiento PINN y generación GIF
for epoch = 1:numEpochs
   % Calcular pérdida y gradientes
   [loss,grads] = dlfeval(@modelLoss,pinn,x_physics,x0_dl,y0_target);
   % Actualización Adam
   [pinn,averageGrad,averageSqGrad] = adamupdate( ...
       pinn,grads,averageGrad,averageSqGrad, ...
       epoch,learningRate);
   % Actualizar gráficos cada 50 épocas para GIF fluido
   if mod(epoch,50)==0 || epoch==1
       currentLoss = double(extractdata(loss));
       % Subplot pérdida
       subplot(1,2,1)
       addpoints(hLoss,epoch,currentLoss);
       % Subplot solución PINN
       y_pred = predict(pinn,x_physics);
       subplot(1,2,2)
       set(hPinn,'XData',extractdata(x_physics), ...
                 'YData',extractdata(y_pred));
       title(['Época: ', num2str(epoch), ...
              '  |  Loss: ', num2str(currentLoss,'%.2e')])
       drawnow;
       % Guardar frame GIF
       frame = getframe(fig);
       im = frame2im(frame);
       [imind, cm] = rgb2ind(im, 256);
       if epoch == 1
           imwrite(imind, cm, gifFilename, 'gif', ...
               'Loopcount', inf, 'DelayTime', 0.15);
       else
           imwrite(imind, cm, gifFilename, 'gif', ...
               'WriteMode', 'append', 'DelayTime', 0.15);
       end
   end
end
%% 6. Función de pérdida PINN
function [loss,grads] = modelLoss(net,x_phys,x0,y0_target)
   % Predicción de la red
   y = forward(net,x_phys);
   % Derivada automática: y'
   dy_dx = dlgradient(sum(y,'all'),x_phys, ...
       'EnableHigherDerivatives',true);
   % Residuo EDO: y' - y = 0
   f = dy_dx - y;
   loss_physics = mean(f.^2);
   % Condición inicial y(0)=1
   y0_pred = forward(net,x0);
   loss_bc = (y0_pred - y0_target)^2;
   % Pérdida total combinando términos
   loss = loss_physics + loss_bc;
   % Gradientes respecto a parámetros de la red
   grads = dlgradient(loss,net.Learnables);
end
