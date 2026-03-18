%% PINN Ej (b)
clear; clc; close all;
%% 1. Configuración de la Red (Arquitectura de 4 Capas y 28 Neuronas)
% Según el documento, esta configuración es ideal para las oscilaciones.
numNeurons = 7;
layers = [
  featureInputLayer(1, 'Name', 'input')
   fullyConnectedLayer(numNeurons)
  tanhLayer
   fullyConnectedLayer(numNeurons)
  tanhLayer
 
  fullyConnectedLayer(numNeurons)
  tanhLayer
 
  fullyConnectedLayer(numNeurons)
  tanhLayer
 
  fullyConnectedLayer(1, 'Name', 'output')
];
pinn = dlnetwork(layers);
%% 2. Parámetros de Entrenamiento y Datos
numEpochs = 3000;
learningRate = 0.02; % Tasa alta para ver cambios rápidos
gifFilename = 'pinn_ej_b.gif';
% Formato 'CB' (Canales, Batch) para evitar errores de dimensión
x_physics = dlarray(linspace(0, 2, 300), 'CB');
x0 = dlarray(0, 'CB');
y0_target = 1;
% Solución de referencia (ODE45) para comparar
[t_ref, y_ref] = ode45(@(t,y) 1 - y/(2 + 1.5*sin(4*pi*t)), [0 2], 1);
averageGrad = []; averageSqGrad = [];
%% 3. Gráficos y Preparación de GIF
fig = figure('Color', 'w', 'Position', [100 100 1000 450]);
subplot(1,2,1);
hLoss = animatedline('Color', [0.85 0.33 0.1], 'LineWidth', 1.5);
title('Evolución de la Pérdida (Loss)'); grid on;
subplot(1,2,2);
plot(t_ref, y_ref, 'r-', 'LineWidth', 2, 'DisplayName', 'Referencia'); hold on;
hPinn = plot(NaN, NaN, 'b--', 'LineWidth', 2, 'DisplayName', 'PINN');
title('Aproximación de la Solución'); ylim([0.8 1.4]); grid on;
legend('Location', 'northeast');
%% 4. Bucle de Entrenamiento (Actualización cada 150 épocas)
for epoch = 1:numEpochs
  [loss, grads] = dlfeval(@modelLoss, pinn, x_physics, x0, y0_target);
  [pinn, averageGrad, averageSqGrad] = adamupdate(pinn, grads, ...
      averageGrad, averageSqGrad, epoch, learningRate);
   if mod(epoch, 150) == 0 || epoch == 1
      currentLoss = double(extractdata(loss));
      addpoints(hLoss, epoch, currentLoss);
    
      y_pred = predict(pinn, x_physics);
      set(hPinn, 'XData', extractdata(x_physics), 'YData', extractdata(y_pred));
    
      % Captura de fotograma para GIF
      drawnow limitrate;
      frame = getframe(fig);
      im = frame2im(frame);
      [imind, cm] = rgb2ind(im, 256);
      if epoch == 1
          imwrite(imind, cm, gifFilename, 'gif', 'Loopcount', inf, 'DelayTime', 0.25);
      else
          imwrite(imind, cm, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.25);
      end
  end
end
%% 5. Función de Pérdida (Physics-Informed)
function [loss, grads] = modelLoss(net, x_phys, x0, y0_target)
  % Diferenciación automática: clave de las PINN
  y = forward(net, x_phys);
  dy_dx = dlgradient(sum(y, 'all'), x_phys, 'EnableHigherDerivatives', true);
   % Residuo de la EDO (Información física)
  den = 2 + 1.5 * sin(4 * pi * x_phys);
  f = dy_dx - (1 - y ./ den);
  loss_physics = mean(f.^2);
   % Error en condición inicial y(0)=1
  y0_pred = forward(net, x0);
  loss_boundary = (y0_pred - y0_target)^2;
   % Pérdida total combinada
  loss = loss_physics + loss_boundary;
  grads = dlgradient(loss, net.Learnables);
end
