%% PINN Ejercicio (c): Oscilador Armónico con Neumann BC
clear; clc; close all;
%% 1. Configuración de la Red
% Aumentamos ligeramente la profundidad (3 capas ocultas) para capturar
% mejor la segunda derivada (curvatura).
numNeurons = 6;
layers = [
   featureInputLayer(1, 'Name', 'input')
  
   fullyConnectedLayer(numNeurons, 'Name', 'fc1')
   tanhLayer('Name', 'act1')
  
   fullyConnectedLayer(numNeurons, 'Name', 'fc2')
   tanhLayer('Name', 'act2')
   fullyConnectedLayer(numNeurons, 'Name', 'fc3')
   tanhLayer('Name', 'act3')
  
   fullyConnectedLayer(1, 'Name', 'output')
];
% Inicializamos la red dlnetwork
pinn = dlnetwork(layers);
%% 2. Parámetros y Datos
numEpochs = 1000;       % Más épocas porque 2da derivada es más difícil de aprender
learningRate = 0.01;
gifFilename = 'ejercicio_c_harmonica.gif';
% --- Puntos de Colocación (Dominio) ---
% Usamos formato 'CB' (Channel, Batch)
x_physics = dlarray(linspace(0, 2, 400), 'CB');
% --- Puntos de Borde ---
x_bc1 = dlarray(0, 'CB');   % Para y(0) = 0
x_bc2 = dlarray(1, 'CB');   % Para y'(1) = 1 (Neumann)
% --- Solución Analítica (Exacta) ---
% La solución de y'' + pi^2*y = 0 con y(0)=0, y'(1)=1 es:
% y(x) = -1/pi * sin(pi*x)
t_ref = linspace(0, 2, 200);
y_ref = -(1/pi) * sin(pi * t_ref);
%% 3. Inicialización de Gráficos
fig = figure('Color', 'w', 'Position', [100 100 1000 500]);
% Subplot Pérdida
subplot(1,2,1);
hLoss = animatedline('Color', '#D95319', 'LineWidth', 1.5);
title('Pérdida (Loss)'); xlabel('Época'); grid on; set(gca, 'YScale', 'log');
% Subplot Solución
subplot(1,2,2);
plot(t_ref, y_ref, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Solución Exacta'); hold on;
hPinn = plot(NaN, NaN, 'r--', 'LineWidth', 2.5, 'DisplayName', 'Predicción PINN');
title('Solución: y'''' + \pi^2 y = 0');
xlabel('x'); ylabel('y'); legend('Location', 'northeast'); grid on;
ylim([-0.5 0.5]); xlim([0 2]);
% Optimizador Adam
averageGrad = []; averageSqGrad = [];
%% 4. Bucle de Entrenamiento
fprintf('Entrenando PINN para 2da derivada...\n');
startTick = tic;
for epoch = 1:numEpochs
   % Evaluar gradientes y pérdida
   [loss, grads] = dlfeval(@modelLoss, pinn, x_physics, x_bc1, x_bc2);
  
   % Actualizar pesos (Adam)
   [pinn, averageGrad, averageSqGrad] = adamupdate(pinn, grads, ...
       averageGrad, averageSqGrad, epoch, learningRate);
  
   % Actualización Visual (Cada 100 épocas para eficiencia)
   if mod(epoch, 100) == 0 || epoch == 1
       % Extraer datos para plotear
       currentLoss = extractdata(loss);
       y_pred = predict(pinn, x_physics);
      
       % Actualizar líneas
       addpoints(hLoss, epoch, currentLoss);
       set(hPinn, 'XData', extractdata(x_physics), 'YData', extractdata(y_pred));
       title(subplot(1,2,2), sprintf('Época %d | Loss: %.2e', epoch, currentLoss));
      
       drawnow; % Forzar actualización gráfica
      
       % --- Generación de GIF Optimizada ---
       frame = getframe(fig);
       im = frame2im(frame);
       [imind, cm] = rgb2ind(im, 256);
      
       if epoch == 1
           imwrite(imind, cm, gifFilename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
       else
           imwrite(imind, cm, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
       end
   end
end
toc(startTick);
%% 5. Función de Pérdida (Con 2da Derivada y Neumann)
function [loss, grads] = modelLoss(net, x, x_bc1, x_bc2)
  
   % --- 1. Residuo de la EDO (Physics Loss) ---
   y = forward(net, x);
  
   % Primera derivada: dy/dx
   dy = dlgradient(sum(y, 'all'), x, 'EnableHigherDerivatives', true);
  
   % Segunda derivada: d2y/dx2
   % Importante: derivamos la primera derivada respecto a x
   ddy = dlgradient(sum(dy, 'all'), x, 'EnableHigherDerivatives', true);
  
   % Ecuación: y'' + pi^2 * y = 0
   res_ode = ddy + (pi^2) * y;
   loss_physics = mean(res_ode.^2);
  
   % --- 2. Condición de Borde 1: Dirichlet y(0) = 0 ---
   y_0 = forward(net, x_bc1);
   loss_bc1 = (y_0 - 0)^2;
  
   % --- 3. Condición de Borde 2: Neumann y'(1) = 1 ---
   % Calculamos forward en x=1
   y_1 = forward(net, x_bc2);
   % Calculamos derivada en x=1
   dy_1 = dlgradient(sum(y_1, 'all'), x_bc2, 'EnableHigherDerivatives', true);
   % Error: (y'(1) - 1)^2
   loss_bc2 = (dy_1 - 1)^2;
  
   % --- 4. Pérdida Total ---
   % Ponderamos las BCs un poco más alto para forzar convergencia
   loss = loss_physics + loss_bc1 + loss_bc2;
  
   % Calcular gradientes para actualizar la red
   grads = dlgradient(loss, net.Learnables);
end
