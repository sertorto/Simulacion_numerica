%% PINN Ejercicio (d): Ecuación de Transporte (Advección)
clear; clc; close all;
%% 1. Configuración de la Red
% CAMBIO IMPORTANTE: La entrada ahora es 2 (x, t)
numNeurons = 8;
layers = [
   featureInputLayer(2, 'Name', 'input_xt') % 2 Entradas: x, t
  
   fullyConnectedLayer(numNeurons, 'Name', 'fc1')
   tanhLayer('Name', 'act1')
  
   fullyConnectedLayer(numNeurons, 'Name', 'fc2')
   tanhLayer('Name', 'act2')
  
   fullyConnectedLayer(numNeurons, 'Name', 'fc3')
   tanhLayer('Name', 'act3')
  
   fullyConnectedLayer(1, 'Name', 'output')
];
pinn = dlnetwork(layers);
%% 2. Parámetros y Datos
numEpochs = 1500;
learningRate = 0.01;
gifFilename = 'ejercicio_d_transporte.gif';
% --- Generación de Malla (Grid) para Entrenamiento ---
N_x = 50; N_t = 50;
x_vals = linspace(0, 1, N_x);
t_vals = linspace(0, 0.5, N_t);
[X_mesh, T_mesh] = meshgrid(x_vals, t_vals);
% Aplanamos y concatenamos para tener formato 2 x N
% Fila 1: x, Fila 2: t
XT_physics = dlarray([X_mesh(:)'; T_mesh(:)'], 'CB');
% --- Datos Condición Inicial (t=0) ---
% u(x,0) = max(1 - |2x - 1|, 0)
x_ic = linspace(0, 1, 100);
t_ic = zeros(size(x_ic));
XT_ic = dlarray([x_ic; t_ic], 'CB');
u_ic_target = max(1 - abs(2*x_ic - 1), 0);
u_ic_target = dlarray(u_ic_target, 'CB');
% --- Datos Condición de Borde (x=0) ---
% Asumimos "inflow" nulo u(0,t) = 0 ya que el pulso viaja a la derecha
t_bc = linspace(0, 0.5, 50);
x_bc = zeros(size(t_bc));
XT_bc = dlarray([x_bc; t_bc], 'CB');
u_bc_target = dlarray(zeros(size(t_bc)), 'CB');
% --- Solución Exacta para Comparar ---
% La solución es una onda viajera: u(x,t) = u0(x - t)
exact_sol = @(x,t) max(1 - abs(2*(x - t) - 1), 0);
U_exact = exact_sol(X_mesh, T_mesh);
%% 3. Inicialización de Gráficos
fig = figure('Color', 'w', 'Position', [100 100 1100 500]);
% Subplot 1: Pérdida
subplot(1,2,1);
hLoss = animatedline('Color', '#D95319', 'LineWidth', 1.5);
title('Pérdida (Loss)'); xlabel('Época'); grid on; set(gca, 'YScale', 'log');
% Subplot 2: Superficie 3D (x, t, u)
subplot(1,2,2);
% Graficamos la malla exacta como referencia (transparente)
surf(X_mesh, T_mesh, U_exact, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'b');
hold on;
% Graficamos la predicción PINN (inicialmente plana)
hSurf = surf(X_mesh, T_mesh, zeros(size(X_mesh)), 'FaceAlpha', 0.8, 'FaceColor', 'r');
title('Solución u_t + u_x = 0');
xlabel('Espacio (x)'); ylabel('Tiempo (t)'); zlabel('u(x,t)');
legend('Exacta (Ref)', 'PINN (Predicción)', 'Location', 'best');
view(3); grid on; zlim([0 1.2]);
averageGrad = []; averageSqGrad = [];
%% 4. Bucle de Entrenamiento
fprintf('Entrenando PINN Ecuación Transporte...\n');
startTick = tic;
for epoch = 1:numEpochs
  
   % Evaluar gradientes y pérdida
   [loss, grads] = dlfeval(@modelLoss, pinn, XT_physics, XT_ic, u_ic_target, XT_bc, u_bc_target);
  
   % Actualizar pesos (Adam)
   [pinn, averageGrad, averageSqGrad] = adamupdate(pinn, grads, ...
       averageGrad, averageSqGrad, epoch, learningRate);
  
   % Visualización y GIF
   if mod(epoch, 50) == 0 || epoch == 1
       % Predecir en toda la malla
       currentLoss = extractdata(loss);
       U_pred_flat = predict(pinn, XT_physics);
       U_pred_grid = reshape(extractdata(U_pred_flat), size(X_mesh));
      
       % Actualizar Loss
       addpoints(hLoss, epoch, currentLoss);
      
       % Actualizar Superficie
       set(hSurf, 'ZData', U_pred_grid);
       title(subplot(1,2,2), sprintf('Época %d | Loss: %.4f', epoch, currentLoss));
      
       drawnow;
      
       % Capturar GIF
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
%% 5. Función de Pérdida (PDE + IC + BC)
function [loss, grads] = modelLoss(net, xt, xt_ic, u_ic_targ, xt_bc, u_bc_targ)
  
   % --- 1. Residuo de la PDE: u_t + u_x = 0 ---
   u = forward(net, xt);
  
   % Calculamos el gradiente respecto a la entrada (x,t)
   % grads_input será una matriz de 2 x N (Fila 1: du/dx, Fila 2: du/dt)
   grads_input = dlgradient(sum(u, 'all'), xt, 'EnableHigherDerivatives', true);
  
   u_x = grads_input(1, :);
   u_t = grads_input(2, :);
  
   res_pde = u_t + u_x; % Ecuación de transporte
   loss_physics = mean(res_pde.^2);
  
   % --- 2. Condición Inicial (t=0) ---
   u_pred_ic = forward(net, xt_ic);
   loss_ic = mean((u_pred_ic - u_ic_targ).^2);
  
   % --- 3. Condición de Borde (x=0) ---
   u_pred_bc = forward(net, xt_bc);
   loss_bc = mean((u_pred_bc - u_bc_targ).^2);
  
   % --- 4. Pérdida Total ---
   loss = loss_physics + loss_ic + loss_bc;
  
   grads = dlgradient(loss, net.Learnables);
end
