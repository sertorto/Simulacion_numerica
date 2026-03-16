%% Comparativa: Lambda Adaptativo vs Estándar (Guardado en Escritorio)
clear; close all; clc;
%% ---------------- DATOS DE ENTRENAMIENTO ----------------
x_train = linspace(-2,2,10);
y_train = 4 - x_train.^2;
f = @(x) 4 - x.^2;
%% ---------------- PARÁMETROS DE LA RED ------------------
N = 2;                 
A  = @(x) tanh(x);
dA = @(x) 1 - tanh(x).^2;
eta = 0.01;
epochs = 500;
crecimiento_lambda = 0.05;
%% ---------------- INICIALIZACIÓN (Mismos pesos) ---------
rng(1)
a_init = randn(1,N); w_init = 2*randn(1,N);
b_init = randn(1,N); c_init = randn;
% Red 1: Con Lambda Adaptativo
a1 = a_init; w1 = w_init; b1 = b_init; c1 = c_init;
lambda = ones(size(x_train));
loss_hist1 = zeros(1, epochs + 1);
% Red 2: Sin Lambda
a2 = a_init; w2 = w_init; b2 = b_init; c2 = c_init;
loss_hist2 = zeros(1, epochs + 1);
%% ---------------- RUTA DEL ESCRITORIO -------------------
% Detectar la ruta del escritorio de forma automática
if ispc % Windows
   desktopPath = fullfile(getenv('USERPROFILE'), 'Desktop');
else % macOS o Linux
   desktopPath = fullfile(getenv('HOME'), 'Desktop');
end
gif_filename = fullfile(desktopPath, 'comparativa_red_neuronal.gif');
%% ---------------- PREPARAR FIGURA -----------------------
x_plot = linspace(-2.2,2.2,400);
y_true = f(x_plot);
h = figure('Color', 'w', 'Position', [100, 100, 1000, 750]);
%% ---------------- ENTRENAMIENTO DUAL --------------------
for epoch = 0:epochs
  
   % --- RED 1: CON LAMBDA ADAPTATIVO ---
   z1 = w1' * x_train + b1';
   y_p1 = a1 * A(z1) + c1;
   err1 = y_p1 - y_train;
  
   lambda(1) = lambda(1) + crecimiento_lambda * abs(err1(1));
   lambda(end) = lambda(end) + crecimiento_lambda * abs(err1(end));
   loss_hist1(epoch+1) = 0.5 * sum(lambda .* err1.^2);
  
   da1 = (lambda .* err1) * A(z1)';
   dw1 = (a1 .* ((lambda .* err1) * (dA(z1) .* x_train)'))';
   db1 = (a1 .* ((lambda .* err1) * dA(z1)'))';
   dc1 = sum(lambda .* err1);
   % --- RED 2: SIN LAMBDA ---
   z2 = w2' * x_train + b2';
   y_p2 = a2 * A(z2) + c2;
   err2 = y_p2 - y_train;
   loss_hist2(epoch+1) = 0.5 * sum(err2.^2);
  
   da2 = err2 * A(z2)';
   dw2 = (a2 .* (err2 * (dA(z2) .* x_train)'))';
   db2 = (a2 .* (err2 * dA(z2)'))';
   dc2 = sum(err2);
   % --- ACTUALIZACIÓN ---
   a1 = a1 - eta*da1; w1 = w1 - eta*dw1'; b1 = b1 - eta*db1'; c1 = c1 - eta*dc1;
   a2 = a2 - eta*da2; w2 = w2 - eta*dw2'; b2 = b2 - eta*db2'; c2 = c2 - eta*dc2;
   % --- VISUALIZACIÓN ---
   if mod(epoch,20)==0 || epoch == epochs
       % Ajuste Con Lambda
       subplot(2,2,1);
       y_net1 = a1 * A(w1' * x_plot + b1') + c1;
       plot(x_plot,y_true,'b','LineWidth',2); hold on;
       plot(x_plot,y_net1,'r--','LineWidth',2);
       scatter(x_train,y_train,40,'k','filled'); hold off;
       grid on; axis([-2.5 2.5 -1 5]);
       title(['Ajuste CON Lambda (E: ',num2str(epoch),')']);
       % Ajuste Sin Lambda
       subplot(2,2,2);
       y_net2 = a2 * A(w2' * x_plot + b2') + c2;
       plot(x_plot,y_true,'b','LineWidth',2); hold on;
       plot(x_plot,y_net2,'r--','LineWidth',2);
       scatter(x_train,y_train,40,'k','filled'); hold off;
       grid on; axis([-2.5 2.5 -1 5]);
       title('Ajuste SIN Lambda (Estándar)');
       % Pérdida Con Lambda
       subplot(2,2,3);
       plot(0:epoch, loss_hist1(1:epoch+1), 'g', 'LineWidth', 1.5);
       grid on; xlim([0 epochs]); title('Pérdida (Con \lambda)');
       % Pérdida Sin Lambda
       subplot(2,2,4);
       plot(0:epoch, loss_hist2(1:epoch+1), 'm', 'LineWidth', 1.5);
       grid on; xlim([0 epochs]); title('Pérdida (Sin \lambda)');
      
       drawnow;
       % --- GUARDAR GIF ---
       frame = getframe(h);
       im = frame2im(frame);
       [A_gif,map] = rgb2ind(im,256,'nodither');
       if epoch == 0
           imwrite(A_gif,map,gif_filename,'gif','LoopCount',Inf,'DelayTime',0.1);
       else
           imwrite(A_gif,map,gif_filename,'gif','WriteMode','append','DelayTime',0.1);
       end
   end
end
fprintf('El GIF se ha guardado en: %s\n', gif_filename);
