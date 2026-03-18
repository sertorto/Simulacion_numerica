%% Aproximacion de f(x) = 4 - x^2
% Red neuronal PROFUNDA 2x2
% Activacion sigmoide
% Entrenamiento con descenso del gradiente
% Visualizacion en GIF y funcion de perdida
clear; close all; clc;
%% ---------------- DATOS ----------------
x_train = linspace(-2,2,8);
y_train = 4 - x_train.^2;
f = @(x) 4 - x.^2;
%% ---------------- ACTIVACION ------------
A  = @(x) 1 ./ (1 + exp(-x));
dA = @(x) A(x) .* (1 - A(x));
%% ---------------- INICIALIZACION --------
rng(1)
% Capa 1
w = 2*randn(1,2);
b = randn(1,2);
% Capa 2
V = randn(2,2);
d = randn(1,2);
% Salida
a = randn(1,2);
c = randn;
%% ---------------- ENTRENAMIENTO ---------
eta = 0.04;
epochs = 300;
Loss_hist = zeros(1, epochs+1);  % <-- almacenar funcion de perdida
%% ---------------- DOMINIO PARA GRAFICAS -------
x_plot = linspace(-2.2,2.2,400);
y_true = f(x_plot);
%% ---------------- GIF -------------------
gif_filename = fullfile(getenv('USERPROFILE'), ...
  'Downloads','entrenamiento_profunda_2x2.gif');
figure
for epoch = 0:epochs
  %% -------- FORWARD --------
  y1 = A(w(1)*x_train + b(1));
  y2 = A(w(2)*x_train + b(2));
 
  z3 = V(1,1)*y1 + V(1,2)*y2 + d(1);
  z4 = V(2,1)*y1 + V(2,2)*y2 + d(2);
 
  y3 = A(z3);
  y4 = A(z4);
 
  y_pred = a(1)*y3 + a(2)*y4 + c;
 
  %% -------- ERROR ---------
  error = y_pred - y_train;
  Loss_hist(epoch+1) = sum(error.^2);   % <-- guardar loss
  %% -------- GRADIENTES ---------
  da(1) = sum(error .* y3);
  da(2) = sum(error .* y4);
  dc    = sum(error);
 
  dz3 = error .* a(1) .* dA(z3);
  dz4 = error .* a(2) .* dA(z4);
 
  dV(1,1) = sum(dz3 .* y1);
  dV(1,2) = sum(dz3 .* y2);
  dV(2,1) = sum(dz4 .* y1);
  dV(2,2) = sum(dz4 .* y2);
  dd(1) = sum(dz3);
  dd(2) = sum(dz4);
 
  dy1 = dz3.*V(1,1) + dz4.*V(2,1);
  dy2 = dz3.*V(1,2) + dz4.*V(2,2);
 
  dw(1) = sum(dy1 .* dA(w(1)*x_train + b(1)) .* x_train);
  dw(2) = sum(dy2 .* dA(w(2)*x_train + b(2)) .* x_train);
  db(1) = sum(dy1 .* dA(w(1)*x_train + b(1)));
  db(2) = sum(dy2 .* dA(w(2)*x_train + b(2)));
 
  %% -------- UPDATE ---------
  a = a - eta*da;
  c = c - eta*dc;
  V = V - eta*dV;
  d = d - eta*dd;
  w = w - eta*dw;
  b = b - eta*db;
  %% -------- VISUALIZACION --------
  if mod(epoch,20)==0 || epoch==epochs
      y1p = A(w(1)*x_plot + b(1));
      y2p = A(w(2)*x_plot + b(2));
      y3p = A(V(1,1)*y1p + V(1,2)*y2p + d(1));
      y4p = A(V(2,1)*y1p + V(2,2)*y2p + d(2));
      y_net = a(1)*y3p + a(2)*y4p + c;
     
      plot(x_plot,y_true,'b','LineWidth',2); hold on
      plot(x_plot,y_net,'r--','LineWidth',2)
      scatter(x_train,y_train,40,'k','filled')
      hold off
      grid on
      axis([-2.2 2.2 -1 5])
      xlabel('x')
      ylabel('f(x)')
      title(['Red profunda 2x2 - Epoca ',num2str(epoch)])
      legend('f(x)','Red neuronal','Datos','Location','South')
      drawnow
     
      %% --------- GUARDAR GIF -----
      frame = getframe(gcf);
      im = frame2im(frame);
      [A_gif,map] = rgb2ind(im,256,'nodither');
      if epoch == 0
          imwrite(A_gif,map,gif_filename,'gif', 'LoopCount',Inf,'DelayTime',0.5);
      else
          imwrite(A_gif,map,gif_filename,'gif', 'WriteMode','append','DelayTime',0.5);
      end
  end
end
