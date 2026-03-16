%% Aproximacion de f(x) = 4 - x^2
% Red neuronal con 1 capa oculta
% Activacion tanh con desplazamiento (bias por neurona)
% Entrenamiento mediante descenso del gradiente
clear; close all; clc;
%% ---------------- DATOS DE ENTRENAMIENTO ----------------
x_train = linspace(-2,2,25);
y_train = 4 - x_train.^2;
%% ---------------- FUNCION OBJETIVO ----------------------
f = @(x) 4 - x.^2;
%% ---------------- PARAMETROS DE LA RED ------------------
N = 2;                  % numero de neuronas ocultas
A  = @(x) tanh(x);
dA = @(x) 1 - tanh(x).^2;
%% ---------------- INICIALIZACION ------------------------
rng(1)
a = randn(1,N);         % pesos de salida
w = 2*randn(1,N);       % pesos de entrada
b = randn(1,N);         % desplazamientos
c = randn;              % bias final
%% ---------------- ENTRENAMIENTO -------------------------
eta = 0.01;
epochs = 200;
%% ---------------- DOMINIO PARA GRAFICAS -----------------
x_plot = linspace(-2.2,2.2,400);
y_true = f(x_plot);
%% ---------------- PREPARAR GIF --------------------------
gif_filename = fullfile(getenv('USERPROFILE'),'Downloads','entrenamiento_tanh_bias.gif');
figure
for epoch = 0:epochs
  
   %% --------- FORWARD ---------
   y_pred = zeros(size(x_train));
   for i = 1:N
       y_pred = y_pred + a(i)*A(w(i)*x_train + b(i));
   end
   y_pred = y_pred + c;
  
   error = y_pred - y_train;
  
   %% --------- GRADIENTES -------
   da = zeros(1,N);
   dw = zeros(1,N);
   db = zeros(1,N);
  
   for i = 1:N
       z = w(i)*x_train + b(i);
       da(i) = sum(error .* A(z));
       dw(i) = sum(error .* a(i) .* dA(z) .* x_train);
       db(i) = sum(error .* a(i) .* dA(z));
   end
  
   dc = sum(error);
  
   %% --------- ACTUALIZACION ----
   a = a - eta*da;
   w = w - eta*dw;
   b = b - eta*db;
   c = c - eta*dc;
  
   %% --------- VISUALIZACION ----
   if mod(epoch,20)==0 || epoch==epochs
      
       y_net = zeros(size(x_plot));
       for i = 1:N
           y_net = y_net + a(i)*A(w(i)*x_plot + b(i));
       end
       y_net = y_net + c;
      
       plot(x_plot,y_true,'b','LineWidth',2); hold on
       plot(x_plot,y_net,'r--','LineWidth',2)
       scatter(x_train,y_train,40,'k','filled')
       hold off
      
       grid on
       xlabel('x')
       ylabel('f(x)')
       title(['Epoca ',num2str(epoch)])
       legend('f(x) real','Red neuronal','Datos','Location','South')
       axis([-2.2 2.2 -1 5])
       drawnow
      
       %% --------- GUARDAR GIF -----
       frame = getframe(gcf);
       im = frame2im(frame);
       [A_gif,map] = rgb2ind(im,256,'nodither');
      
       if epoch == 0
           imwrite(A_gif,map,gif_filename,'gif', ...
               'LoopCount',Inf,'DelayTime',0.5);
       else
           imwrite(A_gif,map,gif_filename,'gif', ...
               'WriteMode','append','DelayTime',0.5);
       end
   end
end
