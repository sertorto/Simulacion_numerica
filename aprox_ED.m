clear all; clc; clf %borramos variables, limpiamos consola y figura
%Datos
% f es la no homogeneidad del Problema de Contorno
f = @(x) x.*(1-x);
a=0; %punto inicial
b=1; %punto final
N_values = [1,5,8,10,20,50,75,100]; %número de puntos interiores
gifname = 'diferencias_finitas.gif';
%en total tenemos N+2 = 7 puntos
for idx = 1:length(N_values)
   N = N_values(idx);
   h=(b-a)/(N+1); %paso de la malla
   % Construcción de la matriz L
   L = zeros(N, N);
  
   %llenamos la diagonal y las subdiagonales
   if N == 1
       L(1,1) = 2;
   else
       L(1,1) = 2; L(1,2) = -1; %primera fila
       L(N,N) = 2; L(N,N-1) = -1; %ultima fila
       for j = 2:N-1 %filas intermedias
           L(j,j) = 2;
           L(j,j-1) = -1;
           L(j,j+1) = -1;
       end
   end
   %calculamos ahora la malla y del término fuente
   x(1)=a;
   x(N+2)=b;
   fun(1)=f(x(1));
   fun(N+2)=f(x(N+2));
   %puntos interiores
   for k=2:N+1
       x(k)=x(k-1)+h;
       fun(k)=f(x(k)); %valores de f(x) evaluados en los nodos
   end
   %por último, resolvemos el sistema lineal
   u(2:N+1)=(h^2)*inv(L)*fun(2:N+1)';
   u(1)=0; u(N+2)=0; %condiciones de contorno
 
   plot(x,u,'r','LineWidth',2) %representación gráfica
   xlabel('x')
   ylabel('u(x)')
   title(['Solución numérica con N = ', num2str(N)])
   grid on
   axis([0 1 0 0.03])
   drawnow
   % Captura del frame
   frame = getframe(gcf);
   im = frame2im(frame);
   [imind,cm] = rgb2ind(im,256);
   % Crear o añadir al GIF
   if idx == 1
       imwrite(imind,cm,gifname,'gif','Loopcount',inf,'DelayTime',1);
   else
       imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',1);
   end
end
