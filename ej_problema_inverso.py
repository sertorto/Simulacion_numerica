import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ===============================
# 1. ARQUITECTURA KAN
# ===============================
class KANLayer(nn.Module):
   def __init__(self, input_size, output_size, degree):
       super().__init__()
       self.input_size = input_size
       self.output_size = output_size
       self.degree = degree
       self.weights = nn.Parameter(torch.randn(output_size, input_size*(degree+1)) * 0.1)
       self.bias = nn.Parameter(torch.zeros(output_size, 1))
  
   def forward(self, x):
       Phi = [x**d for d in range(self.degree+1)]
       Phi = torch.cat(Phi, dim=1)
       return Phi @ self.weights.T + self.bias.T


class KANNetwork(nn.Module):
   def __init__(self, num_neurons=15, degree=3):
       super().__init__()
       self.kan1 = KANLayer(1, num_neurons, degree)
       self.kan2 = KANLayer(num_neurons, num_neurons, degree)
       self.out = nn.Linear(num_neurons, 1)
  
   def forward(self, x):
       x = torch.tanh(self.kan1(x))
       x = torch.tanh(self.kan2(x))
       return self.out(x)


# ===============================
# 2. CONFIGURACIÓN (Dominio -1 a 1)
# ===============================
N_points = 600
x_phys = torch.linspace(-1, 1, N_points).view(-1, 1).requires_grad_(True)


# Bordes para u: u(-1)=0, u(1)=0
x_bc_u = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
u_bc_target = torch.tensor([[0.0], [0.0]], dtype=torch.float32)


# Borde para D: D(0)=1 (Anclamos el centro de la campana)
x_bc_D = torch.tensor([[0.0]], dtype=torch.float32)
D_bc_target = torch.tensor([[1.0]], dtype=torch.float32)


# Función real D(x) para generar los datos sintéticos
def D_exact_func(x):
   return torch.exp(-x**2)


# ===============================
# 3. FASE 1: CALCULAR u(x) (PRE-CÁLCULO)
# ===============================
print("--- FASE 1: Calculando u(x) en [-1, 1]... ---")


net_u = KANNetwork()
opt_u = optim.Adam(net_u.parameters(), lr=0.005)


# Entrenamos u(x) rápidamente (sin gráficos)
for epoch in range(2000):
   opt_u.zero_grad()
   u = net_u(x_phys)
   u_x = torch.autograd.grad(u, x_phys, torch.ones_like(u), create_graph=True)[0]
   u_xx = torch.autograd.grad(u_x, x_phys, torch.ones_like(u_x), create_graph=True)[0]
  
   # Física conocida: D(x) = exp(-x^2)
   D_val = D_exact_func(x_phys)
   D_val_x = -2 * x_phys * D_val
  
   # Ecuación Directa: -(D u')' + u = 1
   res_u = -(D_val_x * u_x + D_val * u_xx) + u - 1.0
   loss_u = torch.mean(res_u**2) + 20.0 * torch.mean((net_u(x_bc_u) - u_bc_target)**2)
  
   loss_u.backward()
   opt_u.step()


print("¡u(x) calculada! Congelando red...")


# Congelamos u(x) para la fase 2
for param in net_u.parameters():
   param.requires_grad = False


# ===============================
# 4. FASE 2: RECONSTRUIR D(x) (CON GIF)
# ===============================
print("--- FASE 2: Reconstruyendo D(x) (Problema Inverso)... ---")


net_D = KANNetwork()
opt_D = optim.Adam(net_D.parameters(), lr=0.005)
epochs_D = 3000
frames = []


# Preparar gráfico estático
plt.ioff()
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Problema Inverso en $[-1, 1]$: Reconstrucción de $D(x) = e^{-x^2}$', fontsize=16)


# Datos fijos para graficar
x_plot = torch.linspace(-1, 1, 200).view(-1, 1)
x_plot_np = x_plot.numpy()
D_real_np = D_exact_func(x_plot).numpy()
u_fixed_np = net_u(x_plot).detach().numpy() # u(x) ya es fija


for epoch in range(1, epochs_D + 1):
   opt_D.zero_grad()
  
   # Datos de u (Extraídos de la red fija)
   u_fixed = net_u(x_phys)
   u_x_fixed = torch.autograd.grad(u_fixed, x_phys, torch.ones_like(u_fixed), create_graph=True)[0]
   u_xx_fixed = torch.autograd.grad(u_x_fixed, x_phys, torch.ones_like(u_x_fixed), create_graph=True)[0]
  
   # Predicción de D (Lo que estamos entrenando)
   D_pred = net_D(x_phys)
   D_x_pred = torch.autograd.grad(D_pred, x_phys, torch.ones_like(D_pred), create_graph=True)[0]
  
   # Ecuación Inversa: -D' u' - D u'' + u - 1 = 0
   res_inverse = -(D_x_pred * u_x_fixed + D_pred * u_xx_fixed) + u_fixed - 1.0
  
   loss_ode_D = torch.mean(res_inverse**2)
   # Condición de anclaje D(0) = 1
   loss_bc_D = torch.mean((net_D(x_bc_D) - D_bc_target)**2)
  
   loss_D = loss_ode_D + 10.0 * loss_bc_D
   loss_D.backward()
   opt_D.step()
  
   # Captura de frames cada 50 épocas
   if epoch % 50 == 0 or epoch == 1:
       D_current_np = net_D(x_plot).detach().numpy()
      
       axs[0].clear()
       axs[1].clear()
      
       # Panel Izquierdo: u(x) (Dato de entrada)
       axs[0].plot(x_plot_np, u_fixed_np, 'b-', lw=3, alpha=0.6)
       axs[0].set_title("Dato: Solución $u(x)$ (Simétrica)")
       axs[0].set_xlabel("x")
       axs[0].set_ylabel("u(x)")
       axs[0].set_ylim([0, 0.6])
       axs[0].grid(True, ls="--")
      
       # Panel Derecho: D(x) (Reconstrucción)
       axs[1].plot(x_plot_np, D_real_np, 'k-', lw=4, alpha=0.2, label='D(x) Real ($e^{-x^2}$)')
       axs[1].plot(x_plot_np, D_current_np, 'r--', lw=3, label='D(x) Reconstruida')
       axs[1].set_title(f"Reconstrucción D(x) | Epoch {epoch}")
       axs[1].set_xlabel("x")
       axs[1].set_ylabel("D(x)")
       axs[1].set_ylim([0.2, 1.2])
       axs[1].legend()
       axs[1].grid(True, ls="--")
      
       fig.canvas.draw()
       # IMPORTANTE: .copy() para que el GIF no sea estático
       image_rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
       frames.append(image_rgba[:, :, :3])


# ===============================
# 5. GUARDAR GIF
# ===============================
save_path = "/Users/molab/Documents/simnu/KAN_Inversa_D_Variable_Simetrica.gif"
os.makedirs(os.path.dirname(save_path), exist_ok=True)


imgs = [Image.fromarray(img) for img in frames]
imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=80, loop=0)


print(f"\n¡GIF simétrico generado! Guardado en:\n{save_path}")
plt.close(fig)
