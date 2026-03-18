import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Evitar conflictos de librerías en macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ===============================
# 1. DEFINICIÓN DE ARQUITECTURAS (Matched Budgets)
# ===============================


class KANLayer(nn.Module):
   def __init__(self, input_size, output_size, degree):
       super().__init__()
       self.input_size = input_size
       self.output_size = output_size
       self.degree = degree
       self.weights = nn.Parameter(torch.randn(output_size, input_size*(degree+1)) * 0.5)
       self.bias = nn.Parameter(torch.zeros(output_size, 1))
  
   def forward(self, x):
       Phi = [x**d for d in range(self.degree+1)]
       Phi = torch.cat(Phi, dim=1)
       y = Phi @ self.weights.T + self.bias.T
       return y


class KANNetwork(nn.Module):
   # Presupuesto ajustado: degree=3, neurons=12 (~1297 parámetros)
   def __init__(self, input_dim=2, num_neurons=12, degree=3):
       super().__init__()
       self.kan1 = KANLayer(input_dim, num_neurons, degree)
       self.kan2 = KANLayer(num_neurons, num_neurons, degree)
       self.kan3 = KANLayer(num_neurons, num_neurons, degree)
       self.out = nn.Linear(num_neurons, 1)
  
   def forward(self, x):
       x = torch.tanh(self.kan1(x))
       x = torch.tanh(self.kan2(x))
       x = torch.tanh(self.kan3(x))
       x = self.out(x)
       return x


class PINN_Standard(nn.Module):
   # Presupuesto ajustado: 4 capas, 20 neuronas (~1341 parámetros)
   def __init__(self):
       super().__init__()
       self.net = nn.Sequential(
           nn.Linear(2, 20), nn.Tanh(),
           nn.Linear(20, 20), nn.Tanh(),
           nn.Linear(20, 20), nn.Tanh(),
           nn.Linear(20, 20), nn.Tanh(),
           nn.Linear(20, 1)
       )
      
   def forward(self, x):
       return self.net(x)


# ===============================
# 2. PREPARACIÓN DE DATOS (Fisher-Kolmogorov)
# ===============================
N_physics = 2500
N_ic = 500
N_bc = 500


alpha_diff = 0.05  # Coeficiente de difusión
rho_react = 5.0    # Tasa de crecimiento logístico (Fisher)


# Puntos Físicos: x en [-1, 1], t en [0, 1]
x_phys = torch.rand((N_physics, 1)) * 2.0 - 1.0
t_phys = torch.rand((N_physics, 1))
XT_phys = torch.cat([x_phys, t_phys], dim=1).requires_grad_(True)


# Condición Inicial u(x,0) = exp(-10 * x^2) (Un pulso gaussiano en el centro)
x_ic = torch.rand((N_ic, 1)) * 2.0 - 1.0
t_ic = torch.zeros((N_ic, 1))
XT_ic = torch.cat([x_ic, t_ic], dim=1)
u_ic_target = torch.exp(-10.0 * x_ic**2)


# Condiciones de Borde (Extremos fríos/inertes: u(-1,t)=0, u(1,t)=0)
x_bc0 = -torch.ones((N_bc // 2, 1))
t_bc0 = torch.rand((N_bc // 2, 1))
x_bc1 = torch.ones((N_bc // 2, 1))
t_bc1 = torch.rand((N_bc // 2, 1))
XT_bc = torch.cat([torch.cat([x_bc0, t_bc0], dim=1),
                  torch.cat([x_bc1, t_bc1], dim=1)], dim=0)
u_bc_target = torch.zeros((N_bc, 1))


# ===============================
# 3. ENTRENAMIENTO Y FUNCIÓN DE PÉRDIDA
# ===============================
model_pikan = KANNetwork()
model_pinn = PINN_Standard()


opt_pikan = optim.Adam(model_pikan.parameters(), lr=0.005)
opt_pinn = optim.Adam(model_pinn.parameters(), lr=0.005)


num_epochs = 3000
loss_hist_pikan = []
loss_hist_pinn = []
frames = []


def compute_loss(model):
   u_pred = model(XT_phys)
  
   # Derivadas de primer orden
   du_dxt = torch.autograd.grad(outputs=u_pred, inputs=XT_phys,
                                grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
   u_x = du_dxt[:, 0:1]
   u_t = du_dxt[:, 1:2]
  
   # Derivada de segundo orden (Difusión)
   du_x_dxt = torch.autograd.grad(outputs=u_x, inputs=XT_phys,
                                  grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
   u_xx = du_x_dxt[:, 0:1]
  
   # ECUACIÓN DE FISHER-KOLMOGOROV
   # u_t - alpha * u_xx - rho * u * (1 - u) = 0
   res_pde = u_t - alpha_diff * u_xx - rho_react * u_pred * (1.0 - u_pred)
  
   loss_pde = torch.mean(res_pde**2)
  
   # Condiciones Iniciales y de Borde
   loss_ic = torch.mean((model(XT_ic) - u_ic_target)**2)
   loss_bc = torch.mean((model(XT_bc) - u_bc_target)**2)
  
   return loss_pde + loss_ic + loss_bc


# ===============================
# 4. BUCLE PRINCIPAL
# ===============================
plt.ioff()
fig, ax_loss = plt.subplots(figsize=(8, 6))


print("Iniciando entrenamiento (Fisher-Kolmogorov: PINN vs PIKAN)...")
for epoch in range(1, num_epochs + 1):
  
   # PIKAN
   opt_pikan.zero_grad()
   loss_k = compute_loss(model_pikan)
   loss_k.backward()
   opt_pikan.step()
   loss_hist_pikan.append(loss_k.item())
  
   # PINN
   opt_pinn.zero_grad()
   loss_p = compute_loss(model_pinn)
   loss_p.backward()
   opt_pinn.step()
   loss_hist_pinn.append(loss_p.item())
  
   # Generar gráficos para el GIF
   if epoch % 50 == 0 or epoch == 1:
       print(f"Epoch {epoch}/{num_epochs} | Loss PIKAN: {loss_k.item():.5f} | Loss PINN: {loss_p.item():.5f}")
      
       ax_loss.clear()
       ax_loss.plot(range(1, epoch + 1), loss_hist_pikan, 'r-', lw=2.5, label='PIKAN')
       ax_loss.plot(range(1, epoch + 1), loss_hist_pinn, 'b-', lw=2.5, label='PINN')
       ax_loss.set_yscale('log')
       ax_loss.set_xlim(0, num_epochs)
       ax_loss.set_title('Pérdidas (Ecuación de Fisher-Kolmogorov)', fontsize=14)
       ax_loss.set_xlabel('Epoch', fontsize=12)
       ax_loss.set_ylabel('Loss (Log)', fontsize=12)
       ax_loss.legend(fontsize=12)
       ax_loss.grid(True, which="both", ls="--", alpha=0.6)
      
       fig.tight_layout()
       fig.canvas.draw()
      
       image_rgba = np.asarray(fig.canvas.buffer_rgba())
       image_rgb = image_rgba[:, :, :3]
       frames.append(image_rgb)


# ===============================
# 5. GUARDAR GIF
# ===============================
save_path = "/Users/molab/Documents/simnu/PIKAN_Fisher_Loss.gif"
os.makedirs(os.path.dirname(save_path), exist_ok=True)


# Guardar GIF infinito
imgs = [Image.fromarray(img) for img in frames]
imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
plt.close(fig)


print(f"\n¡Entrenamiento finalizado! GIF guardado en: {save_path}\n")
