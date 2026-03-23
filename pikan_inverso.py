import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ===============================
# 1. ARQUITECTURA KAN (PIKAN INVERSA)
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
       return Phi @ self.weights.T + self.bias.T


class InverseKANNetwork(nn.Module):
   def __init__(self, num_neurons=12, degree=3):
       super().__init__()
       self.kan1 = KANLayer(1, num_neurons, degree)
       self.kan2 = KANLayer(num_neurons, num_neurons, degree)
       self.out = nn.Linear(num_neurons, 1)
      
       # EL SECRETO: Parámetro 'A' entrenable.
       # Lo inicializamos mal a propósito (A=0.5) para que la red lo descubra.
       self.A_pred = nn.Parameter(torch.tensor([0.5]))
  
   def forward(self, x):
       x = torch.tanh(self.kan1(x))
       x = torch.tanh(self.kan2(x))
       return self.out(x)


# ===============================
# 2. SOLUCIÓN EXACTA Y DATOS (SENSORES)
# ===============================
# Ecuación: -u_xx + u = A  |  u(0)=0, u(1)=0
def solucion_exacta(x_np, A):
   sh1 = np.sinh(1.0)
   ch1 = np.cosh(1.0)
   # Solución analítica para este problema con término constante A
   return A * (1.0 - np.cosh(x_np) + ((ch1 - 1.0) / sh1) * np.sinh(x_np))


# Puntos del dominio
x_dom_np = np.linspace(0, 1, 200)
x_dom_tensor = torch.tensor(x_dom_np, dtype=torch.float32).view(-1, 1).requires_grad_(True)


# "Sensores": 15 puntos de datos extraídos de la solución directa
x_data_np = np.linspace(0.05, 0.95, 15)
x_data = torch.tensor(x_data_np, dtype=torch.float32).view(-1, 1)


# Datos medidos para los 3 casos (A=1, A=2, A=3)
u_data_A1 = torch.tensor(solucion_exacta(x_data_np, 1.0), dtype=torch.float32).view(-1, 1)
u_data_A2 = torch.tensor(solucion_exacta(x_data_np, 2.0), dtype=torch.float32).view(-1, 1)
u_data_A3 = torch.tensor(solucion_exacta(x_data_np, 3.0), dtype=torch.float32).view(-1, 1)


# Bordes
x_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
u_bc = torch.tensor([[0.0], [0.0]], dtype=torch.float32)


# ===============================
# 3. ENTRENAMIENTO DE LAS 3 REDES (EN PARALELO)
# ===============================
model_A1 = InverseKANNetwork()
model_A2 = InverseKANNetwork()
model_A3 = InverseKANNetwork()


opt_A1 = optim.Adam(model_A1.parameters(), lr=0.01)
opt_A2 = optim.Adam(model_A2.parameters(), lr=0.01)
opt_A3 = optim.Adam(model_A3.parameters(), lr=0.01)


num_epochs = 1500
frames = []


# Historiales para graficar cómo evoluciona el parámetro A
hist_A1, hist_A2, hist_A3 = [], [], []


def compute_inverse_loss(model, u_data_target):
   # 1. Pérdida Física (-u_xx + u = A_pred)
   u_pred = model(x_dom_tensor)
   u_x = torch.autograd.grad(u_pred, x_dom_tensor, torch.ones_like(u_pred), create_graph=True)[0]
   u_xx = torch.autograd.grad(u_x, x_dom_tensor, torch.ones_like(u_x), create_graph=True)[0]
  
   # Aquí introducimos la incógnita A_pred en la ecuación
   res_pde = -u_xx + u_pred - model.A_pred
   loss_pde = torch.mean(res_pde**2)
  
   # 2. Pérdida de Datos (ajuste a los sensores)
   u_data_pred = model(x_data)
   loss_data = torch.mean((u_data_pred - u_data_target)**2)
  
   # 3. Pérdida Borde
   loss_bc = torch.mean((model(x_bc) - u_bc)**2)
  
   # Priorizamos enormemente los datos y bordes para forzar al modelo a ajustar 'A'
   return loss_pde + 20.0 * loss_data + 10.0 * loss_bc


# ===============================
# 4. PREPARACIÓN DE GRÁFICOS
# ===============================
plt.ioff()
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Problema Inverso (PIKAN): Descubriendo el parámetro oculto A', fontsize=18, fontweight='bold', y=0.96)


print("Iniciando Problema Inverso para A=1, A=2 y A=3...")


for epoch in range(1, num_epochs + 1):
  
   # Entrenamiento Paso a Paso
   opt_A1.zero_grad(); loss1 = compute_inverse_loss(model_A1, u_data_A1); loss1.backward(); opt_A1.step()
   opt_A2.zero_grad(); loss2 = compute_inverse_loss(model_A2, u_data_A2); loss2.backward(); opt_A2.step()
   opt_A3.zero_grad(); loss3 = compute_inverse_loss(model_A3, u_data_A3); loss3.backward(); opt_A3.step()
  
   # Registrar el valor de A predicho en cada época
   hist_A1.append(model_A1.A_pred.item())
   hist_A2.append(model_A2.A_pred.item())
   hist_A3.append(model_A3.A_pred.item())
  
   # Visualización y captura GIF
   if epoch % 30 == 0 or epoch == 1:
       u_pred1 = model_A1(x_dom_tensor).detach().numpy()
       u_pred2 = model_A2(x_dom_tensor).detach().numpy()
       u_pred3 = model_A3(x_dom_tensor).detach().numpy()
      
       for ax in axs.flatten(): ax.clear()
      
       # --- FILA 1: CURVAS u(x) ---
       # A = 1
       axs[0, 0].plot(x_dom_np, solucion_exacta(x_dom_np, 1.0), 'k-', lw=3, alpha=0.2, label='Física Real')
       axs[0, 0].plot(x_dom_np, u_pred1, 'r--', lw=2.5, label='PIKAN')
       axs[0, 0].scatter(x_data_np, u_data_A1.numpy(), c='k', zorder=5, label='Datos')
       axs[0, 0].set_title(f"A = 1 (Predicho: {hist_A1[-1]:.3f})")
       axs[0, 0].set_ylim([-0.02, 0.15])
      
       # A = 2
       axs[0, 1].plot(x_dom_np, solucion_exacta(x_dom_np, 2.0), 'k-', lw=3, alpha=0.2)
       axs[0, 1].plot(x_dom_np, u_pred2, 'g--', lw=2.5)
       axs[0, 1].scatter(x_data_np, u_data_A2.numpy(), c='k', zorder=5)
       axs[0, 1].set_title(f"A = 2 (Predicho: {hist_A2[-1]:.3f})")
       axs[0, 1].set_ylim([-0.02, 0.3])
      
       # A = 3
       axs[0, 2].plot(x_dom_np, solucion_exacta(x_dom_np, 3.0), 'k-', lw=3, alpha=0.2)
       axs[0, 2].plot(x_dom_np, u_pred3, 'b--', lw=2.5)
       axs[0, 2].scatter(x_data_np, u_data_A3.numpy(), c='k', zorder=5)
       axs[0, 2].set_title(f"A = 3 (Predicho: {hist_A3[-1]:.3f})")
       axs[0, 2].set_ylim([-0.02, 0.45])
      
       for ax in axs[0, :]:
           ax.grid(True, ls='--'); ax.set_ylabel('$u(x)$'); ax.set_xlabel('$x$')
       axs[0, 0].legend()
      
       # --- FILA 2: EVOLUCIÓN DEL PARÁMETRO A ---
       epocas_rango = range(1, epoch + 1)
      
       axs[1, 0].axhline(1.0, color='k', ls='-', alpha=0.5, lw=2)
       axs[1, 0].plot(epocas_rango, hist_A1, 'r-', lw=2.5)
       axs[1, 0].set_ylabel('Valor de A')
      
       axs[1, 1].axhline(2.0, color='k', ls='-', alpha=0.5, lw=2)
       axs[1, 1].plot(epocas_rango, hist_A2, 'g-', lw=2.5)
      
       axs[1, 2].axhline(3.0, color='k', ls='-', alpha=0.5, lw=2)
       axs[1, 2].plot(epocas_rango, hist_A3, 'b-', lw=2.5)
      
       for ax in axs[1, :]:
           ax.set_ylim([0, 3.5])
           ax.set_xlim([0, num_epochs])
           ax.set_xlabel('Época')
           ax.grid(True, ls='--')
          
       fig.tight_layout(rect=[0, 0.03, 1, 0.95])
       fig.canvas.draw()
      
       image_rgba = np.asarray(fig.canvas.buffer_rgba())
       frames.append(image_rgba[:, :, :3])


# ===============================
# 5. GUARDAR GIF
# ===============================
save_path = "/Users/molab/Documents/simnu/PIKAN_Inverso_Constante.gif"
os.makedirs(os.path.dirname(save_path), exist_ok=True)


imgs = [Image.fromarray(img) for img in frames]
imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)


print(f"\n¡Entrenamiento completado! GIF infinito guardado en:\n{save_path}")


print("\n=== VALORES FINALES DESCUBIERTOS ===")
print(f"Caso A=1 -> Descubierto A = {hist_A1[-1]:.5f}")
print(f"Caso A=2 -> Descubierto A = {hist_A2[-1]:.5f}")
print(f"Caso A=3 -> Descubierto A = {hist_A3[-1]:.5f}")
