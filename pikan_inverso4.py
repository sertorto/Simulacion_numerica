import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ===============================
# 1. ARQUITECTURA KAN (PIKAN INVERSA POTENCIADA)
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


class InverseKANNetwork3P(nn.Module):
   # RED MÁS POTENTE: 20 neuronas y grado 4
   def __init__(self, num_neurons=25, degree=4):
       super().__init__()
       self.kan1 = KANLayer(1, num_neurons, degree)
       self.kan2 = KANLayer(num_neurons, num_neurons, degree)
       self.kan2 = KANLayer(num_neurons, num_neurons, degree)
       self.out = nn.Linear(num_neurons, 1)
      
       # Inicializamos en 0.1 para evitar simetrías nulas al arrancar
       self.A_pred = nn.Parameter(torch.tensor([0.1]))
       self.B_pred = nn.Parameter(torch.tensor([0.1]))
       self.C_pred = nn.Parameter(torch.tensor([0.1]))
  
   def forward(self, x):
       x = torch.tanh(self.kan1(x))
       x = torch.tanh(self.kan2(x))
       return self.out(x)


# ===============================
# 2. SOLUCIÓN EXACTA Y DATOS
# ===============================
def solucion_exacta(x_np, A, B, C):
   sh1 = np.sinh(1.0)
   ch1 = np.cosh(1.0)
  
   c1 = -(C + 2.0 * A)
   c2 = ((C + 2.0 * A) * ch1 - 3.0 * A - B - C) / sh1
  
   u = c1 * np.cosh(x_np) + c2 * np.sinh(x_np) + A * x_np**2 + B * x_np + C + 2.0 * A
   return u


x_dom_np = np.linspace(0, 1, 200)
x_dom_tensor = torch.tensor(x_dom_np, dtype=torch.float32).view(-1, 1).requires_grad_(True)


x_data_np = np.linspace(0.05, 0.95, 20) # Subimos un poco los sensores a 20
x_data = torch.tensor(x_data_np, dtype=torch.float32).view(-1, 1)


u_data_1 = torch.tensor(solucion_exacta(x_data_np, A=1.0, B=1.0, C=1.0), dtype=torch.float32).view(-1, 1)
u_data_2 = torch.tensor(solucion_exacta(x_data_np, A=2.0, B=-1.0, C=0.0), dtype=torch.float32).view(-1, 1)
u_data_3 = torch.tensor(solucion_exacta(x_data_np, A=-1.0, B=2.0, C=1.0), dtype=torch.float32).view(-1, 1)


x_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
u_bc = torch.tensor([[0.0], [0.0]], dtype=torch.float32)


# ===============================
# 3. ENTRENAMIENTO
# ===============================
model_1 = InverseKANNetwork3P()
model_2 = InverseKANNetwork3P()
model_3 = InverseKANNetwork3P()


# LR MÁS BAJO: 0.002 para mayor estabilidad en el descenso
opt_1 = optim.Adam(model_1.parameters(), lr=0.002)
opt_2 = optim.Adam(model_2.parameters(), lr=0.002)
opt_3 = optim.Adam(model_3.parameters(), lr=0.002)


# MÁS ÉPOCAS: 5000 iteraciones
num_epochs = 5000
frames = []


def compute_inverse_loss(model, u_data_target):
   u_pred = model(x_dom_tensor)
   u_x = torch.autograd.grad(u_pred, x_dom_tensor, torch.ones_like(u_pred), create_graph=True)[0]
   u_xx = torch.autograd.grad(u_x, x_dom_tensor, torch.ones_like(u_x), create_graph=True)[0]
  
   F_pred = model.A_pred * (x_dom_tensor**2) + model.B_pred * x_dom_tensor + model.C_pred
   res_pde = -u_xx + u_pred - F_pred
   loss_pde = torch.mean(res_pde**2)
  
   loss_data = torch.mean((model(x_data) - u_data_target)**2)
   loss_bc = torch.mean((model(x_bc) - u_bc)**2)
  
   # Aumentamos ligeramente el peso de los datos para anclar mejor los coeficientes
   return loss_pde + 40.0 * loss_data + 20.0 * loss_bc


# ===============================
# 4. BUCLE Y GRÁFICOS
# ===============================
plt.ioff()
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('PIKAN Inversa Potenciada: Descubriendo $F(x) = Ax^2 + Bx + C$', fontsize=18, fontweight='bold', y=0.96)


print("Iniciando Problema Inverso Potenciado (5000 épocas)...")


for epoch in range(1, num_epochs + 1):
  
   opt_1.zero_grad(); loss1 = compute_inverse_loss(model_1, u_data_1); loss1.backward(); opt_1.step()
   opt_2.zero_grad(); loss2 = compute_inverse_loss(model_2, u_data_2); loss2.backward(); opt_2.step()
   opt_3.zero_grad(); loss3 = compute_inverse_loss(model_3, u_data_3); loss3.backward(); opt_3.step()
  
   # Capturamos frames cada 100 épocas para no hacer un GIF gigantesco
   if epoch % 100 == 0 or epoch == 1:
       u_pred1 = model_1(x_dom_tensor).detach().numpy()
       u_pred2 = model_2(x_dom_tensor).detach().numpy()
       u_pred3 = model_3(x_dom_tensor).detach().numpy()
      
       A1, B1, C1 = model_1.A_pred.item(), model_1.B_pred.item(), model_1.C_pred.item()
       A2, B2, C2 = model_2.A_pred.item(), model_2.B_pred.item(), model_2.C_pred.item()
       A3, B3, C3 = model_3.A_pred.item(), model_3.B_pred.item(), model_3.C_pred.item()
      
       for ax in axs.flatten(): ax.clear()
      
       # --- FILA 1: CURVAS u(x) ---
       axs[0, 0].plot(x_dom_np, solucion_exacta(x_dom_np, 1, 1, 1), 'k-', lw=3, alpha=0.2, label='Física Real')
       axs[0, 0].plot(x_dom_np, u_pred1, 'r--', lw=2.5, label='PIKAN')
       axs[0, 0].scatter(x_data_np, u_data_1.numpy(), c='k', zorder=5)
       axs[0, 0].set_title("Caso 1: $x^2+x+1$", fontsize=14)
       axs[0, 0].set_ylim([-0.05, 0.45])
      
       axs[0, 1].plot(x_dom_np, solucion_exacta(x_dom_np, 2, -1, 0), 'k-', lw=3, alpha=0.2)
       axs[0, 1].plot(x_dom_np, u_pred2, 'g--', lw=2.5)
       axs[0, 1].scatter(x_data_np, u_data_2.numpy(), c='k', zorder=5)
       axs[0, 1].set_title("Caso 2: $2x^2-x$", fontsize=14)
       axs[0, 1].set_ylim([-0.05, 0.1])
      
       axs[0, 2].plot(x_dom_np, solucion_exacta(x_dom_np, -1, 2, 1), 'k-', lw=3, alpha=0.2)
       axs[0, 2].plot(x_dom_np, u_pred3, 'b--', lw=2.5)
       axs[0, 2].scatter(x_data_np, u_data_3.numpy(), c='k', zorder=5)
       axs[0, 2].set_title("Caso 3: $-x^2+2x+1$", fontsize=14)
       axs[0, 2].set_ylim([-0.05, 0.25])
      
       for ax in axs[0, :]:
           ax.grid(True, ls='--'); ax.set_ylabel('$u(x)$'); ax.set_xlabel('$x$')
       axs[0, 0].legend()
      
       # --- FILA 2: EVOLUCIÓN F(x) ---
       F_real_1 = 1.0*x_dom_np**2 + 1.0*x_dom_np + 1.0
       F_pred_1 = A1*x_dom_np**2 + B1*x_dom_np + C1
       axs[1, 0].plot(x_dom_np, F_real_1, 'k-', lw=3, alpha=0.2, label='Real')
       axs[1, 0].plot(x_dom_np, F_pred_1, 'r-', lw=2.5, label='Predicha')
       axs[1, 0].set_title(f"A={A1:.2f}, B={B1:.2f}, C={C1:.2f}")
       axs[1, 0].set_ylim([-0.5, 3.5])
      
       F_real_2 = 2.0*x_dom_np**2 - 1.0*x_dom_np + 0.0
       F_pred_2 = A2*x_dom_np**2 + B2*x_dom_np + C2
       axs[1, 1].plot(x_dom_np, F_real_2, 'k-', lw=3, alpha=0.2)
       axs[1, 1].plot(x_dom_np, F_pred_2, 'g-', lw=2.5)
       axs[1, 1].set_title(f"A={A2:.2f}, B={B2:.2f}, C={C2:.2f}")
       axs[1, 1].set_ylim([-1.0, 1.5])
      
       F_real_3 = -1.0*x_dom_np**2 + 2.0*x_dom_np + 1.0
       F_pred_3 = A3*x_dom_np**2 + B3*x_dom_np + C3
       axs[1, 2].plot(x_dom_np, F_real_3, 'k-', lw=3, alpha=0.2)
       axs[1, 2].plot(x_dom_np, F_pred_3, 'b-', lw=2.5)
       axs[1, 2].set_title(f"A={A3:.2f}, B={B3:.2f}, C={C3:.2f}")
       axs[1, 2].set_ylim([-0.5, 2.5])
      
       for ax in axs[1, :]:
           ax.set_xlim([0, 1])
           ax.set_xlabel('$x$')
           ax.set_ylabel('$F(x)$')
           ax.grid(True, ls='--')
       axs[1, 0].legend()
          
       fig.tight_layout(rect=[0, 0.03, 1, 0.95])
       fig.canvas.draw()
      
       image_rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
       frames.append(image_rgba[:, :, :3])


# ===============================
# 5. GUARDAR GIF
# ===============================
save_path = "/Users/molab/Documents/simnu/PIKAN_Inverso_Ax2_Bx_C_Potenciado.gif"
os.makedirs(os.path.dirname(save_path), exist_ok=True)


imgs = [Image.fromarray(img) for img in frames]
imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)


print(f"\n¡Entrenamiento completado! GIF guardado en:\n{save_path}")


print("\n=== COEFICIENTES FINALES DESCUBIERTOS ===")
print(f"Caso 1 (Target: 1, 1, 1)   -> A={model_1.A_pred.item():.4f}, B={model_1.B_pred.item():.4f}, C={model_1.C_pred.item():.4f}")
print(f"Caso 2 (Target: 2, -1, 0)  -> A={model_2.A_pred.item():.4f}, B={model_2.B_pred.item():.4f}, C={model_2.C_pred.item():.4f}")
print(f"Caso 3 (Target: -1, 2, 1)  -> A={model_3.A_pred.item():.4f}, B={model_3.B_pred.item():.4f}, C={model_3.C_pred.item():.4f}")
