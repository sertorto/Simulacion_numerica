import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio


# Solucionar posible error de librerías duplicadas en entornos locales
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ===============================
# 1. DEFINICIÓN DE ARQUITECTURAS
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
   def __init__(self, input_dim=2, num_neurons=15, degree=4):
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
# 2. PREPARACIÓN DE DATOS (Calor)
# ===============================
N_physics = 2500
N_ic = 500
N_bc = 500
alpha_diff = 0.1


x_phys = torch.rand((N_physics, 1))
t_phys = torch.rand((N_physics, 1)) * 0.5
XT_phys = torch.cat([x_phys, t_phys], dim=1).requires_grad_(True)


x_ic = torch.rand((N_ic, 1))
t_ic = torch.zeros((N_ic, 1))
XT_ic = torch.cat([x_ic, t_ic], dim=1)
u_ic_target = torch.sin(torch.pi * x_ic)


x_bc0 = torch.zeros((N_bc // 2, 1))
t_bc0 = torch.rand((N_bc // 2, 1)) * 0.5
x_bc1 = torch.ones((N_bc // 2, 1))
t_bc1 = torch.rand((N_bc // 2, 1)) * 0.5
XT_bc = torch.cat([torch.cat([x_bc0, t_bc0], dim=1), torch.cat([x_bc1, t_bc1], dim=1)], dim=0)
u_bc_target = torch.zeros((N_bc, 1))


x_vals = np.linspace(0, 1, 50)
t_vals = np.linspace(0, 0.5, 50)
X_mesh, T_mesh = np.meshgrid(x_vals, t_vals)
XT_flat = np.hstack((X_mesh.reshape(-1, 1), T_mesh.reshape(-1, 1)))
XT_tensor_plot = torch.tensor(XT_flat, dtype=torch.float32)


U_exact = np.exp(-alpha_diff * (np.pi**2) * T_mesh) * np.sin(np.pi * X_mesh)


# ===============================
# 3. ENTRENAMIENTO Y CÁLCULO LOSS
# ===============================
model_pikan = KANNetwork()
model_pinn = PINN_Standard()


opt_pikan = optim.Adam(model_pikan.parameters(), lr=0.005)
opt_pinn = optim.Adam(model_pinn.parameters(), lr=0.005)


num_epochs = 2000
loss_hist_pikan = []
loss_hist_pinn = []
frames = [] # Lista real donde guardaremos las imágenes copiadas


def compute_loss(model):
   u_pred = model(XT_phys)
   du_dxt = torch.autograd.grad(outputs=u_pred, inputs=XT_phys,
                                grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
   u_x = du_dxt[:, 0:1]
   u_t = du_dxt[:, 1:2]
  
   du_x_dxt = torch.autograd.grad(outputs=u_x, inputs=XT_phys,
                                  grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
   u_xx = du_x_dxt[:, 0:1]
  
   res_pde = u_t - alpha_diff * u_xx
   loss_pde = 10.0 * torch.mean(res_pde**2)
  
   loss_ic = torch.mean((model(XT_ic) - u_ic_target)**2)
   loss_bc = torch.mean((model(XT_bc) - u_bc_target)**2)
  
   return loss_pde + loss_ic + loss_bc


# ===============================
# 4. GRÁFICOS Y BUCLE
# ===============================
plt.ioff()
fig = plt.figure(figsize=(18, 5))
ax_loss = fig.add_subplot(1, 3, 1)
ax_pinn = fig.add_subplot(1, 3, 2, projection='3d')
ax_pikan = fig.add_subplot(1, 3, 3, projection='3d')


print("Entrenando Ecuación del Calor (PINN vs PIKAN)...")


for epoch in range(1, num_epochs + 1):
   opt_pikan.zero_grad()
   loss_k = compute_loss(model_pikan)
   loss_k.backward()
   opt_pikan.step()
   loss_hist_pikan.append(loss_k.item())
  
   opt_pinn.zero_grad()
   loss_p = compute_loss(model_pinn)
   loss_p.backward()
   opt_pinn.step()
   loss_hist_pinn.append(loss_p.item())
  
   if epoch % 50 == 0 or epoch == 1:
       print(f"Epoch {epoch}/{num_epochs} | Loss PIKAN: {loss_k.item():.5f} | Loss PINN: {loss_p.item():.5f}")
      
       ax_loss.clear()
       ax_loss.plot(range(1, epoch + 1), loss_hist_pikan, 'r-', lw=2, label='PIKAN')
       ax_loss.plot(range(1, epoch + 1), loss_hist_pinn, 'b-', lw=2, label='PINN')
       ax_loss.set_yscale('log')
       ax_loss.set_xlim(0, num_epochs)
       ax_loss.set_title('Pérdidas (Difusión)')
       ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('Loss (Log)')
       ax_loss.legend(); ax_loss.grid(True)
      
       u_pinn_plot = model_pinn(XT_tensor_plot).detach().numpy().reshape(50, 50)
       u_pikan_plot = model_pikan(XT_tensor_plot).detach().numpy().reshape(50, 50)
      
       ax_pinn.clear()
       ax_pinn.plot_surface(X_mesh, T_mesh, U_exact, alpha=0.2, color='gray', edgecolor='none')
       ax_pinn.plot_surface(X_mesh, T_mesh, u_pinn_plot, alpha=0.8, color='blue', edgecolor='none')
       ax_pinn.set_title('Calor (PINN)')
       ax_pinn.set_zlim([0, 1.0])
      
       ax_pikan.clear()
       ax_pikan.plot_surface(X_mesh, T_mesh, U_exact, alpha=0.2, color='gray', edgecolor='none')
       ax_pikan.plot_surface(X_mesh, T_mesh, u_pikan_plot, alpha=0.8, color='red', edgecolor='none')
       ax_pikan.set_title('Calor (PIKAN)')
       ax_pikan.set_zlim([0, 1.0])
      
       fig.tight_layout()
       fig.canvas.draw()
      
       # --- LA SOLUCIÓN ESTÁ AQUÍ ---
       # Extraemos el buffer, lo pasamos a array, y CREAMOS UNA COPIA INDEPENDIENTE
       image_rgba = np.asarray(fig.canvas.buffer_rgba())
       image_rgb = np.copy(image_rgba[:, :, :3]) # <--- El .copy() evita que se sobreescriba
       frames.append(image_rgb)


# ===============================
# 5. GUARDAR GIF ANIMADO
# ===============================
save_path = "/Users/molab/Documents/simnu/PIKAN_Calor.gif"
os.makedirs(os.path.dirname(save_path), exist_ok=True)


# Guardamos la lista de frames independientes en un solo archivo GIF animado
imageio.mimsave(save_path, frames, fps=10, loop=0)
print(f"\n¡GIF animado guardado con éxito en:\n{save_path}")


plt.close(fig)
