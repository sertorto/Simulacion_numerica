import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===============================
# 1. PARÁMETROS
# ===============================
alpha = 0.1
epochs = 3000

# ===============================
# 2. ARQUITECTURAS
# ===============================

class KANLayer(nn.Module):
    def __init__(self, input_size, output_size, degree):
        super().__init__()
        self.degree = degree
        self.weights = nn.Parameter(torch.randn(output_size, input_size*(degree+1)) * 0.5)
        self.bias = nn.Parameter(torch.zeros(output_size, 1))

    def forward(self, x):
        Phi = [x**d for d in range(self.degree+1)]
        Phi = torch.cat(Phi, dim=1)
        return Phi @ self.weights.T + self.bias.T

class KANNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.kan1 = KANLayer(2, 20, 4)
        self.kan2 = KANLayer(20, 20, 4)
        self.kan3 = KANLayer(20, 20, 4)
        self.out = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.tanh(self.kan1(x))
        x = torch.tanh(self.kan2(x))
        x = torch.tanh(self.kan3(x))
        return self.out(x)

class PINN(nn.Module):
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
# 3. DATOS
# ===============================

N_f = 2500
N_ic = 500
N_bc = 500

# Física
x_f = torch.rand((N_f,1))
t_f = torch.rand((N_f,1))*0.5
XT_f = torch.cat([x_f,t_f], dim=1).requires_grad_(True)

# Inicial
x_ic = torch.rand((N_ic,1))
t_ic = torch.zeros((N_ic,1))
XT_ic = torch.cat([x_ic,t_ic], dim=1)
u_ic = torch.sin(np.pi*x_ic)

# Bordes
x_bc1 = torch.zeros((N_bc//2,1))
t_bc1 = torch.rand((N_bc//2,1))*0.5
XT_bc1 = torch.cat([x_bc1,t_bc1], dim=1)

x_bc2 = torch.ones((N_bc//2,1))
t_bc2 = torch.rand((N_bc//2,1))*0.5
XT_bc2 = torch.cat([x_bc2,t_bc2], dim=1)

# Malla visualización
x_vals = np.linspace(0,1,50)
t_vals = np.linspace(0,0.5,50)
X,T = np.meshgrid(x_vals,t_vals)
XT_plot = torch.tensor(np.hstack((X.reshape(-1,1),T.reshape(-1,1))), dtype=torch.float32)

U_exact = np.exp(-alpha*np.pi**2*T)*np.sin(np.pi*X)

# ===============================
# 4. MODELOS
# ===============================

model_kan = KANNetwork()
model_pinn = PINN()

opt_kan = optim.Adam(model_kan.parameters(), lr=0.005)
opt_pinn = optim.Adam(model_pinn.parameters(), lr=0.005)

# ===============================
# 5. FUNCIÓN DE PÉRDIDA
# ===============================

def loss_fn(model):
    u = model(XT_f)
    grads = torch.autograd.grad(u, XT_f,
                                grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_t = grads[:,1:2]
    u_x = grads[:,0:1]
    u_xx = torch.autograd.grad(u_x, XT_f,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:,0:1]
    pde = u_t - alpha*u_xx
    loss_pde = torch.mean(pde**2)
    loss_ic = torch.mean((model(XT_ic)-u_ic)**2)
    loss_bc1 = torch.mean(model(XT_bc1)**2)
    loss_bc2 = torch.mean(model(XT_bc2)**2)
    return loss_pde + loss_ic + loss_bc1 + loss_bc2

# ===============================
# 6. ENTRENAMIENTO + CREAR DOS GIFs SEPARADOS
# ===============================

loss_hist_kan = []
loss_hist_pinn = []
frames_loss = []
frames_approx = []

plt.ioff()

# Figuras para GIF de pérdida
fig_loss, ax_loss = plt.subplots(figsize=(8,5))

# Figuras para GIF de aproximaciones
fig_approx = plt.figure(figsize=(12,5))
ax_pinn = fig_approx.add_subplot(1, 2, 1, projection='3d')
ax_kan = fig_approx.add_subplot(1, 2, 2, projection='3d')

print("Entrenando...")

for epoch in range(1, epochs+1):

    # Entrenamiento KAN
    opt_kan.zero_grad()
    l_kan = loss_fn(model_kan)
    l_kan.backward()
    opt_kan.step()
    loss_hist_kan.append(l_kan.item())

    # Entrenamiento PINN
    opt_pinn.zero_grad()
    l_pinn = loss_fn(model_pinn)
    l_pinn.backward()
    opt_pinn.step()
    loss_hist_pinn.append(l_pinn.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | KAN: {l_kan.item():.6f} | PINN: {l_pinn.item():.6f}")

        # GIF pérdida
        ax_loss.clear()
        ax_loss.plot(loss_hist_kan, 'r', label='KAN')
        ax_loss.plot(loss_hist_pinn, 'b', label='PINN')
        ax_loss.set_yscale('log')
        ax_loss.set_title('Evolución de la pérdida')
        ax_loss.set_xlabel('Época')
        ax_loss.set_ylabel('Loss (log)')
        ax_loss.legend()
        ax_loss.grid(True)
        fig_loss.canvas.draw()
        img_loss = np.array(fig_loss.canvas.renderer.buffer_rgba())
        frames_loss.append(img_loss.copy())

        # GIF aproximaciones
        u_pinn_plot = model_pinn(XT_plot).detach().numpy().reshape(50,50)
        u_kan_plot = model_kan(XT_plot).detach().numpy().reshape(50,50)

        ax_pinn.clear()
        ax_pinn.plot_surface(X, T, U_exact, alpha=0.2, color='gray', edgecolor='none')
        ax_pinn.plot_surface(X, T, u_pinn_plot, alpha=0.8, color='blue', edgecolor='none')
        ax_pinn.set_title('Predicción PINN')
        ax_pinn.set_xlabel('x')
        ax_pinn.set_ylabel('t')
        ax_pinn.set_zlabel('u')
        ax_pinn.set_zlim([0, 1.2])

        ax_kan.clear()
        ax_kan.plot_surface(X, T, U_exact, alpha=0.2, color='gray', edgecolor='none')
        ax_kan.plot_surface(X, T, u_kan_plot, alpha=0.8, color='red', edgecolor='none')
        ax_kan.set_title('Predicción KAN')
        ax_kan.set_xlabel('x')
        ax_kan.set_ylabel('t')
        ax_kan.set_zlabel('u')
        ax_kan.set_zlim([0, 1.2])

        fig_approx.canvas.draw()
        img_approx = np.array(fig_approx.canvas.renderer.buffer_rgba())
        frames_approx.append(img_approx.copy())

# ===============================
# 7. GUARDAR LOS DOS GIFs
# ===============================

save_loss = os.path.join(os.getcwd(), "Heat_PINN_vs_KAN_Loss.gif")
save_approx = os.path.join(os.getcwd(), "Heat_PINN_vs_KAN_Approx.gif")

# Puedes ajustar fps para velocidad; 5 es buen punto medio
imageio.mimsave(save_loss, frames_loss, fps=5, loop=0)
imageio.mimsave(save_approx, frames_approx, fps=5, loop=0)

plt.close(fig_loss)
plt.close(fig_approx)
print(f"GIF de pérdida guardado en: {save_loss}")
print(f"GIF de aproximaciones guardado en: {save_approx}")
