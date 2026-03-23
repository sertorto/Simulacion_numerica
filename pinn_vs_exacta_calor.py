import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Configuración
# ------------------------------
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device("cpu")

# ------------------------------
# Dominio y condiciones
# ------------------------------
# Dominio espacial y temporal
x_dom = torch.linspace(0,1,200).view(-1,1)
t_dom = torch.linspace(0,1,200).view(-1,1)

# Puntos aleatorios para entrenamiento (collocation)
N_f = 10000
x_f = torch.rand(N_f,1)
t_f = 5*torch.rand(N_f,1)

# Condiciones iniciales
x0 = torch.linspace(0,1,100).view(-1,1)
t0 = torch.zeros_like(x0)
u0 = torch.sin(np.pi*x0)

# Condiciones de frontera
tb = torch.linspace(0,1,100).view(-1,1)
xb0 = torch.zeros_like(tb)
xb1 = torch.ones_like(tb)

# ------------------------------
# Red neuronal
# ------------------------------
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = torch.tanh(self.layers[i](x))
        return self.layers[-1](x)

# Entrada: (x,t) -> Salida: u(x,t)
pinn = PINN([2, 32, 32, 32, 1]).to(device)

# ------------------------------
# Función de pérdida
# ------------------------------
def loss_fn():
    # Colocación
    x_f_t = torch.cat([x_f, t_f], dim=1).requires_grad_(True)
    u = pinn(x_f_t)

    u_t = torch.autograd.grad(u, x_f_t, torch.ones_like(u), create_graph=True)[0][:,1:2]
    u_x = torch.autograd.grad(u, x_f_t, torch.ones_like(u), create_graph=True)[0][:,0:1]
    u_xx = torch.autograd.grad(u_x, x_f_t, torch.ones_like(u_x), create_graph=True)[0][:,0:1]

    f = -u_t + u_xx
    loss_f = torch.mean(f**2)

    # Condición inicial
    u0_pred = pinn(torch.cat([x0,t0],dim=1))
    loss_ic = torch.mean((u0_pred - u0)**2)

    # Condición frontera
    ub0_pred = pinn(torch.cat([xb0,tb],dim=1))
    ub1_pred = pinn(torch.cat([xb1,tb],dim=1))
    loss_bc = torch.mean(ub0_pred**2) + torch.mean(ub1_pred**2)

    return loss_f + 50*loss_ic + 50*loss_bc

# ------------------------------
# Optimización
# ------------------------------
optimizer = optim.Adam(pinn.parameters(), lr=0.01)
epochs = 2000

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss = {loss.item():.6e}")

# ------------------------------
# Predicción y visualización
# ------------------------------
# Asumiendo que tienes ya la red entrenada y haces la predicción como antes:

x_plot = np.linspace(0, 1, 100)
t_plot = np.arange(0, 0.5 + 0.1, 0.1)  # tiempos: 0,0.1,...,0.5
X, T = np.meshgrid(x_plot, t_plot)

X_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
T_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32)  # sin normalizar aquí

pinn.eval()
with torch.no_grad():
    u_pred = pinn(torch.cat([X_flat, T_flat], dim=1)).cpu().numpy().reshape(X.shape)

# Crear la cuadrícula para mostrar los resultados
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.flatten()

for i, t_val in enumerate(t_plot):
    axs[i].plot(x_plot, u_pred[i, :], 'r-', linewidth=2)
    axs[i].set_title(f't = {t_val:.1f}')
    axs[i].set_ylim([-0.1, 1.1])
    axs[i].grid(True)
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('u(x,t)')

plt.tight_layout()
plt.show()
