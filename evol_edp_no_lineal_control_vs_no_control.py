import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(0)
np.random.seed(0)

# ==========================================
# 1. Parámetros y Redes Neuronales (Base del PDF)
# ==========================================
T_final = 3.0
times = np.arange(0.5, 3.1, 0.5)

class MLP(nn.Module):
    def __init__(self, inp, out, hidden=64, depth=3):
        super().__init__()
        layers = [nn.Linear(inp, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out)]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# Redes pedidas en el PDF
net_u_nc = MLP(2, 1).to('cpu')        # u(x,t) Sin control
net_u_c = MLP(2, 1).to('cpu')         # u(x,t) Con control
net_f = MLP(1, 1, 32, 2).to('cpu')    # f(t) Red Termostato (Control)

# ==========================================
# 2. Muestreo de Puntos
# ==========================================
def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

def u0(x):
    return torch.sin(np.pi * x)

N_f, N_ic, N_bc, N_T = 800, 200, 200, 200

def sample_interior(n):
    x = torch.rand(n, 1, requires_grad=True)
    t = T_final * torch.rand(n, 1, requires_grad=True)
    return x, t

def sample_initial(n):
    x = torch.rand(n, 1, requires_grad=True)
    t = torch.zeros(n, 1, requires_grad=True)
    return x, t

def sample_left(n):
    x = torch.zeros(n, 1, requires_grad=True)
    t = T_final * torch.rand(n, 1, requires_grad=True)
    return x, t

def sample_right(n):
    x = torch.ones(n, 1, requires_grad=True)
    t = T_final * torch.rand(n, 1, requires_grad=True)
    return x, t

def sample_terminal(n):
    x = torch.rand(n, 1, requires_grad=True)
    t = T_final * torch.ones(n, 1, requires_grad=True)
    return x, t


def loss_nc():
    x, t = sample_interior(N_f)
    u = net_u_nc(torch.cat([x, t], 1))
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    
    # -> EL TOQUE DE VUESTRO GRUPO: Añadimos la no linealidad u*(1-u)
    # Ecuación: 10*u_t - u_xx - u*(1-u) = 0
    pde = torch.mean((10 * u_t - u_xx - u * (1.0 - u))**2) 
    
    x0, t0 = sample_initial(N_ic)
    ic = torch.mean((net_u_nc(torch.cat([x0, t0], 1)) - u0(x0))**2)
    
    x1, t1 = sample_left(N_bc)
    bc0 = torch.mean((net_u_nc(torch.cat([x1, t1], 1)))**2)
    
    xr, tr = sample_right(N_bc)
    u_r = net_u_nc(torch.cat([xr, tr], 1))
    u_x_r = grad(u_r, xr)
    bc1 = torch.mean((u_x_r)**2)
    
    return pde + 30*ic + 30*bc0 + 20*bc1

def loss_c():
    x, t = sample_interior(N_f)
    u = net_u_c(torch.cat([x, t], 1))
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    
    # -> EL TOQUE DE VUESTRO GRUPO: Añadimos la no linealidad u*(1-u)
    pde = torch.mean((10 * u_t - u_xx - u * (1.0 - u))**2)
    
    x0, t0 = sample_initial(N_ic)
    ic = torch.mean((net_u_c(torch.cat([x0, t0], 1)) - u0(x0))**2)
    
    x1, t1 = sample_left(N_bc)
    bc0 = torch.mean((net_u_c(torch.cat([x1, t1], 1)))**2)
    
    xr, tr = sample_right(N_bc)
    ur = net_u_c(torch.cat([xr, tr], 1))
    uxr = grad(ur, xr)
    f = net_f(tr)
    bc1 = torch.mean((uxr - f)**2) 
    
    xT, tT = sample_terminal(N_T)
    final = torch.mean((net_u_c(torch.cat([xT, tT], 1)))**2)
    
    reg = torch.mean((net_f(tr))**2)
    
    return pde + 30*ic + 30*bc0 + 20*bc1 + 120*final + 1e-3*reg

# ==========================================
# 4. Entrenamiento de las Redes
# ==========================================
print("Entrenando Caso 1 (No Lineal SIN Control)...")
opt_nc = torch.optim.Adam(net_u_nc.parameters(), lr=1e-3)
for i in range(1500):
    opt_nc.zero_grad()
    l = loss_nc()
    l.backward()
    opt_nc.step()

print("Entrenando Caso 2 (No Lineal CON Control IA)...")
opt_c = torch.optim.Adam(list(net_u_c.parameters()) + list(net_f.parameters()), lr=1e-3)
for i in range(2000):
    opt_c.zero_grad()
    l = loss_c()
    l.backward()
    opt_c.step()

print("[OK] ¡Entrenamiento completado!")

# ==========================================
# 5. Gráficas y Resultados 
# ==========================================
x_plot = np.linspace(0, 1, 300).reshape(-1, 1)

def eval_u(net, t):
    XT = np.hstack([x_plot, t * np.ones_like(x_plot)])
    XT = torch.tensor(XT, dtype=torch.float32)
    with torch.no_grad():
        u = net(XT).numpy().flatten()
    return u

# --- Comparativa 1x6 ---
fig, axes = plt.subplots(1, len(times), figsize=(18, 3.5), sharey=True)
fig.suptitle("Evolución de EDP No Lineal: Sin Control vs Con Control", fontsize=15, y=1.05)

for i, t in enumerate(times):
    u_nc = eval_u(net_u_nc, t)
    u_c = eval_u(net_u_c, t)
    axes[i].plot(x_plot, u_nc, label="Sin control", color='tab:blue')
    axes[i].plot(x_plot, u_c, label="Con control", color='tab:orange', linestyle='--')
    axes[i].set_title(f"t = {t}")
    axes[i].grid(True, alpha=0.5)
    axes[i].set_xlabel("x")
    if i == 0:
        axes[i].set_ylabel("Temperatura u(x,t)")

axes[-1].legend()
plt.tight_layout()
plt.show()

# --- Gráfica de la Función de Control f(t) ---
t_plot = np.linspace(0, T_final, 300).reshape(-1, 1)
t_tensor = torch.tensor(t_plot, dtype=torch.float32)
with torch.no_grad():
    f_plot = net_f(t_tensor).numpy().flatten()

plt.figure(figsize=(7, 4))
plt.plot(t_plot, f_plot, color='tab:red', linewidth=2)
plt.title("Acción de la Red de Control: Función $f(t)$ aprendida")
plt.xlabel("Tiempo (t)")
plt.ylabel("Flujo extraído $f(t)$")
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()
