# ============================================================
# CONTROL EN LA FRONTERA DE REACCION-DIFUSION 2D CON PINNs
# ============================================================
# ecuacion interior
# 10 u_t - (u_xx + u_yy) - u(1-u) = 0
# dominio
# x, y in (0,1) x (0,1)
# t in (0,3)
# dato inicial
# u(x,y,0) = sin(pi x) * sin(pi y)
# bordes fríos (Dirichlet)
# u(0,y,t) = u(x,0,t) = u(x,1,t) = 0
# borde controlado (Neumann en x=1)
# SIN control: u_x(1,y,t) = 0
# CON control: u_x(1,y,t) = f(t)
# objetivo
# u(x,y,3) ≈ 0

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

T_final = 3.0
times = np.arange(0.5, 3.1, 0.5)

# Tamaños muestreo
N_f = 2000
N_ic = 500
N_bc_dir = 900  # 300 por cada borde frío (x=0, y=0, y=1)
N_bc_neu = 300  # 300 para el borde controlado (x=1)
N_T = 500
epochs_nc = 2000
epochs_c = 3000

# ------------------------------------------------------------
# dato inicial
# ------------------------------------------------------------
def u0(x, y):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)

# ------------------------------------------------------------
# red neuronal
# ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, inp, out, hidden=64, depth=4):
        super().__init__()
        layers = [nn.Linear(inp, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

net_u_nc = MLP(3, 1).to(device)
net_u_c = MLP(3, 1).to(device)
net_f = MLP(1, 1, 32, 2).to(device)

# ------------------------------------------------------------
# derivadas
# ------------------------------------------------------------
def grad(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]

# ------------------------------------------------------------
# muestreo 2D por fronteras
# ------------------------------------------------------------
def sample_interior(n):
    x = torch.rand(n, 1, device=device, requires_grad=True)
    y = torch.rand(n, 1, device=device, requires_grad=True)
    t = T_final * torch.rand(n, 1, device=device, requires_grad=True)
    return x, y, t

def sample_initial(n):
    x = torch.rand(n, 1, device=device, requires_grad=True)
    y = torch.rand(n, 1, device=device, requires_grad=True)
    t = torch.zeros(n, 1, device=device, requires_grad=True)
    return x, y, t

def sample_dirichlet(n):
    # Bordes donde u=0 (x=0, y=0, y=1)
    n_edge = n // 3
    # Borde x=0
    x1 = torch.zeros(n_edge, 1, device=device, requires_grad=True)
    y1 = torch.rand(n_edge, 1, device=device, requires_grad=True)
    # Borde y=0
    x2 = torch.rand(n_edge, 1, device=device, requires_grad=True)
    y2 = torch.zeros(n_edge, 1, device=device, requires_grad=True)
    # Borde y=1
    x3 = torch.rand(n_edge, 1, device=device, requires_grad=True)
    y3 = torch.ones(n_edge, 1, device=device, requires_grad=True)
    
    x_bc = torch.cat([x1, x2, x3], dim=0)
    y_bc = torch.cat([y1, y2, y3], dim=0)
    t_bc = T_final * torch.rand(n_edge * 3, 1, device=device, requires_grad=True)
    return x_bc, y_bc, t_bc

def sample_neumann(n):
    # Borde controlado x=1
    x = torch.ones(n, 1, device=device, requires_grad=True)
    y = torch.rand(n, 1, device=device, requires_grad=True)
    t = T_final * torch.rand(n, 1, device=device, requires_grad=True)
    return x, y, t

def sample_terminal(n):
    x = torch.rand(n, 1, device=device, requires_grad=True)
    y = torch.rand(n, 1, device=device, requires_grad=True)
    t = T_final * torch.ones(n, 1, device=device, requires_grad=True)
    return x, y, t

# ------------------------------------------------------------
# perdida sin control
# ------------------------------------------------------------
def loss_nc():
    # 1. Interior
    x, y, t = sample_interior(N_f)
    u = net_u_nc(torch.cat([x, y, t], 1))
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_y = grad(u, y)
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)
    pde = torch.mean((10 * u_t - (u_xx + u_yy) - u * (1.0 - u))**2)
    
    # 2. Inicial
    x0, y0, t0 = sample_initial(N_ic)
    ic = torch.mean((net_u_nc(torch.cat([x0, y0, t0], 1)) - u0(x0, y0))**2)
    
    # 3. Dirichlet (u=0)
    xd, yd, td = sample_dirichlet(N_bc_dir)
    bc_dir = torch.mean(net_u_nc(torch.cat([xd, yd, td], 1))**2)
    
    # 4. Neumann (u_x=0)
    xn, yn, tn = sample_neumann(N_bc_neu)
    u_n = net_u_nc(torch.cat([xn, yn, tn], 1))
    u_x_n = grad(u_n, xn)
    bc_neu = torch.mean(u_x_n**2)
    
    return pde + 30 * ic + 30 * bc_dir + 20 * bc_neu

# ------------------------------------------------------------
# perdida con control
# ------------------------------------------------------------
def loss_c():
    # 1. Interior
    x, y, t = sample_interior(N_f)
    u = net_u_c(torch.cat([x, y, t], 1))
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_y = grad(u, y)
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)
    pde = torch.mean((10 * u_t - (u_xx + u_yy) - u * (1.0 - u))**2)
    
    # 2. Inicial
    x0, y0, t0 = sample_initial(N_ic)
    ic = torch.mean((net_u_c(torch.cat([x0, y0, t0], 1)) - u0(x0, y0))**2)
    
    # 3. Dirichlet (u=0)
    xd, yd, td = sample_dirichlet(N_bc_dir)
    bc_dir = torch.mean(net_u_c(torch.cat([xd, yd, td], 1))**2)
    
    # 4. Neumann Controlado (u_x = f(t))
    xn, yn, tn = sample_neumann(N_bc_neu)
    u_n = net_u_c(torch.cat([xn, yn, tn], 1))
    u_x_n = grad(u_n, xn)
    f_val = net_f(tn)
    bc_neu = torch.mean((u_x_n - f_val)**2)
    
    # 5. Objetivo Terminal
    xT, yT, tT = sample_terminal(N_T)
    final = torch.mean(net_u_c(torch.cat([xT, yT, tT], 1))**2)
    
    # 6. Regularización
    t_reg = T_final * torch.rand(N_f, 1, device=device)
    reg = torch.mean(net_f(t_reg)**2)
    
    return pde + 30 * ic + 30 * bc_dir + 20 * bc_neu + 150 * final + 1e-3 * reg

# ------------------------------------------------------------
# entrenamiento
# ------------------------------------------------------------
opt_nc = torch.optim.Adam(net_u_nc.parameters(), lr=1e-3)
print("Entrenando red 2D SIN control...")
for i in range(epochs_nc):
    opt_nc.zero_grad()
    l = loss_nc()
    l.backward()
    opt_nc.step()
    if (i+1) % 500 == 0:
        print(f"Época {i+1}/{epochs_nc} | Pérdida: {l.item():.5f}")

opt_c = torch.optim.Adam(
    list(net_u_c.parameters()) + list(net_f.parameters()),
    lr=1e-3
)
print("\nEntrenando red 2D CON control de frontera en x=1...")
for i in range(epochs_c):
    opt_c.zero_grad()
    l = loss_c()
    l.backward()
    opt_c.step()
    if (i+1) % 500 == 0:
        print(f"Época {i+1}/{epochs_c} | Pérdida: {l.item():.5f}")

# ------------------------------------------------------------
# evaluacion (Mapas de calor 2D)
# ------------------------------------------------------------
n_pts = 60
x_lin = np.linspace(0, 1, n_pts)
y_lin = np.linspace(0, 1, n_pts)
X_mesh, Y_mesh = np.meshgrid(x_lin, y_lin)

X_flat = X_mesh.reshape(-1, 1)
Y_flat = Y_mesh.reshape(-1, 1)

def eval_u_2d(net, t_val):
    T_flat = np.full_like(X_flat, t_val)
    XYT = np.hstack([X_flat, Y_flat, T_flat])
    XYT_tensor = torch.tensor(XYT, dtype=torch.float32, device=device)
    with torch.no_grad():
        u_pred = net(XYT_tensor).cpu().numpy()
    return u_pred.reshape(n_pts, n_pts)

# ------------------------------------------------------------
# COMPARACION: MAPAS DE CALOR (Filas: Caso, Columnas: Tiempo)
# ------------------------------------------------------------
fig, axes = plt.subplots(2, len(times), figsize=(18, 6), sharex=True, sharey=True)
fig.suptitle("Evolución Térmica 2D: Sin Control vs Con Control en Frontera ($x=1$)", fontsize=16)

vmin, vmax = 0.0, 1.0 

for i, t in enumerate(times):
    # Fila 0: Sin control
    u_nc = eval_u_2d(net_u_nc, t)
    im0 = axes[0, i].contourf(X_mesh, Y_mesh, u_nc, levels=40, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0, i].set_title(f"t={t}")
    if i == 0: axes[0, i].set_ylabel("y (Sin Control)")
    
    # Fila 1: Con control
    u_c = eval_u_2d(net_u_c, t)
    im1 = axes[1, i].contourf(X_mesh, Y_mesh, u_c, levels=40, cmap='inferno', vmin=vmin, vmax=vmax)
    if i == 0: axes[1, i].set_ylabel("y (Con Control)")
    axes[1, i].set_xlabel("x")
    
    # Resaltar la frontera controlada (x=1)
    axes[1, i].axvline(x=1.0, color='cyan', linestyle='--', lw=3)

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(im0, cax=cbar_ax, label="u(x,y,t)")

plt.show()

# ------------------------------------------------------------
# CONTROL APRENDIDO
# ------------------------------------------------------------
t_plot = np.linspace(0, T_final, 300).reshape(-1, 1)
t_tensor = torch.tensor(t_plot, dtype=torch.float32, device=device)

with torch.no_grad():
    f_plot = net_f(t_tensor).cpu().numpy().flatten()
    
plt.figure(figsize=(8, 4))
plt.plot(t_plot, f_plot, 'r-', lw=2)
plt.title("Evolución del Control $f(t)$ en la frontera $x=1$")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.grid(True)
plt.show()
