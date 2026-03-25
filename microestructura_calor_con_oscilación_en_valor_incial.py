import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(0)
np.random.seed(0)

# ==========================================
# 1. Parámetros y Redes Neuronales
# ==========================================
T_final = 3.0
times = [0.0, 0.002, 0.01, 0.1, 0.5, 3.0]

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

# ==========================================
# 2. Utilidades y Muestreo
# ==========================================
def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

def u0_micro(x, epsilon):
    # Condición Inicial: Base macroscópica + Ruido microscópico
    return torch.sin(np.pi * x) + 0.5 * torch.sin(2.0 * np.pi * x / epsilon)

N_f, N_ic, N_bc, N_T = 1500, 400, 300, 300

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

# ==========================================
# 3. Función Principal de Experimento
# ==========================================
def ejecutar_experimento(epsilon):
    print(f"\n{'-'*50}")
    print(f"🚀 INICIANDO EXPERIMENTO PURA DIFUSIÓN CON EPSILON = {epsilon}")
    print(f"{'-'*50}")

    net_u_nc = MLP(2, 1).to('cpu')
    net_u_c = MLP(2, 1).to('cpu')
    net_f = MLP(1, 1, 32, 2).to('cpu')

    def pde_micro(u, x, t):
        u_t = grad(u, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        
        coef = 10.0 * (1.1 + torch.sin(np.pi * x / epsilon))
        reaccion = u * (1.0 - u)
        
        # Ecuación pura sin reducir la difusión
        return torch.mean((coef * u_t - u_xx - reaccion)**2)

    def loss_nc():
        x, t = sample_interior(N_f)
        u = net_u_nc(torch.cat([x, t], 1))
        pde = pde_micro(u, x, t)
        
        x0, t0 = sample_initial(N_ic)
        ic = torch.mean((net_u_nc(torch.cat([x0, t0], 1)) - u0_micro(x0, epsilon))**2)
        
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
        pde = pde_micro(u, x, t)
        
        x0, t0 = sample_initial(N_ic)
        ic = torch.mean((net_u_c(torch.cat([x0, t0], 1)) - u0_micro(x0, epsilon))**2)
        
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
        
        return pde + 30*ic + 30*bc0 + 20*bc1 + 150*final + 1e-3*reg

    epochs_nc = 1500
    epochs_c = 2500

    print("Entrenando Caso 1 (SIN Control)...")
    opt_nc = torch.optim.Adam(net_u_nc.parameters(), lr=1e-3)
    for i in range(epochs_nc):
        opt_nc.zero_grad()
        l = loss_nc()
        l.backward()
        opt_nc.step()

    print("Entrenando Caso 2 (CON Control IA)...")
    opt_c = torch.optim.Adam(list(net_u_c.parameters()) + list(net_f.parameters()), lr=1e-3)
    for i in range(epochs_c):
        opt_c.zero_grad()
        l = loss_c()
        l.backward()
        opt_c.step()

    print(f"[OK] Entrenamiento epsilon={epsilon} completado.")

    x_plot = np.linspace(0, 1, 300).reshape(-1, 1)

    def eval_u(net, t_val):
        XT = np.hstack([x_plot, t_val * np.ones_like(x_plot)])
        XT = torch.tensor(XT, dtype=torch.float32)
        with torch.no_grad():
            u = net(XT).numpy().flatten()
        return u

    # Evolución PDE (Se pinta para cada epsilon)
    fig, axes = plt.subplots(1, len(times), figsize=(20, 4), sharey=True)
    fig.suptitle(f"Difusión Pura con Microestructura Inicial ($\epsilon={epsilon}$)", fontsize=16, y=1.05)

    for i, t in enumerate(times):
        if t == 0.0:
            u_exacta = u0_micro(torch.tensor(x_plot), epsilon).numpy()
            axes[i].plot(x_plot, u_exacta, color='dimgray', lw=2, label="IC Exacta (t=0)")
        else:
            u_nc = eval_u(net_u_nc, t)
            u_c = eval_u(net_u_c, t)
            axes[i].plot(x_plot, u_nc, label="Sin control", color='tab:blue', lw=2)
            axes[i].plot(x_plot, u_c, label="Con control", color='tab:orange', linestyle='--', lw=2)
            
        axes[i].set_title(f"t = {t}")
        axes[i].grid(True, alpha=0.5)
        axes[i].set_xlabel("x")
        axes[i].set_ylim([-1.0, 2.0]) 
        
        if i == 0:
            axes[i].set_ylabel("u(x,t)")
            axes[i].legend(loc="upper right")
        elif i == 1:
            axes[i].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Extraemos y devolvemos el control f(t) para graficarlo luego
    t_plot = np.linspace(0, T_final, 300).reshape(-1, 1)
    t_tensor = torch.tensor(t_plot, dtype=torch.float32)
    with torch.no_grad():
        f_plot = net_f(t_tensor).numpy().flatten()
        
    return t_plot, f_plot

# ==========================================
# 4. Ejecutar y Comparar los Controles f(t)
# ==========================================
controles = {}
for eps in [0.1, 0.01]:
    t_plot, f_plot = ejecutar_experimento(eps)
    controles[eps] = f_plot

# Gráfica Final Comparativa del Control f(t)
plt.figure(figsize=(9, 5))
colores = {0.1: 'tab:blue', 0.01: 'tab:green'}

for eps, f_plot in controles.items():
    plt.plot(t_plot, f_plot, label=f'Control para $\epsilon={eps}$', color=colores[eps], linewidth=2)

plt.title("Comparativa del Control Macro $f(t)$ aprendido", fontsize=14)
plt.xlabel("Tiempo (t)", fontsize=12)
plt.ylabel("Flujo de extracción $f(t)$ en x=1", fontsize=12)
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.grid(True, alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
