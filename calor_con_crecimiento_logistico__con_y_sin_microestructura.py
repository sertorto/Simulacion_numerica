import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(0)
np.random.seed(0)

# ==========================================
# 1. Parámetros
# ==========================================
T_final = 3.0
times = [0.0, 0.01, 0.1, 0.5, 1.0, 1.5, 3.0]

N_f, N_ic, N_bc, N_T = 1500, 400, 300, 300

# ==========================================
# 2. Red neuronal
# ==========================================
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
# 3. Utilidades
# ==========================================
def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True)[0]

def u0(x):
    return torch.sin(2.0 * np.pi * x)

# ==========================================
# 4. Muestreo
# ==========================================
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
# 5. PDEs
# ==========================================
def pde_micro(u, x, t, epsilon):
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    coef = (1.1 + torch.sin(np.pi * x / epsilon))
    reaccion = u * (1.0 - u)
    return torch.mean((coef * u_t - u_xx - reaccion)**2)

def pde_macro(u, x, t):
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    coef = 1.1
    reaccion = u * (1.0 - u)
    return torch.mean((coef * u_t - u_xx - reaccion)**2)

# ==========================================
# 6. Experimento
# ==========================================
def ejecutar_experimento(epsilon, usar_micro, usar_control):

    print(f"\n--- eps={epsilon} | micro={usar_micro} | control={usar_control} ---")

    net_u = MLP(2,1)
    net_f = MLP(1,1,32,2) if usar_control else None

    def pde(u,x,t):
        if usar_micro:
            return pde_micro(u,x,t,epsilon)
        else:
            return pde_macro(u,x,t)

    def loss():
        x, t = sample_interior(N_f)
        u = net_u(torch.cat([x,t],1))
        loss_pde = pde(u,x,t)

        # IC
        x0,t0 = sample_initial(N_ic)
        ic = torch.mean((net_u(torch.cat([x0,t0],1)) - u0(x0))**2)

        # BC izquierda
        xL,tL = sample_left(N_bc)
        bc0 = torch.mean(net_u(torch.cat([xL,tL],1))**2)

        # BC derecha
        xR,tR = sample_right(N_bc)
        uR = net_u(torch.cat([xR,tR],1))
        u_xR = grad(uR,xR)

        if usar_control:
            f = net_f(tR)
            bc1 = torch.mean((u_xR - f)**2)
        else:
            bc1 = torch.mean((u_xR)**2)

        # terminal
        if usar_control:
            xT,tT = sample_terminal(N_T)
            final = torch.mean(net_u(torch.cat([xT,tT],1))**2)
            reg = torch.mean(net_f(tR)**2)
        else:
            final = 0
            reg = 0

        return loss_pde + 30*ic + 100*bc0 + 20*bc1 + 50*final + 1e-3*reg

    params = list(net_u.parameters())
    if usar_control:
        params += list(net_f.parameters())

    opt = torch.optim.Adam(params, lr=1e-3)
    epochs = 3000 if usar_control else 2000

    for i in range(epochs):
        opt.zero_grad()
        l = loss()
        l.backward()
        opt.step()

    # ======================================
    # Evaluación
    # ======================================
    x_plot = np.linspace(0,1,300).reshape(-1,1)

    def eval_u(t_val):
        XT = np.hstack([x_plot, t_val*np.ones_like(x_plot)])
        XT = torch.tensor(XT, dtype=torch.float32)
        with torch.no_grad():
            return net_u(XT).numpy().flatten()

    t_plot = np.linspace(0, T_final, 200)

    u_norm = []
    for t in t_plot:
        u = eval_u(t)
        u_norm.append(np.sqrt(np.mean(u**2)))
    u_norm = np.array(u_norm)

    if usar_control:
        t_tensor = torch.tensor(t_plot.reshape(-1,1), dtype=torch.float32)
        with torch.no_grad():
            f = net_f(t_tensor).numpy().flatten()
        f_norm = np.abs(f)
    else:
        f = None
        f_norm = None

    # Plot evolución
    fig, axes = plt.subplots(1,len(times), figsize=(18,3))
    title = f"eps={epsilon} | micro={usar_micro} | control={usar_control}"
    fig.suptitle(title)

    for i,t in enumerate(times):
        u = eval_u(t)
        axes[i].plot(x_plot, u)
        axes[i].set_title(f"t={t}")
        axes[i].grid()

    plt.tight_layout()
    plt.show()

    return t_plot, u_norm, f_norm

# ==========================================
# 7. EJECUCIÓN CORREGIDA
# ==========================================
resultados = {}

# 🔹 CASOS SIN MICRO (solo una vez)
for control in [False, True]:
    key = ("macro", control)
    resultados[key] = ejecutar_experimento(
        epsilon=1, usar_micro=False, usar_control=control
    )

# 🔹 CASOS CON MICRO (para cada epsilon)
epsilons = [1, 0.1, 0.01]

for eps in epsilons:
    for control in [False, True]:
        key = (eps, "micro", control)
        resultados[key] = ejecutar_experimento(
            epsilon=eps, usar_micro=True, usar_control=control
        )

# ==========================================
# 8. NORMA DE u
# ==========================================
plt.figure(figsize=(10,5))

# macro
for control in [False, True]:
    t_plot, u_norm, _ = resultados[("macro", control)]
    plt.plot(t_plot, u_norm, label=f"macro, ctrl={control}", linewidth=3)

# micro
for eps in epsilons:
    for control in [False, True]:
        t_plot, u_norm, _ = resultados[(eps, "micro", control)]
        plt.plot(t_plot, u_norm, '--', label=f"eps={eps}, ctrl={control}")

plt.title("Norma L2 de la solución")
plt.xlabel("t")
plt.ylabel("||u||")
plt.legend()
plt.grid()
plt.show()

# ==========================================
# 9. NORMA DEL CONTROL
# ==========================================
plt.figure(figsize=(10,5))

# macro
t_plot, _, f_norm = resultados[("macro", True)]
plt.plot(t_plot, f_norm, label="macro", linewidth=3)

# micro
for eps in epsilons:
    t_plot, _, f_norm = resultados[(eps, "micro", True)]
    plt.plot(t_plot, f_norm, '--', label=f"eps={eps}")

plt.title("Norma del control")
plt.xlabel("t")
plt.ylabel("|f(t)|")
plt.legend()
plt.grid()
plt.show()
