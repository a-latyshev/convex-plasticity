import ufl
from dolfinx import fem, io, common

def eps(v):
    e = ufl.sym(ufl.grad(v))
    return ufl.as_tensor([[e[0, 0], e[0, 1], 0],
                          [e[0, 1], e[1, 1], 0],
                          [0, 0, 0]])
                            
def sigma(eps, lambda_: float, mu_: float):
    return lambda_ * ufl.tr(eps)*ufl.Identity(3) + 2*mu_ * eps

def as_3D_tensor(X):
    return ufl.as_tensor([[X[0], X[3], 0],
                          [X[3], X[1], 0],
                          [0, 0, X[2]]])
def ppos(x):
    return (x + ufl.sqrt(x**2))/2.

def proj_sig(deps, old_sig, old_p, lambda_: float, mu_: float, sig0: float, H: float):
    sig_n = as_3D_tensor(old_sig)
    sig_elas = sig_n + sigma(deps, lambda_, mu_)
    s = ufl.dev(sig_elas)
    sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
    f_elas = sig_eq - sig0 - H*old_p
    dp = ppos(f_elas) / (3*mu_ + H)
    n_elas = s/sig_eq * ppos(f_elas)/f_elas
    beta = 3*mu_ * dp / sig_eq
    new_sig = sig_elas - beta*s
    return ufl.as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
           ufl.as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]), \
           beta, dp     

def sigma_tang(e, n_elas, beta, lambda_: float, mu_: float, H: float):
    N_elas = as_3D_tensor(n_elas)
    return sigma(e, lambda_, mu_) - 3*mu_*(3*mu_ / (3*mu_ + H) - beta) * ufl.inner(N_elas, e)*N_elas - 2*mu_ * beta * ufl.dev(e) 

# def proj_sig(deps, old_sig, old_p):
#     sig_n = as_3D_tensor(old_sig)
#     sig_elas = sig_n + UtilityFuncs.sigma(deps)
#     s = ufl.dev(sig_elas)
#     sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
#     f_elas = sig_eq - sig0 - H*old_p
#     dp = ppos(f_elas)/(3*mu+H)
#     beta = 3*mu*dp/sig_eq
#     new_sig = sig_elas-beta*s
#     return ufl.as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
#     dp       
# self.proj_sig = proj_sig

def deps_p(deps, old_sig, old_p, lambda_: float, mu_: float, sig0: float, H: float):
    sig_n = as_3D_tensor(old_sig)
    sig_elas = sig_n + sigma(deps, lambda_, mu_)
    s = ufl.dev(sig_elas)
    sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
    f_elas = sig_eq - sig0 - H*old_p
    dp_sig_eq = ufl.conditional(f_elas > 0, f_elas/(3*mu_+H)/sig_eq, 0) # sig_eq is equal to 0 on the first iteration
    # dp = ppos(f_elas)/(3*mu+H) # this approach doesn't work with ufl.derivate
    return 3./2. * dp_sig_eq * s 