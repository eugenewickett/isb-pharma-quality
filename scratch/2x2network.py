import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

np.set_printoptions(legacy='1.25')


numpts = 30
alph = 0.2
b = 2
edgeset = [2, 3, 4]

# x: supplier SFP prob
x = np.outer(np.linspace(0, 0.5, numpts), np.ones(numpts))
# y: retailer SFP prob
y = x.copy().T

# edges = 2
fig = plt.figure()
ax = plt.axes(projection='3d')
for edgenum in edgeset:
    if edgenum == 2:
        z = (2*x) - (x**2)
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='green', alpha=alph)
        z = 1 - ((1 - y) ** 2) * ((1 - x) ** 2)
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='orange', alpha=alph)
        z = y + (1-y)*x
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='blue', alpha=alph)
        ax.view_init(20, 150)

        ax.set_title('Prob ANY detection vs. $\lambda_S,\lambda_R$\nb='+str(b)+
             ' $|E|=$'+str(edgenum))
plt.xlabel('$\lambda_S$')
plt.ylabel('$\lambda_R$')
ax.set_zlabel('Prob detection')
ax.set_zlim([0, 1])
plt.show()

# edges = 3
fig = plt.figure()
ax = plt.axes(projection='3d')
for edgenum in edgeset:
    if edgenum == 3:
        z = (2*x) - (x**2)
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='green', alpha=alph)
        z = x
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='orange', alpha=alph)
        z = x + (1-x)*y
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='blue', alpha=alph)
        z = 1 - ((1 - y) ** 2) * ((1 - x) ** 2)
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='purple', alpha=alph)
        ax.view_init(20, 150)
        ax.set_title('Prob ANY detection vs. $\lambda_S,\lambda_R$\nb='+str(b)+
             ' $|E|=$'+str(edgenum))
plt.xlabel('$\lambda_S$')
plt.ylabel('$\lambda_R$')
ax.set_zlabel('Prob detection')
ax.set_zlim([0, 1])
plt.show()

# edges = 4
fig = plt.figure()
ax = plt.axes(projection='3d')
for edgenum in edgeset:
    if edgenum == 4:
        z = (2*x) - (x**2)
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='green', alpha=alph)
        z = 1 - (1-y)*((1-x)**2)
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='orange', alpha=alph)
        z = 1 - ((1 - y) ** 2) * ((1 - x) ** 2)
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='blue', alpha=alph)
        ax.view_init(20, 150)
        ax.set_title('Prob ANY detection vs. $\lambda_S,\lambda_R$\nb='+str(b)+
             ' $|E|=$'+str(edgenum))
plt.xlabel('$\lambda_S$')
plt.ylabel('$\lambda_R$')
ax.set_zlabel('Prob detection')
ax.set_zlim([0, 1])
plt.show()

# SOURCE detection, w increasing edges

fig = plt.figure()
ax = plt.axes(projection='3d')
for edgenum in edgeset:
    if edgenum == 2:
        z1 = y*(1-x) + x
        z2 = (2 * x) - (x ** 2)
        z = np.maximum(z1, z2)
        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='orange', alpha=alph)
    if edgenum == 4:

        ax.plot_surface(x, y, z2, cmap='viridis', edgecolor='green', alpha=alph)


    ax.view_init(10, 190)

ax.set_title('Prob SOURCE detection vs. $\lambda_S,\lambda_R$\nb='+str(b))
plt.xlabel('$\lambda_S$')
plt.ylabel('$\lambda_R$')
ax.set_zlabel('Prob detection')
ax.set_zlim([0, 1])
plt.show()

##############
# TRADE-OFF BETWEEN INSP QUALITY AND ADD'L TESTING
##############
# Consider 1 vs 2 tests in partial sourcing, 2x2 supply chain
# When does inspection quality drop enough that it's better to just get 1 test?
numpts = 30
insp_rho_vec = np.arange(0, 1, 0.001)
insp_rho = 0.85

rate_retail, rate_supply = 0.9, 0.3
rate_supplier = np.outer(np.linspace(0, 0.8, numpts), np.ones(numpts))
rate_retailer = rate_supplier.copy().T

lhs = 1 - (1-rate_retailer)*((1-rate_supplier)**2)
sens_vec = np.empty(lhs.shape)
for i in range(numpts):
    for j in range(numpts):
        currrate_supp = rate_supplier[i, j]
        currrate_ret = rate_retailer[i, j]

        temp = 1 - ((1-insp_rho_vec*currrate_ret)**2)*((1-insp_rho*currrate_supp)**2)
        try:
            ind = next(k for k in range(len(temp)) if temp[k] >= lhs[i, j])
            sens_vec[i, j] = temp[ind]
        except StopIteration:
            sens_vec[i, j] = np.nan
# rhs = 1 - ((1-insp_rho*rate_retailer)**2)*((1-insp_rho*rate_supplier)**2)

alph = 0.1
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(rate_retailer, rate_supplier, sens_vec, cmap='viridis',
                edgecolor='orange', alpha=alph)
# ax.plot_surface(rate_retailer, rate_supplier, lhs, cmap='viridis',
#                 edgecolor='green', alpha=alph)
ax.view_init(40, -210)
ax.set_title('Trade-off sensitivity vs. $\lambda_S,\lambda_R$')
plt.xlabel('$\lambda_R$')
plt.ylabel('$\lambda_S$')
ax.set_zlabel('Sensitivity')
ax.set_zlim([0, 1])
plt.show()

ind = next(i for i in range(len(rhs)) if rhs[i] > lhs)
print(rhs[ind])

