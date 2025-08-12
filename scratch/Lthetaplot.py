import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
import matplotlib.patheffects as pe

matplotlib.use('qt5agg')  # pycharm backend doesn't support interactive plots, so we use qt here

np.set_printoptions(precision=2, suppress=True)

# Define policy function when all parameters are fixed
alph_0 = 0.8
b_0 = 0.6
c_0 = 0.02
cSup_0 = 0.12
w_0 = 0.075
lambretlo_0, lambrethi_0 = 0.8, 0.9
sens_0, fpr_0 = 0.8, 0.01
lambsuplo_0, lambsuphi_0 = 0.85, 0.99
lambsup, lambsup1, lambsup2 = 0.9, 0.9, 0.9
Ltheta_max = 0.3

# Function returning retailer utilities under each of 7 possible policies
def UtilsRet_Scen5(lambsup1, lambsup2, Ltheta, b, c, w1, w2, lambretlo, lambrethi, sens, fpr):
    util_list = []
    # What is retailer's utility as a function of different policies?
    # Returns a list of utilities for each policy under the given inputs
    # 0={Y12}, 1={N12}, 2={Y1}, 3={N1}, 4={Y2}, 5={N2}, 6={N}
    # Policy {Y12}
    q1 = max((1 - b - c + b*c - w1 + b*w2) / (2 * (1-(b**2))), 0)
    q2 = max((1 - b - c + b*c - w2 + b*w1) / (2 * (1-(b**2))), 0)
    util_list.append((1 - b - c + b*c - w1 + b*w2)*(1 - c - w1 - b*q2 - q1) / (2 * (1-b**2)) + \
                     (1 - b - c + b*c + b*w1 - w2)*(1 - c - w2 - b*q1 - q2) / (2 * (1-b**2)) - \
                     Ltheta * (fpr + (sens - fpr) * (1 - lambrethi * lambsup1 * lambsup2)))

    # Policy {N12}
    q1 = max((1 - b - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
    q2 = max((1 - b - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
    util_list.append((1 - b - w1 + b*w2) * (1 - w1 - b*q2 - q1) / (2 * (1 - b ** 2)) + \
                     (1 - b + b*w1 - w2) * (1 - w2 - b*q1 - q2) / (2 * (1 - b ** 2)) - \
                     Ltheta * (fpr + (sens - fpr) * (1 - lambretlo * lambsup1 * lambsup2)))
    # Policy {Y1}
    util_list.append(0.5 * (1 - c - w1) * (1 - c - w1 + 0.5 * (-1 + c + w1)) - Ltheta * (
                fpr + (sens - fpr) * (1 - lambrethi * lambsup1)))
    # Policy {N1}
    util_list.append(0.5 * (1 - w1) * (1 - w1 + 0.5 * (-1 + w1)) - Ltheta * (
            fpr + (sens - fpr) * (1 - lambretlo * lambsup1)))
    # Policy {Y2}
    util_list.append(0.5 * (1 - c - w2) * (1 - c - w2 + 0.5 * (-1 + c + w2)) - Ltheta * (
            fpr + (sens - fpr) * (1 - lambrethi * lambsup2)))
    # Policy {N2}
    util_list.append(0.5 * (1 - w2) * (1 - w2 + 0.5 * (-1 + w2)) - Ltheta * (
            fpr + (sens - fpr) * (1 - lambretlo * lambsup2)))
    # Policy {N}
    util_list.append(0)
    return util_list

def RetOrderQuantsFromStrat(stratind, b, c, w1, w2):
    # Returns order quantities under inputs. If w2=-1, single sourcing quantity given
    strat = int(stratind)
    if strat == 0:
        q1 = max((1 - b - c + b * c - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
        q2 = max((1 - b - c + b * c - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
    elif strat == 1:
        q1 = max((1 - b - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
        q2 = max((1 - b - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
    elif strat == 2:
        q1 = max((1 - c - w1) / 2, 0)
        q2 = 0
    elif strat == 3:
        q1 = max((1 - w1) / 2, 0)
        q2 = 0
    elif strat == 4:
        q1 = 0
        q2 = max((1 - c - w2) / 2, 0)
    elif strat == 5:
        q1 = 0
        q2 = max((1 - w2) / 2, 0)
    elif strat == 6:
        q1 = 0
        q2 = 0
    return q1, q2

def SupSymPrices(b, c, cSup):
    # Returns wholesale prices under symmetric supplier quality levels
    w = max((1 - b - c + b * c + cSup) / (2 - b), 0)
    return w, w

def SupAsymPrices(b, c, cSup):
    # Returns wholesale prices under asymmetric supplier quality levels, assuming S1 is low-quality
    w1 = max(((-1+b) * (2+b-c*(2+b))-b*cSup ) / (-4+b**2), 0)
    w2 = max(((-1+b) * (2+b-c*(2+b))-2*cSup ) / (-4+b**2), 0)
    return w1, w2

def WholesalePricesFromEq(eqind, b, c, cSup):
    # Returns wholesale prices under given equilibrium
    eq = int(eqind)
    if eq == 0:  # {HHY12}
        w1, w2 = SupSymPrices(b, c, cSup)
    elif eq == 1:  # {HHN12}
        w1, w2 = SupSymPrices(b, 0, cSup)
    elif eq == 2:  # {LLY12}
        w1, w2 = SupSymPrices(b, c, 0)
    elif eq == 3:  # {LLN12}
        w1, w2 = SupSymPrices(b, 0, 0)
    elif eq == 4:  # {LHY12}
        w1, w2 = SupAsymPrices(b, c, cSup)
    elif eq == 5:  # {LHN12}
        w1, w2 = SupAsymPrices(b, 0, cSup)
    else:
        print('Invalid equilibrium')
    return w1, w2

def SupUtil(q, w, cSup, Ls, lambret, lambsup, sens, fpr):
    # Returns supplier utility
    return q*(w-cSup) - Ls*((sens-fpr)*(1-lambret*lambsup)+fpr)

numLpts = 40  # Refinement along each axis for plotting
Ltheta_vec = np.arange(0, Ltheta_max+Ltheta_max/numLpts, Ltheta_max/numLpts)

# Loop through possible equilibria and generate matrices demarcating where they are Nash equilibria
# 0:{HHY12}, 1:{HHN12}, 2:{LLY12}, 3:{LLN12}, 4:{LHY12}, 5:{LHN12}
numWpts = 30  # Refinement for deviation prices
w_vec = np.arange(1/numWpts, 1, 1/numWpts)

printUpdates = False

eq_list = ['HHY12', 'HHN12', 'LLY12', 'LLN12', 'LHY12', 'LHN12', 'HHY1']
eqMat_list = []

for eqind, eq_curr in enumerate(eq_list):
    # lambsup_eq
    if eq_curr in ['HHY12', 'HHN12']:
        lambsup1_eq = lambsuphi_0
        lambsup2_eq = lambsuphi_0
    elif eq_curr in ['LLY12', 'LLN12']:
        lambsup1_eq = lambsuplo_0
        lambsup2_eq = lambsuplo_0
    elif eq_curr in ['LHY12', 'LHN12']:
        lambsup1_eq = lambsuplo_0
        lambsup2_eq = lambsuphi_0
    # supplier sourcing
    if eq_curr in ['HHY12', 'HHN12', 'LLY12', 'LLN12', 'LHY12', 'LHN12']:
        bothSupsSourced = True
    else:
        bothSupsSourced = False
    # symmetric suppliers
    if eq_curr in ['HHY12', 'HHN12', 'LLY12', 'LLN12']:
        bothSupsSym = True
    elif eq_curr in ['LHY12', 'LHN12']:
        bothSupsSym = False
    # retailer strategy
    if eq_curr in ['HHY12', 'LLY12', 'LHY12']:
        retStrat_eq = 0
        lambret_eq = lambrethi_0
    elif eq_curr in ['HHN12', 'LLN12', 'LHN12']:
        retStrat_eq = 1
        lambret_eq = lambretlo_0

    # wholesale prices
    w1_eq, w2_eq = WholesalePricesFromEq(eqind, b_0, c_0, cSup_0)
    # Retailer order quantities
    q1_eq, q2_eq = RetOrderQuantsFromStrat(retStrat_eq, b_0, c_0, w1_eq, w2_eq)

    # Loop through Ltheta; rows are Lr, cols are Ls
    eq_mat = np.zeros((numLpts+1, numLpts+1))
    if printUpdates is True:
        print('Current eq point: '+eq_curr)
    for Lrind, Lr in enumerate(Ltheta_vec):
        if printUpdates is True:
            print('Lr: ' + str(Lr))
        for Lsind, Ls in enumerate(Ltheta_vec):
            if printUpdates is True:
                print('Ls: ' + str(Ls))
            # Initialize matrix storage value to 1, and update to 0 if any equilibrium conditions violated
            matVal = 1
            # Check that retailer prefers the equilibrium strategy
            utilRetVec_eq = UtilsRet_Scen5(lambsup1_eq, lambsup2_eq, Lr, b_0, c_0, w1_eq, w2_eq, lambretlo_0,
                                           lambrethi_0, sens_0, fpr_0)
            if np.argmax(utilRetVec_eq) != retStrat_eq:
                matVal = 0
                if printUpdates is True:
                    print('The retailer prefers strategy '+str(np.argmax(utilRetVec_eq)))
            # Check that suppliers have positive utility
            if eq_curr in ['HHY12', 'HHN12']:
                s1util_eq = SupUtil(q1_eq, w1_eq, cSup_0, Ls, lambret_eq, lambsup1_eq, sens_0, fpr_0)
                s2util_eq = SupUtil(q2_eq, w2_eq, cSup_0, Ls, lambret_eq, lambsup2_eq, sens_0, fpr_0)
            elif eq_curr in ['LLY12', 'LLN12']:
                s1util_eq = SupUtil(q1_eq, w1_eq, 0, Ls, lambret_eq, lambsup1_eq, sens_0, fpr_0)
                s2util_eq = SupUtil(q2_eq, w2_eq, 0, Ls, lambret_eq, lambsup2_eq, sens_0, fpr_0)
            elif eq_curr in ['LHY12', 'LHN12']:
                s1util_eq = SupUtil(q1_eq, w1_eq, 0, Ls, lambret_eq, lambsup1_eq, sens_0, fpr_0)
                s2util_eq = SupUtil(q2_eq, w2_eq, cSup_0, Ls, lambret_eq, lambsup2_eq, sens_0, fpr_0)

            if s1util_eq <= 0 or s2util_eq <= 0:
                matVal = 0
                if printUpdates is True:
                    print('A supplier utility is non-positive')
            # Now check if supplier wants to deviate
            if lambsup1_eq == lambsuphi_0:
                lambsup1_dev = lambsuplo_0
            elif lambsup1_eq == lambsuplo_0:
                lambsup1_dev = lambsuphi_0
            # Iterate through possible deviation prices for S1
            if matVal == 1:
                for w1_dev in w_vec:
                    if w1_dev != w1_eq:  # Deviation price cannot be the equilibrium price
                        # What is retailer policy under deviation?
                        utilRetVec_eq = UtilsRet_Scen5(lambsup1_dev, lambsup2_eq, Lr, b_0, c_0, w1_dev, w2_eq, lambretlo_0,
                                                       lambrethi_0, sens_0, fpr_0)
                        retStrat_dev = np.argmax(utilRetVec_eq)
                        if retStrat_dev in [0,2,4]:
                            lambret_dev = lambrethi_0
                        else:
                            lambret_dev = lambretlo_0
                        # Get q1 from retailer's preferred strategy
                        q1_dev, q2_dev = RetOrderQuantsFromStrat(retStrat_dev, b_0, c_0, w1_dev, w2_eq)
                        if lambsup1_dev == lambsuphi_0:
                            utilSup1_dev = SupUtil(q1_dev, w1_dev, cSup_0, Ls, lambret_dev, lambsup1_dev, sens_0, fpr_0)
                        elif lambsup1_dev == lambsuplo_0:
                            utilSup1_dev = SupUtil(q1_dev, w1_dev, 0, Ls, lambret_dev, lambsup1_dev, sens_0, fpr_0)
                        if utilSup1_dev > s1util_eq:
                            if printUpdates is True:
                                print('S1 does better with w1='+str(w1_dev))
                            matVal = 0
            # Check S2 if asymmetric
            if bothSupsSym is False and bothSupsSourced is True:
                if lambsup2_eq == lambsuphi_0:
                    lambsup2_dev = lambsuplo_0
                elif lambsup2_eq == lambsuplo_0:
                    lambsup2_dev = lambsuphi_0
                for w2_dev in w_vec:
                    if w2_dev != w2_eq:  # Deviation price cannot be the equilibrium price
                        # What is retailer policy under deviation?
                        utilRetVec_eq = UtilsRet_Scen5(lambsup1_eq, lambsup2_dev, Lr, b_0, c_0, w1_eq, w2_dev, lambretlo_0,
                                                       lambrethi_0, sens_0, fpr_0)
                        retStrat_dev = np.argmax(utilRetVec_eq)
                        if retStrat_dev in [0, 2, 4]:
                            lambret_dev = lambrethi_0
                        else:
                            lambret_dev = lambretlo_0
                        # Get q2 from retailer's preferred strategy
                        q1_dev, q2_dev = RetOrderQuantsFromStrat(retStrat_dev, b_0, c_0, w1_eq, w2_dev)
                        if lambsup2_dev == lambsuphi_0:
                            utilSup2_dev = SupUtil(q2_dev, w2_dev, cSup_0, Ls, lambret_dev, lambsup2_dev, sens_0, fpr_0)
                        elif lambsup2_dev == lambsuplo_0:
                            utilSup2_dev = SupUtil(q2_dev, w2_dev, 0, Ls, lambret_dev, lambsup2_dev, sens_0, fpr_0)
                        if utilSup2_dev > s2util_eq:
                            if printUpdates is True:
                                print('S2 does better with w2=' + str(w1_dev))
                            matVal = 0
            eq_mat[Lrind, Lsind] = matVal
    eqMat_list.append(eq_mat)

d = np.linspace(0, Ltheta_max, numLpts+1)

alval = 0.7  # Transparency for plots
values = [0, 1]
labels = ['{HHY12}', '{HHN12}', '{LLY12}', '{LLN12}', '{LHY12}', '{LHN12}']
eqcolors = ['royalblue', 'deepskyblue', 'red', 'pink', 'darkorange', 'bisque']

fig = plt.figure()
fig.suptitle('Nash equilibria vs. '+r'$L^{R}_{\theta}$'+', '+r'$L^{S}_{\theta}$', fontsize=18, fontweight='bold')
ax1 = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.3)

eqMat_list_plot = []
for eqM in eqMat_list:
    eqM_p = eqM.copy()
    eqM_p[eqM==0.] = np.nan
    eqMat_list_plot.append(eqM_p)

# colors = [im1.cmap(im1.norm(value)) for value in values]
imlist = []
for eqind in range(len(labels)):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[eqind]], name='from_list', N=None)
    im = ax1.imshow(eqMat_list_plot[eqind].T, vmin=0, vmax=1,
                    extent=(Ltheta_vec.min(), Ltheta_vec.max(), Ltheta_vec.min(), Ltheta_vec.max()),
                    origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)

ax1.set_xlabel(r'$L^{R}_{\theta}$', fontsize=12)
ax1.set_ylabel(r'$L^{S}_{\theta}$', rotation=0, fontsize=12, labelpad=3)
plt.show(block=True)

# Add sliders for changing the parameters
slstrtval = 0.43
b_slider_ax = fig.add_axes([0.1, slstrtval, 0.65, 0.02])
b_slider = Slider(b_slider_ax, 'b', 0.01, 0.99, valinit=b_0)
c_slider_ax = fig.add_axes([0.1, slstrtval-0.04, 0.65, 0.02])
c_slider = Slider(c_slider_ax, 'c', 0.01, 0.99, valinit=c_0)
w_slider_ax = fig.add_axes([0.1, slstrtval-0.08, 0.65, 0.02])
w_slider = Slider(w_slider_ax, 'w', 0.01, 0.99, valinit=w_0)
lambretlo_slider_ax = fig.add_axes([0.1, slstrtval-0.12, 0.65, 0.02])
lambretlo_slider = Slider(lambretlo_slider_ax, r'$\lambda^{lo}$', 0.01, 0.99, valinit=lambretlo_0)
lambrethi_slider_ax = fig.add_axes([0.1, slstrtval-0.16, 0.65, 0.02])
lambrethi_slider = Slider(lambrethi_slider_ax, r'$\lambda^{hi}$', 0.01, 0.99, valinit=lambrethi_0)
sens_slider_ax = fig.add_axes([0.1, slstrtval-0.2, 0.65, 0.02])
sens_slider = Slider(sens_slider_ax, r'$\rho$', 0.5, 0.99, valinit=sens_0)
fpr_slider_ax = fig.add_axes([0.1, slstrtval-0.24, 0.65, 0.02])
fpr_slider = Slider(fpr_slider_ax, r'$\phi$', 0.01, 0.2, valinit=fpr_0)

lambsup_slider_ax = fig.add_axes([0.3, slstrtval-0.31, 0.35, 0.02])
lambsup_slider = Slider(lambsup_slider_ax, r'$\Lambda$', 0.01, 0.99, valinit=lambsup)
alpha_slider_ax = fig.add_axes([0.3, slstrtval-0.35, 0.35, 0.02])
alpha_slider = Slider(alpha_slider_ax, r'$\alpha$', 0.01, 0.99, valinit=alph)


# create a patch (proxy artist) for every color
patches = [mpatches.Patch(color=colors[i], edgecolor='black', label=labels[i]) for i in range(len(values))]
# put those patched as legend-handles into the legend
ax2.legend(handles=patches, bbox_to_anchor=(-0.6, 1.0), loc='upper left', borderaxespad=0.1, fontsize=14)
def sliders_on_changed(val):
    retpolicy_mat = RetPolicyWRTLtheta_Scen1(lambsup_slider.val, alpha_slider.val, Ltheta_vec, b_slider.val, c_slider.val,
                                       w_slider.val, lambretlo_slider.val, lambrethi_slider.val, sens_slider.val,
                                       fpr_slider.val)
    line0.set_ydata(retpolicy_mat[0])
    line1.set_ydata(retpolicy_mat[1])
    line2.set_ydata(retpolicy_mat[2])
    line3.set_ydata(retpolicy_mat[3])
    line4.set_ydata(retpolicy_mat[4])

    minmaxgap = np.nanmax(retpolicy_mat) - np.nanmin(retpolicy_mat)
    ax1.set_ylim([np.nanmin(retpolicy_mat) - 0.05*minmaxgap, np.nanmax(retpolicy_mat) + 0.05*minmaxgap])

    im.set_data(inducedpolicyfuncforplot_Scen1(numpts, Ltheta_vec, b_slider.val, c_slider.val, w_slider.val,
                                         lambretlo_slider.val, lambrethi_slider.val, sens_slider.val,
                                         fpr_slider.val))
    fig.canvas.draw_idle()
b_slider.on_changed(sliders_on_changed)
c_slider.on_changed(sliders_on_changed)
w_slider.on_changed(sliders_on_changed)
lambretlo_slider.on_changed(sliders_on_changed)
lambrethi_slider.on_changed(sliders_on_changed)
sens_slider.on_changed(sliders_on_changed)
fpr_slider.on_changed(sliders_on_changed)
lambsup_slider.on_changed(sliders_on_changed)
alpha_slider.on_changed(sliders_on_changed)
plt.show(block=True)

























######################
######################
# SCENARIO 5
######################
######################
w12_delt_0 = 0.05
lamb12_delt_0 = 0.05


lambsup1, lambsup2 = 0.9, 0.9-lamb12_delt_0
Ltheta = 0.12
UtilsRet_Scen5(lambsup1, lambsup2, Ltheta, b_0, c_0, w_0, w_0-w12_delt_0, lambretlo_0, lambrethi_0, sens_0, fpr_0)

alph = 0.9
b, c, w1, w2, lambretlo, lambrethi, sens, fpr = b_0, c_0, w_0, w_0-w12_delt_0,lambretlo_0,lambrethi_0,sens_0,fpr_0

def InducedPolicy_Scen5(alph, lambsup1, lambsup2, Ltheta_vec, b, c, w1, w2, lambretlo, lambrethi, sens, fpr):
    # Evaluate retailer's preferred policies WRT Ltheta_vec values, add to a policy list if it's not yet included
    retpolicy_list = []
    for Ltheta in Ltheta_vec:
        curr_retutils = UtilsRet_Scen5(lambsup1, lambsup2, Ltheta, b, c, w1, w2, lambretlo, lambrethi, sens, fpr)
        curr_retpolicy = np.argmax(curr_retutils)
        if not curr_retpolicy in retpolicy_list:
            retpolicy_list.append(curr_retpolicy)
    # Identify possible retailer policy that is preferred by SP
    SPutil_list = []
    for policy in retpolicy_list:
        if policy == 0:  # {Y12}
            q1 = max((1 - b - c + b * c - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
            q2 = max((1 - b - c + b * c - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
            SPutil_list.append(UtilSP(q1, q2, lambrethi, lambsup1, lambsup2, alph))
        if policy == 1:  # {N12}
            q1 = max((1 - b - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
            q2 = max((1 - b - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
            SPutil_list.append(UtilSP(q1, q2, lambretlo, lambsup1, lambsup2, alph))
        if policy == 2:  # {Y1}
            q1 = (1 - c - w1)/2
            q2 = 0
            SPutil_list.append(UtilSP(q1, q2, lambrethi, lambsup1, lambsup2, alph))
        if policy == 3:  # {N1}
            q1 = (1 - w1) / 2
            q2 = 0
            SPutil_list.append(UtilSP(q1, q2, lambretlo, lambsup1, lambsup2, alph))
        if policy == 4:  # {Y2}
            q1 = 0
            q2 = (1 - c - w2) / 2
            SPutil_list.append(UtilSP(q1, q2, lambrethi, lambsup1, lambsup2, alph))
        if policy == 5:  # {N2}
            q1 = 0
            q2 = (1 - w2) / 2
            SPutil_list.append(UtilSP(q1, q2, lambretlo, lambsup1, lambsup2, alph))
        if policy == 6:  # {N}
            SPutil_list.append(UtilSP(0, 0, lambrethi, lambsup1, lambsup2,  alph))
    SPpreferpol = int(retpolicy_list[np.argmax(SPutil_list)])

    return int(SPpreferpol)

def RetPolicyWRTLtheta_Scen5(lambsup1, lambsup2, alph, Ltheta_vec, b, c, w1, w2,
                             lambretlo, lambrethi, sens, fpr):
    # Returns an updated set of plottable policy lines for each value in Ltheta_vec
    retpolicy_mat = np.empty((7, Ltheta_vec.shape[0]))
    retpolicy_mat[:] = np.nan
    for Lthetaind, Ltheta in enumerate(Ltheta_vec):
        curr_retutils = UtilsRet_Scen5(lambsup1, lambsup2, Ltheta, b, c, w1, w2, lambretlo, lambrethi, sens, fpr)
        # Obtain SPs utility for retailer's chosen policy
        if np.argmax(curr_retutils) == 0:  # {Y12}
            q1 = max((1 - b - c + b * c - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
            q2 = max((1 - b - c + b * c - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
            lambret = lambrethi
        if np.argmax(curr_retutils) == 1:  # {N12}
            q1 = max((1 - b - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
            q2 = max((1 - b - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
            lambret = lambretlo
        if np.argmax(curr_retutils) == 2:  # {Y1}
            q1, q2 = 0.5 * (1 - c - w1), 0
            lambret = lambrethi
        if np.argmax(curr_retutils) == 3:  # {N1}
            q1, q2 = 0.5 * (1 - w1), 0
            lambret = lambretlo
        if np.argmax(curr_retutils) == 4:  # {Y2}
            q1, q2 = 0, 0.5 * (1 - c - w2)
            lambret = lambrethi
        if np.argmax(curr_retutils) == 5:  # {N2}
            q1, q2 = 0, 0.5 * (1 - w2)
            lambret = lambretlo
        if np.argmax(curr_retutils) == 6:  # {N}
            q1, q2 = 0, 0
            lambret = lambretlo
        retpolicy_mat[np.argmax(curr_retutils), Lthetaind] = UtilSP(q1, q2, lambret, lambsup1, lambsup2, alph)
    return retpolicy_mat

testmat = RetPolicyWRTLtheta_Scen5(lambsup1, lambsup2, alph, Ltheta_vec, b, c, w1, w2,lambretlo, lambrethi, sens, fpr)

def inducedpolforplot_Scen5_deltlamb(numpts, Ltheta_vec, lambsup1, b, c, w1, wdelt, lambretlo, lambrethi, sens, fpr):
    # Returns an updated induced-policy matrix for the given parameters
    inducepolicymat = np.empty((numpts, numpts))
    for lsdeltind, lsdelt in enumerate(np.linspace(0.01, lambsup1-0.01, numpts)):  # delta Lambda
        for aind, a in enumerate(np.linspace(0.01, 0.99, numpts)):  # alpha value for SP
            inducepolicymat[lsdeltind, aind] = int(round(InducedPolicy_Scen5(a, lambsup1, lambsup1-lsdelt, Ltheta_vec,
                                                                             b, c, w1, w1-wdelt, lambretlo, lambrethi,
                                                                             sens, fpr)))
            #print(str(ls) + ' ' + str(a) + ': ' + str(inducepolicymat[lsind, aind]))
    return inducepolicymat

def inducedpolforplot_Scen5_deltw(numpts, Ltheta_vec, lambsup1, lambdelt, b, c, w1, lambretlo, lambrethi, sens, fpr):
    # Returns an updated induced-policy matrix for the given parameters
    inducepolicymat = np.empty((numpts, numpts))
    for wdeltind, wdelt in enumerate(np.linspace(0.01, w1-0.01, numpts)):  # delta w
        for aind, a in enumerate(np.linspace(0.01, 0.99, numpts)):  # alpha value for SP
            inducepolicymat[wdeltind, aind] = int(round(InducedPolicy_Scen5(a, lambsup1, lambsup1-lambdelt, Ltheta_vec,
                                                                             b, c, w1, w1-wdelt, lambretlo, lambrethi,
                                                                             sens, fpr)))
            #print(str(ls) + ' ' + str(a) + ': ' + str(inducepolicymat[lsind, aind]))
    return inducepolicymat

numpts = 15  # MODIFY AS NEEDED

inducepolicymat_deltlamb = inducedpolforplot_Scen5_deltlamb(numpts, Ltheta_vec, lambsup1, b_0, c_0, w_0, w12_delt_0,
                                                     lambretlo_0, lambrethi_0, sens_0, fpr_0)

inducepolicymat_wlamb = inducedpolforplot_Scen5_deltw(numpts, Ltheta_vec, lambsup1, lamb12_delt_0, b_0, c_0, w_0,
                                                     lambretlo_0, lambrethi_0, sens_0, fpr_0)

d1 = np.linspace(0.01, 0.99, numpts)
d2_lamb = np.linspace(0.01, lambsup1-0.01, numpts)
d2_w = np.linspace(0.01, w1-0.01, numpts)

al_lambgrid, deltlambgrid = np.meshgrid(d1, d2_lamb)
al_wgrid, deltwgrid = np.meshgrid(d1, d2_w)

values = [0, 1, 2, 3, 4, 5, 6]
labels = ['{Y12}', '{N12}', '{Y1}', '{N1}', '{Y2}', '{N2}', '{N}']
cmapname = 'viridis'

fig = plt.figure(figsize=(20,10))
fig.suptitle('Scenario 5: '+r'$\Lambda_1=\Lambda_2+\Delta\Lambda,w_1=w_2+\Delta w$', fontsize=18, fontweight='bold')
ax1 = plt.subplot2grid((3, 2), (0,0), colspan=2)
ax2 = plt.subplot2grid((3, 2), (1,0), rowspan=2)
ax3 = plt.subplot2grid((3, 2), (1,1), rowspan=2)

fig.subplots_adjust(bottom=0.3)

im2 = ax2.imshow(inducepolicymat_deltlamb, vmin=0, vmax=values[-1],
                extent=(al_lambgrid.min(), al_lambgrid.max(), deltlambgrid.min(), deltlambgrid.max()),
                origin="lower", cmap=cmapname, interpolation='none')
im3 = ax3.imshow(inducepolicymat_wlamb, vmin=0, vmax=values[-1],
                extent=(al_wgrid.min(), al_wgrid.max(),deltwgrid.min(), deltwgrid.max()),
                origin="lower", cmap=cmapname, interpolation='none')
colors = [im2.cmap(im2.norm(value)) for value in values]

lambsup1, alph = 0.8, 0.9
retpolicy_mat = RetPolicyWRTLtheta_Scen5(lambsup1, lambsup1-lamb12_delt_0, alph, Ltheta_vec, b_0, c_0, w_0,
                                         w_0-w12_delt_0, lambretlo_0, lambrethi_0, sens_0, fpr_0)

[line0] = ax1.plot(Ltheta_vec, retpolicy_mat[0], linewidth=7, color=colors[0])
[line1] = ax1.plot(Ltheta_vec, retpolicy_mat[1], linewidth=7, color=colors[1])
[line2] = ax1.plot(Ltheta_vec, retpolicy_mat[2], linewidth=7, color=colors[2])
[line3] = ax1.plot(Ltheta_vec, retpolicy_mat[3], linewidth=7, color=colors[3])
[line4] = ax1.plot(Ltheta_vec, retpolicy_mat[4], linewidth=7, color=colors[4])
[line5] = ax1.plot(Ltheta_vec, retpolicy_mat[5], linewidth=7, color=colors[5])
[line6] = ax1.plot(Ltheta_vec, retpolicy_mat[6], linewidth=7, color=colors[6])

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable

ax1.set_xlim([0, 2.0])
ax1.set_ylim([-0.6, 0.8])
ax1.set_title('SP utility vs retailer strategy, as function of '+r'$L_{\theta}$')
ax1.set_xlabel(r'$L_\theta$', fontsize=12)
ax1.set_ylabel(r'$U_{SP}$', rotation=0, fontsize=12, labelpad=15)

# Add sliders for changing the parameters
slstrtval = 0.28
slht = 0.01
slvertgap = 0.02
b_slider_ax = fig.add_axes([0.1, slstrtval, 0.65, slht])
b_slider = Slider(b_slider_ax, 'b', 0.01, 0.99, valinit=b_0)
c_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap, 0.65, slht])
c_slider = Slider(c_slider_ax, 'c', 0.01, 0.99, valinit=c_0)
w_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*2, 0.65, slht])
w_slider = Slider(w_slider_ax, r'$w_1$', 0.01, 0.99, valinit=w_0)
wdelt_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*3, 0.65, slht])
wdelt_slider = Slider(wdelt_slider_ax, r'$\Delta w$', 0.01, 0.5, valinit=w12_delt_0)
lambretlo_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*4, 0.65, slht])
lambretlo_slider = Slider(lambretlo_slider_ax, r'$\lambda^{lo}$', 0.01, 0.99, valinit=lambretlo_0)
lambrethi_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*5, 0.65, slht])
lambrethi_slider = Slider(lambrethi_slider_ax, r'$\lambda^{hi}$', 0.01, 0.99, valinit=lambrethi_0)
sens_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*6, 0.65, slht])
sens_slider = Slider(sens_slider_ax, r'$\rho$', 0.5, 0.99, valinit=sens_0)
fpr_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*7, 0.65, slht])
fpr_slider = Slider(fpr_slider_ax, r'$\phi$', 0.01, 0.2, valinit=fpr_0)
lambsup1_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*8, 0.65, slht])
lambsup1_slider = Slider(lambsup1_slider_ax, r'$\Lambda_1$', 0.01, 0.99, valinit=lambsup1)
lambdelt_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*9, 0.65, slht])
lambdelt_slider = Slider(lambdelt_slider_ax, r'$\Delta\Lambda$', 0.01, 0.5, valinit=lamb12_delt_0)

alpha_slider_ax = fig.add_axes([0.3, slstrtval-slvertgap*9-slht*4, 0.35, slht])
alpha_slider = Slider(alpha_slider_ax, r'$\alpha$', 0.01, 0.99, valinit=alph)


ax2.set_xlim([0, 1])
ax2.set_title('SPs best induced retailer strategy')
ax2.set_xlabel(r'$\alpha$', fontsize=12)
ax2.set_ylabel(r'$\Delta\Lambda$', rotation=0, fontsize=12, labelpad=15)

ax3.set_xlim([0, 1])
ax3.set_title('SPs best induced retailer strategy')
ax3.set_xlabel(r'$\alpha$', fontsize=12)
ax3.set_ylabel(r'$\Delta w$', rotation=0, fontsize=12, labelpad=15)

# create a patch (proxy artist) for every color
patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]
# put those patched as legend-handles into the legend
fig.legend(handles=patches, loc='upper left', borderaxespad=0.05, fontsize=12)
def sliders_on_changed(val):
    retpolicy_mat = RetPolicyWRTLtheta_Scen5(lambsup1_slider.val, lambsup1_slider.val-lambdelt_slider.val,
                                             alpha_slider.val, Ltheta_vec, b_slider.val, c_slider.val,
                                             w_slider.val, w_slider.val-wdelt_slider.val, lambretlo_slider.val,
                                             lambrethi_slider.val, sens_slider.val, fpr_slider.val)

    line0.set_ydata(retpolicy_mat[0])
    line1.set_ydata(retpolicy_mat[1])
    line2.set_ydata(retpolicy_mat[2])
    line3.set_ydata(retpolicy_mat[3])
    line4.set_ydata(retpolicy_mat[4])
    line5.set_ydata(retpolicy_mat[5])
    line6.set_ydata(retpolicy_mat[6])

    im2.set_data(inducedpolforplot_Scen5_deltlamb(numpts, Ltheta_vec, lambsup1_slider.val, b_slider.val, c_slider.val,
                                                  w_slider.val, wdelt_slider.val, lambretlo_slider.val,
                                                  lambrethi_slider.val, sens_slider.val, fpr_slider.val))
    im2.set_extent(extent=(al_wgrid.min(), al_wgrid.max(), deltlambgrid.min(), lambsup1_slider.val))
    im3.set_data(inducedpolforplot_Scen5_deltw(numpts, Ltheta_vec, lambsup1_slider.val, lambdelt_slider.val,
                                               b_slider.val, c_slider.val, w_slider.val, lambretlo_0, lambrethi_0,
                                               sens_0, fpr_0))
    im3.set_extent(extent=(al_wgrid.min(), al_wgrid.max(), deltwgrid.min(), w_slider.val))

    fig.canvas.draw_idle()

b_slider.on_changed(sliders_on_changed)
c_slider.on_changed(sliders_on_changed)
w_slider.on_changed(sliders_on_changed)
wdelt_slider.on_changed(sliders_on_changed)
lambretlo_slider.on_changed(sliders_on_changed)
lambrethi_slider.on_changed(sliders_on_changed)
sens_slider.on_changed(sliders_on_changed)
fpr_slider.on_changed(sliders_on_changed)
lambsup1_slider.on_changed(sliders_on_changed)
lambdelt_slider.on_changed(sliders_on_changed)
alpha_slider.on_changed(sliders_on_changed)

make_axes_area_auto_adjustable(ax1)
make_axes_area_auto_adjustable(ax2)
make_axes_area_auto_adjustable(ax3)

plt.show(block=True)











