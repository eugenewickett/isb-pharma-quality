from numpy import pi, sin
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button, RadioButtons

matplotlib.use('qt5agg')  # pycharm backend doesn't support interactive plots, so we use qt here

np.set_printoptions(precision = 2, suppress = True)

# Define policy function when all parameters are fixed
alph_0 = 0.8
b_0 = 0.7
c_0 = 0.075
w_0 = 0.1
lambretlo_0 = 0.65
lambrethi_0 = 0.95
sens_0 = 0.9
fpr_0 = 0.01
lambsup = 0.9
Ltheta_max = 8
Ltheta_vec = np.arange(0, Ltheta_max, 0.02)

def UtilSP(q1, q2, lambret, lambsup1, lambsup2, alph):
    # Social planner's utility
    return q1*(alph+(lambret*lambsup1)-1) + q2*(alph+(lambret*lambsup2)-1)

def UtilsRet(lambsup1, lambsup2, Ltheta, b, c, w1, w2, lambretlo, lambrethi, sens, fpr):
    util_list = []
    # What is retailer's utility as a function of different policies?
    # 0={Y12}, 1={N12}, 2={Y1}, 3={N1}
    # Policy {Y12}
    q1 = max((1 - b - c + b * c - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
    q2 = max((1 - b - c + b * c - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
    util_list.append((1 - b - c + b * c - w1 + b * w2) * (1 - c - w1 - b * q2 - q1) / (2 * (1 - b ** 2)) + \
                     (1 - b - c + b * c + b * w1 - w2) * (1 - c - w2 - b * q1 - q2) / (2 * (1 - b ** 2)) - \
                     Ltheta * (fpr + (sens - fpr) * (1 - lambrethi * lambsup1 * lambsup2)))

    # Policy {N12}
    q1 = max((1 - b - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
    q2 = max((1 - b - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
    util_list.append((1 - b - w1 + b * w2) * (1 - w1 - b * q2 - q1) / (2 * (1 - b ** 2)) + \
                     (1 - b + b * w1 - w2) * (1 - w2 - b * q1 - q2) / (2 * (1 - b ** 2)) - \
                     Ltheta * (fpr + (sens - fpr) * (1 - lambretlo * lambsup1 * lambsup2)))
    # Policy {Y1}
    util_list.append(0.5 * (1 - c - w1) * (1 - c - w1 + 0.5 * (-1 + c + w1)) - Ltheta * (
                fpr + (sens - fpr) * (1 - lambrethi * lambsup1)))
    # Policy {N1}
    util_list.append(0.5 * (1 - w1) * (1 - w1 + 0.5 * (-1 + w1)) - Ltheta * (
            fpr + (sens - fpr) * (1 - lambretlo * lambsup1)))
    # Policy {N}
    util_list.append(0)
    return util_list

def InducedPolicy(alph, lambsup, Ltheta_vec, b, c, w, lambretlo, lambrethi, sens, fpr):
    # Evaluate retailer's preferred policies WRT Ltheta_vec values add to a policy list if it's not yet included
    retpolicy_list = []
    for Ltheta in Ltheta_vec:
        curr_retutils = UtilsRet(lambsup, lambsup, Ltheta, b, c, w, w, lambretlo, lambrethi, sens, fpr)
        curr_retpolicy = np.argmax(curr_retutils)
        if not curr_retpolicy in retpolicy_list:
            retpolicy_list.append(curr_retpolicy)
    # Identify possible retailer policy that is preferred by SP
    SPutil_list = []
    for policy in retpolicy_list:
        if policy == 0:  # {Y12}
            q1 = max((1 - b - c + b * c - w + b * w) / (2 * (1 - (b ** 2))), 0)
            q2 = max((1 - b - c + b * c - w + b * w) / (2 * (1 - (b ** 2))), 0)
            SPutil_list.append(UtilSP(q1, q2, lambrethi, lambsup, lambsup, alph))
        if policy == 1:  # {N12}
            q1 = max((1 - b - w + b * w) / (2 * (1 - (b ** 2))), 0)
            q2 = max((1 - b - w + b * w) / (2 * (1 - (b ** 2))), 0)
            SPutil_list.append(UtilSP(q1, q2, lambretlo, lambsup, lambsup, alph))
        if policy == 2:  # {Y1}
            SPutil_list.append(UtilSP(0.5 * (1 - c - w), 0, lambrethi, lambsup, lambsup, alph))
        if policy == 3:  # {N1}
            SPutil_list.append(UtilSP(0.5 * (1 - w), 0, lambretlo, lambsup, lambsup, alph))
        if policy == 4:  # {N}
            SPutil_list.append(UtilSP(0, 0, lambretlo, lambsup, lambsup, alph))
    SPpreferpol = int(retpolicy_list[np.argmax(SPutil_list)])

    return int(SPpreferpol)

def RetPolicyWRTLtheta(lambsup, alph, Ltheta_vec, b, c, w, lambretlo, lambrethi, sens, fpr):
    # Returns an updated set of plottable policy lines for each value in Ltheta_vec
    retpolicy_mat = np.empty((5, Ltheta_vec.shape[0]))
    retpolicy_mat[:] = np.nan
    for Lthetaind, Ltheta in enumerate(Ltheta_vec):
        curr_retutils = UtilsRet(lambsup, lambsup, Ltheta, b, c, w, w, lambretlo, lambrethi, sens, fpr)
        # Obtain SPs utility for retailer's chosen policy
        if np.argmax(curr_retutils) == 0:  # {Y12}
            q1 = max((1 - b - c + b * c - w + b * w) / (2 * (1 - (b ** 2))), 0)
            q2 = max((1 - b - c + b * c - w + b * w) / (2 * (1 - (b ** 2))), 0)
            lambret = lambrethi
        if np.argmax(curr_retutils) == 1:  # {N12}
            q1 = max((1 - b - w + b * w) / (2 * (1 - (b ** 2))), 0)
            q2 = max((1 - b - w + b * w) / (2 * (1 - (b ** 2))), 0)
            lambret = lambretlo
        if np.argmax(curr_retutils) == 2:  # {Y1}
            q1, q2 = 0.5 * (1 - c - w), 0
            lambret = lambrethi
        if np.argmax(curr_retutils) == 3:  # {N1}
            q1, q2 = 0.5 * (1 - w), 0
            lambret = lambretlo
        if np.argmax(curr_retutils) == 4:  # {N}
            q1, q2 = 0, 0
            lambret = lambretlo
        retpolicy_mat[np.argmax(curr_retutils), Lthetaind] = UtilSP(q1, q2, lambret, lambsup, lambsup, alph)
    return retpolicy_mat

def inducedpolicyfuncforplot(numpts, Ltheta_vec, b, c, w, lambretlo, lambrethi, sens, fpr):
    # Returns an updated induced-policy matrix for the given parameters
    inducepolicymat = np.empty((numpts, numpts))
    for lsind, ls in enumerate(np.linspace(0.01, 0.99, numpts)):
        for aind, a in enumerate(np.linspace(0.01, 0.99, numpts)):
            inducepolicymat[lsind, aind] = int(round(InducedPolicy(a, ls, Ltheta_vec, b, c, w,
                                                         lambretlo, lambrethi, sens, fpr)))
            #print(str(ls) + ' ' + str(a) + ': ' + str(inducepolicymat[lsind, aind]))
    return inducepolicymat

numpts = 25  # Refinement along each axis
inducepolicymat = inducedpolicyfuncforplot(numpts, Ltheta_vec, b_0, c_0, w_0,
                                                     lambretlo_0, lambrethi_0, sens_0, fpr_0)


d = np.linspace(0.01, 0.99, numpts)
lambsupplier, al = np.meshgrid(d, d)

values = [0, 1, 2, 3, 4]
labels = ['{Y12}', '{N12}', '{Y1}', '{N1}', '{N}']
cmapname = 'viridis'

fig = plt.figure()
fig.suptitle('Scenario 1: '+r'$\Lambda_1=\Lambda_2,w_1=w_2$', fontsize=18, fontweight='bold')
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.subplots_adjust(bottom=0.5)
im = ax2.imshow(inducepolicymat, vmin=0, vmax=4,
                extent=(lambsupplier.min(), lambsupplier.max(), al.min(), al.max()),
                origin="lower", cmap=cmapname)
colors = [im.cmap(im.norm(value)) for value in values]

lambsup, alph = 0.8, 0.9
retpolicy_mat = RetPolicyWRTLtheta(lambsup, alph, Ltheta_vec, b_0, c_0, w_0, lambretlo_0, lambrethi_0, sens_0, fpr_0)

[line0] = ax1.plot(Ltheta_vec, retpolicy_mat[0], linewidth=7, color=colors[0])
[line1] = ax1.plot(Ltheta_vec, retpolicy_mat[1], linewidth=7, color=colors[1])
[line2] = ax1.plot(Ltheta_vec, retpolicy_mat[2], linewidth=7, color=colors[2])
[line3] = ax1.plot(Ltheta_vec, retpolicy_mat[3], linewidth=7, color=colors[3])
[line4] = ax1.plot(Ltheta_vec, retpolicy_mat[4], linewidth=7, color=colors[4])

ax1.set_xlim([0, Ltheta_max])
ax1.set_ylim([-0.6, 0.8])
ax1.set_title('Retailer strategy vs SP utility')
ax1.set_xlabel(r'$L_\theta$', fontsize=12)
ax1.set_ylabel(r'$U_{SP}$', rotation=0, fontsize=12, labelpad=3)


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

ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_title('SPs best induced retailer strategy')
ax2.set_xlabel(r'$\alpha$', fontsize=12)
ax2.set_ylabel(r'$\Lambda_{1}$', rotation=0, fontsize=12, labelpad=3)
# create a patch (proxy artist) for every color
patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]
# put those patched as legend-handles into the legend
ax2.legend(handles=patches, bbox_to_anchor=(-0.6, 1.0), loc='upper left', borderaxespad=0.1, fontsize=14)
def sliders_on_changed(val):
    retpolicy_mat = RetPolicyWRTLtheta(lambsup_slider.val, alpha_slider.val, Ltheta_vec, b_slider.val, c_slider.val,
                                       w_slider.val, lambretlo_slider.val, lambrethi_slider.val, sens_slider.val,
                                       fpr_slider.val)
    line0.set_ydata(retpolicy_mat[0])
    line1.set_ydata(retpolicy_mat[1])
    line2.set_ydata(retpolicy_mat[2])
    line3.set_ydata(retpolicy_mat[3])
    line4.set_ydata(retpolicy_mat[4])

    im.set_data(inducedpolicyfuncforplot(numpts, Ltheta_vec, b_slider.val, c_slider.val, w_slider.val,
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


# SCENARIO 5
d = np.linspace(0.01, 0.99, numpts)
lambsupplier, al = np.meshgrid(d, d)

values = [0, 1, 2, 3, 4]
labels = ['{Y12}', '{N12}', '{Y1}', '{N1}', '{N}']
cmapname = 'viridis'

fig = plt.figure()
fig.suptitle('Scenario 1: '+r'$\Lambda_1=\Lambda_2,w_1=w_2$', fontsize=18, fontweight='bold')
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.subplots_adjust(bottom=0.5)
im = ax2.imshow(inducepolicymat, vmin=0, vmax=4,
                extent=(lambsupplier.min(), lambsupplier.max(), al.min(), al.max()),
                origin="lower", cmap=cmapname)
colors = [im.cmap(im.norm(value)) for value in values]

lambsup, alph = 0.8, 0.9
retpolicy_mat = RetPolicyWRTLtheta(lambsup, alph, Ltheta_vec, b_0, c_0, w_0, lambretlo_0, lambrethi_0, sens_0, fpr_0)

[line0] = ax1.plot(Ltheta_vec, retpolicy_mat[0], linewidth=7, color=colors[0])
[line1] = ax1.plot(Ltheta_vec, retpolicy_mat[1], linewidth=7, color=colors[1])
[line2] = ax1.plot(Ltheta_vec, retpolicy_mat[2], linewidth=7, color=colors[2])
[line3] = ax1.plot(Ltheta_vec, retpolicy_mat[3], linewidth=7, color=colors[3])
[line4] = ax1.plot(Ltheta_vec, retpolicy_mat[4], linewidth=7, color=colors[4])

ax1.set_xlim([0, Ltheta_max])
ax1.set_ylim([-0.6, 0.8])
ax1.set_title('Retailer strategy vs SP utility')
ax1.set_xlabel(r'$L_\theta$', fontsize=12)
ax1.set_ylabel(r'$U_{SP}$', rotation=0, fontsize=12, labelpad=3)


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

ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_title('SPs best induced retailer strategy')
ax2.set_xlabel(r'$\alpha$', fontsize=12)
ax2.set_ylabel(r'$\Lambda_{1}$', rotation=0, fontsize=12, labelpad=3)
# create a patch (proxy artist) for every color
patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]
# put those patched as legend-handles into the legend
ax2.legend(handles=patches, bbox_to_anchor=(-0.6, 1.0), loc='upper left', borderaxespad=0.1, fontsize=14)
def sliders_on_changed(val):
    retpolicy_mat = RetPolicyWRTLtheta(lambsup_slider.val, alpha_slider.val, Ltheta_vec, b_slider.val, c_slider.val,
                                       w_slider.val, lambretlo_slider.val, lambrethi_slider.val, sens_slider.val,
                                       fpr_slider.val)
    line0.set_ydata(retpolicy_mat[0])
    line1.set_ydata(retpolicy_mat[1])
    line2.set_ydata(retpolicy_mat[2])
    line3.set_ydata(retpolicy_mat[3])
    line4.set_ydata(retpolicy_mat[4])

    im.set_data(inducedpolicyfuncforplot(numpts, Ltheta_vec, b_slider.val, c_slider.val, w_slider.val,
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











