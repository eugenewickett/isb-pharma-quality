import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
import matplotlib.patheffects as pe

matplotlib.use('qt5agg')  # pycharm backend doesn't support interactive plots, so we use qt here

np.set_printoptions(precision=2, suppress=True)


###########################
# ASSUME LAMBDA_1 = LAMBDA_2
###########################
# Generate an interactive plot WRT Supplier 1's decision for w1

b_0 = 0.7
c_0 = 0.045
w2_0 = 0.2
lambretlo_0, lambrethi_0 = 0.65, 0.95
sens_0, fpr_0 = 0.9, 0.01
lambsup_0 = 0.9
Ltheta_max = 8
Ltheta_0 = 0.3
Ltheta_vec = np.arange(0, Ltheta_max, 0.02)

def UtilsRet(lambsup1, lambsup2, Ltheta, b, c, w1, w2, lambretlo, lambrethi, sens, fpr):
    util_list = []
    # What is retailer's utility as a function of different policies?
    # 0={Y12}, 1={N12}, 2={Y1}, 3={N1}, 4={Y2}, 5={N2}, 6={N}
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
    # Policy {Y2}
    util_list.append(0.5 * (1 - c - w2) * (1 - c - w2 + 0.5 * (-1 + c + w2)) - Ltheta * (
            fpr + (sens - fpr) * (1 - lambrethi * lambsup2)))
    # Policy {N2}
    util_list.append(0.5 * (1 - w2) * (1 - w2 + 0.5 * (-1 + w2)) - Ltheta * (
            fpr + (sens - fpr) * (1 - lambretlo * lambsup2)))
    # Policy {N}
    util_list.append(0)
    return util_list

UtilsRet(lambsup_0, lambsup_0, Ltheta_0, b_0, c_0, 0.1, w2_0, lambretlo_0, lambrethi_0, sens_0, fpr_0)

def UtilSup(q, w, c, qualInvest):
    # Supplier utility; qualInvest is a boolean indicating whether the quality investment was made at the supplier
    return q*(w-c*qualInvest)

def RetPolicyWRTw1(numpts, lambsup1, lambsup2, Ltheta, b, c, w1_dummy, w2, lambretlo, lambrethi, sens, fpr):
    # Returns an updated set of plottable policy lines for each value in Ltheta_vec
    w1_vec = np.linspace(0.001, 0.999, numpts)
    retpolicy_mat = np.empty((7, w1_vec.shape[0]))
    retpolicy_mat[:] = np.nan
    for w1ind, w1 in enumerate(w1_vec):
        curr_retutils = UtilsRet(lambsup1, lambsup2, Ltheta, b, c, w1, w2, lambretlo, lambrethi, sens, fpr)
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
        retpolicy_mat[np.argmax(curr_retutils), w1ind] = UtilSup(q1, w1, 0, 0)
    return retpolicy_mat

numpts = 500  # Refinement along each axis for plotting
w1_vec = np.linspace(0.001, 0.999, numpts)

retpolicy_mat = RetPolicyWRTw1(numpts, lambsup_0, lambsup_0, Ltheta_0, b_0, c_0, 0, w2_0, lambretlo_0,
                               lambrethi_0, sens_0, fpr_0)

values = [0, 1, 2, 3, 4, 5, 6]
labels = ['{Y12}', '{N12}', '{Y1}', '{N1}', '{Y2}', '{N2}', '{N}']
cmapname = 'viridis'
cmapobj = plt.get_cmap(cmapname)
colors = ['darkgreen', 'lime', 'darkblue', 'cornflowerblue', 'darkred', 'tomato', 'black']

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.5)

ax.set_xlim([0, 1])
minmaxgap = np.nanmax(retpolicy_mat) - np.nanmin(retpolicy_mat)
ax.set_ylim([np.nanmin(retpolicy_mat) - 0.05*minmaxgap, np.nanmax(retpolicy_mat) + 0.05*minmaxgap])
ax.set_title('Retailer strategy vs Supplier 1 utility')
ax.set_xlabel(r'$w_1$', fontsize=12)
ax.set_ylabel(r'$U_{S_1}$', rotation=0, fontsize=12, labelpad=3)

patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]
# put those patched as legend-handles into the legend
fig.legend(handles=patches, loc='upper left', borderaxespad=0.05, fontsize=12, title='Retailer strategy')

# Add sliders for changing the parameters
slstrtval = 0.43
b_slider_ax = fig.add_axes([0.1, slstrtval, 0.65, 0.02])
b_slider = Slider(b_slider_ax, r'$b$', 0.01, 0.99, valinit=b_0)
c_slider_ax = fig.add_axes([0.1, slstrtval-0.04, 0.65, 0.02])
c_slider = Slider(c_slider_ax, r'$c$', 0.01, 0.99, valinit=c_0)
w2_slider_ax = fig.add_axes([0.1, slstrtval-0.08, 0.65, 0.02])
w2_slider = Slider(w2_slider_ax, r'$w_2$', 0.01, 0.99, valinit=w2_0)
lambretlo_slider_ax = fig.add_axes([0.1, slstrtval-0.12, 0.65, 0.02])
lambretlo_slider = Slider(lambretlo_slider_ax, r'$\lambda^{lo}$', 0.01, 0.99, valinit=lambretlo_0)
lambrethi_slider_ax = fig.add_axes([0.1, slstrtval-0.16, 0.65, 0.02])
lambrethi_slider = Slider(lambrethi_slider_ax, r'$\lambda^{hi}$', 0.01, 0.99, valinit=lambrethi_0)
sens_slider_ax = fig.add_axes([0.1, slstrtval-0.2, 0.65, 0.02])
sens_slider = Slider(sens_slider_ax, r'$\rho$', 0.5, 0.99, valinit=sens_0)
fpr_slider_ax = fig.add_axes([0.1, slstrtval-0.24, 0.65, 0.02])
fpr_slider = Slider(fpr_slider_ax, r'$\phi$', 0.01, 0.2, valinit=fpr_0)
lambsup_slider_ax = fig.add_axes([0.1, slstrtval-0.28, 0.65, 0.02])
lambsup_slider = Slider(lambsup_slider_ax, r'$\Lambda$', 0.01, 0.99, valinit=lambsup_0)
Ltheta_slider_ax = fig.add_axes([0.1, slstrtval-0.32, 0.65, 0.02])
Ltheta_slider = Slider(Ltheta_slider_ax, r'$L_{\theta}$', 0.001, 4.0, valinit=Ltheta_0)


lindwd = 5
[line0] = ax.plot(w1_vec, retpolicy_mat[0], linewidth=lindwd, color=colors[0],
                   path_effects=[pe.Stroke(linewidth=lindwd+1, foreground='black'), pe.Normal()])
[line1] = ax.plot(w1_vec, retpolicy_mat[1], linewidth=lindwd, color=colors[1],
                   path_effects=[pe.Stroke(linewidth=lindwd+1, foreground='black'), pe.Normal()])
[line2] = ax.plot(w1_vec, retpolicy_mat[2], linewidth=lindwd, color=colors[2],
                   path_effects=[pe.Stroke(linewidth=lindwd+1, foreground='black'), pe.Normal()])
[line3] = ax.plot(w1_vec, retpolicy_mat[3], linewidth=lindwd, color=colors[3],
                   path_effects=[pe.Stroke(linewidth=lindwd+1, foreground='black'), pe.Normal()])
[line4] = ax.plot(w1_vec, retpolicy_mat[4], linewidth=lindwd, color=colors[4],
                   path_effects=[pe.Stroke(linewidth=lindwd+1, foreground='black'), pe.Normal()])
[line5] = ax.plot(w1_vec, retpolicy_mat[5], linewidth=lindwd, color=colors[5],
                   path_effects=[pe.Stroke(linewidth=lindwd+1, foreground='black'), pe.Normal()])
[line6] = ax.plot(w1_vec, retpolicy_mat[6], linewidth=lindwd, color=colors[6],
                   path_effects=[pe.Stroke(linewidth=lindwd+1, foreground='black'), pe.Normal()])

def sliders_on_changed(val):
    retpolicy_mat = RetPolicyWRTw1(numpts, lambsup_slider.val, lambsup_slider.val, Ltheta_slider.val,
                                   b_slider.val, c_slider.val, 0, w2_slider.val, lambretlo_slider.val,
                                   lambrethi_slider.val, sens_slider.val, fpr_slider.val)
    line0.set_ydata(retpolicy_mat[0])
    line1.set_ydata(retpolicy_mat[1])
    line2.set_ydata(retpolicy_mat[2])
    line3.set_ydata(retpolicy_mat[3])
    line4.set_ydata(retpolicy_mat[4])
    line5.set_ydata(retpolicy_mat[5])
    line6.set_ydata(retpolicy_mat[6])

    minmaxgap = np.nanmax(retpolicy_mat) - np.nanmin(retpolicy_mat)
    ax.set_ylim([np.nanmin(retpolicy_mat) - 0.05*minmaxgap, np.nanmax(retpolicy_mat) + 0.05*minmaxgap])

    fig.canvas.draw_idle()

b_slider.on_changed(sliders_on_changed)
c_slider.on_changed(sliders_on_changed)
w2_slider.on_changed(sliders_on_changed)
lambretlo_slider.on_changed(sliders_on_changed)
lambrethi_slider.on_changed(sliders_on_changed)
sens_slider.on_changed(sliders_on_changed)
fpr_slider.on_changed(sliders_on_changed)
lambsup_slider.on_changed(sliders_on_changed)
Ltheta_slider.on_changed(sliders_on_changed)
plt.show(block=True)






