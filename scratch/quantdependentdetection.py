'''
This script aims to simulate the strategy choices by the retailer and social planner under quantity-dependent
detection probabilities for inspection. We hope to obtain plots similar to those derived in SPregionplot.py.
'''
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
b_0 = 0.8
c_0 = 0.04
w_0 = 0.075
lambretlo_0, lambrethi_0 = 0.6, 0.9
sens_0, fpr_0 = 0.75, 0.01
lambsup, lambsup1, lambsup2 = 0.9, 0.9, 0.9
Ltheta_max = 8
Ltheta_vec = np.arange(0, Ltheta_max, 0.02)

numpts = 50  # Refinement along each axis for plotting

def UtilSP(q1, q2, lambret, lambsup1, lambsup2, alph):
    # Social planner's utility
    return q1*(alph+(lambret*lambsup1)-1) + q2*(alph+(lambret*lambsup2)-1)

######################
######################
# SCENARIO 1
######################
######################
def UtilsRet_Scen1(lambsup1, lambsup2, Ltheta, b, c, w1, w2, lambretlo, lambrethi, sens, fpr):
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

def InducedPolicy_Scen1(alph, lambsup, Ltheta_vec, b, c, w, lambretlo, lambrethi, sens, fpr):
    # Evaluate retailer's preferred policies WRT Ltheta_vec values add to a policy list if it's not yet included
    retpolicy_list = []
    for Ltheta in Ltheta_vec:
        curr_retutils = UtilsRet_Scen1(lambsup, lambsup, Ltheta, b, c, w, w, lambretlo, lambrethi, sens, fpr)
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

def RetPolicyWRTLtheta_Scen1(lambsup, alph, Ltheta_vec, b, c, w, lambretlo, lambrethi, sens, fpr):
    # Returns an updated set of plottable policy lines for each value in Ltheta_vec
    retpolicy_mat = np.empty((5, Ltheta_vec.shape[0]))
    retpolicy_mat[:] = np.nan
    for Lthetaind, Ltheta in enumerate(Ltheta_vec):
        curr_retutils = UtilsRet_Scen1(lambsup, lambsup, Ltheta, b, c, w, w, lambretlo, lambrethi, sens, fpr)
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

def inducedpolicyfuncforplot_Scen1(numpts, Ltheta_vec, b, c, w, lambretlo, lambrethi, sens, fpr):
    # Returns an updated induced-policy matrix for the given parameters
    inducepolicymat = np.empty((numpts, numpts))
    for lsind, ls in enumerate(np.linspace(0.01, 0.99, numpts)):
        for aind, a in enumerate(np.linspace(0.01, 0.99, numpts)):
            inducepolicymat[lsind, aind] = int(round(InducedPolicy_Scen1(a, ls, Ltheta_vec, b, c, w,
                                                         lambretlo, lambrethi, sens, fpr)))
            #print(str(ls) + ' ' + str(a) + ': ' + str(inducepolicymat[lsind, aind]))
    return inducepolicymat

##############
# Here we verify we can replicate the region plot of strategy preferences for the retailer
##############
def regionplotfuncforretailer_Scen1(numpts, lambsup_max, w_max, Ltheta, b, c, lambretlo, lambrethi, sens, fpr):
    # Returns a region plot plotting matrix with wholesale price x-axis and quality-rate y-axis
    regionplotmat = np.empty((numpts, numpts))
    for wind, w in enumerate(np.linspace(0.01, w_max, numpts)):
        for lsind, ls in enumerate(np.linspace(0.01, lambsup_max, numpts)):
            regionplotmat[lsind, wind] = int(round(np.argmax(UtilsRet_Scen1(ls, ls, Ltheta, b, c, w, w,
                                                                            lambretlo, lambrethi, sens, fpr))))
    return regionplotmat

numpts = 200

Ltheta_0 = 0.08
regplotmat = regionplotfuncforretailer_Scen1(numpts, 0.99, 0.99, Ltheta_0, b_0, c_0, lambretlo_0, lambrethi_0, sens_0,
                                             fpr_0)
d = np.linspace(0.01, 0.99, numpts)
lambsupplier, wprice = np.meshgrid(d, d)
values = [0, 1, 2, 3, 4]
labels = ['{Y12}', '{N12}', '{Y1}', '{N1}', '{N}']
cmapname = 'plasma'
cmapname = 'Greys'

fig = plt.figure()
fig.suptitle('Scenario 1: '+r'$\Lambda_1=\Lambda_2,w_1=w_2$', fontsize=18, fontweight='bold')
ax1 = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.4)
im = ax1.imshow(regplotmat, vmin=-1, vmax=4,
                extent=(lambsupplier.min(), lambsupplier.max(), wprice.min(), wprice.max()),
                origin="lower", cmap=cmapname)
colors = [im.cmap(im.norm(value)) for value in values]

#ax1.set_xlim([0, 2])
#minmaxgap = np.nanmax(retpolicy_mat) - np.nanmin(retpolicy_mat)
#ax1.set_ylim([np.nanmin(retpolicy_mat) - 0.05*minmaxgap, np.nanmax(retpolicy_mat) + 0.05*minmaxgap])
ax1.set_title('Retailer strategy vs supplier decisions')
ax1.set_xlabel(r'$w_1=w_2$', fontsize=12)
ax1.set_ylabel(r'$\Lambda_1=\Lambda_2$', rotation=0, fontsize=12, labelpad=3)

# Add sliders for changing the parameters
slstrtval = 0.32
b_slider_ax = fig.add_axes([0.1, slstrtval, 0.65, 0.02])
b_slider = Slider(b_slider_ax, 'b', 0.01, 0.99, valinit=b_0)
c_slider_ax = fig.add_axes([0.1, slstrtval-0.04, 0.65, 0.02])
c_slider = Slider(c_slider_ax, 'c', 0.01, 0.99, valinit=c_0)
lambretlo_slider_ax = fig.add_axes([0.1, slstrtval-0.08, 0.65, 0.02])
lambretlo_slider = Slider(lambretlo_slider_ax, r'$\lambda^{lo}$', 0.01, 0.99, valinit=lambretlo_0)
lambrethi_slider_ax = fig.add_axes([0.1, slstrtval-0.12, 0.65, 0.02])
lambrethi_slider = Slider(lambrethi_slider_ax, r'$\lambda^{hi}$', 0.01, 0.99, valinit=lambrethi_0)
sens_slider_ax = fig.add_axes([0.1, slstrtval-0.16, 0.65, 0.02])
sens_slider = Slider(sens_slider_ax, r'$\rho$', 0.5, 0.99, valinit=sens_0)
fpr_slider_ax = fig.add_axes([0.1, slstrtval-0.2, 0.65, 0.02])
fpr_slider = Slider(fpr_slider_ax, r'$\phi$', 0.01, 0.2, valinit=fpr_0)
Ltheta_slider_ax = fig.add_axes([0.1, slstrtval-0.24, 0.65, 0.02])
Ltheta_slider = Slider(Ltheta_slider_ax, r'$L_{\theta}$', 0.01, 4.0, valinit=Ltheta_0)

# create a patch (proxy artist) for every color
patches = [mpatches.Patch(color=colors[i], edgecolor='black', label=labels[i]) for i in range(len(values))]
# put those patched as legend-handles into the legend
ax1.legend(handles=patches, bbox_to_anchor=(-0.6, 1.0), loc='upper left', borderaxespad=0.1, fontsize=14)
def sliders_on_changed(val):
    regplotmat = regionplotfuncforretailer_Scen1(numpts, 0.99, 0.99, Ltheta_slider.val, b_slider.val, c_slider.val,
                                                 lambretlo_slider.val, lambrethi_slider.val, sens_slider.val,
                                                 fpr_slider.val)

    im.set_data(regplotmat)

    #minmaxgap = np.nanmax(retpolicy_mat) - np.nanmin(retpolicy_mat)
    #ax1.set_ylim([np.nanmin(retpolicy_mat) - 0.05*minmaxgap, np.nanmax(retpolicy_mat) + 0.05*minmaxgap])

    fig.canvas.draw_idle()
b_slider.on_changed(sliders_on_changed)
c_slider.on_changed(sliders_on_changed)
lambretlo_slider.on_changed(sliders_on_changed)
lambrethi_slider.on_changed(sliders_on_changed)
sens_slider.on_changed(sliders_on_changed)
fpr_slider.on_changed(sliders_on_changed)
Ltheta_slider.on_changed(sliders_on_changed)
plt.show(block=True)

########################
# Now incorporate quantity-dependent detections at inspection
########################
def RetPrice(q1, q2, b):
    return 1 - q1 - b*q2
def RetProfit(q1, q2, b, w1, w2, c):
    return (q1*(RetPrice(q1, q2, b) - w1 - c)) + (q2*(RetPrice(q2, q1, b) - w2 - c))
def RetLowQualProb_quant(lambret, lambsup1, lambsup2, q1, q2, detect_const=0):
    return 1 - (lambret*(((q1+detect_const/2)/(q1+q2+detect_const))*lambsup1 +\
                         ((q2+detect_const/2)/(q1+q2+detect_const))*lambsup2))

def RetLowQualProb_actor(lambret, lambsup1, lambsup2, q1, q2, detect_const=0):
    if q1>0:
        if q2>0:
            retval = 1 - lambret*lambsup1*lambsup2
        else:
            retval = 1 - lambret*lambsup1
    else:
        retval = 1 - lambret*lambsup2
    return retval


def UtilsRet_Scen1_quantdetect(detect_const, lambsup1, lambsup2, Ltheta, b, c, w1, w2, lambretlo, lambrethi,
                               sens, fpr):
    util_list = []
    # What is retailer's utility as a function of different policies?
    # 0={Y12}, 1={N12}, 2={Y1}, 3={N1}
    # Maximizing quantities can no longer be derived; need to obtain numerically
    qvec = np.linspace(0.01, 0.99, 100)
    # Policy {Y12}
    currmaxutil, q1max, q2max = -0.0001, 0, 0  # initialize best policy values
    for curr_q1 in qvec:
        for curr_q2 in qvec:
            currutil = RetProfit(curr_q1, curr_q2, b, w1, w2, c) - \
                Ltheta*(fpr + (sens-fpr)*(RetLowQualProb_quant(lambrethi, lambsup1, lambsup2, curr_q1, curr_q2)))
            if currutil > currmaxutil:
                q1max, q2max = curr_q1, curr_q2
                currmaxutil = currutil
    util_list.append(currmaxutil)
    # Policy {N12}
    currmaxutil, q1max, q2max = -0.0001, 0, 0  # initialize best policy values
    for curr_q1 in qvec:
        for curr_q2 in qvec:
            currutil = RetProfit(curr_q1, curr_q2, b, w1, w2, 0) - \
                       Ltheta * (fpr + (sens - fpr) * (
                        RetLowQualProb_quant(lambretlo, lambsup1, lambsup2, curr_q1, curr_q2)))
            if currutil > currmaxutil:
                q1max, q2max = curr_q1, curr_q2
                currmaxutil = currutil
    util_list.append(currmaxutil)
    # Policy {Y1}
    util_list.append(0.5 * (1 - c - w1) * (1 - c - w1 + 0.5 * (-1 + c + w1)) - Ltheta * (
                fpr + (sens - fpr) * (1 - lambrethi * lambsup1)))
    # Policy {N1}
    util_list.append(0.5 * (1 - w1) * (1 - w1 + 0.5 * (-1 + w1)) - Ltheta * (
            fpr + (sens - fpr) * (1 - lambretlo * lambsup1)))
    # Policy {N}
    util_list.append(0)
    return util_list




def regionplotfuncforretailer_Scen1_quantdetect(numpts, detect_const, lambsup_max, w_max, Ltheta, b, c, lambretlo,
                                                lambrethi, sens, fpr):
    # Returns a region plot plotting matrix with wholesale price x-axis and quality-rate y-axis
    regionplotmat = np.empty((numpts, numpts))
    for wind, w in enumerate(np.linspace(0.01, w_max, numpts)):
        for lsind, ls in enumerate(np.linspace(0.01, lambsup_max, numpts)):
            regionplotmat[lsind, wind] = int(round(np.argmax(UtilsRet_Scen1_quantdetect(detect_const, ls, ls, Ltheta,
                                                                                        b, c, w, w, lambretlo,
                                                                                        lambrethi, sens, fpr))))
    return regionplotmat

numpts = 10

Ltheta_0 = 0.08
regplotmat = regionplotfuncforretailer_Scen1_quantdetect(numpts, 0, 0.99, 0.99, Ltheta_0, b_0, c_0, lambretlo_0,
                                                         lambrethi_0, sens_0, fpr_0)
d = np.linspace(0.01, 0.99, numpts)
lambsupplier, wprice = np.meshgrid(d, d)
values = [0, 1, 2, 3, 4]
labels = ['{Y12}', '{N12}', '{Y1}', '{N1}', '{N}']
cmapname = 'plasma'
cmapname = 'Greys'

fig = plt.figure()
fig.suptitle('Scenario 1: '+r'$\Lambda_1=\Lambda_2,w_1=w_2$', fontsize=18, fontweight='bold')
ax1 = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.5)
im = ax1.imshow(regplotmat, vmin=-1, vmax=4,
                extent=(lambsupplier.min(), lambsupplier.max(), wprice.min(), wprice.max()),
                origin="lower", cmap=cmapname)
colors = [im.cmap(im.norm(value)) for value in values]

#ax1.set_xlim([0, 2])
#minmaxgap = np.nanmax(retpolicy_mat) - np.nanmin(retpolicy_mat)
#ax1.set_ylim([np.nanmin(retpolicy_mat) - 0.05*minmaxgap, np.nanmax(retpolicy_mat) + 0.05*minmaxgap])
ax1.set_title('Retailer strategy vs supplier decisions')
ax1.set_xlabel(r'$w_1=w_2$', fontsize=12)
ax1.set_ylabel(r'$\Lambda_1=\Lambda_2$', rotation=0, fontsize=12, labelpad=3)

# Add sliders for changing the parameters
slstrtval = 0.43
b_slider_ax = fig.add_axes([0.1, slstrtval, 0.65, 0.02])
b_slider = Slider(b_slider_ax, 'b', 0.01, 0.99, valinit=b_0)
c_slider_ax = fig.add_axes([0.1, slstrtval-0.04, 0.65, 0.02])
c_slider = Slider(c_slider_ax, 'c', 0.01, 0.99, valinit=c_0)
lambretlo_slider_ax = fig.add_axes([0.1, slstrtval-0.12, 0.65, 0.02])
lambretlo_slider = Slider(lambretlo_slider_ax, r'$\lambda^{lo}$', 0.01, 0.99, valinit=lambretlo_0)
lambrethi_slider_ax = fig.add_axes([0.1, slstrtval-0.16, 0.65, 0.02])
lambrethi_slider = Slider(lambrethi_slider_ax, r'$\lambda^{hi}$', 0.01, 0.99, valinit=lambrethi_0)
sens_slider_ax = fig.add_axes([0.1, slstrtval-0.2, 0.65, 0.02])
sens_slider = Slider(sens_slider_ax, r'$\rho$', 0.5, 0.99, valinit=sens_0)
fpr_slider_ax = fig.add_axes([0.1, slstrtval-0.24, 0.65, 0.02])
fpr_slider = Slider(fpr_slider_ax, r'$\phi$', 0.01, 0.2, valinit=fpr_0)
Ltheta_slider_ax = fig.add_axes([0.1, slstrtval-0.28, 0.65, 0.02])
Ltheta_slider = Slider(Ltheta_slider_ax, r'$L_{\theta}$', 0.01, 4.0, valinit=Ltheta_0)

# create a patch (proxy artist) for every color
patches = [mpatches.Patch(color=colors[i], edgecolor='black', label=labels[i]) for i in range(len(values))]
# put those patched as legend-handles into the legend
ax1.legend(handles=patches, bbox_to_anchor=(-0.6, 1.0), loc='upper left', borderaxespad=0.1, fontsize=14)
def sliders_on_changed(val):
    regplotmat = regionplotfuncforretailer_Scen1_quantdetect(numpts, 0, 0.99, 0.99, Ltheta_slider.val, b_slider.val,
                                                             c_slider.val, lambretlo_slider.val, lambrethi_slider.val,
                                                             sens_slider.val, fpr_slider.val)

    im.set_data(regplotmat)

    #minmaxgap = np.nanmax(retpolicy_mat) - np.nanmin(retpolicy_mat)
    #ax1.set_ylim([np.nanmin(retpolicy_mat) - 0.05*minmaxgap, np.nanmax(retpolicy_mat) + 0.05*minmaxgap])

    fig.canvas.draw_idle()
b_slider.on_changed(sliders_on_changed)
c_slider.on_changed(sliders_on_changed)
lambretlo_slider.on_changed(sliders_on_changed)
lambrethi_slider.on_changed(sliders_on_changed)
sens_slider.on_changed(sliders_on_changed)
fpr_slider.on_changed(sliders_on_changed)
Ltheta_slider.on_changed(sliders_on_changed)
plt.show(block=True)







