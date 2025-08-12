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

########################
# Quantity-dependent detections at inspection, where the number of inspections equals the number of available brands
########################
def RetLowQualProb_quant_numsup(lambret, lambsup1, lambsup2, q1, q2, detect_const=0):
    if q1>0 and q2>0:
        numsup = 2
    else:
        numsup = 1
    return 1 - (lambret*((((q1+detect_const/2)/(q1+q2+detect_const))*lambsup1 +\
                         ((q2+detect_const/2)/(q1+q2+detect_const))*lambsup2)**numsup))

def UtilsRet_Scen1_quantdetect_numsup(detect_const, lambsup1, lambsup2, Ltheta, b, c, w1, w2, lambretlo, lambrethi,
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
                Ltheta*(fpr + (sens-fpr)*(RetLowQualProb_quant_numsup(lambrethi, lambsup1, lambsup2, curr_q1, curr_q2)))
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
                        RetLowQualProb_quant_numsup(lambretlo, lambsup1, lambsup2, curr_q1, curr_q2)))
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

def regionplotfuncforretailer_Scen1_quantdetect_numsup(numpts, detect_const, lambsup_max, w_max, Ltheta, b, c, lambretlo,
                                                lambrethi, sens, fpr):
    # Returns a region plot plotting matrix with wholesale price x-axis and quality-rate y-axis
    regionplotmat = np.empty((numpts, numpts))
    for wind, w in enumerate(np.linspace(0.01, w_max, numpts)):
        for lsind, ls in enumerate(np.linspace(0.01, lambsup_max, numpts)):
            regionplotmat[lsind, wind] = int(round(np.argmax(UtilsRet_Scen1_quantdetect_numsup(detect_const, ls, ls,
                                                                                        Ltheta, b, c, w, w, lambretlo,
                                                                                        lambrethi, sens, fpr))))
    return regionplotmat

numpts = 12

Ltheta_0 = 0.08
regplotmat = regionplotfuncforretailer_Scen1_quantdetect_numsup(numpts, 0, 0.99, 0.99, Ltheta_0, b_0, c_0, lambretlo_0,
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
    regplotmat = regionplotfuncforretailer_Scen1_quantdetect_numsup(numpts, 0, 0.99, 0.99, Ltheta_slider.val, b_slider.val,
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

########################
# Now do Scenario 5; need to show quantities of each supplier ordered, *not* the policy
# Generate q1 and q2 density plots next to each other
########################
w12_delt_0 = 0.05
lamb12_delt_0 = 0.05

def UtilsRet_Scen5_quantdetect_numsup(detect_const, lambsup1, lambsup2, Ltheta, b, c, w1, w2, lambretlo, lambrethi,
                               sens, fpr):
    util_list = []
    quants_list = []  # Return the order quantities so we don't have to recalculate
    # What is retailer's utility as a function of different policies?
    # 0={Y12}, 1={N12}, 2={Y1}, 3={N1}, 4={Y2}, 5={N2}, 6={N}
    # Maximizing quantities can no longer be derived for {12} strategies; need to obtain numerically
    qvec = np.linspace(0.01, 0.99, 100)
    # Policy {Y12}
    currmaxutil, q1max, q2max = -0.0001, 0, 0  # initialize best policy values
    for curr_q1 in qvec:
        for curr_q2 in qvec:
            currutil = RetProfit(curr_q1, curr_q2, b, w1, w2, c) - \
                Ltheta*(fpr + (sens-fpr)*(RetLowQualProb_quant_numsup(lambrethi, lambsup1, lambsup2, curr_q1, curr_q2)))
            if currutil > currmaxutil:
                q1max, q2max = curr_q1, curr_q2
                currmaxutil = currutil
    util_list.append(currmaxutil)
    quants_list.append((q1max, q2max))
    # Policy {N12}
    currmaxutil, q1max, q2max = -0.0001, 0, 0  # initialize best policy values
    for curr_q1 in qvec:
        for curr_q2 in qvec:
            currutil = RetProfit(curr_q1, curr_q2, b, w1, w2, 0) - \
                       Ltheta * (fpr + (sens - fpr) * (
                        RetLowQualProb_quant_numsup(lambretlo, lambsup1, lambsup2, curr_q1, curr_q2)))
            if currutil > currmaxutil:
                q1max, q2max = curr_q1, curr_q2
                currmaxutil = currutil
    util_list.append(currmaxutil)
    quants_list.append((q1max, q2max))
    # Policy {Y1}
    q1max = 0.5 * (1 - c - w1)
    util_list.append(q1max * (1 - c - w1 - q1max) - Ltheta * (fpr + (sens - fpr) * (1 - lambrethi * lambsup1)))
    quants_list.append((q1max, 0))
    # Policy {N1}
    q1max = 0.5 * (1 - w1)
    util_list.append(q1max * (1 - w1 - q1max) - Ltheta * (fpr + (sens - fpr) * (1 - lambretlo * lambsup1)))
    quants_list.append((q1max, 0))
    # Policy {Y2}
    q2max = 0.5 * (1 - c - w2)
    util_list.append(q2max * (1 - c - w2 - q2max) - Ltheta * (fpr + (sens - fpr) * (1 - lambrethi * lambsup2)))
    quants_list.append((0, q2max))
    # Policy {N2}
    q2max = 0.5 * (1 - w2)
    util_list.append(q2max * (1 - w2 - q2max) - Ltheta * (fpr + (sens - fpr) * (1 - lambretlo * lambsup2)))
    quants_list.append((0, q2max))
    # Policy {N}
    util_list.append(0)
    quants_list.append((0, 0))
    return util_list, quants_list

lambsup1, lambsup2 = 0.9-lamb12_delt_0, 0.9
Ltheta = 0.12
pol_list, q_list = UtilsRet_Scen5_quantdetect_numsup(0, lambsup1, lambsup2, Ltheta, b_0, c_0, w_0-w12_delt_0, w_0,
                                                     lambretlo_0, lambrethi_0, sens_0, fpr_0)

def RetQuantInvestPlots_WRT_w(numpts, detect_const, lambsup1, lambsup2, Ltheta, b, c, lambretlo, lambrethi, sens, fpr):
    """
    Return 3 matrices denoting the retailer quantities and quality investments for different w1, w2.
    Assumed w2>w1.
    """
    q1mat, q2mat, investmat = np.empty((numpts, numpts)), np.empty((numpts, numpts)), np.empty((numpts, numpts))
    for currw1ind, currw1 in enumerate(np.linspace(0.01, 0.99, numpts)):
        for currw2ind, currw2 in enumerate(np.linspace(0.01, 0.99, numpts)):
            if currw2 >= currw1:  # Only calculate for lower diagonal values
                curr_ulist, curr_qlist = UtilsRet_Scen5_quantdetect_numsup(detect_const, lambsup1, lambsup2, Ltheta, b,
                                                                c, currw1, currw2, lambretlo, lambrethi, sens, fpr)
                currpolind = np.argmax(curr_ulist)
                currqtup = curr_qlist[currpolind]
                q1mat[currw2ind, currw1ind] = currqtup[0]
                q2mat[currw2ind, currw1ind] = currqtup[1]
                if currpolind in [0, 2, 4]:
                    investmat[currw2ind, currw1ind] = 1
                else:
                    investmat[currw2ind, currw1ind] = 0
            else:
                q1mat[currw2ind, currw1ind], q2mat[currw2ind, currw1ind] = np.nan, np.nan
                investmat[currw2ind, currw1ind] = np.nan
    return q1mat, q2mat, investmat

numpts = 20  # MODIFY AS NEEDED
q1mat, q2mat, investmat = RetQuantInvestPlots_WRT_w(numpts, 0, lambsup2-lamb12_delt_0, lambsup2, Ltheta_0, b_0, c_0,
                                                    lambretlo_0, lambrethi_0, sens_0, fpr_0)

d1 = np.linspace(0.01, 0.99, numpts)
grid = np.meshgrid(d1, d1)


#values = [0, 1, 2, 3, 4, 5, 6]
#labels = ['{Y12}', '{N12}', '{Y1}', '{N1}', '{Y2}', '{N2}', '{N}']
cmapname = 'Greys'

fig = plt.figure(figsize=(20, 10))
fig.suptitle('Scenario 5: '+r'$\Delta\Lambda=\Lambda_2-\Lambda_1,\Delta w=w_2-w_1$', fontsize=18, fontweight='bold')
ax1 = plt.subplot2grid((1, 3), (0, 0))
ax2 = plt.subplot2grid((1, 3), (0, 1))
ax3 = plt.subplot2grid((1, 3), (0, 2))

fig.subplots_adjust(bottom=0.3)

im1 = ax1.imshow(q1mat.T, vmin=0, vmax=1,
                extent=(0, 1, 0, 1),
                origin="lower", cmap=cmapname, interpolation='none')

im2 = ax2.imshow(q2mat.T, vmin=0, vmax=1,
                extent=(0, 1, 0, 1),
                origin="lower", cmap=cmapname, interpolation='none')

im3 = ax3.imshow(investmat.T, vmin=0, vmax=1,
                extent=(0, 1, 0, 1),
                origin="lower", cmap=cmapname, interpolation='none')

ax1.set_title(r'$q_{1}$')
ax1.set_xlabel(r'$w_2$', fontsize=12)
ax1.set_ylabel(r'$w_1$', rotation=0, fontsize=12, labelpad=15)

ax2.set_title(r'$q_{2}$')
ax2.set_xlabel(r'$w_2$', fontsize=12)
ax2.set_ylabel(r'$w_1$', rotation=0, fontsize=12, labelpad=15)

ax3.set_title('Retailer quality investment')
ax3.set_xlabel(r'$w_2$', fontsize=12)
ax3.set_ylabel(r'$w_1$', rotation=0, fontsize=12, labelpad=15)

# Add sliders for changing the parameters
slstrtval = 0.28
slht = 0.01
slvertgap = 0.02
b_slider_ax = fig.add_axes([0.1, slstrtval, 0.65, slht])
b_slider = Slider(b_slider_ax, 'b', 0.01, 0.99, valinit=b_0)
c_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap, 0.65, slht])
c_slider = Slider(c_slider_ax, 'c', 0.01, 0.99, valinit=c_0)
lambretlo_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*2, 0.65, slht])
lambretlo_slider = Slider(lambretlo_slider_ax, r'$\lambda^{lo}$', 0.01, 0.99, valinit=lambretlo_0)
lambrethi_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*3, 0.65, slht])
lambrethi_slider = Slider(lambrethi_slider_ax, r'$\lambda^{hi}$', 0.01, 0.99, valinit=lambrethi_0)
sens_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*4, 0.65, slht])
sens_slider = Slider(sens_slider_ax, r'$\rho$', 0.5, 0.99, valinit=sens_0)
fpr_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*5, 0.65, slht])
fpr_slider = Slider(fpr_slider_ax, r'$\phi$', 0.01, 0.2, valinit=fpr_0)
lambsup2_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*6, 0.65, slht])
lambsup2_slider = Slider(lambsup2_slider_ax, r'$\Lambda_2$', 0.01, 0.99, valinit=lambsup1)
lambdelt_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*7, 0.65, slht])
lambdelt_slider = Slider(lambdelt_slider_ax, r'$\Delta\Lambda$', 0.01, 0.5, valinit=lamb12_delt_0)
Ltheta_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*8, 0.65, slht])
Ltheta_slider = Slider(Ltheta_slider_ax, r'$L_\theta$', 0.01, 0.5, valinit=Ltheta_0)

# create a patch (proxy artist) for every color
#patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]
# put those patched as legend-handles into the legend
#fig.legend(handles=patches, loc='upper left', borderaxespad=0.05, fontsize=12)
def sliders_on_changed(val):
    q1mat, q2mat, investmat = RetQuantInvestPlots_WRT_w(numpts, 0, lambsup2_slider.val-lambdelt_slider.val,
                                                        lambsup2_slider.val, Ltheta_slider.val, b_slider.val,
                                                        c_slider.val, lambretlo_slider.val, lambrethi_slider.val,
                                                        sens_slider.val, fpr_slider.val)
    im1.set_data(q1mat.T)
    im2.set_data(q2mat.T)
    im3.set_data(investmat.T)

    fig.canvas.draw_idle()

b_slider.on_changed(sliders_on_changed)
c_slider.on_changed(sliders_on_changed)
lambretlo_slider.on_changed(sliders_on_changed)
lambrethi_slider.on_changed(sliders_on_changed)
sens_slider.on_changed(sliders_on_changed)
fpr_slider.on_changed(sliders_on_changed)
lambsup2_slider.on_changed(sliders_on_changed)
lambdelt_slider.on_changed(sliders_on_changed)
Ltheta_slider.on_changed(sliders_on_changed)

plt.show(block=True)


###############
# DIFFERENT DETECTION MECHANISM
###############
w12_delt_0 = 0.05
lamb12_delt_0 = 0.05

def UtilsRet_Scen5_quantdetect(detect_const, lambsup1, lambsup2, Ltheta, b, c, w1, w2, lambretlo, lambrethi,
                               sens, fpr):
    util_list = []
    quants_list = []  # Return the order quantities so we don't have to recalculate
    # What is retailer's utility as a function of different policies?
    # 0={Y12}, 1={N12}, 2={Y1}, 3={N1}, 4={Y2}, 5={N2}, 6={N}
    # Maximizing quantities can no longer be derived for {12} strategies; need to obtain numerically
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
    quants_list.append((q1max, q2max))
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
    quants_list.append((q1max, q2max))
    # Policy {Y1}
    q1max = 0.5 * (1 - c - w1)
    util_list.append(q1max * (1 - c - w1 - q1max) - Ltheta * (fpr + (sens - fpr) * (1 - lambrethi * lambsup1)))
    quants_list.append((q1max, 0))
    # Policy {N1}
    q1max = 0.5 * (1 - w1)
    util_list.append(q1max * (1 - w1 - q1max) - Ltheta * (fpr + (sens - fpr) * (1 - lambretlo * lambsup1)))
    quants_list.append((q1max, 0))
    # Policy {Y2}
    q2max = 0.5 * (1 - c - w2)
    util_list.append(q2max * (1 - c - w2 - q2max) - Ltheta * (fpr + (sens - fpr) * (1 - lambrethi * lambsup2)))
    quants_list.append((0, q2max))
    # Policy {N2}
    q2max = 0.5 * (1 - w2)
    util_list.append(q2max * (1 - w2 - q2max) - Ltheta * (fpr + (sens - fpr) * (1 - lambretlo * lambsup2)))
    quants_list.append((0, q2max))
    # Policy {N}
    util_list.append(0)
    quants_list.append((0, 0))
    return util_list, quants_list

lambsup1, lambsup2 = 0.9-lamb12_delt_0, 0.9
Ltheta = 0.12
pol_list, q_list = UtilsRet_Scen5_quantdetect(0, lambsup1, lambsup2, Ltheta, b_0, c_0, w_0-w12_delt_0, w_0,
                                                     lambretlo_0, lambrethi_0, sens_0, fpr_0)

def RetQuantInvestPlots_WRT_w(numpts, detect_const, lambsup1, lambsup2, Ltheta, b, c, lambretlo, lambrethi, sens, fpr):
    """
    Return 3 matrices denoting the retailer quantities and quality investments for different w1, w2.
    Assumed w2>w1.
    """
    q1mat, q2mat, investmat = np.empty((numpts, numpts)), np.empty((numpts, numpts)), np.empty((numpts, numpts))
    for currw1ind, currw1 in enumerate(np.linspace(0.01, 0.99, numpts)):
        for currw2ind, currw2 in enumerate(np.linspace(0.01, 0.99, numpts)):
            if currw2 >= currw1:  # Only calculate for lower diagonal values
                curr_ulist, curr_qlist = UtilsRet_Scen5_quantdetect(detect_const, lambsup1, lambsup2, Ltheta, b,
                                                                c, currw1, currw2, lambretlo, lambrethi, sens, fpr)
                currpolind = np.argmax(curr_ulist)
                currqtup = curr_qlist[currpolind]
                q1mat[currw2ind, currw1ind] = currqtup[0]
                q2mat[currw2ind, currw1ind] = currqtup[1]
                if currpolind in [0, 2, 4]:
                    investmat[currw2ind, currw1ind] = 1
                else:
                    investmat[currw2ind, currw1ind] = 0
            else:
                q1mat[currw2ind, currw1ind], q2mat[currw2ind, currw1ind] = np.nan, np.nan
                investmat[currw2ind, currw1ind] = np.nan
    return q1mat, q2mat, investmat

numpts = 20  # MODIFY AS NEEDED
q1mat, q2mat, investmat = RetQuantInvestPlots_WRT_w(numpts, 0, lambsup2-lamb12_delt_0, lambsup2, Ltheta_0, b_0, c_0,
                                                    lambretlo_0, lambrethi_0, sens_0, fpr_0)

d1 = np.linspace(0.01, 0.99, numpts)
grid = np.meshgrid(d1, d1)


#values = [0, 1, 2, 3, 4, 5, 6]
#labels = ['{Y12}', '{N12}', '{Y1}', '{N1}', '{Y2}', '{N2}', '{N}']
cmapname = 'Greys'

fig = plt.figure(figsize=(20, 10))
fig.suptitle('Scenario 5: '+r'$\Delta\Lambda=\Lambda_2-\Lambda_1,\Delta w=w_2-w_1$', fontsize=18, fontweight='bold')
ax1 = plt.subplot2grid((1, 3), (0, 0))
ax2 = plt.subplot2grid((1, 3), (0, 1))
ax3 = plt.subplot2grid((1, 3), (0, 2))

fig.subplots_adjust(bottom=0.3)

im1 = ax1.imshow(q1mat.T, vmin=0, vmax=0.5,
                extent=(0, 1, 0, 1),
                origin="lower", cmap=cmapname, interpolation='none')

im2 = ax2.imshow(q2mat.T, vmin=0, vmax=0.5,
                extent=(0, 1, 0, 1),
                origin="lower", cmap=cmapname, interpolation='none')

im3 = ax3.imshow(investmat.T, vmin=0, vmax=1,
                extent=(0, 1, 0, 1),
                origin="lower", cmap=cmapname, interpolation='none')

ax1.set_title(r'$q_{1}$')
ax1.set_xlabel(r'$w_2$', fontsize=12)
ax1.set_ylabel(r'$w_1$', rotation=0, fontsize=12, labelpad=15)

ax2.set_title(r'$q_{2}$')
ax2.set_xlabel(r'$w_2$', fontsize=12)
ax2.set_ylabel(r'$w_1$', rotation=0, fontsize=12, labelpad=15)

ax3.set_title('Retailer quality investment')
ax3.set_xlabel(r'$w_2$', fontsize=12)
ax3.set_ylabel(r'$w_1$', rotation=0, fontsize=12, labelpad=15)

# Add sliders for changing the parameters
slstrtval = 0.28
slht = 0.01
slvertgap = 0.02
b_slider_ax = fig.add_axes([0.1, slstrtval, 0.65, slht])
b_slider = Slider(b_slider_ax, 'b', 0.01, 0.99, valinit=b_0)
c_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap, 0.65, slht])
c_slider = Slider(c_slider_ax, 'c', 0.01, 0.99, valinit=c_0)
lambretlo_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*2, 0.65, slht])
lambretlo_slider = Slider(lambretlo_slider_ax, r'$\lambda^{lo}$', 0.01, 0.99, valinit=lambretlo_0)
lambrethi_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*3, 0.65, slht])
lambrethi_slider = Slider(lambrethi_slider_ax, r'$\lambda^{hi}$', 0.01, 0.99, valinit=lambrethi_0)
sens_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*4, 0.65, slht])
sens_slider = Slider(sens_slider_ax, r'$\rho$', 0.5, 0.99, valinit=sens_0)
fpr_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*5, 0.65, slht])
fpr_slider = Slider(fpr_slider_ax, r'$\phi$', 0.01, 0.2, valinit=fpr_0)
lambsup2_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*6, 0.65, slht])
lambsup2_slider = Slider(lambsup2_slider_ax, r'$\Lambda_2$', 0.01, 0.99, valinit=lambsup1)
lambdelt_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*7, 0.65, slht])
lambdelt_slider = Slider(lambdelt_slider_ax, r'$\Delta\Lambda$', 0.01, 0.5, valinit=lamb12_delt_0)
Ltheta_slider_ax = fig.add_axes([0.1, slstrtval-slvertgap*8, 0.65, slht])
Ltheta_slider = Slider(Ltheta_slider_ax, r'$L_\theta$', 0.01, 0.5, valinit=Ltheta_0)

# create a patch (proxy artist) for every color
#patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]
# put those patched as legend-handles into the legend
#fig.legend(handles=patches, loc='upper left', borderaxespad=0.05, fontsize=12)
def sliders_on_changed(val):
    q1mat, q2mat, investmat = RetQuantInvestPlots_WRT_w(numpts, 0, lambsup2_slider.val-lambdelt_slider.val,
                                                        lambsup2_slider.val, Ltheta_slider.val, b_slider.val,
                                                        c_slider.val, lambretlo_slider.val, lambrethi_slider.val,
                                                        sens_slider.val, fpr_slider.val)
    im1.set_data(q1mat.T)
    im2.set_data(q2mat.T)
    im3.set_data(investmat.T)

    fig.canvas.draw_idle()

b_slider.on_changed(sliders_on_changed)
c_slider.on_changed(sliders_on_changed)
lambretlo_slider.on_changed(sliders_on_changed)
lambrethi_slider.on_changed(sliders_on_changed)
sens_slider.on_changed(sliders_on_changed)
fpr_slider.on_changed(sliders_on_changed)
lambsup2_slider.on_changed(sliders_on_changed)
lambdelt_slider.on_changed(sliders_on_changed)
Ltheta_slider.on_changed(sliders_on_changed)

plt.show(block=True)

####################
# Plot quantities and investment decisions from *all* detection paradigms
####################


