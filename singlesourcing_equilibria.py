# Simulation of potential equilibria when the retailer has a single-sourcing option
# 15-SEP-26

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import textwrap
from scipy.optimize import fsolve
from matplotlib.widgets import Slider
import scipy.optimize as scipyOpt
from matplotlib.widgets import RadioButtons
from numpy.core.multiarray import ndarray

# matplotlib.use('qt5agg',force=True)  # pycharm backend doesn't support interactive plots, so we use qt here
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
plt.rcParams["font.family"] = "serif"

def Sup1Util(supVec, retVec, envirDict, supPen):
    # Returns supplier 1 utility
    # Get quality investment cost
    if supVec[1] == envirDict['H']: # Yes qual investment
        cS = envirDict['cS']
    elif supVec[1] == envirDict['L']:  # No qual investment
        cS = 0
    else:
        print('INVALID SUPPLIER QUALITY LEVEL')
        exit()
    q = retVec[0] # S1
    return q*(supVec[0]-cS) - (1-supVec[1])*supPen

def Sup2Util(supVec, retVec, envirDict, supPen):
    # Returns supplier 2 utility
    # Get quality investment cost
    if supVec[1] == envirDict['H']: # Yes qual investment
        cS = envirDict['cS']
    elif supVec[1] == envirDict['L']:  # No qual investment
        cS = 0
    else:
        print('INVALID SUPPLIER QUALITY LEVEL')
        exit()
    q = retVec[1]  # S2
    return q*(supVec[0]-cS) - (1-supVec[1])*supPen

def invDemPrice(qi, qj, b):
    return 1 - qi - b*qj

def q1Opt(w1, w2, b):
    # Returns optimal order quantities from S1 under dual sourcing
    return max(0, (1-b-w1+b*w2)/(2*(1-(b**2))))

def q2Opt(w1, w2, b):
    # Returns optimal order quantities from S1 under dual sourcing
    return max(0, (1-b+b*w1-w2)/(2*(1-(b**2))))

def RetUtilDual(sup1Vec, sup2Vec, envirDict, retPen):
    # Returns retailer's utility under dual sourcing
    w1, w2, qual1, qual2 = sup1Vec[0], sup2Vec[0], sup1Vec[1], sup2Vec[1]
    b = envirDict['b']
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    prof1, prof2 = q1 * (invDemPrice(q1, q2, b) - w1), q2 * (invDemPrice(q2, q1, b) - w2)
    insppen = retPen * (1 - qual1 * qual2)
    return prof1 + prof2 - insppen, q1, q2

def RetUtilSingSup1(sup1Vec, sup2Vec, envirDict, retPen):
    # Returns retailer's utility under single sourcing from Supplier 1
    w1, w2, qual1, qual2 = sup1Vec[0], sup2Vec[0], sup1Vec[1], sup2Vec[1]
    b = 0
    q1, q2 = q1Opt(w1, w2, b), 0
    prof1 = q1 * (invDemPrice(q1, q2, b) - w1)
    insppen = retPen * (1 - qual1)
    return prof1 - insppen, q1, q2

def RetUtilSingSup2(sup1Vec, sup2Vec, envirDict, retPen):
    # Returns retailer's utility under single sourcing from Supplier 2
    w1, w2, qual1, qual2 = sup1Vec[0], sup2Vec[0], sup1Vec[1], sup2Vec[1]
    b = 0
    q1, q2 = 0, q2Opt(w1, w2, b)
    prof2 = q2 * (invDemPrice(q2, q1, b) - w2)
    insppen = retPen * (1 - qual2)
    return prof2 - insppen, q1, q2

def GetRetOptDecis(sup1Vec, sup2Vec, envirDict, retPen):
    # Returns retailer's best sourcing decision, one of 'Dual', 'Sing1', 'Sing2', or 'None',
    #       as well as the order quantities
    dualUtil, q1Dual, q2Dual = RetUtilDual(sup1Vec, sup2Vec, envirDict, retPen)
    sing1Util, q1Sing1, q2Sing1 = RetUtilSingSup1(sup1Vec, sup2Vec, envirDict, retPen)
    sing2Util, q1Sing2, q2Sing2 = RetUtilSingSup2(sup1Vec, sup2Vec, envirDict, retPen)
    maxRetUtil = max([dualUtil,sing1Util,sing2Util])
    if maxRetUtil < 0:  # All sourcing yields negative utility
        retPol = 'None'
        q1, q2 = 0, 0
    elif dualUtil == maxRetUtil:  # Dual sourcing
        retPol = 'Dual'
        q1, q2 = q1Dual, q2Dual
    elif sing1Util == maxRetUtil:  # Single sourcing from S1
        retPol = 'Sing1'
        q1, q2 = q1Sing1, q2Sing1
    elif sing2Util == maxRetUtil:  # Single sourcing from S1
        retPol = 'Sing2'
        q1, q2 = q1Sing2, q2Sing2
    return retPol, q1, q2


##############################
# PRIMARY SIMULATION BLOCK
##############################
pixWidth = 0.005  # Gap between X,Y pixels
Xmax, Ymax = 0.6, 0.2

wWidth = 0.005  # Gap between possible wholesale prices
wVec = np.arange(wWidth, 0.8, wWidth)  # Assume there are no on-path prices beyond 0.8

envirDict = {'b': 0.7, 'cS': 0.15, 'L':0.8, 'H':1.0}

Xvec, Yvec = np.arange(0, Xmax+pixWidth, pixWidth), np.arange(0, Ymax+pixWidth, pixWidth)

# Initialize return matrices
LLmat = np.full((Xvec.shape[0], Yvec.shape[0]), np.nan)
LHmat, HHmat = LLmat.copy(), LLmat.copy()

for Xind, Xcurr in enumerate(Xvec):
    for Yind, Ycurr in enumerate(Yvec):
        print('Evaluating X='+str(Xcurr)+', Y=' + str(Ycurr)+'...')
        for w1On in wVec:
            for w2On in wVec:
                # Check each quality profile
                ### LL ###
                isEquil = True
                sup1VecOn, sup2VecOn = [w1On, envirDict['L']], [w2On, envirDict['L']]
                # Check retailer on-path decision is dual
                sourcePol, q1On, q2On = GetRetOptDecis(sup1VecOn, sup2VecOn, envirDict, Xcurr)
                qVecOn = [q1On, q2On]
                if not sourcePol == 'Dual':
                    isEquil = False
                # If valid, check supplier off-path moves
                # S1 first
                if isEquil:
                    sup1UtilOn = Sup1Util(sup1VecOn, qVecOn, envirDict, Ycurr)
                    for w1Off in wVec:
                        if w1Off != w1On and isEquil:
                            # Check L first
                            sup1VecOffL = [w1Off, envirDict['L']]
                            # Check retailer off-path decision
                            sourcePolOff, q1Off, q2Off = GetRetOptDecis(sup1VecOffL, sup2VecOn, envirDict, Xcurr)
                            qVecOff = [q1Off, q2Off]
                            sup1UtilOff = Sup1Util(sup1VecOffL, qVecOff, envirDict, Ycurr)
                            if sup1UtilOff > sup1UtilOn:  # There exists a better off-path move
                                isEquil = False
                            # Check H second
                            sup1VecOffH = [w1Off, envirDict['H']]
                            # Check retailer off-path decision
                            sourcePolOff, q1Off, q2Off = GetRetOptDecis(sup1VecOffH, sup2VecOn, envirDict, Xcurr)
                            qVecOff = [q1Off, q2Off]
                            sup1UtilOff = Sup1Util(sup1VecOffH, qVecOff, envirDict, Ycurr)
                            if sup1UtilOff > sup1UtilOn:  # There exists a better off-path move
                                isEquil = False
                # S2 second
                if isEquil:
                    sup2UtilOn = Sup2Util(sup2VecOn, qVecOn, envirDict, Ycurr)
                    for w2Off in wVec:
                        if w2Off != w2On and isEquil:
                            # Check L first
                            sup2VecOffL = [w2Off, envirDict['L']]
                            # Check retailer off-path decision
                            sourcePolOff, q1Off, q2Off = GetRetOptDecis(sup1VecOn, sup2VecOffL, envirDict, Xcurr)
                            qVecOff = [q1Off, q2Off]
                            sup2UtilOff = Sup2Util(sup2VecOffL, qVecOff, envirDict, Ycurr)
                            if sup2UtilOff > sup2UtilOn:  # There exists a better off-path move
                                isEquil = False
                            # Check H second
                            sup2VecOffH = [w2Off, envirDict['H']]
                            # Check retailer off-path decision
                            sourcePolOff, q1Off, q2Off = GetRetOptDecis(sup1VecOn, sup2VecOffH, envirDict, Xcurr)
                            qVecOff = [q1Off, q2Off]
                            sup2UtilOff = Sup2Util(sup2VecOffH, qVecOff, envirDict, Ycurr)
                            if sup2UtilOff > sup2UtilOn:  # There exists a better off-path move
                                isEquil = False
                # If isEquil is still valid, then sum the wholesale prices and add to the LLmat
                if isEquil:
                    LLmat[Xind, Yind] = w1On + w2On
                    print('LL eq found at X='+str(Xcurr)+', Y='+str(Ycurr)+'; w1='+str(w1On)+', w2='+str(w2On))
                ### LH ###
                if w2On > envirDict['cS']:
                    isEquil = True
                else:
                    isEquil = False
                sup1VecOn, sup2VecOn = [w1On, envirDict['L']], [w2On, envirDict['H']]
                # Check retailer on-path decision is dual
                sourcePol, q1On, q2On = GetRetOptDecis(sup1VecOn, sup2VecOn, envirDict, Xcurr)
                qVecOn = [q1On, q2On]
                if not sourcePol == 'Dual':
                    isEquil = False
                # If valid, check supplier off-path moves
                # S1 first
                if isEquil:
                    sup1UtilOn = Sup1Util(sup1VecOn, qVecOn, envirDict, Ycurr)
                    for w1Off in wVec:
                        if w1Off != w1On and isEquil:
                            # Check L first
                            sup1VecOffL = [w1Off, envirDict['L']]
                            # Check retailer off-path decision
                            sourcePolOff, q1Off, q2Off = GetRetOptDecis(sup1VecOffL, sup2VecOn, envirDict, Xcurr)
                            qVecOff = [q1Off, q2Off]
                            sup1UtilOff = Sup1Util(sup1VecOffL, qVecOff, envirDict, Ycurr)
                            if sup1UtilOff > sup1UtilOn:  # There exists a better off-path move
                                isEquil = False
                            # Check H second
                            sup1VecOffH = [w1Off, envirDict['H']]
                            # Check retailer off-path decision
                            sourcePolOff, q1Off, q2Off = GetRetOptDecis(sup1VecOffH, sup2VecOn, envirDict, Xcurr)
                            qVecOff = [q1Off, q2Off]
                            sup1UtilOff = Sup1Util(sup1VecOffH, qVecOff, envirDict, Ycurr)
                            if sup1UtilOff > sup1UtilOn:  # There exists a better off-path move
                                isEquil = False
                # S2 second
                if isEquil:
                    sup2UtilOn = Sup2Util(sup2VecOn, qVecOn, envirDict, Ycurr)
                    for w2Off in wVec:
                        if w2Off != w2On and isEquil:
                            # Check L first
                            sup2VecOffL = [w2Off, envirDict['L']]
                            # Check retailer off-path decision
                            sourcePolOff, q1Off, q2Off = GetRetOptDecis(sup1VecOn, sup2VecOffL, envirDict, Xcurr)
                            qVecOff = [q1Off, q2Off]
                            sup2UtilOff = Sup2Util(sup2VecOffL, qVecOff, envirDict, Ycurr)
                            if sup2UtilOff > sup2UtilOn:  # There exists a better off-path move
                                isEquil = False
                            # Check H second
                            sup2VecOffH = [w2Off, envirDict['H']]
                            # Check retailer off-path decision
                            sourcePolOff, q1Off, q2Off = GetRetOptDecis(sup1VecOn, sup2VecOffH, envirDict, Xcurr)
                            qVecOff = [q1Off, q2Off]
                            sup2UtilOff = Sup2Util(sup2VecOffH, qVecOff, envirDict, Ycurr)
                            if sup2UtilOff > sup2UtilOn:  # There exists a better off-path move
                                isEquil = False
                # If isEquil is still valid, then sum the wholesale prices and add to the LHmat
                if isEquil:
                    LHmat[Xind, Yind] = w1On + w2On
                    print('LH eq found at X=' + str(Xcurr) + ', Y=' + str(Ycurr) + '; w1=' + str(
                        w1On) + ', w2=' + str(w2On))
                ### HH ###
                if w1On > envirDict['cS'] and w2On > envirDict['cS']:
                    isEquil = True
                else:
                    isEquil = False
                sup1VecOn, sup2VecOn = [w1On, envirDict['H']], [w2On, envirDict['H']]
                # Check retailer on-path decision is dual
                sourcePol, q1On, q2On = GetRetOptDecis(sup1VecOn, sup2VecOn, envirDict, Xcurr)
                qVecOn = [q1On, q2On]
                if not sourcePol == 'Dual':
                    isEquil = False
                # If valid, check supplier off-path moves
                # S1 first
                if isEquil:
                    sup1UtilOn = Sup1Util(sup1VecOn, qVecOn, envirDict, Ycurr)
                    for w1Off in wVec:
                        if w1Off != w1On and isEquil:
                            # Check L first
                            sup1VecOffL = [w1Off, envirDict['L']]
                            # Check retailer off-path decision
                            sourcePolOff, q1Off, q2Off = GetRetOptDecis(sup1VecOffL, sup2VecOn, envirDict, Xcurr)
                            qVecOff = [q1Off, q2Off]
                            sup1UtilOff = Sup1Util(sup1VecOffL, qVecOff, envirDict, Ycurr)
                            if sup1UtilOff > sup1UtilOn:  # There exists a better off-path move
                                isEquil = False
                            # Check H second
                            sup1VecOffH = [w1Off, envirDict['H']]
                            # Check retailer off-path decision
                            sourcePolOff, q1Off, q2Off = GetRetOptDecis(sup1VecOffH, sup2VecOn, envirDict, Xcurr)
                            qVecOff = [q1Off, q2Off]
                            sup1UtilOff = Sup1Util(sup1VecOffH, qVecOff, envirDict, Ycurr)
                            if sup1UtilOff > sup1UtilOn:  # There exists a better off-path move
                                isEquil = False
                # S2 second
                if isEquil:
                    sup2UtilOn = Sup2Util(sup2VecOn, qVecOn, envirDict, Ycurr)
                    for w2Off in wVec:
                        if w2Off != w2On and isEquil:
                            # Check L first
                            sup2VecOffL = [w2Off, envirDict['L']]
                            # Check retailer off-path decision
                            sourcePolOff, q1Off, q2Off = GetRetOptDecis(sup1VecOn, sup2VecOffL, envirDict, Xcurr)
                            qVecOff = [q1Off, q2Off]
                            sup2UtilOff = Sup2Util(sup2VecOffL, qVecOff, envirDict, Ycurr)
                            if sup2UtilOff > sup2UtilOn:  # There exists a better off-path move
                                isEquil = False
                            # Check H second
                            sup2VecOffH = [w2Off, envirDict['H']]
                            # Check retailer off-path decision
                            sourcePolOff, q1Off, q2Off = GetRetOptDecis(sup1VecOn, sup2VecOffH, envirDict, Xcurr)
                            qVecOff = [q1Off, q2Off]
                            sup2UtilOff = Sup2Util(sup2VecOffH, qVecOff, envirDict, Ycurr)
                            if sup2UtilOff > sup2UtilOn:  # There exists a better off-path move
                                isEquil = False
                # If isEquil is still valid, then sum the wholesale prices and add to the HHmat
                if isEquil:
                    HHmat[Xind, Yind] = w1On + w2On
                    print('HH eq found at X=' + str(Xcurr) + ', Y=' + str(Ycurr) + '; w1=' + str(
                        w1On) + ', w2=' + str(w2On))

np.save('LLmat.npy', LLmat)
np.save('LHmat.npy', LHmat)
np.save('HHmat.npy', HHmat)

# Plot
b, cS, L, H = envirDict['b'], envirDict['cS'], envirDict['L'], envirDict['H']
alval = 0.9
fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$c_S=$'+str(cS)+', '+r'$L=$'+str(L),
             fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['red',  'blue',  'green']
labels = ['LLFOC',   'LHFOC',  'HHFOC']

matList = [LLmat, LHmat,HHmat]
imlist = []
for xind, x in enumerate(matList):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[xind]], name='from_list', N=None)
    im = ax.imshow(x.T, vmin=-1, vmax=0.8, aspect='auto',
                            extent=(0, Xmax, 0, Ymax),
                            origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)

# Fill in any non-equilibria regions
# Cthdist, Cbedist = (Ctheta_max)/numpts, (Cbeta_max)/numpts
# for i in range(CthetaVec.shape[0]):
#     for j in range(CbetaVec.shape[0]):
#         if np.nansum(eqMats[:, i, j])==0:  # No equilibria here
#             ax.add_patch(matplotlib.patches.Rectangle((CthetaVec[i],CbetaVec[j]),Cthdist,Cbedist,
#                hatch='/////////',fill=False,linewidth=0,snap=False))

legwidth = 20
wraplabels = ['\n'.join(textwrap.wrap(labels[i], width=legwidth)) for i in range(len(labels))]
patches = [mpatches.Patch(color=eqcolors[i], label=wraplabels[i], alpha=alval) for i in range(len(eqcolors))]
          # +[mpatches.Patch(hatch=r'/////////',fill=False,linewidth=0,snap=False,label='1-sup. eq.')]

# put those patched as legend-handles into the legend
ax.legend(handles=patches, bbox_to_anchor=(1.3, 1.0), loc='upper right', borderaxespad=0.1, fontsize=8)
ax.set_xbound(0, Xmax)
ax.set_ybound(0, Ymax)
ax.set_box_aspect(1)
plt.xlabel(r'$X$', fontsize=14)
plt.ylabel(r'$Y$', fontsize=14, rotation=0, labelpad=14)
plt.show()