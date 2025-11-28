# Plot generation for "Strategic Role of Inspections in Pharmaceutical Supply Chains"
# These functions use the simpler version of the model
#   - perfect diagnostic, H=1, no retailer quality choice, retailer sources from both suppliers or neither
# 25-NOV-25

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
import textwrap
import scipy.optimize as scipyOpt
from matplotlib.widgets import RadioButtons
from numpy.core.multiarray import ndarray

# matplotlib.use('qt5agg',force=True)  # pycharm backend doesn't support interactive plots, so we use qt here
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
plt.rcParams["font.family"] = "monospace"

def SPUtil(q1, q2, lambsup1, lambsup2, alph):
    # Social planner's utility
    return q1*(alph+(lambsup1)-1) + q2*(alph+(lambsup2)-1)

def SupUtil(q, w, cSup):
    # Returns supplier utility
    return q*(w-cSup)

# Function returning retailer utilities under each of 7 possible policies
def RetUtil(lambsup1, lambsup2, Ltheta, b, w1, w2):
    # Returns retailer utility under pi=D
    q1 = max((1 - b - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
    q2 = max((1 - b - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
    retval = (1 - b - w1 + b*w2) * (1 - w1 - b*q2 - q1) / (2 * (1 - b ** 2)) + \
                     (1 - b + b*w1 - w2) * (1 - w2 - b*q1 - q2) / (2 * (1 - b ** 2)) - \
                     Ltheta * ((1) * (1 - lambsup1 * lambsup2))
    return retval

# def RetOrderQuantsFromStrat(stratind, b, c, w1, w2):
#     # Returns order quantities under inputs. If w2=-1, single sourcing quantity given
#     strat = int(stratind)
#     if strat == 0:
#         q1 = max((1 - b - c + b * c - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
#         q2 = max((1 - b - c + b * c - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
#     elif strat == 1:
#         q1 = max((1 - b - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
#         q2 = max((1 - b - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
#     elif strat == 2:
#         q1 = max((1 - c - w1) / 2, 0)
#         q2 = 0
#     elif strat == 3:
#         q1 = max((1 - w1) / 2, 0)
#         q2 = 0
#     elif strat == 4:
#         q1 = 0
#         q2 = max((1 - c - w2) / 2, 0)
#     elif strat == 5:
#         q1 = 0
#         q2 = max((1 - w2) / 2, 0)
#     elif strat == 6:
#         q1 = 0
#         q2 = 0
#     return q1, q2

# def retStratToStr(stratInt):
#     # Returns string for integer retailer strategy
#     if stratInt == 0:
#         retStr = ', {Y12}'
#     elif stratInt == 1:
#         retStr = ', {N12}'
#     elif stratInt == 2:
#         retStr = ', {Y1}'
#     elif stratInt == 3:
#         retStr = ', {N1}'
#     elif stratInt == 4:
#         retStr = ', {Y2}'
#     elif stratInt == 5:
#         retStr = ', {N2}'
#     elif stratInt == 6:
#         retStr = ', {N}'
#     return retStr

def SupPriceLL(scDict, Ctheta):
    # Returns on-path LL prices
    w = max((1 - scDict['b']) / (2 - scDict['b']), 0)
    return w, w

def SupPriceHH(scDict, Ctheta):
    # Returns on-path HH prices
    w = max((1 - scDict['b'] + scDict['cSup']) / (2 - scDict['b']), 0)
    return w, w

def SupPriceLHFOC(scDict, Ctheta):
    # Returns on-path LH-FOC prices
    w1 = max((2 - scDict['b'] - scDict['b']**2 + scDict['b']*scDict['cSup']) / (4 - scDict['b']**2), 0)
    w2 = max((2 - scDict['b'] - scDict['b'] ** 2 + 2 * scDict['cSup']) / (4 - scDict['b'] ** 2), 0)
    return w1, w2

def SupPriceLHSqz(scDict, Ctheta):
    # Returns on-path LH-sqz prices
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    w1 = max((b - b*cSup + (b**2)*(3-2*np.sqrt((1+(-2+cSup)*cSup -\
            4*(-4 + 3*b**2)*Ctheta*(-1+supRateLo)) / (-1 + b**2)))+2*(-2 + np.sqrt((1 + (-2 + cSup)* cSup -\
            4*(-4+3*(b**2))*Ctheta*(-1 +supRateLo)) / (-1 + b**2)))) / (-4 + 3*(b**2)), 0)
    w2 = max(0.5*(1+cSup+b*(-1+w1)), 0)
    return w1, w2

def SupPriceLHSqzTwo(scDict, Ctheta):
    # Returns on-path LH-sqz2 prices
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    w1 = max(1 + b*(-1 + cSup + b*np.sqrt((1 + (-2 + cSup)*cSup +4*Ctheta*(-1+supRateLo))/(-1 + (b**2)))) -\
         np.sqrt((1 + (-2+cSup)*cSup + 4*Ctheta*(-1 + supRateLo))/(-1 + (b**2))), 0)
    w2 = max(cSup, 0)
    return w1, w2

def SupPriceLHexp(scDict, Ctheta):
    # Returns on-path LH-exp prices
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    w1 = max((b + (b**2)*(3-2*np.sqrt((1-4*(-4 + 3*(b**2))*Ctheta* (-1 + supRateLo**2))/(-1 + (b**2)))) +\
              2*(-2 + np.sqrt((1 - 4*(-4 + 3*(b**2))*Ctheta*(-1 + (supRateLo**2)))/(-1+ (b**2)))))/(-4+ 3* (b**2)), 0)
    w2 = max(0.5*(1+cSup+b*(-1+w1)), 0)
    return w1, w2

def CthetaHHLB(scDict):
    # Returns the lower bound of the HH region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    retval =  (-1*(((-4 + b*(4 + cSup*(4-4*cSup +b*(-4 + 3*cSup))))*(-1 + supRateLo))/(((-2+b)**2)*(-1 + (b**2)))) +\
               np.sqrt(-1*((((2+b*(-2 + cSup))**2)*cSup*(4-2*cSup + b*(-4 + 3*cSup))*((-1 + supRateLo)**2)) / (((-2 +\
               b)**3)*((-1 + b**2)**2)))))/(8*((-1 + supRateLo)**2))
    return retval

def CthetaLLUB(scDict):
    # Returns the upper bound of the LL region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    retval = 1/(2*((-2 + b)**2)*(1 + b)*(1 - (supRateLo**2)))
    return retval

def cSupHHsqz(w, Ctheta, scDict):
    # Returns the cSupTilde value for designated w and Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    retval = (1 + b + w*(-3 + 2*w)+b*w*(-2+w+np.sqrt((((-1 + w)**2) + 4*Ctheta*(-1 + supRateLo))/(-1 +(b**2)))) +\
              np.sqrt((((-1 + w)**2)+4*Ctheta*(-1 + supRateLo))/(-1 + (b**2))) + (b**2)*(-1+w)*np.sqrt((((-1 + w)**2)+\
              4*Ctheta*(-1 +supRateLo))/(-1 +(b**2)))+4*Ctheta*(-1 + supRateLo)+4*b*Ctheta*(-1 + supRateLo))/(-1 + w)
    return retval

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def GetCritcSupHHsqz(scDict, wincrem=1/1000, cSupincrem=1/1000):
    # Returns the cSup value for which HHsqz prices increase for cSup<critcSup and increase otherwise
    critcSup = 0
    cSupVec = np.arange(cSupincrem, 1, cSupincrem)
    wVec = np.arange(wincrem, 1, wincrem)
    for currcSupInd in range(cSupVec.shape[0]):
        if currcSupInd > 5 and currcSupInd < 995:
            tempDict = scDict.copy()
            tempDict['cSup'] = cSupVec[currcSupInd]
            # Get wVec value closest to the currwHH
            currwHH = find_nearest(wVec, SupPriceHH(tempDict, 0)[0])
            currwHHind = np.where(wVec==currwHH)[0]
            # Check if at a maximum for cSupTilde above and below
            currcSupTilde = cSupHHsqz(wVec[currwHHind], CthetaHHLB(tempDict), tempDict)
            belowcSupTilde = cSupHHsqz(wVec[currwHHind-1], CthetaHHLB(tempDict), tempDict)
            abovecSupTilde = cSupHHsqz(wVec[currwHHind+1], CthetaHHLB(tempDict), tempDict)
            if currcSupTilde > belowcSupTilde and currcSupTilde > abovecSupTilde:
                critcSup = cSupVec[currcSupInd]

    return critcSup

def CthetaHHsqzLB(scDict):
    # Returns the lower bound of the HHsqz region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    critcSup = GetCritcSupHHsqz(scDict)
    currLB = CthetaHHLB(scDict)
    Cthetaincrem = 1/1000
    CthetaBool = True
    if cSup < critcSup:  # Prices increase with decreasing Ctheta
        Wvec = np.arange(SupPriceHH(scDict, 0)[0], 0.999, 1/1000)
        while CthetaBool:
            foundw = False
            newCtheta = currLB - Cthetaincrem
            for w in Wvec:
                temp = cSupHHsqz(w, newCtheta, scDict)
                if temp > cSup:  # There's a valid w
                    foundw = True
            if not foundw:
                CthetaBool = False
            else:
                currLB = newCtheta
    else:  # Prices decrease with decreasing Ctheta
        Wvec = np.arange(cSup, SupPriceHH(scDict, 0)[0], 1 / 1000)
        while CthetaBool:
            foundw = False
            newCtheta = currLB - Cthetaincrem
            for w in Wvec:
                temp = cSupHHsqz(w, newCtheta, scDict)
                if temp >= cSup:  # There's a valid w
                    foundw = True
                    currLB = newCtheta
            if not foundw:
                CthetaBool = False

    return currLB

def SupPriceHHsqz(scDict, Ctheta):
    # Return on-path HHsqz prices
    if Ctheta >= CthetaHHLB(scDict) or Ctheta < CthetaHHsqzLB(scDict):
        print('Given Ctheta is outside the valid HHsqz region')
        w = 0
    else:  # Ctheta is valid
        foundw = False
        b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
        critCsup = GetCritcSupHHsqz(scDict)  # Get critical cSup value
        if cSup < critCsup:  # Prices increase with decreasing Ctheta
            Wvec = np.arange(SupPriceHH(scDict, 0)[0], 0.999, 1 / 1000)
        else:  # Prices decrease with decreasing Ctheta
            Wvec = np.arange(SupPriceHH(scDict, 0)[0], cSup, -1 / 1000)
        currwind = -1
        while not foundw:
            currwind += 1
            if cSupHHsqz(Wvec[currwind], Ctheta, scDict) >= cSup:  # Take first w where this occurs
                w = Wvec[currwind]
                foundw = True
    return w, w

def CthetaLHFOCLB(scDict):
    # Returns the lower bound of the LH FOC region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    retval = (32 + b*(-8*b*(3+b) + 4*(-1 + b)*((2 + b)**2)*cSup + b*(4 - 3*(b**2))*(cSup**2)))/(16*((-4 +\
                (b**2))**2)*(-1 +(b**2))*(-1 + (supRateLo**2)))
    return retval

def cSupexp(scDict, Ctheta):
    # Returns cSup critical value for LH-exp region; if cSup exceeds this value, then LH-exp is valid
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    retval = (2*((b**4)*(10 - 336*Ctheta*(-1 + (supRateLo**2))-42*np.sqrt((1-4*(-4 +3*(b**2))*Ctheta*(-1+\
            (supRateLo**2)))/(-1+(b**2)))) -32*(8*Ctheta*(-1 + (supRateLo**2)) + np.sqrt((1 - 4*(-4 +\
            3*(b**2))*Ctheta*(-1 + (supRateLo**2)))/(-1 +(b**2)))) - 3*(b**7)*(8*Ctheta*(-1 + (supRateLo**2)) +\
            np.sqrt((1 - 4*(-4 +3*(b**2))*Ctheta*(-1 + (supRateLo**2)))/(-1 +(b**2)))) - 4*(b**3)*(1 + 16*Ctheta*(-1 +\
            (supRateLo**2)) + np.sqrt((1 - 4*(-4 +3*(b**2))*Ctheta*(-1 + (supRateLo**2)))/(-1 +(b**2)))) +\
            (b**5)*(1 + 80*Ctheta*(-1 + (supRateLo**2)) +7*np.sqrt((1 - 4*(-4 + 3*(b**2))*Ctheta*(-1 +\
            (supRateLo**2)))/(-1 +(b**2))))+8*(b**2)*(-1 + 64*Ctheta*(-1 + (supRateLo**2)) + 8*np.sqrt((1 -\
            4*(-4 + 3*(b**2))*Ctheta*(-1 + (supRateLo**2)))/(-1 + (b**2)))) + (b**6)*(-3+72*Ctheta*(-1+(supRateLo**2))+\
            10*np.sqrt((1 - 4*(-4 +3*(b**2))*Ctheta*(-1 + (supRateLo**2)))/(-1 +(b**2))))))/(((-2 + b)**2)*b*(-4 +\
            3*(b**2))*(-4 - 3*b + 2*np.sqrt((1-4*(-4 + 3*(b**2))*Ctheta*(-1 + (supRateLo**2)))/(-1 +(b**2))) +\
            2*b*np.sqrt((1-4*(-4 + 3*(b**2))* Ctheta*(-1 + (supRateLo**2)))/(-1 +(b**2)))))
    return retval

# TODO: FINISH
def CthetaLHexpLB(scDict):
    # Returns the lower bound of the LH exp region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    CthetaVec = np.arange(CthetaLHFOCLB(scDict),0,-1/1000)
    currLB = CthetaVec[0]  # Move down from this
    for Cthetaind in range(CthetaVec.shape[0]):
        currcSupVal = cSupexp(scDict, CthetaVec[Cthetaind])
        if cSup >= currcSupVal:
            currLB = CthetaVec[Cthetaind]
    return currLB

def CthetaLHFOCUB(scDict):
    # Returns the upper bound of the LH FOC region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    retval = (8+2*(b**3)*(-1+cSup)+4*(-2+cSup)*cSup-3*(b**2)*(2+(-2+cSup)*cSup))/(4*((-4+\
             (b**2))**2)*(-1+(b**2))*(-1+supRateLo))
    return retval

def CthetaLHsqzUB(scDict):
    # Returns the upper bound of the LH FOC region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    retval1 = 1/(2*((-4+(b**2))**4)*(-1+(b**2))*((-1+supRateLo)**2))*(b*(-b*(80+b*(16+\
              b*(-28+b*(-8+b*(3+b)))))*(-1+supRateLo)-2*((2+b)**2)*(4+(b**2)*(-7+3*b))*cSup*(-1+\
              supRateLo)+(32+(b**2)*(-32+b*(4+(-2+b)*(-1+b)*b)))*(cSup**2)*(-1+supRateLo)+\
              2*b*np.sqrt(((8+b*((-2+b)*b*(3+b-cSup)+4*(-1+cSup))-4*cSup)*cSup*(((-2+b)*(-1+\
              b)*((2+b)**3)-(b**2)*(4+(-4+b)*b*(1+b))*cSup)**2)*((-1+supRateLo)**2))/((-1+b**2)**2)))-\
              2*(32+np.sqrt(((8+b*((-2+b)*b*(3+b-cSup)+4*(-1+cSup))-4*cSup)*cSup*(((-2+b)*(-1+\
              b)*((2+b)**3)-(b**2)*(4+(-4+b)*b*(1+b))*cSup)**2)*((-1+supRateLo)**2))/((-1+(b**2))**2))-\
              32*supRateLo))
    retval2 = -(((-1+cSup)**2)/(4*(b**2)*(-1+supRateLo)))
    return np.min((retval1, retval2))

def CthetaLHsqztwoUB(scDict):
    # Returns the upper bound of the LH FOC region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    tempval = (-3*((-1+cSup)**2)*(-1+supRateLo)+((cSup**2)*(-1+supRateLo))/(-1+b)+((-2+(cSup**2))*(-1+\
             supRateLo))/(1+b)+2*np.sqrt((((1+b*(-1+cSup))**2)*(2+2*b*(-1+cSup)-cSup)*cSup*\
             ((-1+supRateLo)**2))/((-1+(b**2))**2)))/(16*((-1+supRateLo)**2))
    # It's only valid if the squeeze prices are below cSup here
    if SupPriceLHSqz(scDict, tempval)[1] >= cSup:
        retval = CthetaLHsqzUB(scDict)
    else:
        retval = tempval
    return retval

def cSupBar(b):
    # Returns maximum allowable cSup, as per Condition 1 in the paper
    return 1 - (b/(2-(b**2)))

def LthetaEqMatsForPlot(numpts, Ctheta_max, cSup_max, scDict):
    # Generate list of equilibria matrices for plotting
    CthetaVec = np.arange(0, Ctheta_max, (Ctheta_max)/numpts)
    cSupVec = np.arange(0.01, cSup_max, cSup_max/numpts)

    eq_list = ['LL', 'LHexp', 'LHFOC', 'LHsqz', 'HHsqz', 'HH', 'N']
    eqStrat_matList = np.zeros((len(eq_list), cSupVec.shape[0], CthetaVec.shape[0]))
    eqStrat_matList[:] = np.nan

    for currcSupind in range(cSupVec.shape[0]):
        currdict = scDict.copy()
        currdict['cSup'] = cSupVec[currcSupind]
        # Get bounds under current cSup
        CthLLUB, CthHHLB, CthHHsqzLB = CthetaLLUB(currdict), CthetaHHLB(currdict), CthetaHHsqzLB(currdict)
        CthLHFOCLB, CthLHexpLB, CthLHFOCUB = CthetaLHFOCLB(currdict), CthetaLHexpLB(currdict), CthetaLHFOCUB(currdict)
        CthLHsqzUB, CthLHsqztwoUB = CthetaLHsqzUB(currdict), CthetaLHsqztwoUB(currdict)
        # Place "1" where present for each respective equilibrium
        eqStrat_matList[0, currcSupind, np.where(CthetaVec < CthLLUB)] = 1  # LL
        eqStrat_matList[1, currcSupind, np.where((CthetaVec >= CthLHexpLB) & (CthetaVec < CthLHFOCLB))] = 1  # LHexp
        eqStrat_matList[2, currcSupind, np.where((CthetaVec >= CthLHFOCLB) & (CthetaVec <= CthLHFOCUB))] = 1  # LHFOC
        eqStrat_matList[3, currcSupind, np.where((CthetaVec > CthLHFOCUB) &
                                                 ((CthetaVec < CthLHsqzUB) | (CthetaVec < CthLHsqztwoUB)))] = 1  # LHsqz
        eqStrat_matList[4, currcSupind, np.where((CthetaVec >= CthHHsqzLB) & (CthetaVec < CthHHLB))] = 1  # HHsqz
        eqStrat_matList[5, currcSupind, np.where(CthetaVec >= CthHHLB)] = 1  # HH
    # Identify any excessively large cSup
    cSupCond = cSupBar(scDict['b'])
    eqStrat_matList[6, np.where(cSupVec > cSupCond), :] = 1

    return eqStrat_matList

def SPfrictThresh(scDict):
    # Returns the SP friction (alpha) threshold, for which LH-sqz is preferred
    #   for higher alpha values and HH is preferred otherwise
    b, cS, rateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    retval = -(((-2+b)*(1+b)*(b*(2-2* cS+b *np.sqrt((4 *((2+b)**2))/((2+b-(b**2))**2)+1/(((4-5*(b**2)+(b**4))**2)*(-1+\
             rateLo))*(-4*(-1+b)*((2+b)**2)*(8+b*(-4+b*(-6+5*b)))*cS*(-1+rateLo)+(-64+\
             b*(64+b*(80+b*(-64+b*(-28+b*(12+7*b))))))*(cS**2)*(-1+rateLo)-4*((-4+(b**2))**2)*(4-7*(b**2)+\
             3*(b**4))*np.sqrt(-1*((cS*((-2+b+(b**2)-b*cS)**2)*(8-4*b*(1+b)-4*cS+b*(4+b)*cS)*((-1+rateLo)**2))/(((-4+\
             (b**2))**3)*((-1+(b**2))**2)))))))-2*np.sqrt((4*((2+b)**2))/((2+b-(b**2))**2)+1/(((4-5*(b**2)+\
             (b**4))**2)*(-1+rateLo))*(-4*(-1+b)*((2+b)**2)*(8+b*(-4+b*(-6+5*b)))*cS*(-1+rateLo)+(-64+\
             b*(64+b*(80+b*(-64+b*(-28+b*(12+7*b))))))*(cS**2)*(-1+rateLo)-4*((-4+(b**2))**2)*(4-7*(b**2)+\
             3*(b**4))* np.sqrt(-1*((cS*((-2+b+(b**2)-b*cS)**2)*(8-4*b*(1+b)-4*cS+b*(4+b)*cS)*((-1+rateLo)**2))/(((-4+\
             (b**2))**3)*((-1+(b**2))**2)))))))*(-1+rateLo))/((-2+b+(b**2))*(-2*(-2+2*cS+np.sqrt((4*((2+b)**2))/((2+b-\
             (b**2))**2)+1/(((4-5*(b**2)+(b**4))**2)*(-1+rateLo))*(-4*(-1+b)*((2+b)**2)*(8+b*(-4+b*(-6+5*b)))*cS*(-1+\
             rateLo)+(-64+b*(64+b*(80+b*(-64+b*(-28+b*(12+7*b))))))*(cS**2)*(-1+rateLo)-4*((-4+\
             (b**2))**2)*(4-7*(b**2)+3*(b**4))*np.sqrt(-1*((cS*((-2+b+(b**2)-b*cS)**2)*(8-4*b*(1+b)-4*cS+b*(4+\
             b)*cS)*((-1+rateLo)**2))/(((-4+(b**2))**3)*((-1+(b**2))**2)))))))+b*(2-2*cS-np.sqrt((4*((2+b)**2))/((2+\
             b-(b**2))**2)+1/(((4-5*(b**2)+(b**4))**2)*(-1+rateLo))*(-4*(-1+b)*((2+b)**2)*(8+b*(-4+b*(-6+5*b)))*cS*(-1+\
             rateLo)+(-64+b*(64+b*(80+b*(-64+b*(-28+b*(12+7*b))))))*(cS**2)*(-1+rateLo)-4*((-4+(b**2))**2)*(4-7*(b**2)+\
             3*(b**4))*np.sqrt(-1*((cS*((-2+b+(b**2)-b*cS)**2)*(8-4*b*(1+b)-4*cS+b*(4+b)*cS)*((-1+rateLo)**2))/(((-4+\
             (b**2))**3)*((-1+(b**2))**2))))))+b*np.sqrt((4*((2+b)**2))/((2+b-(b**2))**2)+1/(((4-5*(b**2)+\
             (b**4))**2)*(-1+rateLo))*(-4*(-1+b)*((2+b)**2)*(8+b*(-4+b*(-6+5*b)))*cS*(-1+rateLo)+(-64+b*(64+b*(80+\
             b*(-64+b*(-28+b*(12+7*b))))))*(cS**2)*(-1+rateLo)-4*((-4+(b**2))**2)*(4-7*(b**2)+\
             3*(b**4))*np.sqrt(-1*((cS*((-2+b+(b**2)-b*cS)**2)*(8-4*b*(1+b)-4*cS+b*(4+b)*cS)*((-1+rateLo)**2))/(((-4+\
             (b**2))**3)*((-1+(b**2))**2))))))))))
    return retval



#######################
# WHOLESALE PRICE PLOTS
#######################
# for b=[0.6, 0.9]
b, cSup, supRateLo = 0.6, 0.2, 0.8
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo}

CthetaMax = 1.7
CthetaVec = np.arange(0, CthetaMax, 0.001)
CthLLUB, CthHHLB, CthHHsqzLB = CthetaLLUB(scDict), CthetaHHLB(scDict), CthetaHHsqzLB(scDict)
CthLHFOCLB, CthLHexpLB, CthLHFOCUB = CthetaLHFOCLB(scDict), CthetaLHexpLB(scDict), CthetaLHFOCUB(scDict)
CthLHsqzUB, CthLHsqztwoUB = CthetaLHsqzUB(scDict), CthetaLHsqztwoUB(scDict)
LLprices = np.empty((CthetaVec.shape[0],2))
LLprices[:] = np.nan
HHprices, HHsqzprices, LHexpprices, LHFOCprices = LLprices.copy(), LLprices.copy(), LLprices.copy(), LLprices.copy()
LHsqzprices, LHsqztwoprices = LLprices.copy(), LLprices.copy()
# Store prices
for Cthetaind in range(CthetaVec.shape[0]):
    currCtheta = CthetaVec[Cthetaind]
    if currCtheta <= CthLLUB:  # LL
        LLprices[Cthetaind, :] = SupPriceLL(scDict, currCtheta)
    if currCtheta >= CthLHexpLB and currCtheta < CthLHFOCLB:  # LHexp
        LHexpprices[Cthetaind, :] = SupPriceLHexp(scDict, currCtheta)
    if currCtheta >= CthLHFOCLB and currCtheta <= CthLHFOCUB:  # LHFOC
        LHFOCprices[Cthetaind, :] = SupPriceLHFOC(scDict, currCtheta)
    if currCtheta > CthLHFOCUB and currCtheta <= CthLHsqzUB:  # LHsqz1
        LHsqzprices[Cthetaind, :] = SupPriceLHSqz(scDict, currCtheta)
    if currCtheta >= CthLHsqzUB and currCtheta <= CthLHsqztwoUB and currCtheta > CthLHFOCUB:  # LHsqz2
        LHsqztwoprices[Cthetaind, :] = SupPriceLHSqzTwo(scDict, currCtheta)
    if currCtheta >= CthHHsqzLB and currCtheta < CthHHLB:  # HHsqz
        HHsqzprices[Cthetaind, :] = SupPriceHHsqz(scDict, currCtheta)
    if currCtheta >= CthHHLB:
        HHprices[Cthetaind, :] = SupPriceHH(scDict, currCtheta)


fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$c_S=$'+str(cSup)+', '+r'$L=$'+str(supRateLo),
             fontsize=18, fontweight='bold')

al = 0.8
LLcol, HHcol, HHsqzcol = 'red', 'indigo', 'mediumorchid'
LHonecols = ['limegreen', 'seagreen', 'darkgreen']
LHtwocols = ['cornflowerblue', 'blue', 'midnightblue']
lnwd = 5

plt.plot(CthetaVec, LLprices[:, 0], linewidth=lnwd, color=LLcol, alpha=al)
# plt.plot(CthetaVec, LLprices[:, 1], linewidth=lnwd, color=LLcol, alpha=al)
plt.plot(CthetaVec, LHexpprices[:, 0], linewidth=lnwd, color=LHtwocols[0], alpha=al)
plt.plot(CthetaVec, LHexpprices[:, 1], linewidth=lnwd, color=LHonecols[0], alpha=al)
plt.plot(CthetaVec, LHFOCprices[:, 0], linewidth=lnwd, color=LHtwocols[1], alpha=al)
plt.plot(CthetaVec, LHFOCprices[:, 1], linewidth=lnwd, color=LHonecols[1], alpha=al)
plt.plot(CthetaVec, LHsqzprices[:, 0], linewidth=lnwd, color=LHtwocols[2], alpha=al)
plt.plot(CthetaVec, LHsqzprices[:, 1], linewidth=lnwd, color=LHonecols[2], alpha=al)
plt.plot(CthetaVec, LHsqztwoprices[:, 0], linewidth=lnwd, color=LHtwocols[2], alpha=al)
plt.plot(CthetaVec, LHsqztwoprices[:, 1], linewidth=lnwd, color=LHonecols[2], alpha=al)
plt.plot(CthetaVec, HHsqzprices[:, 0], linewidth=lnwd, color=HHsqzcol, alpha=al)
plt.plot(CthetaVec, HHsqzprices[:, 1], linewidth=lnwd, color=HHsqzcol, alpha=al)
plt.plot(CthetaVec, HHprices[:, 0], linewidth=lnwd, color=HHcol, alpha=al)
# plt.plot(CthetaVec, HHprices[:, 1], linewidth=lnwd, color=HHcol, alpha=al)
plt.ylim(0, 0.6)
plt.xlim(0, CthetaMax)
plt.xlabel(r'$C_{\theta}$', fontsize=14)
plt.ylabel(r'$w$', fontsize=14, rotation=0, labelpad=14)
plt.show()

#####################
# Equilibrium plots
#####################
# Use b=[0.6,0.9]
b, cSup, supRateLo = 0.6, 0.1, 0.8
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo}

numpts, CthetaMax, cSupMax = 120, 1.7, 0.35
alval = 0.6

eqStrat_matList = LthetaEqMatsForPlot(numpts, CthetaMax, cSupMax, scDict)
# Fill holes
for csupind in range(eqStrat_matList.shape[1]):
    for cthetaind in range(eqStrat_matList.shape[2]):
        if np.nansum(eqStrat_matList[:,csupind,cthetaind]) == 0:
            eqStrat_matList[4, csupind, cthetaind] = 1

fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$L=$'+str(supRateLo),
             fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['red', 'darkorange', 'gold', 'darkgreen', 'royalblue', 'darkviolet', 'black']
labels = ['LL', 'LHexp', 'LHFOC', 'LHsqz', 'HHsqz', 'HH', 'N']

imlist = []
for eqind in range(len(labels)):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[eqind]], name='from_list', N=None)
    if eqcolors[eqind] == 'black':  # No alpha transparency
        im = ax.imshow(eqStrat_matList[eqind], vmin=0, vmax=1, aspect='auto',
                            extent=(0, CthetaMax, 0, cSupMax),
                            origin="lower", cmap=mycmap, alpha=1)
    else:
        im = ax.imshow(eqStrat_matList[eqind], vmin=0, vmax=1, aspect='auto',
                            extent=(0, CthetaMax, 0, cSupMax),
                            origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)

legwidth = 20
wraplabels = ['\n'.join(textwrap.wrap(labels[i], width=legwidth)) for i in range(len(labels))]
patches = [mpatches.Patch(color=eqcolors[i], edgecolor='black', label=wraplabels[i], alpha=alval) for i in range(len(eqcolors))]
# put those patched as legend-handles into the legend
ax.legend(handles=patches, bbox_to_anchor=(1.3, 1.0), loc='upper right', borderaxespad=0.1, fontsize=8)
ax.set_xbound(0, CthetaMax)
ax.set_ybound(0, cSupMax)
ax.set_box_aspect(1)
plt.xlabel(r'$C_{\theta}$', fontsize=14)
plt.ylabel(r'$c_S$', fontsize=14, rotation=0, labelpad=14)
plt.show(block=True)

#############################
# SP friction threshold plot
#############################
# for b=[0.6,0.9]
b, cSup, supRateLo = 0.9, 0.2, 0.8
scDict1 = {'b': 0.5, 'cSup': cSup, 'supRateLo': supRateLo}
scDict2 = {'b': 0.8, 'cSup': cSup, 'supRateLo': supRateLo}
cSupVec = np.arange(0.01, 0.24, 0.01)
yvec1, yvec2 = [], []
for currcSup in cSupVec:
    currdict1, currdict2 = scDict1.copy(), scDict2.copy()
    currdict1['cSup'] = currcSup
    currdict2['cSup'] = currcSup
    yvec1.append(SPfrictThresh(currdict1))
    yvec2.append(SPfrictThresh(currdict2))


plt.plot(yvec1, cSupVec, '-', linewidth=5, color='indigo')
plt.plot(yvec2, cSupVec, '--', linewidth=5, color='indigo')
# plt.suptitle(r'$b=$'+str(b)+', '+r'$L=$'+str(supRateLo),
#              fontsize=18, fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0, 0.24)
ax.set_box_aspect(1)
plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel(r'$c_S$', fontsize=14, rotation=0, labelpad=14)
plt.show()




