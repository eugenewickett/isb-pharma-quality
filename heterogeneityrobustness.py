# Examining robustness of  for "Strategic Role of Inspections in Pharmaceutical Supply Chains"
# These functions use the simpler version of the model
#   - perfect diagnostic, H=1, no retailer quality choice, retailer sources from both suppliers or neither
# 22-DEC-25

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

def quantOpt(w, wOpp, b):
    # Returns optimal order quantities under dual sourcing
    return max(0, (1-b-w+b*wOpp)/(2*(1-(b**2))))


# Function returning retailer utilities under each of 7 possible policies
def RetUtil(lambsup1, lambsup2, Ltheta, b, w1, w2):
    # Returns retailer utility under pi=D
    q1 = max((1 - b - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
    q2 = max((1 - b - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
    retval = (1 - b - w1 + b*w2) * (1 - w1 - b*q2 - q1) / (2 * (1 - b ** 2)) + \
                     (1 - b + b*w1 - w2) * (1 - w2 - b*q1 - q2) / (2 * (1 - b ** 2)) - \
                     Ltheta * ((1) * (1 - lambsup1 * lambsup2))
    return retval

def SupPriceLL(scDict, Ctheta):
    # Returns on-path LL prices
    w = max((1 - scDict['b']) / (2 - scDict['b']), 0)
    return w, w

def SupPriceLLSqz(scDict, Ctheta):
    # Returns on-path LLsqz prices
    w = max(1 - np.sqrt(-2*((1 + scDict['b'])*Ctheta*(-1 + scDict['supRateLo']**2))), 0)
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

def SupPriceLHexpFOC(scDict, Ctheta):
    # Returns on-path LH-exp (FOC) prices
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    w1 = max((b + (b**2)*(3-2*np.sqrt((1-4*(-4 + 3*(b**2))*Ctheta* (-1 + supRateLo**2))/(-1 + (b**2)))) +\
              2*(-2 + np.sqrt((1 - 4*(-4 + 3*(b**2))*Ctheta*(-1 + (supRateLo**2)))/(-1+ (b**2)))))/(-4+ 3* (b**2)), 0)
    w2 = max(0.5*(1+cSup+b*(-1+w1)), 0)
    return w1, w2

def CritcsupLHexpIR(w, b, Ctheta, supRateLo):
    radterm = np.sqrt((((1-w)**2)+4*Ctheta*(-1 + (supRateLo**2))) / (-1 + (b**2)))
    retval = 1 + b*(-1 + w)-2*np.sqrt(-((-1 + (b**2))*(((1-w)**2) + 4*Ctheta*(-1 + (supRateLo**2)) +\
                                      radterm - b*radterm + b*w*radterm)))
    return retval

def SupPriceLHexpIR(scDict, Ctheta, wstep=1/1000):
    # Returns on-path LH-exp (IR) prices
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    # Start from LH-FOC price for S1; move up until critical cSup value reached
    # Report error if max of critical value function does not exceed cSup
    wVec = np.arange(SupPriceLHFOC(scDict, Ctheta)[0], 0.999, wstep)
    crtiValVec = [CritcsupLHexpIR(w, b, Ctheta, supRateLo) for i, w in enumerate(wVec)]
    if np.max(crtiValVec) < cSup:
        # print("The Ctheta value for LHexpIR is out of bounds.")
        return -1, -1
    # Identify first value above cSup
    retind = np.where(np.array(crtiValVec)>cSup)[0][0]
    w1 = max(wVec[retind], 0)
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

def CthetaLHFOCLB_OLD(scDict):
    # Returns the lower bound of the LH FOC region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    retval = (32 + b*(-8*b*(3+b) + 4*(-1 + b)*((2 + b)**2)*cSup + b*(4 - 3*(b**2))*(cSup**2)))/(16*((-4 +\
                (b**2))**2)*(-1 +(b**2))*(-1 + (supRateLo**2)))
    return retval

def wOffLHFOC(scDict, Ctheta):
    # S2 off-path price, given w1LHFOC, s.t. retailer IR is met
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    radterm = np.sqrt((4 - 64*Ctheta + b*(4 - 4*cSup + b*(((-1 + cSup)**2) - 4*(-8 + (b**2))*Ctheta)) +\
                       4*((-4 + (b**2))**2)*Ctheta*(supRateLo**2)) / (((-4 + (b**2))**2)*(-1 + (b**2))))
    retval = 2 + 2 / (-2 + b) - radterm + (b**2)*(-1*(cSup / (-4 + (b**2))) + radterm)
    return retval

def CthetaLHFOCLB(scDict, step=0.001):
    # Returns the lower bound of the LH FOC region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    # Move left from LHFOC UB until the off-path S2 utility exceeds the on-path utility
    CthetaVec = np.arange(CthetaLHFOCUB(scDict), 0, step*-1)
    # Get on-path S2 utility
    w1on, w2on = SupPriceLHFOC(scDict, 0)
    onUtil = SupUtil(quantOpt(w2on, w1on, b), w2on, cSup)

    retCtheta = -1  # Initialize return value
    found = False
    for currCtheta in CthetaVec:
        if found == False:
            w2off = wOffLHFOC(scDict, currCtheta)
            offUtil = SupUtil(quantOpt(w2off, w1on, b), w2off, 0)
            if offUtil > onUtil:
                retCtheta = currCtheta
                found = True

    return retCtheta

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

def CthetaLHexpFOCLB(scDict):
    # Returns the lower bound of the LH exp (FOC) region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    CthetaVec = np.arange(CthetaLHFOCLB(scDict),0,-1/1000)
    currLB = CthetaVec[0]  # Move down from this
    for Cthetaind in range(CthetaVec.shape[0]):
        currcSupVal = cSupexp(scDict, CthetaVec[Cthetaind])
        if cSup >= currcSupVal:
            currLB = CthetaVec[Cthetaind]
    return currLB

def CthetaLHexpIRLB(scDict, step=1/1000):
    # Returns the lower bound of the LH exp (IR) region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    CthetaVec = np.arange(CthetaLHFOCLB(scDict), CthetaLHexpFOCLB(scDict), -1*step)
    currLB = CthetaVec[0]  # Move down from this
    found = False
    # Get Ctheta bounds, for S1's off-path comparisons
    LLUB, LLsqzUB = CthetaLLUB(scDict), CthetaLLsqzUB(scDict)
    LLutil = SupUtil(quantOpt(SupPriceLL(scDict, 0)[0],SupPriceLL(scDict, 0)[0], b), SupPriceLL(scDict, 0)[0], 0)
    for Cthetaind, currCtheta in enumerate(CthetaVec):
        if found==False:
            w1on, w2on = SupPriceLHexpIR(scDict, currCtheta)
            if w1on ==-1:  # We've already hit the LB
                if Cthetaind > 0:
                    currLB = CthetaVec[Cthetaind-1]
                    return currLB
                else:
                    currLB = CthetaVec[0]
                    return currLB
            w2offFOC = 0.5*(1 + b*(-1 + w1on))  # S2's FOC off-path
            w2offIR = 1 - b + w1on*b - (1 - b*b)*np.sqrt((((-1 + w1on)**2) +\
                        4*currCtheta*(-1 + (supRateLo**2)))/(-1 + (b**2)))  # S2's IR-based off-path
            if w2offIR>w2offFOC and w2on!=-1:
                print('S2 off-path price exceeds off-path FOC price: '+\
                      str(w2offFOC) + ', ' + str(w2offIR) + '; CHECK THIS OUT')
                currLB = currCtheta
                found = True
            # Check S1's utility elsewhere, which is LL if Ctheta<=LLUB, LLsqz if LLUB<Ctheta<=LLsqzUB
            currSup1Util = SupUtil(quantOpt(w1on, w2on, b), w1on, 0)
            LLsqzutil = SupUtil(quantOpt(SupPriceLLSqz(scDict, currCtheta)[0], SupPriceLLSqz(scDict, currCtheta)[1], b),
                                SupPriceLLSqz(scDict, currCtheta)[0], 0)
            if currCtheta <= LLUB and currSup1Util <= LLutil:
                currLB = currCtheta
                found = True
            elif currCtheta <= LLsqzUB and LLUB < currCtheta and currSup1Util <= LLsqzutil:
                currLB = currCtheta
                found = True
            # Check if retailer IR constraint satisfied
            currRetUtil = RetUtil(supRateLo, 1, currCtheta, b, w1on, w2on)
            if currRetUtil < 0:
                currLB = currCtheta + step
                found = True
    return currLB

def CthetaLHFOCUB(scDict):
    # Returns the upper bound of the LH FOC region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    retval = (8+2*(b**3)*(-1+cSup)+4*(-2+cSup)*cSup-3*(b**2)*(2+(-2+cSup)*cSup))/(4*((-4+\
             (b**2))**2)*(-1+(b**2))*(-1+supRateLo))
    return retval

def CthetaLHsqzUB(scDict, Cthincr=1/1000):
    # Returns the upper bound of the LH FOC region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    retval1 = 1/(2*((-4+(b**2))**4)*(-1+(b**2))*((-1+supRateLo)**2))*(b*(-b*(80+b*(16+\
              b*(-28+b*(-8+b*(3+b)))))*(-1+supRateLo)-2*((2+b)**2)*(4+(b**2)*(-7+3*b))*cSup*(-1+\
              supRateLo)+(32+(b**2)*(-32+b*(4+(-2+b)*(-1+b)*b)))*(cSup**2)*(-1+supRateLo)+\
              2*b*np.sqrt(((8+b*((-2+b)*b*(3+b-cSup)+4*(-1+cSup))-4*cSup)*cSup*(((-2+b)*(-1+\
              b)*((2+b)**3)-(b**2)*(4+(-4+b)*b*(1+b))*cSup)**2)*((-1+supRateLo)**2))/((-1+b**2)**2)))-\
              2*(32+np.sqrt(((8+b*((-2+b)*b*(3+b-cSup)+4*(-1+cSup))-4*cSup)*cSup*(((-2+b)*(-1+\
              b)*((2+b)**3)-(b**2)*(4+(-4+b)*b*(1+b))*cSup)**2)*((-1+supRateLo)**2))/((-1+(b**2))**2))-\
              32*supRateLo))  # Off-path move to HH preferred
    retval2 = -(((-1+cSup)**2)/(4*(b**2)*(-1+supRateLo)))  # LHsqz prices hit cSup
    # Possible that on-path LLFOC/LLsqz is preferable to S1; check that here
    CthLLsqzUB, CthLLFOCUB, CthLHFOCUB = CthetaLLsqzUB(scDict), CthetaLLUB(scDict), CthetaLHFOCUB(scDict)  # Need these
    if np.max((CthLLsqzUB, CthLLFOCUB)) > CthLHFOCUB:  # Need to check that on-path LL is not preferred
        # Increase incrementally and check
        CthetaVec = np.arange(CthLHFOCUB, np.min((retval1, retval2, CthLLsqzUB)), Cthincr)
        foundBd, retval3 = False, 10000
        for currCtheta in CthetaVec:
            if currCtheta <= CthLLFOCUB:  # Compare w LLFOC
                currw1LL, currw2LL = SupPriceLL(scDict, currCtheta)  # Get LLFOC prices
                currw1LHsqz, currw2LHsqz = SupPriceLHSqz(scDict, currCtheta)  # Get LHsqz prices
                currLLutil = SupUtil(quantOpt(currw1LL, currw2LL, b), currw1LL, 0)
                currLHsqzutil = SupUtil(quantOpt(currw1LHsqz, currw2LHsqz, b), currw1LHsqz, 0)
                if currLLutil > currLHsqzutil and (not foundBd):  # S1 prefers LL to LHsqz
                    retval3 = currCtheta
                    foundBd = True
            if currCtheta > CthLLFOCUB:  # Compare w LLsqz
                currw1LLsqz, currw2LLsqz = SupPriceLLSqz(scDict, currCtheta)  # Get LLsqz prices
                currw1LHsqz, currw2LHsqz = SupPriceLHSqz(scDict, currCtheta)  # Get LHsqz prices
                currLLsqzutil = SupUtil(quantOpt(currw1LLsqz, currw2LLsqz, b), currw1LLsqz, 0)
                currLHsqzutil = SupUtil(quantOpt(currw1LHsqz, currw2LHsqz, b), currw1LHsqz, 0)
                if currLLsqzutil > currLHsqzutil and (not foundBd):  # S1 prefers LLsqz to LHsqz
                    retval3 = currCtheta
                    foundBd = True
    else:
        retval3 = 10000

    retval = np.min((retval1, retval2, retval3))
    if retval == retval2:  # 2nd LHsqz region is valid
        LHsqztwoValid = True
    else:
        LHsqztwoValid = False
    return retval, LHsqztwoValid

def CthetaLHsqztwoUB(scDict, Cthincr=1/1000):
    # Returns the upper bound of the LH FOC region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    # Depends on previously found bound value
    prevUB, sqztwoValid = CthetaLHsqzUB(scDict)  # sqztwoValid indicates cSup was reached
    if sqztwoValid:
        tempval = (-3*((-1+cSup)**2)*(-1+supRateLo)+((cSup**2)*(-1+supRateLo))/(-1+b)+((-2+(cSup**2))*(-1+\
             supRateLo))/(1+b)+2*np.sqrt((((1+b*(-1+cSup))**2)*(2+2*b*(-1+cSup)-cSup)*cSup*\
             ((-1+supRateLo)**2))/((-1+(b**2))**2)))/(16*((-1+supRateLo)**2))
        # Need to compare with on-path LLsqz as well, potentially
        CthLLsqzUB = CthetaLLsqzUB(scDict)
        foundBd = False
        if CthLLsqzUB > prevUB:
            CthetaVec = np.arange(prevUB, CthLLsqzUB, Cthincr)
            for currCtheta in CthetaVec:
                currw1LLsqz, currw2LLsqz = SupPriceLLSqz(scDict, currCtheta)  # Get LLsqz prices
                currw1LHsqz2, currw2LHsqz2 = SupPriceLHSqzTwo(scDict, currCtheta)  # Get LHsqz2 prices
                currLLsqzutil = SupUtil(quantOpt(currw1LLsqz, currw2LLsqz, b), currw1LLsqz, 0)
                currLHsqz2util = SupUtil(quantOpt(currw1LHsqz2, currw2LHsqz2, b), currw1LHsqz2, 0)
                if currLLsqzutil > currLHsqz2util and (not foundBd):  # S1 prefers LLsqz to LHsqz
                    newbd = currCtheta
                    foundBd = True
        if foundBd:
            retval = newbd
        else:
            retval = tempval
    else:
        retval = prevUB
    return retval

def CthetaLLsqzUBoffIR(scDict, maxCtheta=3.0, Cstep = 1/1000):
    # Return the Ctheta where off-path IR-induced price is preferable to on-path LLsqz
    # maxCtheta is max considered Ctheta
    # Only valid if the following is non-negative
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    if (1 - b - supRateLo - b* supRateLo) < 0:
        return 10000
    else:
        CthetaVec = np.arange(0, maxCtheta, Cstep)
        cSupList = []
        for currCtheta in CthetaVec:
            currradval = np.sqrt(-1*((currCtheta*(-1 + supRateLo)*(-1 + b + supRateLo + b*supRateLo))/(-1 + (b**2))))
            currradval2 = np.sqrt(-1*((1+b)*currCtheta*(-1+(supRateLo**2))))
            newval = 1/((1+b)*currradval)*(-2*np.sqrt(2)*currCtheta*(-1+supRateLo)*supRateLo+currradval +\
                     np.sqrt(2)*b*currCtheta*(1+(2-3*supRateLo)*supRateLo)-currradval2 +\
                     b*currradval*(1-np.sqrt(2)*currradval2)+np.sqrt(2)*(b**2)*(currCtheta-currCtheta*(supRateLo**2) -\
                     currradval*currradval2))
            cSupList.append(newval)
        if np.nanmin(cSupList)<cSup and np.nanmax(cSupList)>cSup:  # We have a matching value
            retVal = CthetaVec[np.where(np.array(cSupList) > cSup)[0][0]]
        else: # Return standard non-value
            return 10000
        return retVal

def CthetaLLsqzUB(scDict, printterms=False):
    # Returns the upper bound of the LL sqz region WRT Ctheta
    b, cSup, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    term1 = -1*(1/(2*(1 + b)*(-1 + (supRateLo**2))))  # Ctheta where on-path prices are 0
    # Ctheta where off-path FOC price is preferred
    term2 = (-4*b*(-1+(-4+cSup)*cSup)*(-1+(supRateLo**2))+4*(-1+(-2+cSup)*cSup)*(-1+(supRateLo**2))-\
             (b**2)*(1+cSup*(6+cSup))*(-1+(supRateLo**2))+(4+4*b)*np.sqrt(((-1+b)*cSup*(-2+b+cSup)*((-2+b+\
             b*cSup)**2)*((-1+(supRateLo**2))**2))/((1+b)**2)))/(2*((-2+b)**4)*(1+b)*((-1+(supRateLo**2))**2))
    # MUST be coupled with off-path retailer IR >= 0
    w1on, w2on = SupPriceLLSqz(scDict, term2)
    w1FOCoff = 0.5*(1 + cSup - b*w2on)
    retUtilFOCoff = RetUtil(1, supRateLo, term2, b, w1FOCoff, w2on)
    if (w1FOCoff <= 0) or (retUtilFOCoff <= 0):
        term2 = 10000  # Placeholder value, as this mechanism is invalid
    term3 = CthetaLLsqzUBoffIR(scDict)
    if printterms==True:
        print('w_i=0: '+str(term1))
        print('FOCoff: ' + str(term2)+', retIR='+str(retUtilFOCoff))
        print('IRoff: ' + str(term3))
    return min(term1, term2, term3)

def cSupBar(b):
    # Returns maximum allowable cSup, as per Condition 1 in the paper
    return 1 - (b/(2-(b**2)))

def cSupDotDeriv(scDict):
    # Evaluates derivative of cSupHHsqz value; positive indicates HHsqz prices are higher than HH prices; negative
    #   indicates lower HHsqz prices than HH prices
    b, cS, rateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    Aterm = np.sqrt((((-1+(-1+b-cSup)/(-2+b))**2) +\
            (-1*(((-4+b*(4+cSup*(4-4*cSup+b*(-4+3*cSup))))*(-1+rateLo))/(((-2+b)**2)*(-1+(b**2))))+\
            np.sqrt(-1*((((2+b*(-2+cSup))**2)*cSup*(4-2*cSup+b*(-4+3*cSup))*((-1+rateLo)**2))/(((-2+\
            b)**3)*((-1+(b**2))**2)))))/(2*(-1+rateLo)))/(-1+(b**2)))

    retval = (1 / (-1 + (-1 + b - cSup) / (-2 + b)))*(-3 + (4*(-1+b-cSup))/(-2+b) + (b*(-1+b-cSup)*(1+(-1+(-1+b-\
              cSup)/(-2 + b))/(((-1+(b**2))*Aterm))))/(-2+b)+ b*(-2+(-1+b-cSup)/(-2+b) + Aterm) +\
              (-1+(-1+b-cSup) / (-2 + b)) / ((-1 + (b**2))*Aterm) +\
              ((b**2)*((-1 + (-1 + b - cSup) / (-2 + b))**2)) / ((-1 + (b**2))*Aterm) +\
              (b**2)*Aterm) - (1 / ((-1 + (-1 + b - cSup) / (-2 + b))**2))*(1+b+\
              (((-3+(2*(-1+b-cSup))/(-2 + b))*(-1+b-cSup))/(-2+b))+\
              ((b*(-1 + b - cSup)*(-2 + ((-1 + b - cSup) / (-2 + b)) + Aterm)) / (-2 + b)) + Aterm +\
              (b**2)*(-1+((-1+b-cSup) / (-2 + b)))*Aterm +\
              ((-1*(((-4+b*(4+cSup*(4-4*cSup+b*(-4+3*cSup))))*(-1+rateLo)) / (((-2+b)**2)*(-1+(b**2))))+\
              np.sqrt(-1*((((2 + b*(-2 + cSup))**2)*cSup*(4-2*cSup+b*(-4+3*cSup))*((-1 +rateLo)**2)) / (((-2+\
              b)**3)*((-1 + (b**2))**2))))) / (2*(-1 +rateLo))) +\
              (b*(-1*(((-4 + b*(4 + cSup*(4-4*cSup+b*(-4+3*cSup))))*(-1 +rateLo)) / (((-2+b)**2)*(-1+(b**2))))+\
              np.sqrt(-1*((((2 + b*(-2 + cSup))**2)*cSup*(4-2*cSup+b*(-4+3*cSup))*((-1 +rateLo)**2)) / (((-2+\
              b)**3)*((-1 + (b**2))**2)))))) / (2*(-1 +rateLo)))
    return retval

def LthetaEqMatsForPlot(numpts, Ctheta_max, cSup_max, scDict):
    # Generate list of equilibria matrices for plotting
    CthetaVec = np.arange(0, Ctheta_max, (Ctheta_max)/numpts)
    cSupVec = np.arange(0.01, cSup_max, cSup_max/numpts)

    eq_list = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHsqz', 'HH', 'N']
    eqStrat_matList = np.zeros((len(eq_list), cSupVec.shape[0], CthetaVec.shape[0]))
    eqStrat_matList[:] = np.nan

    for currcSupind in range(cSupVec.shape[0]):
        currdict = scDict.copy()
        currdict['cSup'] = cSupVec[currcSupind]
        # Get bounds under current cSup
        CthLLUB, CthHHLB, CthHHsqzLB = CthetaLLUB(currdict), CthetaHHLB(currdict), CthetaHHsqzLB(currdict)
        CthLHFOCLB, CthLHexpLB, CthLHFOCUB = CthetaLHFOCLB(currdict), CthetaLHexpIRLB(currdict), CthetaLHFOCUB(currdict)
        (CthLHsqzUB, _), CthLHsqztwoUB  = CthetaLHsqzUB(currdict), CthetaLHsqztwoUB(currdict)
        CthLLsqzUB = CthetaLLsqzUB(scDict)
        # Place "1" where present for each respective equilibrium
        eqStrat_matList[0, currcSupind, np.where(CthetaVec < CthLLUB)] = 1  # LL
        eqStrat_matList[1, currcSupind, np.where((CthetaVec > CthLLUB) & (CthetaVec <= CthLLsqzUB))] = 1  # LLsqz
        eqStrat_matList[2, currcSupind, np.where((CthetaVec >= CthLHexpLB) & (CthetaVec < CthLHFOCLB))] = 1  # LHexp
        eqStrat_matList[3, currcSupind, np.where((CthetaVec >= CthLHFOCLB) & (CthetaVec <= CthLHFOCUB))] = 1  # LHFOC
        eqStrat_matList[4, currcSupind, np.where((CthetaVec > CthLHFOCUB) &
                                                 ((CthetaVec < CthLHsqzUB) | (CthetaVec < CthLHsqztwoUB)))] = 1  # LHsqz
        eqStrat_matList[5, currcSupind, np.where((CthetaVec >= CthHHsqzLB) & (CthetaVec < CthHHLB))] = 1  # HHsqz
        eqStrat_matList[6, currcSupind, np.where(CthetaVec >= CthHHLB)] = 1  # HH
    # Identify any excessively large cSup
    # cSupCond = cSupBar(scDict['b'])
    # eqStrat_matList[7, np.where(cSupVec > cSupCond), :] = 1

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

def SPfrictThreshHHsqz(scDict):
    # Returns the SP friction (alpha) threshold, for which LH-sqz is preferred
    #   for higher alpha values and HHsqz is preferred otherwise
    b, cS, rateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    retval = ((1-b+b*cS)*(-1+rateLo))/((-1+b)*cS)
    return retval

def cSupBarBar(b):
    # From Condition 1 in paper
    return (1-b)*((2+b)**2)/(4+b*(4+b*(1-b)))

b, cSup, supRateLo = 0.5, 0.2, 0.8
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo}

CthetaLLsqzUB(scDict)

CthetaLHFOCUB(scDict)
CthetaLHFOCLB(scDict)

csupvec = np.arange(0.02,0.8,0.01)
intvec, csupBvec, csupBBvec = [], [], []
for currcsup in csupvec:
    scDict['cSup'] = currcsup
    brange = np.arange(0.01,0.99,0.001)
    xvec, y1vec, y2vec = [], [], []
    for currb in brange:
        scDict['b'] = currb
        xvec.append(currb)
        y1vec.append(CthetaLHsqztwoUB(scDict))
        y2vec.append(CthetaHHLB(scDict))

    intvec.append(brange[np.where(np.array(y1vec)<np.array(y2vec))[0][0]])
    csupBvec.append(cSupBar(intvec[-1]))
    csupBBvec.append(cSupBarBar(intvec[-1]))

plt.plot(xvec,y1vec,'r')
plt.plot(xvec,y2vec,'b')
plt.show()

plt.plot(csupvec,csupvec,'r')
plt.plot(csupvec,intvec,'b')
plt.plot(csupvec,csupBvec,'g')
plt.plot(csupvec,csupBBvec,'black')
plt.show()

#######################
# WHOLESALE PRICE PLOTS
#######################
# for b=[0.6, 0.9]
b, cSup, supRateLo = 0.5, 0.2, 0.8
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo}

CthLLUB, CthHHLB, CthHHsqzLB = CthetaLLUB(scDict), CthetaHHLB(scDict), CthetaHHsqzLB(scDict)
CthLHFOCLB, CthLHexpLB, CthLHFOCUB = CthetaLHFOCLB(scDict), CthetaLHexpIRLB(scDict), CthetaLHFOCUB(scDict)
(CthLHsqzUB, _), CthLHsqztwoUB, CthLLsqzUB = CthetaLHsqzUB(scDict), CthetaLHsqztwoUB(scDict), CthetaLLsqzUB(scDict)
CthetaMax = 1.2*CthHHLB
CthetaVec = np.arange(0, CthetaMax, 0.001)
LLprices = np.empty((CthetaVec.shape[0], 2))
LLprices[:] = np.nan
HHprices, HHsqzprices, LHexpprices, LHFOCprices = LLprices.copy(), LLprices.copy(), LLprices.copy(), LLprices.copy()
LHsqzprices, LHsqztwoprices, LLsqzprices = LLprices.copy(), LLprices.copy(), LLprices.copy()
# Store prices
for Cthetaind in range(CthetaVec.shape[0]):
    currCtheta = CthetaVec[Cthetaind]
    if currCtheta <= CthLLUB:  # LL
        LLprices[Cthetaind, :] = SupPriceLL(scDict, currCtheta)
    if currCtheta > CthLLUB and currCtheta <= CthLLsqzUB:  # LL sqz
        LLsqzprices[Cthetaind, :] = SupPriceLLSqz(scDict, currCtheta)
    if currCtheta > CthLHexpLB and currCtheta < CthLHFOCLB:  # LHexpFOC
        LHexpprices[Cthetaind, :] = SupPriceLHexpIR(scDict, currCtheta)
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
LLcol, LLsqzcol, HHcol, HHsqzcol = 'red', 'deeppink', 'indigo', 'mediumorchid'
LHonecols = ['limegreen', 'seagreen', 'darkgreen']
LHtwocols = ['cornflowerblue', 'blue', 'midnightblue']
lnwd = 5

plt.plot(CthetaVec, LLprices[:, 0], linewidth=lnwd, color=LLcol, alpha=al)
plt.plot(CthetaVec, LLsqzprices[:, 0], linewidth=lnwd, color=LLsqzcol, alpha=al)
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
plt.ylim(0, 1.0)
plt.xlim(0, CthetaMax)
plt.xlabel(r'$C_{\theta}$', fontsize=14)
plt.ylabel(r'$w$', fontsize=14, rotation=0, labelpad=14)
plt.show()

#####################
# Equilibrium plots
#####################
# Use b=[0.6,0.9]
b, cSup, supRateLo = 0.6, 0.2, 0.8
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo}

numpts, CthetaMax, cSupMax = 100, 2.0, 0.9
alval = 0.7

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

eqcolors = ['#cf0234', 'deeppink', '#021bf9', '#0d75f8', '#82cafc', '#5ca904', '#0b4008', 'black']
labels = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHsqz', 'HH', 'N']

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
plt.show()

#####################
# Equilibrium plots: ONLY LL/LH/HH
#####################
def LthetaEqMatsForPlot_MainEqOnly(numpts, Ctheta_max, cSup_max, scDict):
    # Generate list of equilibria matrices for plotting
    CthetaVec = np.arange(0, Ctheta_max, (Ctheta_max)/numpts)
    cSupVec = np.arange(0.01, cSup_max, cSup_max/numpts)

    eq_list = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHsqz', 'HH', 'N']
    eqStrat_matList = np.zeros((len(eq_list), cSupVec.shape[0], CthetaVec.shape[0]))
    eqStrat_matList[:] = np.nan

    for currcSupind in range(cSupVec.shape[0]):
        currdict = scDict.copy()
        currdict['cSup'] = cSupVec[currcSupind]
        # Get bounds under current cSup
        CthLLUB, CthHHLB, CthHHsqzLB = CthetaLLUB(currdict), CthetaHHLB(currdict), CthetaHHsqzLB(currdict)
        CthLHFOCLB, CthLHexpLB, CthLHFOCUB = CthetaLHFOCLB(currdict), CthetaLHexpIRLB(currdict), CthetaLHFOCUB(currdict)
        (CthLHsqzUB, _), CthLHsqztwoUB  = CthetaLHsqzUB(currdict), CthetaLHsqztwoUB(currdict)
        CthLLsqzUB = CthetaLLsqzUB(scDict)
        # Place "1" where present for each respective equilibrium
        eqStrat_matList[0, currcSupind, np.where(CthetaVec < CthLLUB)] = 1  # LL
        eqStrat_matList[0, currcSupind, np.where((CthetaVec > CthLLUB) & (CthetaVec <= CthLLsqzUB))] = 1  # LLsqz
        eqStrat_matList[1, currcSupind, np.where((CthetaVec >= CthLHexpLB) & (CthetaVec < CthLHFOCLB))] = 1  # LHexp
        eqStrat_matList[1, currcSupind, np.where((CthetaVec >= CthLHFOCLB) & (CthetaVec <= CthLHFOCUB))] = 1  # LHFOC
        eqStrat_matList[1, currcSupind, np.where((CthetaVec > CthLHFOCUB) &
                                                 ((CthetaVec < CthLHsqzUB) | (CthetaVec < CthLHsqztwoUB)))] = 1  # LHsqz
        eqStrat_matList[2, currcSupind, np.where((CthetaVec >= CthHHsqzLB) & (CthetaVec < CthHHLB))] = 1  # HHsqz
        eqStrat_matList[2, currcSupind, np.where(CthetaVec >= CthHHLB)] = 1  # HH
    # Identify any excessively large cSup
    cSupCond = cSupBar(scDict['b'])
    eqStrat_matList[3, np.where(cSupVec > cSupCond), :] = 1

    return eqStrat_matList
# Use b=[0.6,0.9]
b, cSup, supRateLo = 0.9, 0.2, 0.9
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo}

numpts, CthetaMax, cSupMax = 120, 3.0, 0.35
alval = 0.7

eqStrat_matList = LthetaEqMatsForPlot_MainEqOnly(numpts, CthetaMax, cSupMax, scDict)
# Fill holes
for csupind in range(eqStrat_matList.shape[1]):
    for cthetaind in range(eqStrat_matList.shape[2]):
        if np.nansum(eqStrat_matList[:,csupind,cthetaind]) == 0:
            print(csupind,cthetaind)
            eqStrat_matList[2, csupind, cthetaind] = 1

fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$L=$'+str(supRateLo),
             fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['#cf0234',  '#0d75f8',   '#0b4008', 'black']
labels = ['LL', 'LH',   'HH', 'N']

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
plt.show()

#######################
# UTILITY PLOTS
#######################
# for b=[0.6, 0.9]
b, cSup, supRateLo = 0.5, 0.2, 0.8
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo}

CthLLUB, CthHHLB, CthHHsqzLB = CthetaLLUB(scDict), CthetaHHLB(scDict), CthetaHHsqzLB(scDict)
CthLHFOCLB, CthLHexpLB, CthLHFOCUB = CthetaLHFOCLB(scDict), CthetaLHexpIRLB(scDict), CthetaLHFOCUB(scDict)
(CthLHsqzUB, _), CthLHsqztwoUB, CthLLsqzUB = CthetaLHsqzUB(scDict), CthetaLHsqztwoUB(scDict), CthetaLLsqzUB(scDict)
CthetaMax = 1.4*CthHHLB
CthetaVec = np.arange(0, CthetaMax, CthetaMax/1000)

LLprices = np.empty((CthetaVec.shape[0], 2))
LLprices[:] = np.nan
HHprices, HHsqzprices, LHexpprices, LHFOCprices = LLprices.copy(), LLprices.copy(), LLprices.copy(), LLprices.copy()
LHsqzprices, LHsqztwoprices, LLsqzprices = LLprices.copy(), LLprices.copy(), LLprices.copy()
# Store prices
for Cthetaind in range(CthetaVec.shape[0]):
    currCtheta = CthetaVec[Cthetaind]
    if currCtheta <= CthLLUB:  # LL
        LLprices[Cthetaind, :] = SupPriceLL(scDict, currCtheta)
    if currCtheta > CthLLUB and currCtheta <= CthLLsqzUB:  # LL sqz
        LLsqzprices[Cthetaind, :] = SupPriceLLSqz(scDict, currCtheta)
    if currCtheta > CthLHexpLB and currCtheta < CthLHFOCLB:  # LHexpFOC
        LHexpprices[Cthetaind, :] = SupPriceLHexpIR(scDict, currCtheta)
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
# Combine LHsqz and LHsqztwo
(numi, numj) = LLsqzprices.shape
for i in range(numi):
    if not np.isnan(LHsqztwoprices[i, 0]):  # add to LHsqzprices
        LHsqzprices[i, :] = LHsqztwoprices[i, :]

# Capture retailer/supplier utility under each set of prices
utils = np.empty((7, CthetaVec.shape[0], 3))  # for each eq type, Ctheta point, player
utils[:] = np.nan
priceList = [LLprices, LLsqzprices, LHexpprices, LHFOCprices, LHsqzprices, HHsqzprices, HHprices]
for listind, currpricelist in enumerate(priceList):
    # Assign quality levels depending on current list
    if listind in [0, 1]:  # LL
        currS1qual, currS2qual = scDict['supRateLo'], scDict['supRateLo']
        currS1cSup, currS2cSup = 0, 0
    elif listind in [2, 3, 4]:  # LH
        currS1qual, currS2qual = scDict['supRateLo'], 1
        currS1cSup, currS2cSup = 0, cSup
    elif listind in [5, 6]:  # HH
        currS1qual, currS2qual = 1, 1
        currS1cSup, currS2cSup = cSup, cSup
    for currCthind, currCth in enumerate(CthetaVec):
        w1, w2 = currpricelist[currCthind, :]
        if not np.isnan(w1):  # Continue if we have valid eq prices
            # Retailer utility
            utils[listind, currCthind, 0] = RetUtil(currS1qual, currS2qual, currCth, b, w1, w2)
            # Supplier utilities
            q1, q2 = quantOpt(w1, w2, b), quantOpt(w2, w1, b)
            utils[listind, currCthind, 1] = SupUtil(q1, w1, currS1cSup)
            utils[listind, currCthind, 2] = SupUtil(q2, w2, currS2cSup)

# Supplier utilities
fig = plt.figure()
fig.suptitle('Supplier utility\n'+r'$b=$'+str(b)+', '+r'$c_S=$'+str(cSup)+', '+r'$L=$'+str(supRateLo),
             fontsize=18, fontweight='bold')

al = 0.8
LLcol, LLsqzcol, HHcol, HHsqzcol = 'red', 'deeppink', 'indigo', 'mediumorchid'
LHonecols = ['limegreen', 'seagreen', 'darkgreen']
LHtwocols = ['cornflowerblue', 'blue', 'midnightblue']
lnwd = 5

plt.plot(CthetaVec, utils[0, :, 1], linewidth=lnwd, color=LLcol, alpha=al)  # LL
plt.plot(CthetaVec, utils[1, :, 1], linewidth=lnwd, color=LLsqzcol, alpha=al)  # LLsqz
plt.plot(CthetaVec, utils[2, :, 1], linewidth=lnwd, color=LHonecols[0], alpha=al)  # LHexp
plt.plot(CthetaVec, utils[2, :, 2], linewidth=lnwd, color=LHtwocols[0], alpha=al)  # LHexp
plt.plot(CthetaVec, utils[3, :, 1], linewidth=lnwd, color=LHonecols[1], alpha=al)  # LHFOC
plt.plot(CthetaVec, utils[3, :, 2], linewidth=lnwd, color=LHtwocols[1], alpha=al)  # LHFOC
plt.plot(CthetaVec, utils[4, :, 1], linewidth=lnwd, color=LHonecols[2], alpha=al)  # LHsqz
plt.plot(CthetaVec, utils[4, :, 2], linewidth=lnwd, color=LHtwocols[2], alpha=al)  # LHsqz
plt.plot(CthetaVec, utils[5, :, 1], linewidth=lnwd, color=HHsqzcol, alpha=al)  # HHsqz
plt.plot(CthetaVec, utils[6, :, 1], linewidth=lnwd, color=HHcol, alpha=al)  # HHsqz
# plt.plot(CthetaVec, HHprices[:, 1], linewidth=lnwd, color=HHcol, alpha=al)
utilmax = np.nanmax(utils[:, :, 1:])*1.1
plt.ylim(-0.01, utilmax)
plt.xlim(0, CthetaMax)
plt.xlabel(r'$C_{\theta}$', fontsize=14)
plt.ylabel(r'$U^S$', fontsize=14, rotation=0, labelpad=14)
plt.show()

# Retailer utility
fig = plt.figure()
fig.suptitle('Retailer utility\n'+r'$b=$'+str(b)+', '+r'$c_S=$'+str(cSup)+', '+r'$L=$'+str(supRateLo),
             fontsize=18, fontweight='bold')

al = 0.8
LLcol, LLsqzcol, HHcol, HHsqzcol = 'red', 'deeppink', 'indigo', 'mediumorchid'
LHonecols = ['limegreen', 'seagreen', 'darkgreen']
LHtwocols = ['cornflowerblue', 'blue', 'midnightblue']
lnwd = 5

plt.plot(CthetaVec, utils[0, :, 0], linewidth=lnwd, color=LLcol, alpha=al)  # LL
plt.plot(CthetaVec, utils[1, :, 0], linewidth=lnwd, color=LLsqzcol, alpha=al)  # LLsqz
plt.plot(CthetaVec, utils[2, :, 0], linewidth=lnwd, color=LHonecols[0], alpha=al)  # LHexp
plt.plot(CthetaVec, utils[3, :, 0], linewidth=lnwd, color=LHonecols[1], alpha=al)  # LHFOC
plt.plot(CthetaVec, utils[4, :, 0], linewidth=lnwd, color=LHonecols[2], alpha=al)  # LHsqz
plt.plot(CthetaVec, utils[5, :, 0], linewidth=lnwd, color=HHsqzcol, alpha=al)  # HHsqz
plt.plot(CthetaVec, utils[6, :, 0], linewidth=lnwd, color=HHcol, alpha=al)  # HHsqz
# plt.plot(CthetaVec, HHprices[:, 1], linewidth=lnwd, color=HHcol, alpha=al)
utilmax = np.nanmax(utils[:, :, 0])*1.1
plt.ylim(-0.02, utilmax)
plt.xlim(0, CthetaMax)
plt.xlabel(r'$C_{\theta}$', fontsize=14)
plt.ylabel(r'$U^R$', fontsize=14, rotation=0, labelpad=14)
plt.show()


#############################
# SP friction threshold plot
#############################
# for b=[0.5,0.8]
b, cSup, supRateLo = 0.5, 0.2, 0.9
b1, b2 = 0.5, 0.8
cSupMax, alMax = 0.55, 1.0
scDict1 = {'b': b1, 'cSup': cSup, 'supRateLo': supRateLo}
scDict2 = {'b': b2, 'cSup': cSup, 'supRateLo': supRateLo}
cSupVec = np.arange(0.001, cSupMax, 0.005)
alVec = np.arange(0.001, alMax, 0.005)
yvec1, yvec2 = [], []

cSupBB1 = cSupBarBar(scDict1['b'])
cSupBB2 = cSupBarBar(scDict2['b'])

for currcSup in cSupVec:
    currdict1, currdict2 = scDict1.copy(), scDict2.copy()
    currdict1['cSup'] = currcSup
    currdict2['cSup'] = currcSup
    cSupHHsqzBound1 = GetCritcSupHHsqz(currdict1)
    cSupHHsqzBound2 = GetCritcSupHHsqz(currdict2)
    if currcSup > cSupHHsqzBound1 and currcSup < cSupBB1:
        yvec1.append(SPfrictThreshHHsqz(currdict1))
    elif currcSup < cSupBB1:
        yvec1.append(SPfrictThresh(currdict1))
    else:
        yvec1.append(np.nan)
    if currcSup > cSupHHsqzBound2 and currcSup < cSupBB2:
        yvec2.append(SPfrictThreshHHsqz(currdict2))
    elif currcSup < cSupBB2:
        yvec2.append(SPfrictThresh(currdict2))
    else:
        yvec2.append(np.nan)

# Plot 1
plt.plot(yvec1, cSupVec, '-', linewidth=5, color='indigo')
plt.plot(np.arange(0,1,0.01), np.repeat(cSupHHsqzBound1,100), '-', linewidth=5, color='gray')
plt.plot(np.arange(0,1,0.01), np.repeat(cSupBB1,100), '-', linewidth=5, color='red')
# plt.suptitle(r'$b=$'+str(b)+', '+r'$L=$'+str(supRateLo),
#              fontsize=18, fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0, cSupMax)
ax.set_box_aspect(1)
plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel(r'$c_S$', fontsize=14, rotation=0, labelpad=14)
plt.show()

# Plot 2
plt.plot(yvec2, cSupVec, '-', linewidth=5, color='indigo')
plt.plot(np.arange(0,1,0.01), np.repeat(cSupHHsqzBound2,100), '-', linewidth=2, color='gray')
plt.plot(np.arange(0,1,0.01), np.repeat(cSupBB2,100), '-', linewidth=2, color='red')
# plt.suptitle(r'$b=$'+str(b)+', '+r'$L=$'+str(supRateLo),
#              fontsize=18, fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0, cSupMax)
ax.set_box_aspect(1)
plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel(r'$c_S$', fontsize=14, rotation=0, labelpad=14)
plt.show()

# Make region plots
# Ordered as HH, LH (sqz), HH (sqz), NA (cSup too big)
SPreg_list1 = np.zeros((4, cSupVec.shape[0], alVec.shape[0]))
SPreg_list1[:] = np.nan
SPreg_list2 = np.zeros((4, cSupVec.shape[0], alVec.shape[0]))
SPreg_list2[:] = np.nan
cSupBB1 = cSupBarBar(scDict1['b'])
cSupBB2 = cSupBarBar(scDict2['b'])
for currcSupind, currcSup in enumerate(cSupVec):
    currdict1, currdict2 = scDict1.copy(), scDict2.copy()
    currdict1['cSup'] = currcSup
    currdict2['cSup'] = currcSup
    cDotDeriv1, cDotDeriv2 = cSupDotDeriv(currdict1), cSupDotDeriv(currdict2)
    cSupHHsqzBound1, cSupHHsqzBound2 = GetCritcSupHHsqz(currdict1), GetCritcSupHHsqz(currdict2)
    for curralind, curral in enumerate(alVec):
        # First dict
        if currcSup < cSupBB1 and currcSup <= cSupHHsqzBound1:  # LH (sqz) or HH
            if curral < yvec1[currcSupind]:  # HH
                SPreg_list1[0, currcSupind, curralind] = 1
            else:  # LH (sqz)
                SPreg_list1[1, currcSupind, curralind] = 1
        elif currcSup < cSupBB1 and currcSup > cSupHHsqzBound1:  # LH (sqz) or HH (sqz)
            if curral < yvec1[currcSupind]:  # HH (sqz)
                SPreg_list1[2, currcSupind, curralind] = 1
            else:  # LH (sqz)
                SPreg_list1[1, currcSupind, curralind] = 1
        else:  # NA
            SPreg_list1[3, currcSupind, curralind] = 1
        # Second dict
        if currcSup < cSupBB2 and currcSup <= cSupHHsqzBound2:  # LH (sqz) or HH
            if curral < yvec2[currcSupind]:  # HH
                SPreg_list2[0, currcSupind, curralind] = 1
            else:  # LH (sqz)
                SPreg_list2[1, currcSupind, curralind] = 1
        elif currcSup < cSupBB2 and currcSup <= cSupHHsqzBound1:  # LH (sqz) or HH (sqz)
            if curral < yvec2[currcSupind]:  # HH
                SPreg_list2[2, currcSupind, curralind] = 1
            else:  # LH (sqz)
                SPreg_list2[1, currcSupind, curralind] = 1
        else:  # NA
            SPreg_list2[3, currcSupind, curralind] = 1

# Plot 1
fig = plt.figure()
fig.suptitle(r'$b=$'+str(b1)+', '+r'$L=$'+str(supRateLo), fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)
eqcolors = ['#0b4008', '#82cafc', '#5ca904',  'black']
labels = ['HH', 'LHsqz', 'HHsqz', 'N']
imlist = []
for eqind in range(len(labels)):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[eqind]], name='from_list', N=None)
    if eqcolors[eqind] == 'black':  # No alpha transparency
        im = ax.imshow(SPreg_list1[eqind], vmin=0, vmax=1, aspect='auto',
                            extent=(0, alMax, 0, cSupMax),
                            origin="lower", cmap=mycmap, alpha=1)
    else:
        im = ax.imshow(SPreg_list1[eqind], vmin=0, vmax=1, aspect='auto',
                            extent=(0, alMax, 0, cSupMax),
                            origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)

legwidth = 20
wraplabels = ['\n'.join(textwrap.wrap(labels[i], width=legwidth)) for i in range(len(labels))]
patches = [mpatches.Patch(color=eqcolors[i], edgecolor='black', label=wraplabels[i], alpha=alval) for i in range(len(eqcolors))]
# put those patched as legend-handles into the legend
ax.legend(handles=patches, bbox_to_anchor=(1.3, 1.0), loc='upper right', borderaxespad=0.1, fontsize=8)
ax.set_xbound(0, alMax)
ax.set_ybound(0, cSupMax)
ax.set_box_aspect(1)
plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel(r'$c_S$', fontsize=14, rotation=0, labelpad=14)
plt.show()

# Plot 2
fig = plt.figure()
fig.suptitle(r'$b=$'+str(b2)+', '+r'$L=$'+str(supRateLo), fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)
eqcolors = ['#0b4008', '#82cafc', '#5ca904',  'black']
labels = ['HH', 'LHsqz', 'HHsqz', 'N']
imlist = []
for eqind in range(len(labels)):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[eqind]], name='from_list', N=None)
    if eqcolors[eqind] == 'black':  # No alpha transparency
        im = ax.imshow(SPreg_list2[eqind], vmin=0, vmax=1, aspect='auto',
                            extent=(0, alMax, 0, cSupMax),
                            origin="lower", cmap=mycmap, alpha=1)
    else:
        im = ax.imshow(SPreg_list2[eqind], vmin=0, vmax=1, aspect='auto',
                            extent=(0, alMax, 0, cSupMax),
                            origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)

legwidth = 20
wraplabels = ['\n'.join(textwrap.wrap(labels[i], width=legwidth)) for i in range(len(labels))]
patches = [mpatches.Patch(color=eqcolors[i], edgecolor='black', label=wraplabels[i], alpha=alval) for i in range(len(eqcolors))]
# put those patched as legend-handles into the legend
ax.legend(handles=patches, bbox_to_anchor=(1.3, 1.0), loc='upper right', borderaxespad=0.1, fontsize=8)
ax.set_xbound(0, alMax)
ax.set_ybound(0, cSupMax)
ax.set_box_aspect(1)
plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel(r'$c_S$', fontsize=14, rotation=0, labelpad=14)
plt.show()


#############################
# Plot of various cSup conditions
#############################
bVec = np.arange(0.01,0.99,0.01)
cVec1, cVec2 = cSupBar(bVec), cSupBarBar(bVec)
cVec3 = np.zeros(bVec.shape[0])
for i in range(bVec.shape[0]):
    newdict = scDict1.copy()
    newdict['b'] = bVec[i]
    cVec3[i] = GetCritcSupHHsqz(newdict)
plt.plot(bVec, cVec1, '-', linewidth=5, color='darkgreen')
plt.plot(bVec, cVec2, '--', linewidth=5, color='firebrick')
plt.plot(bVec, cVec3, '-.', linewidth=5, color='deepskyblue')
# plt.suptitle(r'$b=$'+str(b)+', '+r'$L=$'+str(supRateLo),
#              fontsize=18, fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_box_aspect(1)
plt.xlabel(r'$b$', fontsize=14)
plt.ylabel(r'$c_S$', fontsize=14, rotation=0, labelpad=14)
plt.show()

# Plot WRT different SPfrict values