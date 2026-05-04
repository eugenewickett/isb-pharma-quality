# Plot generation for "Strategic Role of Inspections in Pharmaceutical Supply Chains"
# 25-NOV-25

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
import textwrap
from matplotlib.widgets import RadioButtons
# matplotlib.use('qt5agg',force=True)  # pycharm backend doesn't support interactive plots, so we use qt here
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
plt.rcParams["font.family"] = "serif"

def sqroot(val):
    # Returns square root of val after checking that val is positive; returns NaN otherwise
    if val<0:
        retval = np.nan
    else:
        retval = np.sqrt(val)
    return retval
def SPUtil(q1, q2, lambsup1, lambsup2, alph):
    # Social planner's utility
    return q1*(alph+(lambsup1)-1) + q2*(alph+(lambsup2)-1)

def SupUtil(q, w, cSup, lambsup, Y):
    # Returns supplier utility
    return q*(w-cSup) - (1-lambsup)*Y

def quantOpt(w, wOpp, b):
    # Returns optimal order quantities under dual sourcing
    return max(0, (1-b-w+b*wOpp)/(2*(1-(b**2))))

def SocWel(uH, uL, q1, q2, cSup1, cSup2, lambsup1, lambsup2, Ctheta):
    # Social welfare
    return q1*(uH*(lambsup1)+uL*(1-lambsup1)-cSup1) + q2*(uH*(lambsup2)+uL*(1-lambsup2)-cSup2) -\
        Ctheta*(1-lambsup1*lambsup2)

def invPrice(qi, qj, b):
    return 1 - qi - b*qj

def RetOptQuants(retStrat, b, cRet, priceSup_1, priceSup_2):
    # retStrat is 1 of the 7 possible retailer strategies under the quality extension
    if retStrat == 0:  # h12
        numer1, denom1 = ((1-b)*(1-cRet)) - priceSup_1 + b*priceSup_2, 2*(1-(b**2))
        numer2, denom2 = (1 - b) * (1 - cRet) - priceSup_2 + b*priceSup_1, 2 * (1 - (b ** 2))
        q1, q2 = numer1 / denom1, numer2 / denom2
    elif retStrat == 1:  # l12
        numer1, denom1 = (1 - b) - priceSup_1 + b * priceSup_2, 2 * (1 - (b ** 2))
        numer2, denom2 = (1 - b) - priceSup_2 + b * priceSup_1, 2 * (1 - (b ** 2))
        q1, q2 = numer1 / denom1, numer2 / denom2
    elif retStrat == 2:  # h1
        q1, q2 = (1-priceSup_1-cRet)/2, 0
    elif retStrat == 3:  # l1
        q1, q2 = (1-priceSup_1)/2, 0
    elif retStrat == 4:  # h2
        q1, q2 = 0, (1-priceSup_2-cRet)/2
    elif retStrat == 5:  # l1
        q1, q2 = 0, (1-priceSup_2)/2
    else:
        q1, q2 = 0, 0
    return q1, q2

# Function returning retailer utilities under each of 7 possible policies
def RetUtil(X, scDict, RetQualBin, rateSup_1, rateSup_2,  q1, q2):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b = scDict['b']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    # Returns retailer utility
    # Get detection rate
    if RetQualBin == 1:
        rateRet = rateRetHi
        investRet = cRet
    else:
        rateRet = rateRetLo
        investRet = 0
    if q1 > 0 and q2 > 0:
        detectRate = 1 - rateRet * rateSup_1 * rateSup_2
    elif q1 > 0 and q2 == 0:
        detectRate = 1 - rateRet * rateSup_1
    elif q2 > 0 and q1 == 0:
        detectRate = 1 - rateRet * rateSup_2
    else:
        detectRate = 0


    profit1 = q1 * (invPrice(q1, q2, b) - priceSup_1 - investRet)
    profit2 = q2 * (invPrice(q2, q1, b) - priceSup_2 - investRet)
    retval = profit1 + profit2 - (X * inspSensRet * detectRate)
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

def GetCthetaBds(scDict, incr = 1/1000):
    # Return list of bounds WRT Ctheta
    CthLHsqzUB, _ = CthetaLHsqzUB(scDict, incr)
    CthLHsqz2UB = CthetaLHsqztwoUB(scDict, incr)
    CthHHexpLB, CthHHFOCLB = CthetaHHsqzLB(scDict), CthetaHHLB(scDict)
    CthLLsqzUB, CthLLFOCUB = CthetaLLsqzUB(scDict),  CthetaLLUB(scDict)
    CthLHexpLB, CthLHFOCLB, CthLHFOCUB = CthetaLHexpIRLB(scDict, incr), CthetaLHFOCLB(scDict, incr), CthetaLHFOCUB(scDict)
    return [CthLLFOCUB, CthLLsqzUB, CthLHexpLB, CthLHFOCLB, CthLHFOCUB, CthLHsqzUB, CthLHsqz2UB, CthHHexpLB, CthHHFOCLB]

def SocWelEqMatsForPlot(numpts, uL_min, uL_max, cSup_max, scDict, Cthstep = 1/1000, printUpdate=False):
    # Generate list of equilibria matrices for plotting
    # Each point has a social-welfare maximizing equilibria
    b, supRateLo = scDict['b'], scDict['supRateLo']
    # cSup to iterate over
    cSupVec = np.arange(0.01, cSup_max, cSup_max/numpts)
    # uL to iterate over
    uLVec = np.arange(uL_min, uL_max, (uL_max-uL_min)/numpts)

    eq_list = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHsqz', 'HH', 'N']
    eq_matList = np.zeros((len(eq_list), uLVec.shape[0], cSupVec.shape[0]))
    Cth_matList = np.zeros((uLVec.shape[0], cSupVec.shape[0]))
    SW_matList = np.zeros((uLVec.shape[0], cSupVec.shape[0]))
    eq_matList[:], Cth_matList[:], SW_matList[:] = np.nan, np.nan, np.nan

    for currcSupind in range(cSupVec.shape[0]):
        if printUpdate:
            print('cSup: ' + str(cSupVec[currcSupind]))
        currdict = scDict.copy()
        currdict['cSup'], cSup = cSupVec[currcSupind], cSupVec[currcSupind]
        # Get bounds under current cSup
        bdList = GetCthetaBds(currdict)
        # FOC prices don't change WRT Ctheta
        w1LL, w2LL = SupPriceLL(currdict, 0)
        q1LL, q2LL = quantOpt(w1LL, w2LL, b), quantOpt(w2LL, w1LL, b)
        w1LHFOC, w2LHFOC = SupPriceLHFOC(currdict, 0)
        q1LHFOC, q2LHFOC = quantOpt(w1LHFOC, w2LHFOC, b), quantOpt(w2LHFOC, w1LHFOC, b)
        w1HH, w2HH = SupPriceHH(currdict, 0)
        q1HH, q2HH = quantOpt(w1HH, w2HH, b), quantOpt(w2HH, w1HH, b)
        # Iterate across uL
        for curruLind in range(uLVec.shape[0]):
            if printUpdate:
                print('uL: ' + str(uLVec[curruLind]))
            # Get best Ctheta WRT social welfare under each equilibrium
            currBestEq, currBestCth, currBestSW = -1, -1, -1  # Initialize
            # Get prices and welfare under each possible equilibrium and check against current best
            swLLFOC = SocWel(1, uLVec[curruLind], q1LL, q2LL, 0, 0, supRateLo, supRateLo, 0)  # LLFOC
            if swLLFOC > currBestSW:
                currBestEq, currBestCth, currBestSW = 0, 0, swLLFOC
            swLHFOC = SocWel(1, uLVec[curruLind], q1LHFOC, q2LHFOC, 0, cSup, supRateLo, 1, bdList[3])  # LHFOC
            if swLHFOC > currBestSW:
                currBestEq, currBestCth, currBestSW = 3, bdList[3], swLHFOC
            swHHFOC = SocWel(1, uLVec[curruLind], q1HH, q2HH, cSup, cSup, 1, 1, bdList[8])  # HHFOC
            if swHHFOC > currBestSW:
                currBestEq, currBestCth, currBestSW = 6, bdList[8], swHHFOC
            # Need to iterate through Ctheta for other equilibria
            # LLsqz
            for cThind, cTh in enumerate(np.arange(bdList[0], bdList[1], Cthstep)):
                w1LLsqz, w2LLsqz = SupPriceLLSqz(currdict, cTh)
                q1LLsqz, q2LLsqz = quantOpt(w1LLsqz, w2LLsqz, b), quantOpt(w2LLsqz, w1LLsqz, b)
                swLLsqz = SocWel(1, uLVec[curruLind], q1LLsqz, q2LLsqz, 0, 0, supRateLo, supRateLo, cTh)
                if swLLsqz > currBestSW:
                    currBestEq, currBestCth, currBestSW = 1, cTh, swLLsqz
            # LHexp
            for cThInd, cTh in enumerate(np.arange(bdList[2], bdList[3], Cthstep)):
                w1LHexp, w2LHexp = SupPriceLHexpIR(currdict, cTh)
                q1LHexp, q2LHexp = quantOpt(w1LHexp, w2LHexp, b), quantOpt(w2LHexp, w1LHexp, b)
                swLHexp = SocWel(1, uLVec[curruLind], q1LHexp, q2LHexp, 0, cSup, supRateLo, 1, cTh)
                if swLHexp > currBestSW:
                    currBestEq, currBestCth, currBestSW = 2, cTh, swLHexp
            # LHsqz1
            for cThind, cTh in enumerate(np.arange(bdList[4], bdList[5], Cthstep)):
                w1LHsqz, w2LHsqz = SupPriceLHSqz(currdict, cTh)
                q1LHsqz, q2LHsqz = quantOpt(w1LHsqz, w2LHsqz, b), quantOpt(w2LHsqz, w1LHsqz, b)
                swLHsqz = SocWel(1, uLVec[curruLind], q1LHsqz, q2LHsqz, 0, cSup, supRateLo, 1, cTh)
                if swLHsqz > currBestSW:
                    currBestEq, currBestCth, currBestSW = 4, cTh, swLHsqz
            # LHsqz2
            if bdList[6] > bdList[5]:
                for cThind, cTh in enumerate(np.arange(bdList[5], bdList[6], Cthstep)):
                    w1LHsqz, w2LHsqz = SupPriceLHSqzTwo(currdict, cTh)
                    q1LHsqz, q2LHsqz = quantOpt(w1LHsqz, w2LHsqz, b), quantOpt(w2LHsqz, w1LHsqz, b)
                    swLHsqz = SocWel(1, uLVec[curruLind], q1LHsqz, q2LHsqz, 0, cSup, supRateLo, 1, cTh)
                    if swLHsqz > currBestSW:
                        currBestEq, currBestCth, currBestSW = 4, cTh, swLHsqz
            # HHexp
            for cThind, cTh in enumerate(np.arange(bdList[7], bdList[8]-Cthstep, Cthstep)):
                w1HHsqz, w2HHsqz = SupPriceHHsqz(currdict, cTh)
                q1HHsqz, q2HHsqz = quantOpt(w1HHsqz, w2HHsqz, b), quantOpt(w2HHsqz, w1HHsqz, b)
                swHHsqz = SocWel(1, uLVec[curruLind], q1HHsqz, q2HHsqz, cSup, cSup, 1, 1, cTh)
                if swHHsqz > currBestSW:
                    currBestEq, currBestCth, currBestSW = 5, cTh, swHHsqz
            eq_matList[currBestEq, curruLind, currcSupind] = 1
            Cth_matList[curruLind, currcSupind] = currBestCth
            SW_matList[curruLind, currcSupind] = currBestSW
            if printUpdate:
                print('best eq: ' + str(currBestEq)+', Cth: ' + str(currBestCth) + ', SW: ' + str(currBestSW))

    return eq_matList, Cth_matList, SW_matList

##################
# 4/15 NEW PRICING/BOUND FUNCTIONS
##################
def PriceLL(scDict, X, Y):
    # Returns on-path LL prices
    w = max((1 - scDict['b']) / (2 - scDict['b']), 0)
    return w, w

def PriceLLsqz(scDict, X, Y):
    # Returns on-path LLsqz prices
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    w = max(1 - sqroot(2 * (1 + scDict['b']) * X * inspSensRet * (1 - scDict['supRateLo'] ** 2)), 0)
    return w, w

def PriceLH(scDict, X, Y):
    # Returns on-path LH-FOC prices
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    w1 = max((2 - b - (b ** 2) + b * cS) / (4 - b ** 2), 0)
    w2 = max((2 - b - (b ** 2) + 2 * cS) / (4 - b ** 2), 0)
    return w1, w2

def PriceLHsqz(scDict, X, Y):
    # Returns on-path LHsqz prices; accounts for w2<cS
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    # radterm = sqroot((4 * (4 - 3 * (b ** 2)) * (1 - supRateLo) * inspSensRet * X + cS * (2 - cS) - 1) / (1 - (b ** 2)))
    w1 = (b - b*cSup + (b**2)*(3-2*np.sqrt((1+(-2+cSup)*cSup -\
            4*(-4 + 3*b**2)*X*(-1+supRateLo)) / (-1 + b**2)))+2*(-2 + sqroot((1 + (-2 + cSup)* cSup -\
            4*(-4+3*(b**2))*X*(-1 +supRateLo)) / (-1 + b**2)))) / (-4 + 3*(b**2))
    w2 = max(0.5 * (1 + cSup + b * (-1 + w1)), 0)
    if w2 < cS:  # Need adjusted w1
        # radterm2 = sqroot((((1 - cS) ** 2) + 4 * inspSensRet * X * (supRateHi * supRateLo - 1)) / (b ** 2 - 1))
        w1 = 1 + b*(-1 + cSup + b*sqroot((1 + (-2 + cSup)*cSup +4*X*(-1+supRateLo))/(-1 + (b**2)))) -\
         np.sqrt((1 + (-2+cSup)*cSup + 4*X*(-1 + supRateLo))/(-1 + (b**2)))
        w2 = cS
    return w1, w2

def PriceLHhld(scDict, X, Y):
    # Returns on-path LH-hld prices; accounts for w2<cS
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    w1 = (cS * (2*b+cS -2) - 8*(b**2 -1)*inspSensSup*Y*(supRateHi-supRateLo))/(2*b*cS)
    #w2 = max(0.5 * (1 + cSup + b * (-1 + w1)), 0)
    w2 = (3*cS/4) - (1/cS)*(2 * (b**2 -1) * inspSensSup*Y*(supRateHi-supRateLo))
    if w2 < cS:  # Need adjusted w1
        radterm2 = sqroot(2*(b**2)*(1-(b**2)*(1-supRateLo)*inspSensSup*Y))
        w1 = (2*radterm2 - b*(1-b))/(b**2)
        w2 = cS
    return w1, w2

def PriceHH(scDict, X, Y):
    # Returns on-path HH-FOC prices
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    w = max((1 - b + cS) / (2 - b), 0)
    return w, w

def PriceHHhld(scDict, X, Y):
    # Returns on-path HH-hld prices
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    radterm = sqroot((1-b)*(2*(1+b)*((2-b)**2)*Y*inspSensSup*(1-supRateLo) - (b-1)*(cS**2) +(b-2)*cS))
    w = ((b**2) +2*(radterm+1+cS) -b*(3+2*cS)) / ((2-b)**2)
    if w < cS:  # Need adjusted w1
        w = -0.01
    return w, w

def XUBLL(scDict, Y):
    # Returns LL-FOC UB in X
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    retval = 1/(2*inspSensRet*((2 -b)**2)*(1 + b)*(1 - (supRateLo**2)))
    return retval

def YUBLL(scDict, X):
    # Returns LL-FOC UB in X
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    retval = (cS*(b*(cS-4)- 2*cS+4)) / (8*(2-b)*(1-(b**2))*inspSensSup* (supRateHi-supRateLo))
    return retval

def XUBLLsqz(scDict, Y):
    # Returns LL-sqz UB in X for Y=0
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    numerator = (4 * (b + 1) * sqroot((b - 1) * cS * (supRateLo ** 2 - 1) ** 2 * inspSensRet ** 2 * (b + cS -
            2) * (b * cS + b - 2) ** 2 / (b + 1) ** 2) - (supRateLo ** 2 - 1) * inspSensRet * ((b * (b + 4) -
            4) * cS ** 2 + 2 * (3 * b - 2) * (b - 2) * cS + (b - 2) ** 2))
    denominator = 2 * (b - 2) ** 4 * (b + 1) * (supRateLo ** 2 - 1) ** 2 * inspSensRet ** 2
    return numerator / denominator

def YUBLLsqz(scDict, X):
    # Returns LL-sqz UB in Y
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    inner = -(b + 1) * (supRateLo ** 2 - 1) * inspSensRet * X
    sqrt_inner = sqroot(inner)
    numerator = (2 * sqroot(2) * b * cS * sqrt_inner - 2 * (b + 1) * (b - 2) ** 2 * (supRateLo ** 2 - 1) * inspSensRet * X
            + 2 * np.sqrt(2) * b * sqrt_inner - 4 * np.sqrt(2) * sqrt_inner + (cS - 1) ** 2)
    denominator = 8 * (b ** 2 - 1) * inspSensSup * (supRateHi - supRateLo)
    return numerator / denominator

def XUBLH(scDict, Y):
    # Returns LH-FOC UB in X
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    numer = (8 - 2 * b ** 3 * (1 - cS) - 3 * b ** 2 * (2 - cS * (2 - cS)) - 4 * cS * (2 - cS))
    denom = 4 * (4 - b ** 2) ** 2 * (1 - b ** 2) * (1 - supRateLo) * inspSensRet
    return numer / denom

def YUBLH(scDict, X):
    # Returns LH-FOC UB in Y
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    numer = cS * (b * (b + 4) * cS - 4 * b * (b + 1) - 4 * cS + 8)
    denom = 8 * (b ** 4 - 5 * b ** 2 + 4) * inspSensSup * (supRateHi - supRateLo)
    return numer / denom

def XLBLHretIR(scDict, Y):
    # Returns LH-FOC LB in X where the Y-IR LB needs to be used for X larger than this X LB
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    numer = (b * (b * (4 - 3 * b ** 2) * cS ** 2 + 4 * (b - 1) * (b + 2) ** 2 * cS - 8 * b * (b + 3)) + 32)
    denom = 16 * (b ** 2 - 4) ** 2 * (b ** 2 - 1) * (supRateLo ** 2 - 1) * inspSensRet
    return numer / denom

def YLBLHIC(scDict, X):
    # Returns LH-FOC LB in Y where the Y-IC LB is used; for X smaller than XLBLHretIR
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    numer = cS * (b * (b * (3 * cS - 4) - 4) - 4 * cS + 8)
    denom = 8 * (b ** 4 - 5 * b ** 2 + 4) * inspSensSup * (supRateHi - supRateLo)
    return numer / denom

def YLBLHIR(scDict, X):
    # Returns LH-FOC LB in Y where the Y-IR LB is used; for X larger than XLBLHretIR
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    A3 = sqroot((4 * (b ** 2 - 4) ** 2 * (supRateLo ** 2 - 1) * inspSensRet * X + (b * (cS - 1) - 2) ** 2) / ((b ** 2 - 4) ** 2 * (b ** 2 - 1)))
    numer = ((b ** 2 - 1) * A3 * ((b ** 2 - 1) * A3 + b ** 2 * cS / (4 - b ** 2) + 2 / (b - 2) + 2)
            + (b ** 2 * (-cS) + b ** 2 + b + 2 * cS - 2) ** 2 / (b ** 2 - 4) ** 2)
    denom = 2 * (b ** 2 - 1) * inspSensSup * (supRateHi - supRateLo)
    return numer / denom

def XLBLHAtY0(scDict,step=0.001):
    # Returns LH-FOC LB in X at Y=0
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    start = XUBLH(scDict, 0)
    # Check if start is even valid at 0
    if YLBLHIR(scDict, start) > 0:
        retval = start
    else:
        stop = False
        newLB = start - step
        while not stop:
            if YLBLHIR(scDict, newLB) > 0:
                stop = True
            else:
                newLB = newLB - step
        retval = newLB
    return retval

def XUBLHsqzAtCost(scDict, Y):
    # X UB on standard LHsqz prices; LHsqzAtCost needs to be used above this UB
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    numerator = (1 - cS) ** 2
    denominator = 4 * b ** 2 * inspSensRet * (1 - supRateHi * supRateLo)
    return numerator / denominator

def YUBLHsqz(scDict, X):
    # Returns LH-FOC LB in Y where the Y-IR LB is used; for X larger than XLBLHretIR
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    A1 = sqroot((b**2 - 1) * (cS - 1)**2 - 4 * (3 * b**4 - 7 * b**2 + 4) * inspSensRet * X * (supRateHi * supRateLo - 1))
    numerator = ( (b - 1) * b * (cS - 1) * (  (b * (b * ((b - 5) * b - 4) + 16) + 16) * cS
            - b ** 2 * (b * (b + 7) + 8) ) + b ** 4 * ( (b ** 2 - 1) * (cS - 1) ** 2
                    - 4 * (3 * b ** 4 - 7 * b ** 2 + 4) * inspSensRet * X * (supRateHi * supRateLo - 1) )
            + 2 * ((b - 4) * b * (b + 1) + 4) * b ** 2 * cS * A1
            - 2 * (b - 2) * (b - 1) * (b + 2) ** 3 * A1 + 32 * (3 * b ** 6 - 13 * b ** 4 + 18 * b ** 2 - 8) * inspSensRet * X * (supRateHi * supRateLo - 1) )
    denominator = 8 * (4 - 3 * b ** 2) ** 2 * (b ** 2 - 1) * inspSensSup * (supRateHi - supRateLo)

    return numerator / denominator

def YUBLHsqzAtCost(scDict, X):
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    A2 = sqroot(((cS - 1)**2 + 4 * inspSensRet * X * (supRateHi * supRateLo - 1)) / (b**2 - 1))
    numerator = ( 4 * b ** 2 * (cS - 1) * A2  + b * (cS * (4 * A2 + 5 * cS - 10) + 16 * inspSensRet * X * (supRateHi * supRateLo - 1) + 5)
            + 4 * A2  + 3 * (cS - 2) * cS  + 16 * inspSensRet * X * (supRateHi * supRateLo - 1)  + 3)
    denominator = 8 * (b + 1) * inspSensSup * (supRateHi - supRateLo)
    return numerator / denominator

def XUBLHsqzAtY0(scDict,step=0.001):
    # Returns LH-sqz LB in X at Y=0
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    XUBAtCost = XUBLHsqzAtCost(scDict, 0)
    # Check if Y UB is non-negative at this XUBAtCost
    if YUBLHsqz(scDict, XUBAtCost) < 0:
        stop = False
        newUB = XUBAtCost - step
        while not stop:
            if YUBLHsqz(scDict, newUB) > 0:
                stop = True
            else:
                newUB = newUB - step
        retval = newUB
    else:
        stop = False
        newUB = XUBAtCost + step
        while not stop:
            if YUBLHsqzAtCost(scDict, newUB) < 0:
                stop = True
            else:
                newUB = newUB + step
        retval = newUB
    return retval

def YLBLHhldLLFOC(scDict, X):
    # Returns LB of LHhld when comparing on-path equilibrium with LLFOC
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    diff = supRateHi - supRateLo
    sqrt_inner = sqroot(
        (b ** 2 - 1) ** 2 * inspSensSup ** 2 * ((b - 1) ** 2 * b ** 3 - 2 * (b - 2) ** 2 * (b ** 2 + b - 2) * cS +
        (b - 2) ** 2 * b * cS ** 2) * diff ** 2 / ((b - 2) ** 2 * b * cS ** 2))
    numerator = cS * ( 2 * (b ** 2 - 1) ** 2 * cS * inspSensSup * diff + b ** 2 * cS * sqrt_inner
            + (b + 1) * ((b - 2) * b - 4) * (b - 1) ** 2 * inspSensSup * diff)
    denominator = 8 * (b ** 2 - 2) * (b ** 2 - 1) ** 2 * inspSensSup ** 2 * diff ** 2
    return numerator / denominator

def YLBHHIC(scDict, X):
    # Returns Y LB for HH when retailer IR is not considered
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    numerator = cS * (4 - 2 * cS - b * (4 - 3 * cS))
    denominator = 8 * (2 - b) * (1 - b ** 2) * (1 - supRateLo) * inspSensSup
    return numerator / denominator

def YLBHHhld(scDict, X):
    # Returns Y LB for HH when retailer IR is not considered
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    numer = cS* (b*(cS-1)-cS+2)
    denom = 2*((b-2)**2)*(b+1) *(supRateHi-supRateLo) *inspSensSup
    return numer / denom

def XLBHHIR(scDict):
    # Returns X juncture for HH when retailer IR must be considered
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    numerator = (b ** 2 * cS * (4 - 3 * cS) + 4 * b * (cS - 2) + 4 * (cS - 2) * cS + 8)
    denominator = 16 * (b - 2) ** 2 * (b ** 2 - 1) * inspSensRet * (supRateHi * supRateLo - 1)
    return numerator / denominator

def YLBHHIR(scDict, X):
    # Returns Y LB for HH when retailer IR is considered
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    cS_bar = 1 - cS
    A1 = sqroot( (4 * (2 - b) ** 2 * X * inspSensRet * (1 - supRateHi * supRateLo) + (2 - cS) * cS - 1)
        / ((2 - b) ** 2 * (1 - b ** 2)))
    numerator = ( b ** 2 * (12 * X * inspSensRet * (1 - supRateHi * supRateLo) - (4 - cS) * A1)
            + 2 * b * cS_bar * (cS_bar - A1) + 4 * (A1 - 4 * X * inspSensRet * (1 - supRateHi * supRateLo))
            + b ** 3 * ((2 - cS) * A1 - 4 * X * inspSensRet * (1 - supRateHi * supRateLo)))
    denominator = 2 * (2 - b) ** 2 * (1 + b) * (supRateHi - supRateLo) * inspSensSup
    return numerator / denominator

def XLBHHAtY0(scDict,step=0.001):
    # Returns HH-FOC LB in X at Y=0
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    start = XUBLH(scDict, 0)
    # Increase until YLBHHIR falls below 0
    if YLBHHIR(scDict, start) < 0:
        retval = start
    else:
        stop = False
        newLB = start + step
        while not stop:
            if YLBHHIR(scDict, newLB) < 0:
                stop = True
            else:
                newLB = newLB + step
        retval = newLB
    return retval

def YLBLHhldAtX0(scDict,step=0.001):
    # Returns LH-hld LB in Y at X=0
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet, inspSensSup = scDict['inspSensRet'], scDict['inspSensSup']
    start = YUBLL(scDict, 0)
    # Decrease until better off-path utility for Supplier 1
    w1LL, w2LL = PriceLL(scDict, 0, 0)
    currw1, currw2 = PriceLHhld(scDict, 0, start)
    currUtil1 = SupUtil(quantOpt(currw1, currw2, b), currw1, 0, supRateLo, Y)
    LLUtil = SupUtil(quantOpt(w1LL, w2LL, b), w1LL, 0, supRateLo, Y)
    if LLUtil > currUtil1:
        retval = start
    else:
        stop = False
        newLB = start - step
        currw1, currw2 = PriceLHhld(scDict, 0, newLB)
        currUtil1 = SupUtil(quantOpt(currw1, currw2, b), currw1, 0, supRateLo, Y)
        while not stop:
            if LLUtil > currUtil1:
                stop = True
            else:
                newLB = newLB - step
                currw1, currw2 = PriceLHhld(scDict, 0, newLB)
                currUtil1 = SupUtil(quantOpt(currw1, currw2, b), currw1, 0, supRateLo, Y)
        retval = newLB
    return retval

def XLBLHJunc(scDict, step=0.001):
    # Returns juncture where YLHhldLB hits YLHFOCIRLB
    currX, found = XLBLHretIR(scDict, 0), False
    targ = YLBLHhldLLFOC(scDict, 0)
    while not found:
        currYLHRetIR = YLBLHIR(scDict, currX)
        if currYLHRetIR < targ:
            found = True
        else:
            currX = currX + step
    return currX

def XLBHHJunc(scDict, step=0.001):
    # Returns juncture where YHHhldLB hits YHHFOCIRLB
    currX, found = XLBHHIR(scDict), False
    targ = YLBHHhld(scDict, 0)
    while not found:
        currYHHRetIR = YLBHHIR(scDict, currX)
        if currYHHRetIR < targ:
            found = True
        else:
            currX = currX + step
    return currX

def XHHDualLB(scDict):
    # Returns the X bound s.t. the wHHDual off-path price is retIR valid
    # wHHDual is a floor on the off-path HH price; thus, X cannot be an HHLB for X larger than this X bound, as that
    #   would imply the use of a lower off-path price than this floor
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    inspSensSup, inspSensRet = scDict['inspSensSup'], scDict['inspSensRet']
    lb = (1 - cS)**2 / (4*(2 - b)**2 * b**2 * (1 - supRateHi*supRateLo) * inspSensRet)
    return lb

def cSHHICbound(b):
    return (2*(-1 + b))/(-2 + (b**2))

def YHHICLB(scDict):
    # Returns the Y bound s.t. supIC off-path move is real-valued for Cbeta below this bound
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    inspSensSup, inspSensRet = scDict['inspSensSup'], scDict['inspSensRet']
    lb = (cS * (4 - 2*cS + b * (-4 + 3*cS))) / (8 * (-2 + b) * (-1 + b) * (1 + b) * (supRateHi -
            supRateLo) * inspSensSup)
    return lb

def XHHIC(scDict):
    # Returns the X bound s.t. supIC off-path move is no longer retIR valid for X above this bound
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    inspSensSup, inspSensRet = scDict['inspSensSup'], scDict['inspSensRet']
    lb = (8 + 4*b*(-2 + cS) + b**2*(4 - 3*cS)*cS + 4*(-2 + cS)*cS) / (16 * (-2 + b)**2 * (-1 + b**2) * (-1 +
            supRateHi * supRateLo) * inspSensRet)
    return lb

def YHHIRLB(scDict, X):
    # Returns the Y bound s.t. supICoff=IRoff at given X
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    inspSensSup, inspSensRet = scDict['inspSensSup'], scDict['inspSensRet']
    sqrt_term = sqroot(((-1 + cS) ** 2 + 4 * (-2 + b) ** 2 * X * (-1 + supRateHi * supRateLo) * inspSensRet) / (
                (-2 + b) ** 2 * (-1 + b ** 2)))
    lb = (2*b*(-1 + cS)*(-1 + cS + sqrt_term) + 4*(4*X*(-1 + supRateHi*supRateLo)*inspSensRet + sqrt_term) + b**2*(-12*X*(-1
        + supRateHi*supRateLo)*inspSensRet + (-4 + cS)*sqrt_term) + b**3*(4*X*(-1 + supRateHi*supRateLo)*inspSensRet -
        (-2 + cS)*sqrt_term)) / (2*(-2 + b)**2*(1 + b)*(supRateHi - supRateLo)*inspSensSup)
    return lb

def GetYHHLB(scDict, Xmax, numpts=1000):
    # Returns a vector of Y points signifying the HHFOC lower bound for the X vector from 0 to Xmax
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    Xvec = np.arange(0, Xmax, Xmax/numpts)
    XdualBD = XHHDualLB(scDict)
    Yvec = []
    # First check if wHHIC is valid WRT wHHDual at the YHHIC bound
    if cS > cSHHICbound(b):
        # Use dual bound
        XLB = XHHDualLB(scDict)
        for currX in Xvec:
            if currX <= XLB:
                Yvec.append(XLB)
            else:
                Yvec.append(0)
    else:
        # Use IC bound
        YICLB = YHHICLB(scDict)
        XICUB = XHHIC(scDict)
        for currX in Xvec:
            if currX <= XICUB:
                Yvec.append(YICLB)
            elif currX <= XdualBD:
                Yvec.append(YHHIRLB(scDict, currX))
            else:
                Yvec.append(-1)
    return Yvec

#####################
# Retailer dual-sourcing validity
#####################
b, cSup, supRateLo, supRateHi, inspSensRet = 0.7, 0.2, 0.8, 0.95, 0.95
X = 0.35
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo, 'supRateHi': supRateLo, 'inspSensRet': inspSensRet}
# This plot shows valid quality decisions that keep the retailer on the market for different wholesale prices
numpts = 2000 # Resolution of wholesale prices
def RetThreshBin(w1, w2, rateSup1, rateSup2, X, scDict):  # Indicates if retailer will stay on the market
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    inspSensRet = scDict['inspSensRet']
    threshold = (2-2*b*(1-w1)*(1-w2) - w1*(2-w1) - w2*(2-w2))/(4*inspSensRet*(1-(b**2))*(1-rateSup1*rateSup2))
    if X > threshold:
        retval = 0
    else:
        retval = 1
    return retval

plotMat = np.empty((numpts, numpts, 4))  # LL, LH, HH, N
plotMat[:] = 0
for w1ind, w1 in enumerate(np.arange(0.01,0.99,(0.99-0.01)/numpts)):
    for w2ind, w2 in enumerate(np.arange(0.01, 0.99, (0.99 - 0.01) / numpts)):
        LLval = RetThreshBin(w1, w2, supRateLo, supRateLo, X, scDict)  # LL
        LHval = RetThreshBin(w1, w2, supRateLo, supRateHi, X, scDict)  # LH
        HHval = RetThreshBin(w1, w2, supRateHi, supRateHi, X, scDict)  # HH
        if LLval + LHval + HHval == 0:
            plotMat[w1ind, w2ind, 3] = 1
        plotMat[w1ind, w2ind, 0] = LLval
        if LLval == 0:
            plotMat[w1ind, w2ind, 1] = LHval
            if LHval == 0:
                plotMat[w1ind, w2ind, 2] = HHval


alval=0.5
fig = plt.figure()
# ax.set_title(rf"Equilibrium regions ($b=0.8,\ c_S={cS}$)", fontsize=12, pad=16)
# fig.suptitle(r'$b=$'+str(b)+', '+r'$L=$'+str(supRateLo),fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['maroon', 'indigo', 'darkblue', 'dimgray']
labels = ['LL', 'LH', 'HH', 'N']

imlist = []
for eqind in range(len(labels)):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[eqind]], name='from_list', N=None)
    # if eqcolors[eqind] == 'black':  # No alpha transparency
    #     im = ax.imshow(eqStrat_matList[eqind], vmin=0, vmax=1, aspect='auto',
    #                         extent=(0, CthetaMax, 0, cSupMax),
    #                         origin="lower", cmap=mycmap, alpha=1)
    # else:
    im = ax.imshow(plotMat[:,:,eqind], vmin=0, vmax=1, aspect='auto',
                            extent=(0, 1, 0, 1),
                            origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)
plt.ylim(0, 1.0)
plt.xlim(0, 1.0)
plt.text(0.8, 0.8, 'N', color='dimgray', fontsize=18)
plt.text(0.55, 0.55, 'HH', color='dimgray', fontsize=18)
plt.text(0.4, 0.4, 'LH', color='dimgray', fontsize=18)
plt.text(0.25, 0.25, 'LL', color='dimgray', fontsize=18)
plt.xlabel(r'$w_1$', fontsize=11)
plt.ylabel(r'$w_2$', fontsize=11, rotation=0, labelpad=14)
plt.savefig('retailerQualityThresholds.png', dpi=300, bbox_inches='tight')
plt.show()

#####################
# Wholesale price plots WRT X,Y
#####################
b, cSup, supRateLo, supRateHi, inspSensRet, inspSensSup = 0.8, 0.05, 0.8, 1.0, 1.0, 1.0
X, Y = 0.0, 0.0
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo, 'supRateHi': supRateHi,
          'inspSensRet': inspSensRet, 'inspSensSup': inspSensSup}

XLLUB, XLLsqzUB, XLHUB, XLHLB = XUBLL(scDict, Y), XUBLLsqz(scDict, Y), XUBLH(scDict, Y), XLBLHAtY0(scDict)
XLHsqzUB, XHHLB = XUBLHsqzAtY0(scDict), XLBHHAtY0(scDict)
# CthLHFOCLB, CthLHexpLB, CthLHFOCUB = CthetaLHFOCLB(scDict), CthetaLHexpIRLB(scDict), CthetaLHFOCUB(scDict)
# (CthLHsqzUB, _), CthLHsqztwoUB, CthLLsqzUB = CthetaLHsqzUB(scDict), CthetaLHsqztwoUB(scDict), CthetaLLsqzUB(scDict)
Xmax = 1.3*XHHLB
Xvec = np.arange(0, Xmax, 0.001)
LLprices = np.empty((Xvec.shape[0], 2))
LLprices[:] = np.nan
HHprices, LHhldprices, LHprices, HHhldprices = LLprices.copy(), LLprices.copy(), LLprices.copy(), LLprices.copy()
LHsqzprices, LLsqzprices = LLprices.copy(), LLprices.copy()
# Store prices
for Xind in range(Xvec.shape[0]):
    currX = Xvec[Xind]
    if currX <= XLLUB:  # LL
        LLprices[Xind, :] = PriceLL(scDict, X, Y)
    if currX > XLLUB and currX <= XLLsqzUB:  # LL sqz
        LLsqzprices[Xind, :] = PriceLLsqz(scDict, currX, Y)
    # if currCtheta > CthLHexpLB and currCtheta < CthLHFOCLB:  # LHhld
    #     LHexpprices[Cthetaind, :] = SupPriceLHexpIR(scDict, currCtheta)
    if currX >= XLHLB and currX <= XLHUB:  # LH
        LHprices[Xind, :] = PriceLH(scDict, currX, Y)
    if currX > XLHUB and currX <= XLHsqzUB:  # LHsqz
        LHsqzprices[Xind, :] = PriceLHsqz(scDict, currX, Y)
    # if currCtheta >= CthHHsqzLB and currCtheta < CthHHLB:  # HHhld
    #     HHsqzprices[Cthetaind, :] = SupPriceHHsqz(scDict, currCtheta)
    if currX >= XHHLB:  # HH
        HHprices[Xind, :] = PriceHH(scDict, currX, Y)

fig = plt.figure()
al = 0.9
LLcol, LLsqzcol, HHcol, HHhldcol = 'red', 'deeppink', 'blue', 'cornflowerblue'
LHcols = ['indigo', 'mediumorchid', 'mediumorchid']  # LHFOC, LHsqz, LHhld
lnwd, textgap = 5, 0.015

plt.plot(Xvec, LLprices[:, 0], linewidth=lnwd, color=LLcol, alpha=al)
plt.plot(Xvec, LLsqzprices[:, 0], linewidth=lnwd, color=LLsqzcol, alpha=al)
# plt.plot(Xvec, LHhldprices[:, 0], linewidth=lnwd, color=LHcols[2], alpha=al)
# plt.plot(Xvec, LHhldprices[:, 1], type='--', linewidth=lnwd, color=LHcols[0], alpha=al)
plt.plot(Xvec, LHprices[:, 0], linewidth=lnwd, color=LHcols[0], alpha=al)
plt.plot(Xvec, LHprices[:, 1], '--', linewidth=lnwd, color=LHcols[0], alpha=al)
plt.plot(Xvec, LHsqzprices[:, 0], linewidth=lnwd, color=LHcols[1], alpha=al)
plt.plot(Xvec, LHsqzprices[:, 1], '--', linewidth=lnwd, color=LHcols[1], alpha=al)
# plt.plot(Xvec, HHhldprices[:, 0], linewidth=lnwd, color=HHsqzcol, alpha=al)
# plt.plot(Xvec, HHhldprices[:, 1], linewidth=lnwd, color=HHsqzcol, alpha=al)
plt.plot(Xvec, HHprices[:, 0], linewidth=lnwd, color=HHcol, alpha=al)
plt.ylim(0, 0.4)
plt.xlim(0, Xmax)
plt.xlabel(r'$X$', fontsize=11)
plt.ylabel(r'$w$', fontsize=11, rotation=0, labelpad=14)
plt.text(0.09, LLprices[0, 0]+textgap, r'$LL$ (FOC)', color=LLcol, fontsize=14)
plt.text(0.4-textgap, LLsqzprices[660, 0], r'$LL$ (sqz)', color=LLsqzcol, fontsize=14)
plt.text(0.78-textgap, LHsqzprices[1050, 0], r'$LH$ (sqz)', color=LHcols[1], fontsize=14)
plt.text(0.63, LHprices[700, 1]+textgap, r'$LH$ (FOC)', color=LHcols[0], fontsize=14)
plt.text(1.07, HHprices[-1, 0]+textgap, r'$HH$ (FOC)', color=HHcol, fontsize=14)
plt.text(XLHsqzUB+textgap, LHsqzprices[1100, 0]-textgap, r'Supplier 1', color='black', fontsize=8, fontstyle='italic')
plt.text(XLHsqzUB+textgap, LHsqzprices[1100, 1]-textgap, r'Supplier 2', color='black', fontsize=8, fontstyle='italic')
plt.savefig('priceWRTX.png', dpi=300, bbox_inches='tight')
plt.show()

# Repeat WRT Y
YLLUB, YLHUB, YLHLB = YUBLL(scDict, X), YUBLH(scDict, X), YLBLHIC(scDict, X)
YLHhldLB, YHHLB, YHHhldLB = YLBLHhldAtX0(scDict), YLBHHIC(scDict, X), YLBHHhld(scDict, X)

Ymax = 1.3*YHHLB
Yvec = np.arange(0, Ymax, 0.001)
LLprices = np.empty((Yvec.shape[0], 2))
LLprices[:] = np.nan
HHprices, LHhldprices, LHprices, HHhldprices = LLprices.copy(), LLprices.copy(), LLprices.copy(), LLprices.copy()
# LHsqzprices, LLsqzprices = LLprices.copy(), LLprices.copy()
# Store prices
for Yind in range(Yvec.shape[0]):
    currY = Yvec[Yind]
    if currY <= YLLUB:  # LL
        LLprices[Yind, :] = PriceLL(scDict, X, currY)
    if currY > YLHhldLB and currY < YLHLB:  # LHhld
        LHhldprices[Yind, :] = PriceLHhld(scDict, X, currY)
    if currY >= YLHLB and currY <= YLHUB:  # LH
        LHprices[Yind, :] = PriceLH(scDict, X, currY)
    if currY >= YHHhldLB and currY < YHHLB:  # HHhld
        HHhldprices[Yind, :] = PriceHHhld(scDict, X, currY)
    if currY >= YHHLB:  # HH
        HHprices[Yind, :] = PriceHH(scDict, X, currY)

fig = plt.figure()
al = 0.9
LLcol, LLsqzcol, HHcol, HHhldcol = 'red', 'deeppink', 'blue', 'cornflowerblue'
LHcols = ['indigo', 'mediumorchid', 'mediumorchid']  # LHFOC, LHsqz, LHhld
lnwd, textgap = 5, 0.0015

plt.plot(Yvec, LLprices[:, 0], linewidth=lnwd, color=LLcol, alpha=al)
plt.plot(Yvec, LHhldprices[:, 0], linewidth=lnwd, color=LHcols[2], alpha=al)
plt.plot(Yvec, LHhldprices[:, 1], '--', linewidth=lnwd, color=LHcols[2], alpha=al)
plt.plot(Yvec, LHprices[:, 0], linewidth=lnwd, color=LHcols[0], alpha=al)
plt.plot(Yvec, LHprices[:, 1], '--', linewidth=lnwd, color=LHcols[0], alpha=al)
plt.plot(Yvec, HHhldprices[:, 0], linewidth=lnwd, color=HHhldcol, alpha=al)
plt.plot(Yvec, HHhldprices[:, 1], linewidth=lnwd, color=HHhldcol, alpha=al)
plt.plot(Yvec, HHprices[:, 0], linewidth=lnwd, color=HHcol, alpha=al)
plt.ylim(0, 0.4)
plt.xlim(0, Ymax)
plt.xlabel(r'$Y$', fontsize=11)
plt.ylabel(r'$w$', fontsize=11, rotation=0, labelpad=14)
plt.text(0.002, 0.18, r'$LL$ (FOC)', color=LLcol, fontsize=14)
plt.text(0.035, 0.295, r'$LH$ (hld)', color=LHcols[2], fontsize=14)
plt.annotate('', xy=(0.05, 0.20), xytext=(0.042, 0.28), arrowprops=dict(arrowstyle="->",
                                        color = LHcols[2]))
plt.text(0.045, 0.355, r'$LH$ (FOC)', color=LHcols[0], fontsize=14)
plt.annotate('', xy=(0.056, 0.21), xytext=(0.052, 0.35), arrowprops=dict(arrowstyle="->",
                                        color = LHcols[0]))
plt.text(0.061, 0.22, r'$HH$ (FOC)', color=HHcol, fontsize=14)
plt.annotate('', xy=(0.05, 0.20), xytext=(0.042, 0.28), arrowprops=dict(arrowstyle="->",
                                        color = LHcols[2]))
plt.text(0.06, 0.09, r'$HH$ (hld)', color=HHhldcol, fontsize=14)
plt.annotate('', xy=(0.059, 0.19), xytext=(0.066, 0.108), arrowprops=dict(arrowstyle="->",
                                        color = HHhldcol))
plt.text(0.039, 0.13, r'Supplier 1', color='black', fontsize=8, fontstyle='italic')
plt.text(0.039, 0.18, r'Supplier 2', color='black', fontsize=8, fontstyle='italic')
plt.savefig('priceWRTY.png', dpi=300, bbox_inches='tight')
plt.show()

#####################
# Equilibrium plots
#####################
b, cSup, supRateLo, supRateHi, inspSensRet, inspSensSup = 0.8, 0.1, 0.8, 1.0, 1.0, 1.0
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo, 'supRateHi': supRateHi,
          'inspSensRet': inspSensRet, 'inspSensSup': inspSensSup}
Xmax, Ymax, step, Kpen = 1.4, 0.15, 0.001, 3
Xvec = np.arange(0, Xmax, step)
YvecLL, YvecLHlo, YvecLHhi, YvecHH = [], [], [], []
# Define breakpoints and Y bounds that are not a function of X
XLLUB, XLHLBJunc, XLHUB, XHHLBJunc = XUBLL(scDict, 0), XLBLHJunc(scDict), XUBLH(scDict, 0), XLBHHJunc(scDict)
YLLUB, YLHLB, YLHUB, YHHLB = YUBLL(scDict, 0), YLBLHhldLLFOC(scDict, 0), YUBLH(scDict, 0), YLBHHhld(scDict, 0)

for Xind in range(Xvec.shape[0]):
    currX = Xvec[Xind]
    # LL line
    if currX <= XLLUB:
        YvecLL.append(YLLUB)
    elif currX > XLLUB:
        YvecLL.append(YUBLLsqz(scDict, currX))
    # LHlo line
    if currX < XLHLBJunc:
        YvecLHlo.append(YLHLB)
    elif currX >= XLHLBJunc and currX < XLHUB:
        YvecLHlo.append(YLBLHIR(scDict, currX))
    else:
        YvecLHlo.append(-1)
    # LHhi line
    if currX < XLHUB:
        YvecLHhi.append(YLHUB)
    else:
        YvecLHhi.append(YUBLHsqz(scDict, currX))
    # HH line
    if currX < XHHLBJunc:
        YvecHH.append(YHHLB)
    else:
        YvecHH.append(YLBHHIR(scDict, currX))

# Adjust to switch to inspection probabilities
# Xvec = Xvec / Kpen
# YvecLL = [YvecLL[i]/Kpen for i in range(len(YvecLL))]
# YvecLHlo = [YvecLHlo[i]/Kpen for i in range(len(YvecLHlo))]
# YvecLHhi = [YvecLHhi[i]/Kpen for i in range(len(YvecLHhi))]
# YvecHH = [YvecHH[i]/Kpen for i in range(len(YvecHH))]
Kpen =1

alval, lnwd = 0.6, 3
LLcol, LHcol, HHcol = 'red', 'purple', 'mediumblue'

labels = ['LL feasible', 'LH feasible', 'HH feasible', 'LL'+r'$\cap$'+'LH', 'LH'+r'$\cap$'+'HH']
colList = [LLcol, LHcol, HHcol]
plt.rcParams['hatch.linewidth'] = 2.3

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(Xvec, YvecLL, linewidth=lnwd, color=LLcol, alpha=alval)
plt.plot(Xvec, YvecLHlo, linewidth=lnwd, color=LHcol, alpha=alval)
plt.plot(Xvec, YvecLHhi, linewidth=lnwd, color=LHcol, alpha=alval)
plt.plot(Xvec, YvecHH, linewidth=lnwd, color=HHcol, alpha=alval)
patches = [mpatches.Patch(color=colList[i], label=labels[i], alpha=alval*0.5) for i in range(len(colList))]
patches.append(mpatches.Patch(facecolor=LLcol, edgecolor=LHcol,hatch='////',alpha=alval*0.4,label=labels[3]))
patches.append(mpatches.Patch(facecolor=HHcol, edgecolor=LHcol,hatch='////',alpha=alval*0.4,label=labels[4]))
ax.legend(handles=patches, loc='upper right', borderaxespad=0.4, fontsize=8)
plt.fill_between(Xvec, YvecLHlo, YvecLL, hatch='////', facecolor=LLcol, edgecolor=LHcol, alpha=alval*0.2)
plt.fill_between(Xvec, YvecLL, np.repeat(-1, len(YvecLL)), facecolor=LLcol, alpha=alval*0.3)
plt.fill_between(Xvec, YvecLL, YvecLHhi, facecolor=LHcol, alpha=alval*0.3)
plt.fill_between(Xvec, YvecLHhi, np.repeat(1, len(YvecLHhi)), facecolor=HHcol, alpha=alval*0.3)
plt.fill_between(Xvec, YvecLHhi, YvecHH, hatch='////', facecolor=HHcol, edgecolor=LHcol, alpha=alval*0.2)
plt.text((Xmax*0.15)/Kpen, Ymax*0.2/Kpen, 'LL', color=LLcol, fontsize=15, fontweight='bold')
plt.text(0.4/Kpen, (Ymax*0.87)/Kpen, 'HH', color=HHcol, fontsize=15, fontweight='bold')
plt.text(Xmax*0.5/Kpen, Ymax*0.6/Kpen, 'LH', color=LHcol, fontsize=15, fontweight='bold')
plt.text(Xmax*0.39/Kpen, Ymax*0.01/Kpen, 'LL\n'+r'$\cap$'+'\nLH', color='black', fontsize=15, fontweight='bold',
         horizontalalignment='center', alpha=0.7)
plt.annotate('', xy=(0.59/Kpen, 0.01/Kpen), xytext=(0.65/Kpen, 0.013/Kpen), arrowprops=dict(arrowstyle="-", color='black'))
plt.text(Xmax*0.71/Kpen, Ymax*0.01/Kpen, 'LH\n'+r'$\cap$'+'\nHH', color='black', fontsize=15, fontweight='bold',
         horizontalalignment='center', alpha=0.7)
plt.annotate('', xy=(1.04/Kpen, 0.01/Kpen), xytext=(1.10/Kpen, 0.013/Kpen), arrowprops=dict(arrowstyle="-", color='black'))
ax.set_xbound(0, Xmax/Kpen)
ax.set_ybound(0, Ymax/Kpen)
# plt.xlabel(r'$\theta^{\text{R}}$', fontsize=11)
# plt.ylabel(r'$\theta^{\text{S}}$', fontsize=11, rotation=0, labelpad=14)
plt.xlabel(r'$X$', fontsize=11)
plt.ylabel(r'$Y$', fontsize=11, rotation=0, labelpad=14)
plt.savefig('eqplot_example.png', dpi=300, bbox_inches='tight')
plt.show()

#######################
# HHFOC Boundary Plots for Case Study
#######################
b, cSup, supRateLo, supRateHi, inspSensSup, inspSensRet = 0.8, 0.2, 0.8, 1.0, 0.85, 0.2
scDictCough = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo, 'supRateHi': supRateHi,
               'inspSensSup':inspSensSup, 'inspSensRet':inspSensRet}
b, cSup, supRateLo, supRateHi, inspSensSup, inspSensRet = 0.8, 0.05, 0.8, 1.0, 0.85, 0.8
scDictParac = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo, 'supRateHi': supRateHi,
               'inspSensSup':inspSensSup, 'inspSensRet':inspSensRet}
b, cSup, supRateLo, supRateHi, inspSensSup, inspSensRet = 0.8, 0.15, 0.8, 1.0, 0.85, 0.2
scDictCoughInterv = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo, 'supRateHi': supRateHi,
                     'inspSensSup':inspSensSup, 'inspSensRet':inspSensRet}
Kpen = 3.0
XmaxParac, XmaxCough = 1.8, 5.0
numpts = 1000
XCough, XParac = np.arange(0, XmaxCough, XmaxCough/numpts), np.arange(0, XmaxParac, XmaxParac/numpts)

XAct, YAct = Kpen*0.06, Kpen*0.06

YCough = GetYHHLB(scDictCough, XmaxCough, numpts=numpts)
YParac = GetYHHLB(scDictParac, XmaxParac, numpts=numpts)

csReducYbd = YHHICLB(scDictCoughInterv)
XInt, YInt = Kpen*0.06, Kpen*0.08

al, fillal, supSize, lnwd, labmult, labmult2 = 0.9, 0.3, 18, 5, 1.25, 0.88
bdColor, fillcolor = 'midnightblue', 'cornflowerblue'

# Adjust to inspection probability space
XAct, YAct = XAct/Kpen, YAct/Kpen
XInt, YInt = XInt/Kpen, YInt/Kpen
XCough, YCough = XCough/Kpen, np.array(YCough)/Kpen
XParac, YParac = XParac/Kpen, np.array(YParac)/Kpen
XmaxCough = XmaxCough / Kpen
csReducYbd = csReducYbd / Kpen

# Cough Syrup
arrowXLoc = 0.3
fig = plt.figure()
fig.suptitle('Cough Syrup', fontsize=supSize, )
line1, = plt.plot(XCough, YCough, linewidth=lnwd, color=bdColor, alpha=al, label='HH boundary, actual')
plt.fill_between(XCough, YCough, 1.0, color=fillcolor, alpha = fillal)
line2 = plt.axhline(csReducYbd, color='darkgray', alpha=al, linewidth=lnwd*0.7, linestyle='--',
            label='HH boundary, PLI')
plt.plot(XAct, YAct, marker='o', color='black', markersize=9)
plt.plot(XInt, YInt, marker='o',markerfacecolor='none', markeredgecolor='black', markersize=9,
         markeredgewidth=2) # new location
plt.text(XAct*labmult, YAct*labmult2, r'$(\theta^{\text{R}}_{act},\theta^{\text{S}}_{act})$', fontsize=10)
plt.legend(handles=[line1, line2], bbox_to_anchor=(0.97, 0.97), borderpad=0.8,
           loc='upper right', fontsize=8)
plt.annotate('', xytext=(arrowXLoc, YCough[10]*0.98), xy=(arrowXLoc, csReducYbd*1.02),
             arrowprops=dict(arrowstyle='-|>',color='darkred')) # arrow for PLI intervention
plt.annotate('', xytext=(XAct,YAct*1.05), xy=(XInt,YInt*0.96),
             arrowprops=dict(arrowstyle='-|>',color='darkgreen')) # arrow for increased supplier inspection
plt.text(XAct*labmult*1., YAct*labmult*0.88, r'$\theta_S:0.06\rightarrow0.08$', fontsize=8, color='darkgreen')
plt.text(arrowXLoc*1.05, csReducYbd*labmult*0.92, r'$c_S:0.20\rightarrow0.15$', fontsize=8, color='darkred')
plt.ylim(0, 0.18)
plt.xlim(0, 1.0)
plt.xlabel(r'$\theta^{\text{R}}$'+' (retailer inspection probability)', fontsize=11)
plt.ylabel(r'$\theta^{\text{S}}$'+' (supplier inspection probability)', fontsize=11)
plt.text(0.025, 0.168, r'$\times$'+' HH needs intervention', color='red',
         bbox=dict(edgecolor='red', facecolor='whitesmoke', alpha=0.7))
plt.text(0.85, csReducYbd*1.05, r'$\theta^{\text{S}}=$'+str(round(csReducYbd,3)),
         color='dimgray',  fontsize=9)
plt.text(0.85, YCough[0]*1.05, r'$\theta^{\text{S}}=$'+str(round(YCough[0],3)),
         color='dimgray',  fontsize=9)
plt.savefig('CS_cough.png', dpi=300, bbox_inches='tight')
plt.show()

# Paracetamol
fig = plt.figure()
fig.suptitle('Paracetamol', fontsize=supSize)
line1, = plt.plot(XParac, YParac, linewidth=lnwd, color=bdColor, alpha=al, label='HH boundary, actual')
plt.fill_between(XParac, YParac, 1.0, color=fillcolor, alpha = fillal)
plt.plot(XAct, YAct, marker='o', color='black', markersize=9)
plt.text(XAct*labmult, YAct*labmult2, r'$(\theta^{\text{R}}_{act},\theta^{\text{S}}_{act})$', fontsize=10)
plt.legend(handles=[line1], bbox_to_anchor=(0.97, 0.97), borderpad=0.8,
           loc='upper right', fontsize=8)
plt.ylim(0, 0.1)
plt.xlim(0, 0.6)
plt.xlabel(r'$\theta^{\text{R}}$'+' (retailer inspection probability)', fontsize=11)
plt.ylabel(r'$\theta^{\text{S}}$'+' (supplier inspection probability)', fontsize=11)
plt.text(0.015, 0.0935, r'$\checkmark$'+' HH is feasible', color='darkgreen',
         bbox=dict(edgecolor='darkgreen', facecolor='whitesmoke', alpha=0.7))
plt.savefig('CS_parac.png', dpi=300, bbox_inches='tight')
plt.show()

#####################
# Equilibrium plots
#####################
def RetPrefh12Overl12(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = cRet * (2 - cRet - priceSup_1 - priceSup_2)
    denominator = 2 * (1 + b) * rateSup_1 * rateSup_2 * (rateRetHi - rateRetLo) * inspSensRet
    thresh = numerator / denominator
    if X >= thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefh1Overh12(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = (1 - b * (1 - cRet - priceSup_1) - cRet - priceSup_2)**2
    denominator = 4 * (1 - b**2) * rateSup_1 * (1 - rateSup_2) * rateRetHi * inspSensRet
    thresh = numerator / denominator
    if X >= thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefh2Overh12(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = (1 - b * (1 - cRet - priceSup_2) - cRet - priceSup_1)**2
    denominator = 4 * (1 - b**2) * rateSup_2 * (1 - rateSup_1) * rateRetHi * inspSensRet
    thresh = numerator / denominator
    if X >= thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefl1Overl12(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = (1 - b * (1 - priceSup_1) - priceSup_2)**2
    denominator = 4 * (1 - b**2) * rateSup_1 * (1 - rateSup_2) * rateRetLo * inspSensRet
    thresh = numerator / denominator
    if X >= thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefl2Overl12(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = (1 - b * (1 - priceSup_2) - priceSup_1)**2
    denominator = 4 * (1 - b**2) * rateSup_2 * (1 - rateSup_1) * rateRetLo * inspSensRet
    thresh = numerator / denominator
    if X >= thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefh1Overl1(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = cRet * (2 - cRet - 2 * priceSup_1)
    denominator = 4 * rateSup_1 * (rateRetHi - rateRetLo) * inspSensRet
    thresh = numerator / denominator
    if X >= thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefh2Overl2(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = cRet * (2 - cRet - 2 * priceSup_2)
    denominator = 4 * rateSup_2 * (rateRetHi - rateRetLo) * inspSensRet
    thresh = numerator / denominator
    if X >= thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefh1Overl12(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = (b ** 2 * (1 - cRet - priceSup_1) ** 2 - 2 * b * (1 - priceSup_1 - priceSup_2 +
                priceSup_1 * priceSup_2) + cRet * (2 - cRet - 2 * priceSup_1) + (1 - priceSup_2) ** 2)
    denominator = 4 * (1 - b ** 2) * rateSup_1 * inspSensRet * (rateRetHi - rateSup_2 * rateRetLo)
    thresh = numerator / denominator
    if X >= thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefh2Overl12(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = (b**2 * (1 - cRet - priceSup_2)**2 - 2 * b * (1 - priceSup_2 - priceSup_1 +
                priceSup_2 * priceSup_1) + cRet * (2 - cRet - 2 * priceSup_2) + (1 - priceSup_1)**2)
    denominator = 4 * (1 - b**2) * rateSup_2 * inspSensRet * (rateRetHi - rateSup_1 * rateRetLo)
    thresh = numerator / denominator
    if X >= thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefh12Overl1(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = (2 * b * (1 - cRet - priceSup_1) * (1 - cRet - priceSup_2) - 2 * cRet**2
                + 2 * cRet * (2 - priceSup_1 - priceSup_2) - b**2 * (1 - priceSup_1)**2 - (1 - priceSup_2)**2)
    denominator = 4 * (1 - b**2) * rateSup_1 * inspSensRet * (rateSup_2 * rateRetHi - rateRetLo)
    thresh = numerator / denominator
    if rateSup_2 * rateRetHi - rateRetLo >= 0 and X >= thresh:
        retval = 1
    elif rateSup_2 * rateRetHi - rateRetLo < 0 and X < thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefh12Overl2(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = (2 * b * (1 - cRet - priceSup_2) * (1 - cRet - priceSup_1) - 2 * cRet**2
                + 2 * cRet * (2 - priceSup_2 - priceSup_1) - b**2 * (1 - priceSup_2)**2 - (1 - priceSup_1)**2)
    denominator = 4 * (1 - b**2) * rateSup_2 * inspSensRet * (rateSup_1 * rateRetHi - rateRetLo)
    thresh = numerator / denominator
    if rateSup_1 * rateRetHi - rateRetLo >= 0 and X >= thresh:
        retval = 1
    elif rateSup_1 * rateRetHi - rateRetLo < 0 and X < thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefh2Overh1(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = (priceSup_2 - priceSup_1) * (2 - 2 * cRet - priceSup_1 - priceSup_2)
    denominator = 4 * rateRetHi * (rateSup_2 - rateSup_1) * inspSensRet
    thresh = numerator / denominator
    if X >= thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefl2Overl1(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = (priceSup_2 - priceSup_1) * (2 - 2 * cRet - priceSup_1 - priceSup_2)
    denominator = 4 * rateRetLo * (rateSup_2 - rateSup_1) * inspSensRet
    thresh = numerator / denominator
    if X >= thresh:
        retval = 1
    else:
        retval = 0
    return retval

def RetPrefl2Overh1(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = (cRet + priceSup_1 - priceSup_2) * (2 - cRet - priceSup_1 - priceSup_2)
    denominator = 4 * inspSensRet * (rateSup_1 * rateRetHi - rateSup_2 * rateRetLo)
    thresh = numerator / denominator
    crucRat = (cRet + priceSup_1 - priceSup_2) / (rateSup_1 * rateRetHi - rateSup_2 * rateRetLo)
    if X >= thresh and crucRat > 0:
        retval = 1
    elif X < thresh and crucRat <= 0:
        retval = 1
    else:
        retval = 0

    return retval

def RetPrefl1Overh2(X, scDict):
    cRet, priceSup_2, priceSup_1 = scDict['cRet'], scDict['priceSup_2'], scDict['priceSup_1']
    b, rateSup_2, rateSup_1 = scDict['b'], scDict['rateSup_2'], scDict['rateSup_1']
    rateRetLo, rateRetHi, inspSensRet = scDict['rateRetLo'], scDict['rateRetHi'], scDict['inspSensRet']
    numerator = (cRet + priceSup_2 - priceSup_1) * (2 - cRet - priceSup_2 - priceSup_1)
    denominator = 4 * inspSensRet * (rateSup_2 * rateRetHi - rateSup_1 * rateRetLo)
    thresh = numerator / denominator
    crucRat = (cRet + priceSup_2 - priceSup_1) / (rateSup_2 * rateRetHi - rateSup_1 * rateRetLo)
    if X >= thresh and crucRat > 0:
        retval = 1
    elif X < thresh and crucRat <= 0:
        retval = 1
    else:
        retval = 0
    return retval

b, cRet, rateSup_1, rateSup_2, rateRetLo, rateRetHi = 0.8, 0.04, 0.92, 0.92, 0.6, 0.9
inspSensRet, priceSup_1, priceSup_2 = 0.75, 0.25, 0.25
X = 0.08
scDict = {'b': b, 'cRet': cRet, 'rateRetLo': rateRetLo, 'rateRetHi': rateRetHi, 'inspSensRet': inspSensRet,
          'rateSup_1': rateSup_1, 'rateSup_2': rateSup_2, 'priceSup_1': priceSup_1, 'priceSup_2': priceSup_2}
# Shows the retailer's preferred strategy for various wholesale prices and quality rates of symmetric suppliers
numpts = 1200  # Resolution of prices and rates

plotMat = np.empty((numpts, numpts, 5))  # h12, l12, h1, l1, N; h2/l2 ignored for symmetric case
plotMat[:] = 0
for wInd, wCurr in enumerate(np.arange(0.01,0.99,(0.99-0.01)/numpts)):
    scDict['priceSup_1'], scDict['priceSup_2'] = wCurr, wCurr
    for supRateInd, supRateCurr in enumerate(np.arange(0.01, 0.99, (0.99 - 0.01) / numpts)):
        scDict['rateSup_1'], scDict['rateSup_2'] = supRateCurr, supRateCurr
        # Identify dominant retailer strategy
        prefh12l12 = RetPrefh12Overl12(X, scDict)
        prefh1h12 = RetPrefh1Overh12(X, scDict)
        prefl1l12 = RetPrefl1Overl12(X, scDict)
        prefh1l1 = RetPrefh1Overl1(X, scDict)
        prefh1l12 = RetPrefh1Overl12(X, scDict)
        prefh12l1 = RetPrefh12Overl1(X, scDict)
        # h12
        if prefh12l12 == 1 and prefh1h12 == 0 and prefh12l1 == 1:
            q1, q2 = RetOptQuants(0, scDict['b'], scDict['cRet'], scDict['priceSup_1'], scDict['priceSup_2'])
            if RetUtil(X, scDict, 1, supRateCurr, supRateCurr, q1, q2) < 0:  # N strategy dominates
                plotMat[wInd, supRateInd, 4] = 1
            else:
                plotMat[wInd, supRateInd, 0] = 1
        # l12
        if prefh12l12 == 0 and prefl1l12 == 0 and prefh1l12 == 0:
            q1, q2 = RetOptQuants(1, scDict['b'], scDict['cRet'], scDict['priceSup_1'], scDict['priceSup_2'])
            if RetUtil(X, scDict, 0, supRateCurr, supRateCurr, q1, q2) < 0:  # N strategy dominates
                plotMat[wInd, supRateInd, 4] = 1
            else:
                plotMat[wInd, supRateInd, 1] = 1
        # h1
        if prefh1h12 == 1 and prefh1l1 == 1 and prefh1l12 == 1:
            q1, q2 = RetOptQuants(2, scDict['b'], scDict['cRet'], scDict['priceSup_1'], scDict['priceSup_2'])
            if RetUtil(X, scDict, 1, supRateCurr, supRateCurr, q1, q2) < 0:  # N strategy dominates
                plotMat[wInd, supRateInd, 4] = 1
            else:
                plotMat[wInd, supRateInd, 2] = 1
        # l12
        if prefl1l12 == 1 and prefh1l1 == 0 and prefh12l1 == 0:
            q1, q2 = RetOptQuants(3, scDict['b'], scDict['cRet'], scDict['priceSup_1'], scDict['priceSup_2'])
            if RetUtil(X, scDict, 0, supRateCurr, supRateCurr, q1, q2) < 0:  # N strategy dominates
                plotMat[wInd, supRateInd, 4] = 1
            else:
                plotMat[wInd, supRateInd, 3] = 1

# Fill in weird high w points w N
plotMat[round(0.9*numpts):, :, 2] = 0.0
plotMat[round(0.9*numpts):, :, 4] = 1.0
# Fill in blanks
for i in range(numpts):
    for j in range(numpts):
        if np.sum(plotMat[i, j, :]) != 1.0:
            print('point ' + str(i) + ' '+ str(j))
            plotMat[i, j, 2] = 1.0

alval=0.5
fig = plt.figure()
# ax.set_title(rf"Equilibrium regions ($b=0.8,\ c_S={cS}$)", fontsize=12, pad=16)
# fig.suptitle(r'$b=$'+str(b)+', '+r'$L=$'+str(supRateLo),fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['royalblue', 'firebrick', 'cornflowerblue', 'indianred', 'dimgray']
labels = ['h12', 'l12', 'h1', 'l1', 'N']

imlist = []
for eqind in reversed(range(len(labels))):
    mycmap = matplotlib.colors.ListedColormap(['none', eqcolors[eqind]], name='from_list', N=None)
    # if eqcolors[eqind] == 'black':  # No alpha transparency
    #     im = ax.imshow(eqStrat_matList[eqind], vmin=0, vmax=1, aspect='auto',
    #                         extent=(0, CthetaMax, 0, cSupMax),
    #                         origin="lower", cmap=mycmap, alpha=1)
    # else:
    im = ax.imshow(plotMat[:,:,eqind].T, vmin=0, vmax=1, aspect='auto',
                            extent=(0, 1, 0, 1),
                            origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)
plt.ylim(0, 1.0)
plt.xlim(0, 1.0)
plt.text(0.85, 0.5, 'N', color='dimgray', fontsize=18)
plt.text(0.37, 0.92, 'h12', color='dimgray', fontsize=18)
plt.text(0.1, 0.3, 'l12', color='dimgray', fontsize=18)
plt.text(0.52, 0.72, 'h1', color='dimgray', fontsize=18)
plt.text(0.48, 0.39, 'l1', color='dimgray', fontsize=18)
plt.xlabel(r'$w_1=w_2$', fontsize=11)
plt.ylabel(r'$\Lambda_1=\Lambda_2$', fontsize=11, rotation=0, labelpad=14)
# plt.savefig('retailerStratPrefs.png', dpi=300, bbox_inches='tight')
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

#############################
# Plot of b condition
#############################
def bhat(cS):
    rootval = (8 + 27*(-1 + cS)*cS*(-8 + 5*cS)+3*sqroot(3*((4 - 3*cS)**2)*(-1 + cS)*cS*(-8 + 5*cS)*(1 + 15*cS)))**(1/3)
    retval = (-10 + 6*cS + 4/rootval + rootval)/(3*(-4 + 3*cS))
    return retval

def bhatEst(cS):
    return 0.5 - 0.5*cS

cSVec = np.arange(0.001,0.999,0.001)
bVec1 = []
for currcS in cSVec:
    bVec1.append(bhat(currcS))
bVec2 = bhatEst(cSVec)
bVec1 = np.array(bVec1)

plt.plot(cSVec, bVec1, '-', linewidth=5, color='darkgreen')
plt.plot(cSVec, bVec2, '--', linewidth=5, color='deepskyblue')
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_box_aspect(1)
plt.xlabel(r'$c_S$', fontsize=14)
plt.ylabel(r'$\hat{b}$', fontsize=14, rotation=0, labelpad=14)
plt.show()

#############################
# Plot of cS condition
#############################
def cShat(b):
    rootval = sqroot((36 - 112*b + 132*(b**2) - 64*(b**3) + 21*(b**4) - 64*(b**5) + 90*(b**6) - 48*(b**7) +
                        9*(b**8))/((-2 + 4*b - 6*(b**2) + 3*(b**3))**2))
    retval = (2 - 11*(b**2) + 12*(b**3) - 3*(b**4))/(2*(-2 + 4*b - 6*(b**2) + 3*(b**3))) + 0.5*rootval
    return retval

def cShat2(b):
    # cS below this cShat2 means that the off-path floor that ensures dual sourcing need never be considered
    retval = (1 - b)/(1 + b + (b**2) - (b**3))
    return retval

bVec = np.arange(0.001,0.999,0.001)
cSVec, cSVec2 = [], []
for currb in bVec:
    cSVec.append(cShat(currb))
    cSVec2.append(cShat2(currb))

plt.plot(bVec, cSVec, '-', linewidth=5, color='darkblue')
plt.plot(bVec, cSVec2, '-', linewidth=5, color='darkred')
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_box_aspect(1)
plt.xlabel(r'$b$', fontsize=14)
plt.ylabel(r'$\hat{c}_S$', fontsize=14, rotation=0, labelpad=14)
plt.show()

