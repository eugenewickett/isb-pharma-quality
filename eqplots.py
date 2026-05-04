# Base model of "Strategic Role of Inspections in Pharmaceutical Supply Chains"
# Supplier-inspection penalties
# These functions use the simpler version of the model
#   - perfect diagnostic, perfect high-quality rate, no retailer quality choice, retailer sources from both suppliers or neither
# 5-MAR-26

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

np.set_printoptions(precision=4, suppress=True)
plt.rcParams["font.family"] = "monospace"

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def sqroot(val):
    # Returns square root of val after checking that val is positive; returns NaN otherwise
    if val<0:
        retval = np.nan
    else:
        retval = np.sqrt(val)
    return retval

def SupUtil(q, w, cSup, lambsup, Cbeta):
    # Returns supplier utility
    return q*(w-cSup) - (1-lambsup)*Cbeta

def invDemandPrice(qi, qj, b):
    return 1 - qi - b*qj

def q1Opt(w1, w2, b):
    # Returns optimal order quantities from S1 under dual sourcing
    return max(0, (1-b-w1+b*w2)/(2*(1-(b**2))))

def q2Opt(w1, w2, b):
    # Returns optimal order quantities from S1 under dual sourcing
    return max(0, (1-b+b*w1-w2)/(2*(1-(b**2))))

def RetUtil(b, w1, w2, Ctheta, lambsup1, lambsup2):
    # Returns retailer utility under pi=D
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    prof1, prof2 = q1*(invDemandPrice(q1, q2, b)-w1), q2*(invDemandPrice(q2, q1, b)-w2)
    insppen = Ctheta * ((1 - lambsup1 * lambsup2))
    return prof1 + prof2 - insppen

def SupPriceLLFOC(scDict, Ctheta, Cbeta):
    # Returns on-path LL prices
    b, cS, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    w1 = max(0, (1 - b)/(2 - b))
    w2 = max(0, (1 - b)/(2 - b))
    return w1, w2

def SupPriceHH(scDict, Ctheta, Cbeta):
    # Returns on-path HH prices
    b, cS, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    w1 = max(0, (1 - b + cS) / (2 - b))
    w2 = max(0, (1 - b + cS) / (2 - b))
    return w1, w2

def SupPriceLHFOC(scDict, Ctheta, Cbeta):
    # Returns on-path LH prices
    b, cS, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    w1 = max(0, (2 - b - (b**2) + b*cS)/(4 - (b**2)))
    w2 = max(0, (2 - b - (b**2) + 2*cS)/(4 - (b**2)))
    return w1, w2

def SupPriceLHsqz(scDict, Ctheta, Cbeta):
    # Returns on-path LHsqz prices
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    radterm = sqroot((-1 + (b**2))*(((1 - cS)**2) - 4*(-4 + 3*(b**2))*Ctheta*(-1 + supRateHi*supRateLo)))
    w1 = (b*(1 + 3*b - cS) + 2*(-2 + radterm))/(-4 + 3*(b**2))
    w2 = (-2*(1 + cS) + (b**2)*(2 + cS) + b*radterm)/(-4 +3*(b**2))
    # if w2 < cS:
    #     w2 = cS
    #     w1 = 1 - b + cS*b + ((b**2) - 1)*sqroot((1 + (-2 + cS)*cS + 4*Ctheta *(-1 + supRateHi*supRateLo))/(-1 + (b**2)))
    return w1, w2

def SupPriceLHsqzAtCost(scDict, Ctheta, Cbeta):
    # Returns on-path LHsqz prices
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    radterm = sqroot((-1 + (b**2))*(((1 - cS)**2) - 4*(-4 + 3*(b**2))*Ctheta*(-1 + supRateHi*supRateLo)))
    w1 = (b*(1 + 3*b - cS) + 2*(-2 + radterm))/(-4 + 3*(b**2))
    w2 = (-2*(1 + cS) + (b**2)*(2 + cS) + b*radterm)/(-4 +3*(b**2))
    if w2 < cS:
        w2 = cS
        w1 = 1 - b + cS*b + ((b**2) - 1)*sqroot((1 + (-2 + cS)*cS + 4*Ctheta *(-1 + supRateHi*supRateLo))/(-1 + (b**2)))
    else:
        w1, w2 = np.nan, np.nan
    return w1, w2

def SupPriceLLsqz(scDict, Ctheta, Cbeta):
    # Returns on-path LLsqz prices
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1 = 1 - sqroot(-2*((1 + b)*Ctheta*(-1 + (supRateLo**2))))
    w2 = w1
    return max(w1, 0), max(w2, 0)

def funcLHexpIRPrice(x, scDict, X, Y):
    # When only S2 wants to move off-path; S1 chooses price s.t. S2 is indifferent, S2 chooses FOC
    # Retailer-penalty induced
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    term1 = -1*(((cS - x[1])*(-1 + b - b*x[0] + x[1])) / (2*(-1 + (b**2)))) + Y*(-1 + supRateHi)
    radterm1 = sqroot((((-1 + x[0])**2) + 4* X*(-1 + (supRateLo**2)))/(-1 + (b**2)))
    term2 = 0.5*(((-1 + x[0])**2) + 2* Y*(-1 +supRateLo) + 4 *X* (-1 + (supRateLo**2)) + (1 - b +
        b * x[0])*radterm1)
    retval1 = term1 - term2
    retval2 = x[1] - (0.5*(1 + cS + b*(-1 + x[0])))
    return [retval1, retval2]

def SupPriceLHexpIR(scDict, X, Y, tol=1E-8):
    # Returns on-path LHexp-IR prices
    xinit = SupPriceLHFOC(scDict, X, Y)
    root, infodict, _, _ = fsolve(funcLHexpIRPrice, [xinit[0], xinit[1]], args=(scDict, X, Y), full_output=True)
    if np.sum(np.abs(infodict['fvec'])) > tol or np.isnan(infodict['fvec'][0]):
        root = [np.nan, np.nan]
    return root[0], root[1]

def funcHHexpIRPrice(x, scDict, Ctheta, Cbeta):
    # x = [w1HHexp, w2HHexp], candidated HHexp prices
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    term1 = -1*(((cS - x[0])*(-1 + b + x[0] - b*x[1]))/(2*(-1 + (b**2)))) + Cbeta*(-1 +supRateHi)
    term2 = 0.5*(((-1 + x[1])**2) + 2*Cbeta*(-1 + supRateLo) + 4*Ctheta*(-1+supRateHi* supRateLo) +
            (1-b+b*x[1])*sqroot((((-1 + x[1])**2) + 4*Ctheta*(-1 + supRateHi*supRateLo))/(-1 + (b**2))))
    term3 = -1 * (((cS - x[1]) * (-1 + b + x[1] - b * x[0])) / (2 * (-1 + (b ** 2)))) + Cbeta * (-1 + supRateHi)
    term4 = 0.5*(((-1 + x[0])**2) + 2*Cbeta*(-1 + supRateLo) + 4*Ctheta*(-1+supRateHi* supRateLo) +
            (1-b+b*x[0])*sqroot((((-1 + x[0])**2) + 4*Ctheta*(-1 + supRateHi*supRateLo))/(-1 + (b**2))))
    retval1 = term1 - term2
    retval2 = term3 - term4
    return [retval1, retval2]

def SupPriceHHexpIR(scDict, Ctheta, Cbeta, tol=1E-8):
    # Returns on-path HHexp prices
    xinit = SupPriceHH(scDict, 0, 0)
    root, infodict, _, _ = fsolve(funcHHexpIRPrice, [xinit[0], xinit[1]], args=(scDict, Ctheta, Cbeta), full_output=True)
    if np.sum(np.abs(infodict['fvec'])) > tol or np.isnan(infodict['fvec'][0]):
        root= [np.nan, np.nan]
    return root[0], root[1]

def SupPriceLHexpIC(scDict, Ctheta, Cbeta):
    # Returns on-path LHexp prices, if the off-path prices accounted for are RetIR-valid
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1 = (2*(-1 + b)*cS + (cS**2) - 8*(-1 + (b**2))*Cbeta*(supRateHi - supRateLo))/(2*b*cS)
    w2 = (3*cS)/4 - (2*(-1 + (b**2))*Cbeta*(supRateHi - supRateLo))/cS
    # Is the off-path price RetIR-valid?
    w2off = 0.5* (1 - b + b*w1)
    if (RetUtil(b, w1, w2off, Ctheta, supRateLo, supRateHi) < 0 or np.isnan(w2off)):
        # w1, w2 = SupPriceHHexpRet(scDict, Ctheta, Cbeta, tol=1E-8)
        w1, w2 = np.nan, np.nan
    return max(w1, 0), max(w2, 0)

def SupPriceLHexpICAtCost(scDict, X, Y):
    # Returns on-path LHexp-ICAtCost prices
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1 = ((-1 + b)*b + 2*sqroot(2*(b**2)*(-1 + (b**2)) * Y * (-1 + supRateLo)))/(b**2)
    w2 = cS
    return max(w1, 0), max(w2, 0)

def SupPriceHHexpIC(scDict, Ctheta, Cbeta):
    # Returns on-path HHexpSup prices, if the off-path prices they account for are RetIR-valid
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1 = ((b**2) - b* (3 + 2*cS) + 2* (1 + cS + sqroot(-1*((-1 + b)*((-2 + b)*cS - (-1 + b)*(cS**2) +
         2*((-2 + b)**2)* (1 + b)*Cbeta*(supRateHi - supRateLo))))))/((-2 + b)**2)
    w2 = w1
    # Is the off-path price RetIR-valid?
    radterm = sqroot((1 - 4*cS*(-1 + w1) + 4*(-1 + w1)*w1 +2*b*(1 + 2*cS - 2*w1)*(-1 + w2) -8*Cbeta*supRateHi +
                    (b**2)*(1 + (-2 + w2)*w2 + 8*Cbeta*(supRateHi - supRateLo)) +8*Cbeta*supRateLo)/((-1 + (b**2))**2))
    w1off = 0.5*(1 - b + b*w2 - (1 - (b**2))*radterm)
    # Check if off-path prices are possible; else use HHexpRet prices
    if (RetUtil(b, w1off, w2, Ctheta, supRateLo, supRateHi)<0 or np.isnan(w1off)):
        # w1, w2 = SupPriceHHexpRet(scDict, Ctheta, Cbeta, tol=1E-8)
        w1, w2 = np.nan, np.nan
    if w1 < cS:
        w1, w2 = cS, cS

    return w1, w2



# These check functions evaluate the validity of each equilibrium
# All IR conditions are checked
# On-path and off-path prices are checked
# Comparisons with other equilibria are NOT checked
# Violation codes: -1: no violation
#   0: RetIR, 1: S1IR, 2:S2IR, 3:S1ICon, 4: S2ICon, 5:S1ICoff, 6:S2ICoff
def CheckLLFOC(scDict, Ctheta, Cbeta, tol=1E-5):
    # LLFOC
    checkBool, violCode = True, -1  # violation code
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1, w2 = SupPriceLLFOC(scDict, Ctheta, Cbeta)
    rateSup1, rateSup2 = supRateLo, supRateLo
    rateSup1off, rateSup2off = supRateHi, supRateHi
    invest1, invest2 = 0, 0  # Qual investment
    invest1off, invest2off = cS, cS  # Qual investment
    # On-path utilities
    uRet = RetUtil(b, w1, w2, Ctheta, rateSup1, rateSup2)
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    uS1, uS2 = SupUtil(q1, w1, invest1, rateSup1, Cbeta), SupUtil(q2, w2, invest1, rateSup2, Cbeta)
    # Check on-path
    if (uRet < 0):
        checkBool, violCode = False, 0
    if (uS1 < 0):
        checkBool, violCode = False, 1
    if (uS2 < 0):
        checkBool, violCode = False, 2
    # IC conditions
    wVec = np.arange(0.005, 1, 0.005)
    if checkBool:  # Check S1 on-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt-w1)>0.01 and SupUtil(q1Alt, w1Alt, invest1, rateSup1, Cbeta)>uS1:
                checkBool, violCode = False, 3
    if checkBool: # Check S2 on-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2)>0.01 and SupUtil(q2Alt, w2Alt, invest2, rateSup2, Cbeta)>uS2:
                checkBool, violCode = False, 4
    if checkBool:  # Check S1 off-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt - w1) > 0.01 and SupUtil(q1Alt, w1Alt, invest1off, rateSup1off, Cbeta) > uS1:
                checkBool, violCode = False, 5
    if checkBool: # Check S2 off-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2)>0.01 and SupUtil(q2Alt, w2Alt, invest2off, rateSup2off, Cbeta)>uS2:
                checkBool, violCode = False, 6
    return checkBool, violCode, w1, w2, uRet, uS1, uS2

def CheckLLsqz(scDict, Ctheta, Cbeta, tol=1E-5, printUpdate=False):
    # LLsqz
    checkBool, violCode = True, -1  # violation code
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1, w2 = SupPriceLLsqz(scDict, Ctheta, Cbeta)
    rateSup1, rateSup2 = supRateLo, supRateLo
    rateSup1off, rateSup2off = supRateHi, supRateHi
    invest1, invest2 = 0, 0  # Qual investment
    invest1off, invest2off = cS, cS  # Qual investment
    # On-path utilities
    uRet = RetUtil(b, w1, w2, Ctheta, rateSup1, rateSup2)
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    uS1, uS2 = SupUtil(q1, w1, invest1, rateSup1, Cbeta), SupUtil(q2, w2, invest1, rateSup2, Cbeta)
    # Check on-path
    if (uRet < -tol):
        checkBool, violCode = False, 0
    if (uS1 < -tol):
        checkBool, violCode = False, 1
    if (uS2 < -tol):
        checkBool, violCode = False, 2
    # IC conditions
    wVec = np.arange(0.005, 1, 0.005)
    if checkBool:  # Check S1 on-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt-w1) > tol and SupUtil(q1Alt, w1Alt, invest1, rateSup1, Cbeta) > uS1+tol and\
                RetUtil(b, w1Alt, w2, Ctheta, rateSup1, rateSup2) > -tol:
                checkBool, violCode = False, 3
    if checkBool: # Check S2 on-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2, rateSup2, Cbeta) > uS2+tol and\
                RetUtil(b, w1, w2Alt, Ctheta, rateSup1, rateSup2) > -tol:
                checkBool, violCode = False, 4
    if checkBool:  # Check S1 off-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt - w1) > tol and SupUtil(q1Alt, w1Alt, invest1off, rateSup1off, Cbeta) > uS1+tol and\
                RetUtil(b, w1Alt, w2, Ctheta, rateSup1off, rateSup2) > -tol:
                checkBool, violCode = False, 5
    if checkBool: # Check S2 off-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2off, rateSup2off, Cbeta)>uS2+tol and\
                RetUtil(b, w1, w2Alt, Ctheta, rateSup1, rateSup2off) > -tol:
                checkBool, violCode = False, 6
    if checkBool:  # Check for nan values
        if np.isnan(w1) or np.isnan(w2):
            checkBool, violCode = False, 7
            if printUpdate:
                print('On-path prices are not real-valued')
    return checkBool, violCode, w1, w2, uRet, uS1, uS2

def CheckHHFOC(scDict, Ctheta, Cbeta, tol=1E-5, printUpdate=False):
    # HHFOC
    checkBool, violCode = True, -1  # violation code
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1, w2 = SupPriceHH(scDict, Ctheta, Cbeta)
    rateSup1, rateSup2 = supRateHi, supRateHi
    rateSup1off, rateSup2off = supRateLo, supRateLo
    invest1, invest2 = cS, cS  # Qual investment
    invest1off, invest2off = 0, 0  # Qual investment off-path
    # On-path utilities
    uRet = RetUtil(b, w1, w2, Ctheta, rateSup1, rateSup2)
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    uS1, uS2 = SupUtil(q1, w1, invest1, rateSup1, Cbeta), SupUtil(q2, w2, invest1, rateSup2, Cbeta)
    # Check on-path
    if (uRet < -tol):
        checkBool, violCode = False, 0
    if (uS1 < -tol):
        checkBool, violCode = False, 1
    if (uS2 < -tol):
        checkBool, violCode = False, 2
    # IC conditions
    wVec = np.arange(0.005, 1, 0.005)
    if checkBool:  # Check S1 on-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt-w1) > tol and SupUtil(q1Alt, w1Alt, invest1, rateSup1, Cbeta) > uS1+tol and\
                RetUtil(b, w1Alt, w2, Ctheta, rateSup1, rateSup2) > -tol:
                checkBool, violCode = False, 3
    if checkBool: # Check S2 on-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2, rateSup2, Cbeta) > uS2+tol and\
                RetUtil(b, w1, w2Alt, Ctheta, rateSup1, rateSup2) > -tol:
                checkBool, violCode = False, 4
    if checkBool:  # Check S1 off-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt - w1) > tol and SupUtil(q1Alt, w1Alt, invest1off, rateSup1off, Cbeta) > uS1+tol and\
                RetUtil(b, w1Alt, w2, Ctheta, rateSup1off, rateSup2) > -tol:
                checkBool, violCode = False, 5
                if printUpdate:
                    print('S1 off-path move found for w1='+str(w1Alt))
    if checkBool: # Check S2 off-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2off, rateSup2off, Cbeta)>uS2+tol and\
                RetUtil(b, w1, w2Alt, Ctheta, rateSup1, rateSup2off) > -tol:
                checkBool, violCode = False, 6
    return checkBool, violCode, w1, w2, uRet, uS1, uS2

def CheckHHexpIC(scDict, X, Y, tol=1E-5, printUpdate=False):
    # HHFOC
    checkBool, violCode = True, -1  # violation code
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1, w2 = SupPriceHHexpIC(scDict, X, Y)
    rateSup1, rateSup2 = supRateHi, supRateHi
    rateSup1off, rateSup2off = supRateLo, supRateLo
    invest1, invest2 = cS, cS  # Qual investment
    invest1off, invest2off = 0, 0  # Qual investment off-path
    # On-path utilities
    uRet = RetUtil(b, w1, w2, X, rateSup1, rateSup2)
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    uS1, uS2 = SupUtil(q1, w1, invest1, rateSup1, Y), SupUtil(q2, w2, invest1, rateSup2, Y)
    # Check on-path
    if (uRet < -tol) or np.isnan(uRet):
        checkBool, violCode = False, 0
    if (uS1 < -tol) or np.isnan(uRet):
        checkBool, violCode = False, 1
    if (uS2 < -tol) or np.isnan(uRet):
        checkBool, violCode = False, 2
    # IC conditions
    wVec = np.arange(0.005, 1, 0.005)
    if checkBool:  # Check S1 on-path, but only for lower prices
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt-w1) > tol and SupUtil(q1Alt, w1Alt, invest1, rateSup1, Y) > uS1+tol and\
                RetUtil(b, w1Alt, w2, X, rateSup1, rateSup2) > -tol and w1Alt < w1:
                checkBool, violCode = False, 3
            if printUpdate:
                print('S1 on-path move found for w1=' + str(w1Alt))
    if checkBool: # Check S2 on-path, but only for lower prices
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2, rateSup2, Y) > uS2+tol and\
                RetUtil(b, w1, w2Alt, X, rateSup1, rateSup2) > -tol and w2Alt < w2:
                checkBool, violCode = False, 4
    if checkBool:  # Check S1 off-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt - w1) > tol and SupUtil(q1Alt, w1Alt, invest1off, rateSup1off, Y) > uS1+tol and\
                RetUtil(b, w1Alt, w2, X, rateSup1off, rateSup2) > -tol:
                checkBool, violCode = False, 5
                if printUpdate:
                    print('S1 off-path move found for w1='+str(w1Alt))
    if checkBool: # Check S2 off-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2off, rateSup2off, Y)>uS2+tol and\
                RetUtil(b, w1, w2Alt, X, rateSup1, rateSup2off) > -tol:
                checkBool, violCode = False, 6
    if checkBool:  # Check for nan values
        if np.isnan(w1) or np.isnan(w2):
            checkBool, violCode = False, 7
            if printUpdate:
                print('On-path prices are not real-valued')
    return checkBool, violCode, w1, w2, uRet, uS1, uS2

def CheckHHexpIR(scDict, X, Y, tol=1E-5, printUpdate=False):
    # HHFOC
    checkBool, violCode = True, -1  # violation code
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1, w2 = SupPriceHHexpIR(scDict, X, Y)
    rateSup1, rateSup2 = supRateHi, supRateHi
    rateSup1off, rateSup2off = supRateLo, supRateLo
    invest1, invest2 = cS, cS  # Qual investment
    invest1off, invest2off = 0, 0  # Qual investment off-path
    # On-path utilities
    uRet = RetUtil(b, w1, w2, X, rateSup1, rateSup2)
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    uS1, uS2 = SupUtil(q1, w1, invest1, rateSup1, Y), SupUtil(q2, w2, invest1, rateSup2, Y)
    # Check on-path
    if (uRet < -tol) or np.isnan(uRet):
        checkBool, violCode = False, 0
    if (uS1 < -tol) or np.isnan(uRet):
        checkBool, violCode = False, 1
    if (uS2 < -tol) or np.isnan(uRet):
        checkBool, violCode = False, 2
    # IC conditions
    wVec = np.arange(0.005, 1, 0.005)
    if checkBool:  # Check S1 on-path, but only for lower prices
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt-w1) > tol and SupUtil(q1Alt, w1Alt, invest1, rateSup1, Y) > uS1+tol and\
                RetUtil(b, w1Alt, w2, X, rateSup1, rateSup2) > -tol and w1Alt < w1:
                checkBool, violCode = False, 3
    if checkBool: # Check S2 on-path, but only for lower prices
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2, rateSup2, Y) > uS2+tol and\
                RetUtil(b, w1, w2Alt, X, rateSup1, rateSup2) > -tol and w2Alt < w2:
                checkBool, violCode = False, 4
    if checkBool:  # Check S1 off-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt - w1) > tol and SupUtil(q1Alt, w1Alt, invest1off, rateSup1off, Y) > uS1+tol and\
                RetUtil(b, w1Alt, w2, X, rateSup1off, rateSup2) > -tol:
                checkBool, violCode = False, 5
                if printUpdate:
                    print('S1 off-path move found for w1='+str(w1Alt))
    if checkBool: # Check S2 off-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2off, rateSup2off, Y)>uS2+tol and\
                RetUtil(b, w1, w2Alt, X, rateSup1, rateSup2off) > -tol:
                checkBool, violCode = False, 6
    if checkBool:  # Check for nan values
        if np.isnan(w1) or np.isnan(w2):
            checkBool, violCode = False, 7
            if printUpdate:
                print('On-path prices are not real-valued')
    return checkBool, violCode, w1, w2, uRet, uS1, uS2

def CheckLHFOC(scDict, Ctheta, Cbeta, tol=1E-5, printUpdate=False):
    # LHFOC
    checkBool, violCode = True, -1  # violation code
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1, w2 = SupPriceLHFOC(scDict, Ctheta, Cbeta)
    rateSup1, rateSup2 = supRateLo, supRateHi
    rateSup1off, rateSup2off = supRateHi, supRateLo
    invest1, invest2 = 0, cS  # Qual investment
    invest1off, invest2off = cS, 0  # Qual investment off-path
    # On-path utilities
    uRet = RetUtil(b, w1, w2, Ctheta, rateSup1, rateSup2)
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    uS1, uS2 = SupUtil(q1, w1, invest1, rateSup1, Cbeta), SupUtil(q2, w2, invest2, rateSup2, Cbeta)
    # Check on-path
    if (uRet < -tol):
        checkBool, violCode = False, 0
    if (uS1 < -tol):
        checkBool, violCode = False, 1
    if (uS2 < -tol):
        checkBool, violCode = False, 2
    # IC conditions
    wVec = np.arange(0.005, 1, 0.005)
    if checkBool:  # Check S1 on-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt-w1) > tol and SupUtil(q1Alt, w1Alt, invest1, rateSup1, Cbeta) > uS1+tol and\
                RetUtil(b, w1Alt, w2, Ctheta, rateSup1, rateSup2) > -tol:
                checkBool, violCode = False, 3
    if checkBool: # Check S2 on-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2, rateSup2, Cbeta) > uS2+tol and\
                RetUtil(b, w1, w2Alt, Ctheta, rateSup1, rateSup2) > -tol:
                checkBool, violCode = False, 4
    if checkBool:  # Check S1 off-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt - w1) > tol and SupUtil(q1Alt, w1Alt, invest1off, rateSup1off, Cbeta) > uS1+tol and\
                RetUtil(b, w1Alt, w2, Ctheta, rateSup1off, rateSup2) > -tol:
                checkBool, violCode = False, 5
                if printUpdate:
                    print('S1 off-path move found for w1='+str(w1Alt))
    if checkBool: # Check S2 off-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2off, rateSup2off, Cbeta)>uS2+tol and\
                RetUtil(b, w1, w2Alt, Ctheta, rateSup1, rateSup2off) > -tol:
                checkBool, violCode = False, 6
                if printUpdate:
                    print('S2 off-path move found for w2='+str(w2Alt))
    if checkBool:  # Check for nan values
        if np.isnan(w1) or np.isnan(w2):
            checkBool, violCode = False, 7
            if printUpdate:
                print('On-path prices are not real-valued')
    return checkBool, violCode, w1, w2, uRet, uS1, uS2

def CheckLHsqz(scDict, Ctheta, Cbeta, tol=1E-5, printUpdate=False):
    # LHFOC
    checkBool, violCode = True, -1  # violation code
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1, w2 = SupPriceLHsqz(scDict, Ctheta, Cbeta)
    rateSup1, rateSup2 = supRateLo, supRateHi
    rateSup1off, rateSup2off = supRateHi, supRateLo
    invest1, invest2 = 0, cS  # Qual investment
    invest1off, invest2off = cS, 0  # Qual investment off-path
    # On-path utilities
    uRet = RetUtil(b, w1, w2, Ctheta, rateSup1, rateSup2)
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    uS1, uS2 = SupUtil(q1, w1, invest1, rateSup1, Cbeta), SupUtil(q2, w2, invest2, rateSup2, Cbeta)
    # Check on-path
    if (uRet < -tol):
        checkBool, violCode = False, 0
    if (uS1 < -tol):
        checkBool, violCode = False, 1
    if (uS2 < -tol):
        checkBool, violCode = False, 2
    # IC conditions
    wVec = np.arange(0.005, 1, 0.005)
    if checkBool:  # Check S1 on-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt-w1) > tol and SupUtil(q1Alt, w1Alt, invest1, rateSup1, Cbeta) > uS1+tol and\
                RetUtil(b, w1Alt, w2, Ctheta, rateSup1, rateSup2) > -tol:
                checkBool, violCode = False, 3
    if checkBool: # Check S2 on-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2, rateSup2, Cbeta) > uS2+tol and\
                RetUtil(b, w1, w2Alt, Ctheta, rateSup1, rateSup2) > -tol:
                checkBool, violCode = False, 4
    if checkBool:  # Check S1 off-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt - w1) > tol and SupUtil(q1Alt, w1Alt, invest1off, rateSup1off, Cbeta) > uS1+tol and\
                RetUtil(b, w1Alt, w2, Ctheta, rateSup1off, rateSup2) > -tol:
                checkBool, violCode = False, 5
                if printUpdate:
                    print('S1 off-path move found for w1='+str(w1Alt))
    if checkBool: # Check S2 off-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2off, rateSup2off, Cbeta)>uS2+tol and\
                RetUtil(b, w1, w2Alt, Ctheta, rateSup1, rateSup2off) > -tol:
                checkBool, violCode = False, 6
                if printUpdate:
                    print('S2 off-path move found for w2='+str(w2Alt))
    if checkBool:  # Check for nan values
        if np.isnan(w1) or np.isnan(w2):
            checkBool, violCode = False, 7
            if printUpdate:
                print('On-path prices are not real-valued')
    return checkBool, violCode, w1, w2, uRet, uS1, uS2

def CheckLHsqzAtCost(scDict, Ctheta, Cbeta, tol=1E-5, printUpdate=False):
    # LHFOC
    checkBool, violCode = True, -1  # violation code
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1, w2 = SupPriceLHsqzAtCost(scDict, Ctheta, Cbeta)
    rateSup1, rateSup2 = supRateLo, supRateHi
    rateSup1off, rateSup2off = supRateHi, supRateLo
    invest1, invest2 = 0, cS  # Qual investment
    invest1off, invest2off = cS, 0  # Qual investment off-path
    # On-path utilities
    uRet = RetUtil(b, w1, w2, Ctheta, rateSup1, rateSup2)
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    uS1, uS2 = SupUtil(q1, w1, invest1, rateSup1, Cbeta), SupUtil(q2, w2, invest2, rateSup2, Cbeta)
    # Check on-path
    if (uRet < -tol):
        checkBool, violCode = False, 0
    if (uS1 < -tol):
        checkBool, violCode = False, 1
    if (uS2 < -tol):
        checkBool, violCode = False, 2
    # IC conditions
    wVec = np.arange(0.005, 1, 0.005)
    if checkBool:  # Check S1 on-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt-w1) > tol and SupUtil(q1Alt, w1Alt, invest1, rateSup1, Cbeta) > uS1+tol and\
                RetUtil(b, w1Alt, w2, Ctheta, rateSup1, rateSup2) > -tol:
                checkBool, violCode = False, 3
    if checkBool: # Check S2 on-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2, rateSup2, Cbeta) > uS2+tol and\
                RetUtil(b, w1, w2Alt, Ctheta, rateSup1, rateSup2) > -tol:
                checkBool, violCode = False, 4
    if checkBool:  # Check S1 off-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt - w1) > tol and SupUtil(q1Alt, w1Alt, invest1off, rateSup1off, Cbeta) > uS1+tol and\
                RetUtil(b, w1Alt, w2, Ctheta, rateSup1off, rateSup2) > -tol:
                checkBool, violCode = False, 5
                if printUpdate:
                    print('S1 off-path move found for w1='+str(w1Alt))
    if checkBool: # Check S2 off-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2off, rateSup2off, Cbeta)>uS2+tol and\
                RetUtil(b, w1, w2Alt, Ctheta, rateSup1, rateSup2off) > -tol:
                checkBool, violCode = False, 6
                if printUpdate:
                    print('S2 off-path move found for w2='+str(w2Alt))
    if checkBool:  # Check for nan values
        if np.isnan(w1) or np.isnan(w2):
            checkBool, violCode = False, 7
            if printUpdate:
                print('On-path prices are not real-valued')
    return checkBool, violCode, w1, w2, uRet, uS1, uS2

def CheckLHexpIC(scDict, X, Y, tol=1E-5, printUpdate=False):
    # LHexp-IC
    checkBool, violCode = True, -1  # violation code
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1, w2 = SupPriceLHexpIC(scDict, X, Y)
    rateSup1, rateSup2 = supRateLo, supRateHi
    rateSup1off, rateSup2off = supRateHi, supRateLo
    invest1, invest2 = 0, cS  # Qual investment
    invest1off, invest2off = cS, 0  # Qual investment off-path
    # On-path utilities
    uRet = RetUtil(b, w1, w2, X, rateSup1, rateSup2)
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    uS1, uS2 = SupUtil(q1, w1, invest1, rateSup1, Y), SupUtil(q2, w2, invest2, rateSup2, Y)
    # Check on-path
    if (uRet < -tol):
        checkBool, violCode = False, 0
    if (uS1 < -tol):
        checkBool, violCode = False, 1
    if (uS2 < -tol):
        checkBool, violCode = False, 2
    # IC conditions
    wVec = np.arange(0.005, 1, 0.005)
    if checkBool:  # Check S1 on-path, but only for lower prices
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt-w1) > tol and SupUtil(q1Alt, w1Alt, invest1, rateSup1, Y) > uS1+tol and\
                RetUtil(b, w1Alt, w2, X, rateSup1, rateSup2) > -tol and w1Alt < w1:
                checkBool, violCode = False, 3
    if checkBool: # Check S2 on-path, but only for lower prices
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2, rateSup2, Y) > uS2+tol and\
                RetUtil(b, w1, w2Alt, X, rateSup1, rateSup2) > -tol and w2Alt < w2:
                checkBool, violCode = False, 4
    if checkBool:  # Check S1 off-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt - w1) > tol and SupUtil(q1Alt, w1Alt, invest1off, rateSup1off, Y) > uS1+tol and\
                RetUtil(b, w1Alt, w2, X, rateSup1off, rateSup2) > -tol:
                checkBool, violCode = False, 5
                if printUpdate:
                    print('S1 off-path move found for w1='+str(w1Alt))
    if checkBool: # Check S2 off-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2off, rateSup2off, Y)>uS2+tol and\
                RetUtil(b, w1, w2Alt, X, rateSup1, rateSup2off) > -tol:
                checkBool, violCode = False, 6
                if printUpdate:
                    print('S2 off-path move found for w2='+str(w2Alt))
    if checkBool:  # Check for nan values
        if np.isnan(w1) or np.isnan(w2):
            checkBool, violCode = False, 7
            if printUpdate:
                print('On-path prices are not real-valued')
    return checkBool, violCode, w1, w2, uRet, uS1, uS2

def CheckLHexpICAtCost(scDict, X, Y, tol=1E-5, printUpdate=False):
    # LHexp-ICAtCost
    checkBool, violCode = True, -1  # violation code
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1, w2 = SupPriceLHexpICAtCost(scDict, X, Y)
    rateSup1, rateSup2 = supRateLo, supRateHi
    rateSup1off, rateSup2off = supRateHi, supRateLo
    invest1, invest2 = 0, cS  # Qual investment
    invest1off, invest2off = cS, 0  # Qual investment off-path
    # On-path utilities
    uRet = RetUtil(b, w1, w2, X, rateSup1, rateSup2)
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    uS1, uS2 = SupUtil(q1, w1, invest1, rateSup1, Y), SupUtil(q2, w2, invest2, rateSup2, Y)
    # Check on-path
    if (uRet < -tol):
        checkBool, violCode = False, 0
    if (uS1 < -tol):
        checkBool, violCode = False, 1
    if (uS2 < -tol):
        checkBool, violCode = False, 2
    # IC conditions
    wVec = np.arange(0.005, 1, 0.005)
    if checkBool:  # Check S1 on-path, but only for lower prices
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt-w1) > tol and SupUtil(q1Alt, w1Alt, invest1, rateSup1, Y) > uS1+tol and\
                RetUtil(b, w1Alt, w2, X, rateSup1, rateSup2) > -tol and w1Alt < w1:
                checkBool, violCode = False, 3
    if checkBool: # Check S2 on-path, but only for lower prices
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2, rateSup2, Y) > uS2+tol and\
                RetUtil(b, w1, w2Alt, X, rateSup1, rateSup2) > -tol and w2Alt < w2:
                checkBool, violCode = False, 4
    if checkBool:  # Check S1 off-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt - w1) > tol and SupUtil(q1Alt, w1Alt, invest1off, rateSup1off, Y) > uS1+tol and\
                RetUtil(b, w1Alt, w2, X, rateSup1off, rateSup2) > -tol:
                checkBool, violCode = False, 5
                if printUpdate:
                    print('S1 off-path move found for w1='+str(w1Alt))
    if checkBool: # Check S2 off-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2off, rateSup2off, Y)>uS2+tol and\
                RetUtil(b, w1, w2Alt, X, rateSup1, rateSup2off) > -tol:
                checkBool, violCode = False, 6
                if printUpdate:
                    print('S2 off-path move found for w2='+str(w2Alt))
    if checkBool:  # Check for nan values
        if np.isnan(w1) or np.isnan(w2):
            checkBool, violCode = False, 7
            if printUpdate:
                print('On-path prices are not real-valued')
    return checkBool, violCode, w1, w2, uRet, uS1, uS2

def CheckLHexpIR(scDict, X, Y, tol=1E-5, printUpdate=False):
    # LHexp-IR
    checkBool, violCode = True, -1  # violation code
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1, w2 = SupPriceLHexpIR(scDict, X, Y)
    rateSup1, rateSup2 = supRateLo, supRateHi
    rateSup1off, rateSup2off = supRateHi, supRateLo
    invest1, invest2 = 0, cS  # Qual investment
    invest1off, invest2off = cS, 0  # Qual investment off-path
    # On-path utilities
    uRet = RetUtil(b, w1, w2, X, rateSup1, rateSup2)
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    uS1, uS2 = SupUtil(q1, w1, invest1, rateSup1, Y), SupUtil(q2, w2, invest2, rateSup2, Y)
    # Check on-path
    if (uRet < -tol):
        checkBool, violCode = False, 0
    if (uS1 < -tol):
        checkBool, violCode = False, 1
    if (uS2 < -tol):
        checkBool, violCode = False, 2
    # IC conditions
    wVec = np.arange(0.005, 1, 0.005)
    if checkBool:  # Check S1 on-path, but only for lower prices
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt-w1) > tol and SupUtil(q1Alt, w1Alt, invest1, rateSup1, Y) > uS1+tol and\
                RetUtil(b, w1Alt, w2, X, rateSup1, rateSup2) > -tol and w1Alt < w1:
                checkBool, violCode = False, 3
    if checkBool: # Check S2 on-path, but only for lower prices
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2, rateSup2, Y) > uS2+tol and\
                RetUtil(b, w1, w2Alt, X, rateSup1, rateSup2) > -tol and w2Alt < w2:
                checkBool, violCode = False, 4
    if checkBool:  # Check S1 off-path
        for w1Alt in wVec:
            q1Alt = q1Opt(w1Alt, w2, b)
            if np.abs(w1Alt - w1) > tol and SupUtil(q1Alt, w1Alt, invest1off, rateSup1off, Y) > uS1+tol and\
                RetUtil(b, w1Alt, w2, X, rateSup1off, rateSup2) > -tol:
                checkBool, violCode = False, 5
                if printUpdate:
                    print('S1 off-path move found for w1='+str(w1Alt))
    if checkBool: # Check S2 off-path
        for w2Alt in wVec:
            q2Alt = q2Opt(w1, w2Alt, b)
            if np.abs(w2Alt-w2) > tol and SupUtil(q2Alt, w2Alt, invest2off, rateSup2off, Y)>uS2+tol and\
                RetUtil(b, w1, w2Alt, X, rateSup1, rateSup2off) > -tol:
                checkBool, violCode = False, 6
                if printUpdate:
                    print('S2 off-path move found for w2='+str(w2Alt))
    if checkBool:  # Check for nan values
        if np.isnan(w1) or np.isnan(w2):
            checkBool, violCode = False, 7
            if printUpdate:
                print('On-path prices are not real-valued')
    return checkBool, violCode, w1, w2, uRet, uS1, uS2


### DERIVED BOUNDARIES
# def LLbdY(scDict):
#     # LLFOC UB in Y


### PLOTTING FUNCTIONS
def CthetaCbetaMatsForPlot(numpts, Ctheta_max, Cbeta_max, scDict):
    # Generate list of equilibria matrices for plotting
    CthetaVec = np.arange(0, Ctheta_max, (Ctheta_max)/numpts)
    CbetaVec = np.arange(0, Cbeta_max, (Cbeta_max) / numpts)

    eq_list = ['LLFOC', 'LLsqz', 'LHexp-IC', 'LHexp-ICAtCost', 'LHexp-IR', 'LHFOC',
               'LHsqz', 'LHsqzAtCost', 'HHexp-IC', 'HHexp-IR', 'HHFOC']
    eqStrat_matList = np.zeros((len(eq_list), CthetaVec.shape[0], CbetaVec.shape[0]))
    eqStrat_matList[:] = np.nan

    # Fixed Ctheta and Cbeta bounds
    # CthLLUB, CthLHUB = CthetaLLFOCUB(scDict), CthetaLHFOCUB(scDict)
    # CbeLHUB, CbeHHLB = CbetaLHFOCUB(scDict, 0), CbetaHHFOCLB(scDict, 0)

    for currCthetaind, currCtheta in enumerate(CthetaVec):
        # CbeLLUB = CbetaLLFOCUB(scDict, currCtheta)
        for currCbetaind, currCbeta in enumerate(CbetaVec):
            # HHFOC used for HHexp, LHFOC used for LHexp
            # HHFOCBool, LHFOCBool = CheckHHFOCLB(scDict, currCtheta, currCbeta), CheckLHFOCLB(scDict, currCtheta, currCbeta)
            if CheckLLFOC(scDict, currCtheta, currCbeta)[0]:  # LLFOC
                eqStrat_matList[0, currCthetaind, currCbetaind] = 1
            if CheckLLsqz(scDict, currCtheta, currCbeta)[0]:  # LLsqz
                eqStrat_matList[1, currCthetaind, currCbetaind] = 1
            if CheckLHexpIC(scDict, currCtheta, currCbeta)[0]:  # LHexp-IC
                eqStrat_matList[2, currCthetaind, currCbetaind] = 1
            if CheckLHexpICAtCost(scDict, currCtheta, currCbeta)[0]:  # LHexp-ICAtCost
                eqStrat_matList[3, currCthetaind, currCbetaind] = 1
            # if CheckLHexpIR(scDict, currCtheta, currCbeta)[0]:  # LHexp-IR
            #     eqStrat_matList[4, currCthetaind, currCbetaind] = 1
            if CheckLHFOC(scDict, currCtheta, currCbeta)[0]:  # LHFOC
                eqStrat_matList[5, currCthetaind, currCbetaind] = 1
            if CheckLHsqz(scDict, currCtheta, currCbeta)[0]:  # LHsqz
                eqStrat_matList[6, currCthetaind, currCbetaind] = 1
            if CheckLHsqzAtCost(scDict, currCtheta, currCbeta)[0]:  # LHsqzAtCost
                eqStrat_matList[7, currCthetaind, currCbetaind] = 1
            if CheckHHexpIC(scDict, currCtheta, currCbeta)[0]:  # HHexp-IC
                eqStrat_matList[8, currCthetaind, currCbetaind] = 1
            # if CheckHHexpIR(scDict, currCtheta, currCbeta)[0]:  # HHexp-IR
            #     eqStrat_matList[9, currCthetaind, currCbetaind] = 1
            if CheckHHFOC(scDict, currCtheta, currCbeta)[0]:  # HHFOC
                eqStrat_matList[10, currCthetaind, currCbetaind] = 1

    return eqStrat_matList

def RemoveDominatedEq(eqStrat_matList, Xmax, Ymax, scDict):
    # Removes dominated equilibria from an X-Y equilibrium matrix, where domination is indicated by the existence of
    #   an alternate equilibria where both suppliers get at least the same utility
    retMat = eqStrat_matList.copy()
    numEq, numpts = eqStrat_matList.shape[0], eqStrat_matList.shape[1]
    Xvec, Yvec = np.arange(0, Xmax, (Xmax) / numpts), np.arange(0, Ymax, (Ymax) / numpts)
    for Xind, currX in enumerate(Xvec):
        for Yind, currY in enumerate(Yvec):
            tempEqList, tempUtilList = np.where(eqStrat_matList[:, Xind, Yind] == 1)[0], []
            if len(tempEqList) > 1:
                for currEq in tempEqList:  # Store all utilities from check functions
                    if currEq == 0:
                        tempUtilList.append(CheckLLFOC(scDict, currX, currY))
                    elif currEq == 1:
                        tempUtilList.append(CheckLLsqz(scDict, currX, currY))
                    elif currEq == 2:
                        tempUtilList.append(CheckLHexpIC(scDict, currX, currY))
                    elif currEq == 3:
                        tempUtilList.append(CheckLHexpICAtCost(scDict, currX, currY))
                    elif currEq == 4:
                        tempUtilList.append(CheckLHexpIR(scDict, currX, currY))
                    elif currEq == 5:
                        tempUtilList.append(CheckLHFOC(scDict, currX, currY))
                    elif currEq == 6:
                        tempUtilList.append(CheckLHsqz(scDict, currX, currY))
                    elif currEq == 7:
                        tempUtilList.append(CheckLHsqzAtCost(scDict, currX, currY))
                    elif currEq == 8:
                        tempUtilList.append(CheckHHexpIC(scDict, currX, currY))
                    elif currEq == 9:
                        tempUtilList.append(CheckHHexpIR(scDict, currX, currY))
                    elif currEq == 10:
                        tempUtilList.append(CheckHHFOC(scDict, currX, currY))
                for currEqInd, currEq in enumerate(tempEqList):  # Compare all utility sets
                    util1, util2 = tempUtilList[currEqInd][5], tempUtilList[currEqInd][6]
                    validEq = True  # Initialize
                    for compEqInd, compEq in enumerate(tempEqList):
                        if not compEqInd==currEqInd:
                            compUtil1, compUtil2 = tempUtilList[compEqInd][5], tempUtilList[compEqInd][6]
                            if (compUtil1 > util1 and compUtil2 >= util2) or (compUtil1 >= util1 and compUtil2 > util2):
                                validEq = False
                    if not validEq:  # Remove from return array
                        retMat[currEq, Xind, Yind] = np.nan
    return retMat

b, cSup, supRateLo, supRateHi = 0.8, 0.05, 0.8, 1.0
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo, 'supRateHi': supRateHi}
numpts = 30

# Ctheta_max, Cbeta_max = 1.2*CthetaHHFOCLBForNoCbeta(scDict), 1.2*CbetaHHFOCLB(scDict, 0)
Ctheta_max, Cbeta_max = 1.3, 0.4

CthetaVec = np.arange(0, Ctheta_max, (Ctheta_max)/numpts)
CbetaVec = np.arange(0, Cbeta_max, (Cbeta_max)/numpts)

eqMats = CthetaCbetaMatsForPlot(numpts, Ctheta_max, Cbeta_max, scDict)
nondomEqMats = RemoveDominatedEq(eqMats, Ctheta_max, Cbeta_max, scDict)

alval = 0.7

fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$c_S=$'+str(cSup)+', '+r'$L=$'+str(supRateLo),
             fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['#DC143C', 'deeppink', '#13EAC9', '#C0C0C0', '#7FFFD4', '#030764', '#0343DF', '#7BC8F6',
            '#9ACD32', '#AAFF32', '#054907']
labels = ['LLFOC', 'LLsqz', 'LHexp-IC', 'LHexp-ICAtCost', 'LHexp-IR', 'LHFOC', 'LHsqz', 'LHsqzAtCost', 'HHexp-IC', 'HHexp-IR', 'HHFOC']

imlist = []
for eqind in range(len(labels)):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[eqind]], name='from_list', N=None)
    im = ax.imshow(nondomEqMats[eqind].T, vmin=0, vmax=1, aspect='auto',
                            extent=(0, Ctheta_max, 0, Cbeta_max),
                            origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)

legwidth = 12
wraplabels = ['\n'.join(textwrap.wrap(labels[i], width=legwidth)) for i in range(len(labels))]
patches = [mpatches.Patch(color=eqcolors[i], label=wraplabels[i], alpha=alval) for i in range(len(eqcolors))]
          # +[mpatches.Patch(hatch=r'/////////',fill=False,linewidth=0,snap=False,label='1-sup. eq.')]

# put those patched as legend-handles into the legend
ax.legend(handles=patches, bbox_to_anchor=(1.32, 1.0), loc='upper right', borderaxespad=0.1, fontsize=8)
ax.set_xbound(0, Ctheta_max)
ax.set_ybound(0, Cbeta_max)
ax.set_box_aspect(1)
plt.xlabel(r'$X$', fontsize=14)
plt.ylabel(r'$Y$', fontsize=14, rotation=0, labelpad=14)
plt.show()

