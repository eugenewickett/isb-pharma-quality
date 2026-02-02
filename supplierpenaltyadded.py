# Extension to basic model of "Strategic Role of Inspections in Pharmaceutical Supply Chains"
# We add supplier-inspection penalties and a high-quality level (H) that is possibly less than 1.0
# These functions use the simpler version of the model
#   - perfect diagnostic, no retailer quality choice, retailer sources from both suppliers or neither
# 15-JAN-25

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

def SPUtil(q1, q2, lambsup1, lambsup2, alph):
    # Social planner's utility
    return q1*(alph+(lambsup1)-1) + q2*(alph+(lambsup2)-1)

def SocWel(uH, uL, w1, w2, cS1, cS2, lambsup1, lambsup2, Ctheta, Cbeta):
    # Social welfare
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    return q1*(uH*(lambsup1)+uL*(1-lambsup1)-cS1) + q2*(uH*(lambsup2)+uL*(1-lambsup2)-cS2) -\
        Ctheta*(1-lambsup1*lambsup2) - Cbeta*(1-lambsup1) - Cbeta*(1-lambsup2)

def SocWelIgnoreCosts(uH, uL, w1, w2, cS1, cS2, lambsup1, lambsup2, Ctheta, Cbeta):
    # Social welfare
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    return q1*(uH*(lambsup1)+uL*(1-lambsup1)) + q2*(uH*(lambsup2)+uL*(1-lambsup2))

def SocWelIgnorePens(uH, uL, w1, w2, cS1, cS2, lambsup1, lambsup2, Ctheta, Cbeta):
    # Social welfare
    q1, q2 = q1Opt(w1, w2, b), q2Opt(w1, w2, b)
    return q1*(uH*(lambsup1)+uL*(1-lambsup1)-cS1) + q2*(uH*(lambsup2)+uL*(1-lambsup2)-cS2)

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

def SupPriceLL(scDict, Ctheta):
    # Returns on-path LL prices
    b, cS, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    w1 = max(0, (1 - b)/(2 - b))
    w2 = max(0, (1 - b)/(2 - b))
    return w1, w2

def SupPriceHH(scDict, Ctheta):
    # Returns on-path HH prices
    b, cS, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    w1 = max(0, (1 - b + cS) / (2 - b))
    w2 = max(0, (1 - b + cS) / (2 - b))
    return w1, w2

def SupPriceLHFOC(scDict, Ctheta):
    # Returns on-path LH prices
    b, cS, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateLo']
    w1 = max(0, (2 - b - (b**2) + b*cS)/(4 - (b**2)))
    w2 = max(0, (2 - b - (b**2) + 2*cS)/(4 - (b**2)))
    return w1, w2

def SupPriceLHsqz(scDict, Ctheta):
    # Returns on-path LHsqz prices
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    radterm = sqroot((-1 + (b**2))*(((1 - cS)**2) - 4*(-4 + 3*(b**2))*Ctheta*(-1 + supRateHi*supRateLo)))
    w1 = (b*(1 + 3*b - cS) + 2*(-2 + radterm))/(-4 + 3*(b**2))
    w2 = (-2*(1 + cS) + (b**2)*(2 + cS) + b*radterm)/(-4 +3*(b**2))
    if w2 < cS:
        w2 = cS
        w1 = 1 - b + cS*b + ((b**2) - 1)*sqroot((1 + (-2 + cS)*cS + 4*Ctheta *(-1 + supRateHi*supRateLo))/(-1 + (b**2)))
    return w1, w2

def SupPriceLLsqz(scDict, Ctheta):
    # Returns on-path LLsqz prices
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1 = 1 - sqroot(-2*((1 + b)*Ctheta*(-1 + (supRateLo**2))))
    w2 = w1
    return max(w1, 0), max(w2, 0)

def SupPriceHHsqz(scDict, Ctheta):
    # Returns on-path HHsqz prices
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1 = 1 - sqroot(-2*((1 + b)*Ctheta*(-1 + (supRateHi**2))))
    w2 = w1
    return max(w1, 0), max(w2, 0)

def funcLHexpRetPrice(x, scDict, Ctheta, Cbeta):
    # When only S2 wants to move off-path; S1 chooses price s.t. S2 is indifferent, S2 chooses FOC
    # Retailer-penalty induced
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    term1 = -1*(((cS - x[1])*(-1 + b - b*x[0] + x[1])) / (2*(-1 + (b**2)))) + Cbeta*(-1 + supRateHi)
    radterm1 = sqroot((((-1 + x[0])**2) + 4* Ctheta*(-1 + (supRateLo**2)))/(-1 + (b**2)))
    term2 = 0.5*(((-1 + x[0])**2) + 2* Cbeta*(-1 +supRateLo) + 4 *Ctheta* (-1 + (supRateLo**2)) + (1 - b +
      b * x[0])*radterm1)
    retval1 = term1 - term2
    retval2 = x[1] - (0.5*(1 + cS + b*(-1 + x[0])))
    return [retval1, retval2]

def SupPriceLHexpRet(scDict, Ctheta, Cbeta, tol=1E-8):
    # Returns on-path LHexp prices
    xinit = SupPriceLHFOC(scDict, 0)
    root, infodict, _, _ = fsolve(funcLHexpRetPrice, [xinit[0], xinit[1]], args=(scDict, Ctheta, Cbeta), full_output=True)
    if np.sum(np.abs(infodict['fvec'])) > tol or np.isnan(infodict['fvec'][0]):
        root= [-1, -1]
    return root[0], root[1]

def funcHHexpRetPrice(x, scDict, Ctheta, Cbeta):
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

def SupPriceHHexpRet(scDict, Ctheta, Cbeta, tol=1E-8):
    # Returns on-path HHexp prices
    xinit = SupPriceHH(scDict, 0)
    root, infodict, _, _ = fsolve(funcHHexpRetPrice, [xinit[0], xinit[1]], args=(scDict, Ctheta, Cbeta), full_output=True)
    if np.sum(np.abs(infodict['fvec'])) > tol or np.isnan(infodict['fvec'][0]):
        root= [-1, -1]
    return root[0], root[1]

def SupPriceLHexpSup(scDict, Ctheta, Cbeta):
    # Returns on-path LHsqz prices, if the off-path prices accounted for are RetIR-valid
    b, cS, supRateLo, supRateHi = scDict['b'], scDict['cSup'], scDict['supRateLo'], scDict['supRateHi']
    w1 = (2*(-1 + b)*cS + (cS**2) - 8*(-1 + (b**2))*Cbeta*(supRateHi - supRateLo))/(2*b*cS)
    w2 = (3*cS)/4 - (2*(-1 + (b**2))*Cbeta*(supRateHi - supRateLo))/cS
    # Is the off-path price RetIR-valid?
    w2off = 0.5* (1 - b + b*w1)
    if RetUtil(b, w1, w2off, Ctheta, supRateLo, supRateLo) < 0:  # Use LHexpRet prices instead
        w1, w2 = SupPriceLHexpRet(scDict, Ctheta, Cbeta)
    return max(w1, 0), max(w2, 0)

def SupPriceHHexpSup(scDict, Ctheta, Cbeta):
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
    if (Ctheta >= CthetaLHFOCUB(scDict)) and (RetUtil(b, w1off, w2, Ctheta, supRateLo, supRateHi)<0 or np.isnan(w1off)):
        w1, w2 = SupPriceHHexpRet(scDict, Ctheta, Cbeta, tol=1E-8)
    if w1 < cS:
        w1, w2 = cS, cS

    return w1, w2



def CbetaLLFOCUB(scDict, Ctheta):
    # LL FOC UB in Cbeta
    # Depends on whether off-path prices violate retailer IR
    # off-path prices must also yield positive order quantities (esp under high b)
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    # First check if supIC off-path price violates retailer IR; WLOG S2 deviates
    w1on, w2on = SupPriceLL(scDict, Ctheta)
    w2off = 0.5*(2 + 2/(-2 + b) + cS)
    if q2Opt(w1on, w2off, b) <= 0 or w2off < cS:  # The profit-maximizing off-path move yields negative order quantities
        # SupIR bound is used instead
        retval = (-1 + b)/(2*((-2 + b)**2)*(1 + b)*(-1 + supRateLo))
    else:
        if RetUtil(b, w1on, w2off, Ctheta, supRateLo, supRateHi) < 0:  # Use retIR-based bound
            radterm = sqroot((1 + 4*((-2 + b)**2)*Ctheta*(-1+supRateHi*supRateLo))/(((-2+b)**2)*(-1+(b**2))))
            retval = -((((-1 + b)**2) + (-2 + b)*(-1 + (b**2))*radterm*(2+2*(-2 + b) - (-2 + b)*cS +(-2 + b)*(-1 +
                        (b**2))*radterm))/(2*((-2 + b)**2)*(-1 + (b**2))*(supRateHi - supRateLo)))
        else:  # Use supIC-based bound
            retval = (((-1 + (b**2))**2)*((4*(-1+b)*cS)/((-2+b)*((-1 + (b**2))**2))-(cS**2)/((-1+(b**2))**2)))/(8*(-1+
                        b)*(1+b)*(-1*supRateHi+supRateLo))

    return retval

def CthetaLLFOCUB(scDict):
    # LL FOC UB in Ctheta
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    retval = 1/(2*((2 - b)**2)*(1 + b)*(1 - (supRateLo**2)))
    return retval

def CbetaLHFOCUB(scDict, Ctheta):
    # LH FOC UB in Cbeta
    # S1 moves to H at this bound; doesn't depend on Ctheta
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    retval = (cS*(8 - 4*b*(1 + b)-4*cS + b*(4+b)*cS))/(8*(4 - 5*(b**2) + (b**4))*(supRateHi-supRateLo))
    return retval

def CthetaLHFOCUB(scDict):
    # LH FOC UB in Ctheta
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    retval = (8 + 2*(b**3)*(-1 + cS) + 4*(-2 + cS)*cS - 3*(b**2)*(2 + (-2 + cS)*cS))/(4*((-4 + (b**2))**2)*(-1 +\
                (b**2))*(-1 + supRateHi*supRateLo))
    return retval

def CheckLHFOCLB(scDict, Ctheta, Cbeta):
    # Returns True if Ctheta/Cbeta combination is sufficiently *high* to sustain LH FOC; else returns False
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    # Initialize as True
    retBool = True
    # First check if Cbeta sufficiently low for S2 to want to move
    CbetaLB1 = (cS*(8 - 4*cS + b*(-4 + b*(-4 + 3*cS))))/(8*(4 - 5*(b**2) + (b**4))*(supRateHi - supRateLo))
    CthetaLB1 = (32 + b*(-8*b*(3+b)+4*(-1 + b)*((2 + b)**2)*cS+b*(4 - 3*(b**2))*(cS**2)))/(16*((-4 + (b**2))**2)*(-1+\
                    (b**2))*(-1 + (supRateLo**2)))
    if Cbeta <= CbetaLB1 and Ctheta <= CthetaLB1:  # There exists feasible and desired S2 move to L
        retBool = False
    # Check if off-path IR move is desirable
    if retBool is True:
        radterm = sqroot((4 - 64*Ctheta + b*(4 - 4*cS + b*(((-1 + cS)**2) - 4*(-8 + (b**2))*Ctheta)) +\
                    4*((-4 + (b**2))**2)*Ctheta*(supRateLo**2))/(((-4 + (b**2))**2)*(-1 + (b**2))))
        Cbetaterm = (((-2 + b + (b**2) + 2*cS - (b**2)*cS)**2)/((-4 + (b**2))**2) + (-1 + (b**2))*radterm*(2 +\
                    2/(-2+b)+((b**2)*cS)/(4 - (b**2))+(-1 +(b**2))*radterm))/(2*(-1 + (b**2))*(supRateHi - supRateLo))
        if Cbeta < Cbetaterm:
            retBool = False

    return retBool

def CbetaHHFOCLB(scDict, Ctheta):
    # HH FOC LB in Cbeta
    # Both suppliers move to H beyond this bound; doesn't depend on Ctheta
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    retval = (cS*(4 - 2*cS + b*(-4 + 3*cS)))/(8*(-2 + b)*(-1 + b)*(1 + b)*(supRateHi - supRateLo))
    return retval

def CthetaHHFOCLBForNoCbeta(scDict):
    # HH FOC LB in Ctheta when Cbeta = 0
    # Facilitates statement of Pareto-type HHFOC frontier
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    retval = (-1*(((-4 + b*(4 + cS*(4 - 4*cS + b*(-4 + 3*cS))))*(-1 + supRateHi*supRateLo))/(((-2 + b)**2)*(-1 +
                (b**2)))) + np.sqrt(-1*((((2 + b*(-2 + cS))**2)*cS*(4 - 2*cS + b*(-4 + 3*cS))*((-1 +
                supRateHi*supRateLo)**2))/(((-2 + b)**3)*((-1 +(b**2))**2)))))/(8*((-1 + supRateHi*supRateLo)**2))
    return retval

def CheckHHFOCLB(scDict, Ctheta, Cbeta):
    # Returns True if Ctheta/Cbeta combination is sufficiently *high* to sustain HH FOC; else returns False
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    # Initialize as True
    retBool = True
    # First check if Cbeta sufficiently low for S2 to want to move
    CbetaLB1 = CbetaHHFOCLB(scDict, 0)
    if Cbeta > CbetaLB1:
        retBool = False
    CthetaLB1 = (8 + 4*b*(-2 + cS) + (b**2)*(4 - 3*cS)*cS + 4*(-2 + cS)*cS)/(16*((-2 + b)**2)*(-1 +
                    (b**2))*(-1 + supRateHi*supRateLo))
    if Cbeta <= CbetaLB1 and Ctheta <= CthetaLB1:  # There exists feasible and desired S2 move to L
        retBool = False
    # Check if off-path IR move is desirable
    if retBool is True:
        radterm = sqroot((1+(-2+cS)*cS+4*((-2+b)**2)*Ctheta*(-1+supRateHi*supRateLo))/(((-2 + b)**2)*(-1 + (b**2))))
        Cbetaterm = 1/(2*((-2 + b)**2)*(1 + b)*(supRateHi - supRateLo))*(2*b*(-1 + cS)*(-1 + cS + radterm) +
                    4*(4*Ctheta*(-1 + supRateHi*supRateLo) + radterm) + (b**2)*(-12*Ctheta*(-1 + supRateHi* supRateLo)+
                    (-4 + cS)*radterm) + (b**3)*(4*Ctheta*(-1 + supRateHi*supRateLo) - (-2 + cS)*radterm))
        if Cbeta < Cbetaterm:
            retBool = False
    return retBool

def CheckLLsqz(scDict, Ctheta, Cbeta):
    # Returns True if LLsqz is valid for the given Ctheta/Cbeta combination
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    # Initialize return value
    retBool = True
    # Get on-path prices
    w1on = 1 - sqroot(-2*((1 + b)*Ctheta*(-1 +(supRateLo**2))))
    if w1on < 0:  # Ctheta too large
        retBool = False
    # Only valid if LLFOC invalid
    if retBool is True and Ctheta <= CthetaLLFOCUB(scDict):
        retBool = False
    # Is supplier IR met?
    if retBool is True:
        if SupUtil(q1Opt(w1on, w1on, b), w1on, 0, supRateLo, Cbeta) < 0:
            retBool = False
    # Check if suppliers have incentive to move using SupIC condition first
    if retBool:
        # First check that Cbeta is high enough s.t. a move to H is preferred (ignoring retailer)
        radterm = sqroot(-2*((1 + b)*Ctheta*(-1 + (supRateLo**2))))
        Cbetaterm = (((-1 + cS)**2) - 2*((-2 + b)**2)*(1 + b)*Ctheta*(-1 + (supRateLo**2))-4*radterm+2*b*radterm +
                     2*b*cS*radterm)/(8*(-1 + (b**2))*(supRateHi - supRateLo))
        # Check that off-path retailer IR is met; w2off cannot be less than cS
        radterm2 = sqroot((1 - 8*(-1 + (b**2))*Cbeta*(supRateHi - supRateLo) - 2*((-2 + b)**2)*(1 + b)*Ctheta*(-1 +
                            (supRateLo**2)) - 4*radterm + 2*b*radterm + cS*(-2 + cS +2*b*radterm))/((-1 + (b**2))**2))
        w2off = 0.5*(1 + cS - b*radterm-radterm2+(b**2)*radterm2)
        # If w2off < cS, try other valid off-path price
        if w2off < cS:
            w2off = 0.5 * (1 + cS - b * radterm + radterm2 - (b ** 2) * radterm2)
        retutil = RetUtil(b, w1on, w2off, Ctheta, supRateHi, supRateLo)
        if Cbeta > Cbetaterm and retutil >= 0 and q2Opt(w1on, w2off, b) > 0:  # Off-path move valid
            retBool = False
    # Check if RetIR price is preferred
    if retBool:
        radterm3 = sqroot((Ctheta*(-1 + b+2*supRateHi*supRateLo - (1 +b)*(supRateLo**2)))/(-1 + (b**2)))
        CbetatermRetIR = (1/(2*(1 +b)*(supRateHi - supRateLo)))*(-2*b*(1 +b)*Ctheta-4*(1 + b)*Ctheta*(supRateHi -
                            supRateLo) *supRateLo + 2*b*(1 + b)*Ctheta*(supRateLo**2) + sqroot(-2*((1 + b)*Ctheta*(-1+
                            (supRateLo**2)))) - sqroot(2)*(1 + b)* radterm3 + sqroot(2)*(1 + b)*cS*radterm3 +
                            2*b*(1 + b)*sqroot(-1*((1 + b)*Ctheta*(-1 + (supRateLo**2))))*radterm3)
        # Off-path price for S2 in this case
        w2off = 1 - sqroot(2*(Ctheta*(-1 + b +2*supRateHi*supRateLo - (1 +b)*(supRateLo**2))) / (-1 +
                (b**2))) + sqroot(2)*b*(-1*sqroot(-1*((1 + b)*Ctheta*(-1 + (supRateLo**2)))) + b*
                sqroot((Ctheta*(-1 + b +2*supRateHi*supRateLo - (1 +b)*(supRateLo**2))) / (-1 + (b**2))))
        if w2off < cS:
            w2off = 1 + sqroot(2*(Ctheta*(-1 + b +2*supRateHi*supRateLo - (1 +b)*(supRateLo**2))) / (-1 +
                (b**2))) - sqroot(2)*b*(-1*sqroot(-1*((1 + b)*Ctheta*(-1 + (supRateLo**2)))) + b*
                sqroot((Ctheta*(-1 + b +2*supRateHi*supRateLo - (1 +b)*(supRateLo**2))) / (-1 + (b**2))))
        if Cbeta > CbetatermRetIR and w2off >= cS:
            retBool = False
    return retBool

def CheckLHsqz(scDict, Ctheta, Cbeta):
    # Returns True if LHsqz is valid for the given Ctheta/Cbeta combination
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    # Initialize return value
    retBool = True
    # Get on-path prices
    w1on, w2on = SupPriceLHsqz(scDict, Ctheta)
    LHsqzExt = False
    if w2on == cS:  # We're in LHsqzExt
        LHsqzExt = True
    if w1on < 0:  # Invalid
        retBool = False
    # Check if S1 has incentive to move to H
    if not LHsqzExt:
        radterm1 = sqroot((-1 + (b**2))*(1 + (-2 + cS)*cS -4*(-4 + 3*(b**2))*Ctheta*(-1 + supRateHi*supRateLo)))
        CbetaUB = (1 / (4*((4 - 3*(b**2))**2)*(-1 + (b**2))*(supRateHi - supRateLo)))*(-8*b*(-1 +
                    cS)*cS - 6*(b**8)*Ctheta*(-1 +supRateHi*supRateLo) +(b**6)*(1 + (-2 + cS)*cS +
                    62*Ctheta*(-1 +supRateHi*supRateLo)) +(b**5)*(-1 + cS)*(-3 - 3*cS + radterm1) -
                    16*(-8*Ctheta + 8*Ctheta*supRateHi*supRateLo + radterm1) - 2*(b**3)*(-1 + cS)*(-2-5*cS +
                    2*radterm1) - 3*(b**4)*(72*Ctheta*(-1 +supRateHi*supRateLo) + (1 + cS)*radterm1) +
                    4*(b**2)*(72*Ctheta*(-1 + supRateHi*supRateLo) + (4 + cS)*radterm1))
    if LHsqzExt:
        radterm1 = sqroot((1 + (-2 + cS)*cS - 4*Ctheta + 4*Ctheta*supRateHi*supRateLo)/(-1 + (b**2)))
        CbetaUB = (3+3*(-2 + cS)*cS + 16*Ctheta*(-1 + supRateHi*supRateLo) + 4* radterm1 + 4*(b**2)*(-1+cS)*radterm1 +
                   b*(5 +16*Ctheta*(-1 + supRateHi*supRateLo) +cS*(-10 + 5*cS +4*radterm1)))/(8*(1 +
                   b)*(supRateHi - supRateLo))
    if Cbeta >= CbetaUB:
        retBool = False
    # Check if S2 has incentive to move using SupIC condition first
    if retBool and not LHsqzExt:
        radterm2 = sqroot((-1 + (b**2))* (1 + (-2 + cS)*cS -4* (-4 + 3* (b**2))* Ctheta* (-1 + supRateHi*supRateLo)))
        CbetaLB1 = (cS*(8 - 8*(b**2) - 4*cS + 5*(b**2)*cS -4*b*radterm2))/(8*(4 - 7*(b**2) +
                        3*(b**4))*(supRateHi - supRateLo))
        # Check that off-path retailer IR is met
        radterm3 = sqroot((-1 + (b**2))*(1 + (-2 + cS)*cS - 4*(-4+3*(b**2))*Ctheta*(-1 + supRateHi*supRateLo)))
        w2off = (1/6)*(4 - (4*(-1 + cS))/(-4 + 3*(b**2)) - cS + (6*b*radterm3)/(-4 +3* (b**2)))
        retutil = RetUtil(b, w1on, w2off, Ctheta, supRateLo, supRateLo)
        if Cbeta <= CbetaLB1 and retutil >= 0:  # Off-path move valid
            retBool = False
    # Check if RetIR price is preferred
    if retBool and not LHsqzExt:
        radterm4 = sqroot((-1 + (b**2))*(1 + (-2 + cS)*cS - 4*(-4+3*(b**2))*Ctheta*(-1 + supRateHi*supRateLo)))
        CbetaLB2 = (1/(2*(-1 +(b**2))*(supRateHi - supRateLo)))*(((2*(-1 + cS) +b*(2*b -2*b*cS +radterm4))**2)/((4 -
                    3*(b**2))**2) + (-1 + (b**2))* sqroot(((1/(((4 - 3*(b**2))**2)*(-1 + (b**2)))))*(-4 -
                    4*(-2 + cS)*cS +64*Ctheta*supRateLo*(-1*supRateHi + supRateLo) +12*(b**4)*Ctheta*(1 -
                    4*supRateHi*supRateLo +3*(supRateLo**2)) -4* b* (-1 +cS)*radterm4 + (b**2)*(5 + 5*(-2 + cS)*cS +
                    16*Ctheta*(-1 +7*supRateHi*supRateLo - 6*(supRateLo**2)))))*((4 - 4*cS)/(-12 + 9*(b**2)) +
                    (4 - cS)/3 + (2*b*radterm4)/(-4 +3*(b**2)) + (-1 + (b**2))* sqroot(((1/(((4 - 3*(b**2))**2)*(-1 +
                    (b**2)))))*(-4 -4*(-2 + cS)*cS +64*Ctheta*supRateLo*(-1*supRateHi + supRateLo) +
                    12*(b**4)*Ctheta*(1 -4*supRateHi*supRateLo +3*(supRateLo**2)) -4* b* (-1 +cS)*radterm4 +
                    (b**2)*(5 + 5*(-2 + cS)*cS +16*Ctheta*(-1 +7*supRateHi*supRateLo - 6*(supRateLo**2)))))))
        if Cbeta <= CbetaLB2:
            retBool = False
    # Now for LHsqzExt
    if retBool and LHsqzExt:
        radterm5 = sqroot((1 + (-2 + cS)*cS + 4*Ctheta*(-1 + supRateHi*supRateLo))/(-1 + (b**2)))
        CbetaLB3 = -1*(((1 + b*(-1*radterm5 + b* (-1 + cS +b*radterm5)))**2)/(8*(-1 + (b**2))*(supRateHi - supRateLo)))
        # Check that off-path retailer IR is met
        radterm6 = sqroot((1 + (-2 + cS)*cS +4*Ctheta*(-1 + supRateHi*supRateLo))/(-1 +(b**2)))
        w2off = 0.5*(1 +(b**2)*(-1 + cS) - (b - (b**3))*radterm6)
        retutil = RetUtil(b, w1on, w2off, Ctheta, supRateLo, supRateLo)
        if Cbeta <= CbetaLB3 and retutil >= 0:  # Off-path move valid
            retBool = False
    # Check if RetIR price is preferred
    if retBool and LHsqzExt:
        radterm7 = sqroot((1 + (-2 + cS)*cS +4*Ctheta*(-1 +supRateHi*supRateLo))/(-1 + (b**2)))
        radterm8 = sqroot((-1 - (-2 + cS)*cS +4*Ctheta*supRateLo*(-1*supRateHi + supRateLo)+2*(b**2)*(1 + (-2+cS)*cS +
                    2*Ctheta*(-1+supRateHi*supRateLo))-2*b*(-1+cS)*radterm7+2*(b**3)*(-1+cS)*radterm7)/(-1 + (b**2)))
        CbetaLB4 = (1/(2*(supRateHi - supRateLo)))*radterm8*(1 - radterm8 +b*(-1*radterm7 + b*(-1 + cS +
                    b*radterm7 + radterm8)))
        if Cbeta <= CbetaLB4:
            retBool = False

    return retBool

def CheckHHexp(scDict, Ctheta, Cbeta, wIncr = 0.01):
    # Returns True if HHexp is valid for the given Ctheta/Cbeta combination
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    # Initialize return value
    retBool = True
    # Get RetIR-valid SupIC HHexp prices
    w1on, w2on = SupPriceHHexpSup(scDict, Ctheta, Cbeta)
    if w1on < cS or np.isnan(w1on):  # Only need to check if these prices are SupIC-valid; existence otherwise indicates HHexp is possible
        retBool = False
    # Check off-path moves
    if retBool is True:
        w1offVec = np.arange(wIncr, 1.0, wIncr)
        contLoop = True
        for w1offInd, w1off in enumerate(w1offVec):
            if contLoop:
                utilOff = SupUtil(q1Opt(w1off, w2on, b), w1off, 0, supRateLo, Cbeta)
                utilOn = SupUtil(q1Opt(w1on, w2on, b), w1on, cS, supRateHi, Cbeta)
                utilOffRet = RetUtil(b, w1off, w2on, Ctheta, supRateLo, supRateHi)
                if utilOff > utilOn and utilOffRet >=0:  # Off-path moves exists
                    retBool = False
                    contLoop = False

    return retBool

def CheckLHexp(scDict, Ctheta, Cbeta):
    # Returns True if LHexp is valid for the given Ctheta/Cbeta combination
    b, cS, supRateHi, supRateLo = scDict['b'], scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    # Initialize return value
    retBool = True
    # Get RetIR-valid SupIC LHexp prices
    w1on, w2on = SupPriceLHexpSup(scDict, Ctheta, Cbeta)
    if np.isnan(w1on):
        retBool = False
    # Check on-path RetIR
    if RetUtil(b, w1on, w2on, Ctheta, supRateLo, supRateHi) < 0:
        retBool = False
    # S1 compares with other 'available' on-path equilibria
    s1utilOn = SupUtil(q1Opt(w1on, w1on, b), w1on, 0, supRateLo, Cbeta)
    if CheckLLsqz(scDict, Ctheta, Cbeta):
        w1LLsqz, w2LLsqz = SupPriceLLsqz(scDict, Ctheta)
        if SupUtil(q1Opt(w1LLsqz, w2LLsqz, b), w1LLsqz, 0, supRateLo, Cbeta) > s1utilOn:
            retBool = False
    if Ctheta < CthetaLLFOCUB(scDict) and Cbeta < CbetaLLFOCUB(scDict, Ctheta):
        w1LL, w2LL = SupPriceLL(scDict, Ctheta)
        if SupUtil(q1Opt(w1LL, w2LL, b), w1LL, 0, supRateLo, Cbeta) > s1utilOn:
            retBool = False
    return retBool

def GetEqPriceList(scDict, Ctheta, Cbeta):
    # Returns a list of 8 sets of equilibrium prices
    LL1, LL2 = SupPriceLL(scDict, Ctheta)
    LLsqz1, LLsqz2 = SupPriceLLsqz(scDict, Ctheta)
    LHexpRet1, LHexpRet2 = SupPriceLHexpRet(scDict, Ctheta, Cbeta)
    LHexpSup1, LHexpSup2 = SupPriceLHexpSup(scDict, Ctheta, Cbeta)
    LHFOC1, LHFOC2 = SupPriceLHFOC(scDict, Ctheta)
    LHsqz1, LHsqz2 = SupPriceLHsqz(scDict, Ctheta)
    HHexpRet1, HHexpRet2 = SupPriceHHexpRet(scDict, Ctheta, Cbeta)
    HHexpSup1, HHexpSup2 = SupPriceHHexpSup(scDict, Ctheta, Cbeta)
    HH1, HH2 = SupPriceHH(scDict, Ctheta)
    HHsqz1, HHsqz2 = SupPriceHHsqz(scDict, Ctheta)
    retList = [(LL1, LL2), (LLsqz1, LLsqz2), (LHexpRet1, LHexpRet2), (LHexpSup1, LHexpSup2), (LHFOC1, LHFOC2),
               (LHsqz1, LHsqz2), (HHexpRet1, HHexpRet2), (HHexpSup1, HHexpSup2), (HH1, HH2), (HHsqz1, HHsqz2)]
    return retList

def CthetaCbetaMatsForPlot(numpts, Ctheta_max, Cbeta_max, scDict):
    # Generate list of equilibria matrices for plotting
    CthetaVec = np.arange(0, Ctheta_max, (Ctheta_max)/numpts)
    CbetaVec = np.arange(0, Cbeta_max, (Cbeta_max) / numpts)

    eq_list = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHexp', 'HH', 'HHsqz', 'N']
    eqStrat_matList = np.zeros((len(eq_list), CthetaVec.shape[0], CbetaVec.shape[0]))
    eqStrat_matList[:] = np.nan

    # Fixed Ctheta and Cbeta bounds
    CthLLUB, CthLHUB = CthetaLLFOCUB(scDict), CthetaLHFOCUB(scDict)
    CbeLHUB, CbeHHLB = CbetaLHFOCUB(scDict, 0), CbetaHHFOCLB(scDict, 0)

    for currCthetaind, currCtheta in enumerate(CthetaVec):
        CbeLLUB = CbetaLLFOCUB(scDict, currCtheta)
        for currCbetaind, currCbeta in enumerate(CbetaVec):
            # HHFOC used for HHexp, LHFOC used for LHexp
            HHFOCBool, LHFOCBool = CheckHHFOCLB(scDict, currCtheta, currCbeta), CheckLHFOCLB(scDict, currCtheta, currCbeta)
            if currCtheta < CthLLUB and currCbeta < CbeLLUB:  # LLFOC
                eqStrat_matList[0, currCthetaind, currCbetaind] = 1
            if CheckLLsqz(scDict, currCtheta, currCbeta):  # LLsqz
                eqStrat_matList[1, currCthetaind, currCbetaind] = 1
            if CheckLHexp(scDict, currCtheta, currCbeta) and not LHFOCBool:  # LHexp
                eqStrat_matList[2, currCthetaind, currCbetaind] = 1
            if LHFOCBool and currCtheta < CthLHUB and currCbeta < CbeLHUB:  # LHFOC
                eqStrat_matList[3, currCthetaind, currCbetaind] = 1
            if CheckLHsqz(scDict, currCtheta, currCbeta) and currCtheta >= CthLHUB:  # LHsqz
                eqStrat_matList[4, currCthetaind, currCbetaind] = 1
            if CheckHHexp(scDict, currCtheta, currCbeta) and currCbeta < CbeHHLB and not HHFOCBool:  # HHexp
                eqStrat_matList[5, currCthetaind, currCbetaind] = 1
            if currCbeta >= CbeHHLB or HHFOCBool:  # HHFOC
                eqStrat_matList[6, currCthetaind, currCbetaind] = 1

    return eqStrat_matList

def GetPricesFromEq(eq, scDict, Ctheta, Cbeta):
    # Returns set of prices for given argruments
    cS, supRateHi, supRateLo = scDict['cSup'], scDict['supRateHi'], scDict['supRateLo']
    if eq == 0:  # LLFOC
        w1, w2 = SupPriceLL(scDict, Ctheta)
        cS1, cS2 = 0, 0
        qual1, qual2 = supRateLo, supRateLo
    if eq == 1:  # LLsqz
        w1, w2 = SupPriceLLsqz(scDict, Ctheta)
        cS1, cS2 = 0, 0
        qual1, qual2 = supRateLo, supRateLo
    if eq == 2:  # LHexp
        w1, w2 = SupPriceLHexpSup(scDict, Ctheta, Cbeta)
        cS1, cS2 = 0, cS
        qual1, qual2 = supRateLo, supRateHi
    if eq == 3:  # LHFOC
        w1, w2 = SupPriceLHFOC(scDict, Ctheta)
        cS1, cS2 = 0, cS
        qual1, qual2 = supRateLo, supRateHi
    if eq == 4:  # LHsqz
        w1, w2 = SupPriceLHsqz(scDict, Ctheta)
        cS1, cS2 = 0, cS
        qual1, qual2 = supRateLo, supRateHi
    if eq == 5:  # HHexp
        w1, w2 = SupPriceHHexpSup(scDict, Ctheta, Cbeta)
        cS1, cS2 = cS, cS
        qual1, qual2 = supRateHi, supRateHi
    if eq == 6:  # HHFOC
        w1, w2 = SupPriceHH(scDict, Ctheta)
        cS1, cS2 = cS, cS
        qual1, qual2 = supRateHi, supRateHi
    if eq < 0 or eq > 6:
        print('Enter a valid equilibrium code')
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    return w1, w2, cS1, cS2, qual1, qual2

def SocWelEqMatsForPlot(numpts, Ctheta_max, Cbeta_max, uH, uL, scDict):
    # Generate list of equilibria matrices for plotting
    # Social-welfare maximizing equilibria is chosen if multiple equilibria exist
    eqList = CthetaCbetaMatsForPlot(numpts, Ctheta_max, Cbeta_max, scDict)
    CthetaVec = np.arange(0, Ctheta_max, (Ctheta_max) / numpts)
    CbetaVec = np.arange(0, Cbeta_max, (Cbeta_max) / numpts)

    retMat = eqList[0].copy()
    retMat[:] = np.nan
    for currCthetaind, currCtheta in enumerate(CthetaVec):
        for currCbetaind, currCbeta in enumerate(CbetaVec):
            possEqList = [i for i in range(eqList.shape[0]) if eqList[i, currCthetaind, currCbetaind]==1]
            if len(possEqList) == 1:
                w1, w2, cS1, cS2, qual1, qual2 = GetPricesFromEq(possEqList[0], scDict, currCtheta, currCbeta)
                retMat[currCthetaind, currCbetaind] = SocWel(uH, uL, w1, w2, cS1, cS2, qual1, qual2, currCtheta, currCbeta)
            if len(possEqList) > 1:
                w1, w2, cS1, cS2, qual1, qual2 = GetPricesFromEq(possEqList[0], scDict, currCtheta, currCbeta)
                currSocWel = SocWel(uH, uL, w1, w2, cS1, cS2, qual1, qual2, currCtheta, currCbeta)
                for i in range(1, len(possEqList)):
                    w1, w2, cS1, cS2, qual1, qual2 = GetPricesFromEq(possEqList[i], scDict, currCtheta, currCbeta)
                    iSocWel = SocWel(uH, uL, w1, w2, cS1, cS2, qual1, qual2, currCtheta, currCbeta)
                    if iSocWel > currSocWel:
                        currSocWel = iSocWel
                retMat[currCthetaind, currCbetaind] = currSocWel

    return retMat

def SocWelEqMatsForPlotIgnoreCosts(numpts, Ctheta_max, Cbeta_max, uH, uL, scDict):
    # Generate list of equilibria matrices for plotting
    # Social-welfare maximizing equilibria is chosen if multiple equilibria exist
    eqList = CthetaCbetaMatsForPlot(numpts, Ctheta_max, Cbeta_max, scDict)
    CthetaVec = np.arange(0, Ctheta_max, (Ctheta_max) / numpts)
    CbetaVec = np.arange(0, Cbeta_max, (Cbeta_max) / numpts)

    retMat = eqList[0].copy()
    retMat[:] = np.nan
    for currCthetaind, currCtheta in enumerate(CthetaVec):
        for currCbetaind, currCbeta in enumerate(CbetaVec):
            possEqList = [i for i in range(eqList.shape[0]) if eqList[i, currCthetaind, currCbetaind]==1]
            if len(possEqList) == 1:
                w1, w2, cS1, cS2, qual1, qual2 = GetPricesFromEq(possEqList[0], scDict, currCtheta, currCbeta)
                retMat[currCthetaind, currCbetaind] = SocWelIgnoreCosts(uH, uL, w1, w2, cS1, cS2, qual1, qual2, currCtheta, currCbeta)
            if len(possEqList) > 1:
                w1, w2, cS1, cS2, qual1, qual2 = GetPricesFromEq(possEqList[0], scDict, currCtheta, currCbeta)
                currSocWel = SocWelIgnoreCosts(uH, uL, w1, w2, cS1, cS2, qual1, qual2, currCtheta, currCbeta)
                for i in range(1, len(possEqList)):
                    w1, w2, cS1, cS2, qual1, qual2 = GetPricesFromEq(possEqList[i], scDict, currCtheta, currCbeta)
                    iSocWel = SocWelIgnoreCosts(uH, uL, w1, w2, cS1, cS2, qual1, qual2, currCtheta, currCbeta)
                    if iSocWel > currSocWel:
                        currSocWel = iSocWel
                retMat[currCthetaind, currCbetaind] = currSocWel

    return retMat

def SocWelEqMatsForPlotIgnorePens(numpts, Ctheta_max, Cbeta_max, uH, uL, scDict):
    # Generate list of equilibria matrices for plotting
    # Social-welfare maximizing equilibria is chosen if multiple equilibria exist
    eqList = CthetaCbetaMatsForPlot(numpts, Ctheta_max, Cbeta_max, scDict)
    CthetaVec = np.arange(0, Ctheta_max, (Ctheta_max) / numpts)
    CbetaVec = np.arange(0, Cbeta_max, (Cbeta_max) / numpts)

    retMat = eqList[0].copy()
    retMat[:] = np.nan
    for currCthetaind, currCtheta in enumerate(CthetaVec):
        for currCbetaind, currCbeta in enumerate(CbetaVec):
            possEqList = [i for i in range(eqList.shape[0]) if eqList[i, currCthetaind, currCbetaind]==1]
            if len(possEqList) == 1:
                w1, w2, cS1, cS2, qual1, qual2 = GetPricesFromEq(possEqList[0], scDict, currCtheta, currCbeta)
                retMat[currCthetaind, currCbetaind] = SocWelIgnorePens(uH, uL, w1, w2, cS1, cS2, qual1, qual2, currCtheta, currCbeta)
            if len(possEqList) > 1:
                w1, w2, cS1, cS2, qual1, qual2 = GetPricesFromEq(possEqList[0], scDict, currCtheta, currCbeta)
                currSocWel = SocWelIgnorePens(uH, uL, w1, w2, cS1, cS2, qual1, qual2, currCtheta, currCbeta)
                for i in range(1, len(possEqList)):
                    w1, w2, cS1, cS2, qual1, qual2 = GetPricesFromEq(possEqList[i], scDict, currCtheta, currCbeta)
                    iSocWel = SocWelIgnorePens(uH, uL, w1, w2, cS1, cS2, qual1, qual2, currCtheta, currCbeta)
                    if iSocWel > currSocWel:
                        currSocWel = iSocWel
                retMat[currCthetaind, currCbetaind] = currSocWel

    return retMat

#######################
# EQUILIBRIUM PLOTS
#######################
b, cSup, supRateLo, supRateHi = 0.6, 0.1, 0.8, 1.0
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo, 'supRateHi': supRateHi}
numpts = 100

Ctheta_max, Cbeta_max = 1.2*CthetaHHFOCLBForNoCbeta(scDict), 1.2*CbetaHHFOCLB(scDict, 0)

CthetaVec = np.arange(0, Ctheta_max, (Ctheta_max)/numpts)
CbetaVec = np.arange(0, Cbeta_max, (Cbeta_max)/numpts)

eqMats = CthetaCbetaMatsForPlot(numpts, Ctheta_max, Cbeta_max, scDict)

alval = 0.7

fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$c_S=$'+str(cSup)+', '+r'$L=$'+str(supRateLo),
             fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['#cf0234', 'deeppink', '#021bf9', '#0d75f8', '#82cafc', '#5ca904', '#0b4008']
labels = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHexp', 'HH']

imlist = []
for eqind in range(len(labels)):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[eqind]], name='from_list', N=None)
    im = ax.imshow(eqMats[eqind].T, vmin=0, vmax=1, aspect='auto',
                            extent=(0, Ctheta_max, 0, Cbeta_max),
                            origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)

# Fill in any non-equilibria regions
Cthdist, Cbedist = (Ctheta_max)/numpts, (Cbeta_max)/numpts
for i in range(CthetaVec.shape[0]):
    for j in range(CbetaVec.shape[0]):
        if np.nansum(eqMats[:, i, j])==0:  # No equilibria here
            ax.add_patch(matplotlib.patches.Rectangle((CthetaVec[i],CbetaVec[j]),Cthdist,Cbedist,
               hatch='/////////',fill=False,linewidth=0,snap=False))

legwidth = 20
wraplabels = ['\n'.join(textwrap.wrap(labels[i], width=legwidth)) for i in range(len(labels))]
patches = [mpatches.Patch(color=eqcolors[i], label=wraplabels[i], alpha=alval) for i in range(len(eqcolors))] +\
          [mpatches.Patch(hatch=r'/////////',fill=False,linewidth=0,snap=False,label='1-sup. eq.')]

# put those patched as legend-handles into the legend
ax.legend(handles=patches, bbox_to_anchor=(1.3, 1.0), loc='upper right', borderaxespad=0.1, fontsize=8)
ax.set_xbound(0, Ctheta_max)
ax.set_ybound(0, Cbeta_max)
ax.set_box_aspect(1)
plt.xlabel(r'$C_{\theta}^R$', fontsize=14)
plt.ylabel(r'$C_{\beta}^S$', fontsize=14, rotation=0, labelpad=14)
plt.show()

#######################
# SOCIAL WELFARE PLOT
#######################
b, cSup, supRateLo, supRateHi = 0.9, 0.1, 0.8, 1.0
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo, 'supRateHi': supRateHi}
numpts = 100
Ctheta_max, Cbeta_max = 1.2*CthetaHHFOCLBForNoCbeta(scDict), 1.2*CbetaHHFOCLB(scDict, 0)
uH = 5.0
uL = -1*uH

SocWelMat = SocWelEqMatsForPlot(numpts, Ctheta_max, Cbeta_max, uH, uL, scDict)

fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$c_S=$'+str(cSup)+', '+r'$L=$'+str(supRateLo)+', '+r'$u_H,u_L=$'+str(uH)+','+str(uL),
             fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['#cf0234', 'deeppink', '#021bf9', '#0d75f8', '#82cafc', '#5ca904', '#0b4008']
labels = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHexp', 'HH']

imlist = []
im = ax.imshow(SocWelMat.T, vmin=np.nanmin(SocWelMat), vmax=np.nanmax(SocWelMat), aspect='auto',
                            extent=(0, Ctheta_max, 0, Cbeta_max), origin="lower", cmap='Oranges')
imlist.append(im)

ax.set_xbound(0, Ctheta_max)
ax.set_ybound(0, Cbeta_max)
ax.set_box_aspect(1)
plt.xlabel(r'$C_{\theta}^R$', fontsize=14)
plt.ylabel(r'$C_{\beta}^S$', fontsize=14, rotation=0, labelpad=14)
plt.show()

#######################
# SOCIAL WELFARE PLOT: IGNORE INVESTMENT COSTS
#######################
b, cSup, supRateLo, supRateHi = 0.9, 0.4, 0.8, 1.0
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo, 'supRateHi': supRateHi}
numpts = 50
Ctheta_max, Cbeta_max = 1.2*CthetaHHFOCLBForNoCbeta(scDict), 1.2*CbetaHHFOCLB(scDict, 0)
uH = 5.0
uL = -1*uH/10

SocWelMatIgnCosts = SocWelEqMatsForPlotIgnoreCosts(numpts, Ctheta_max, Cbeta_max, uH, uL, scDict)

fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$c_S=$'+str(cSup)+', '+r'$L=$'+str(supRateLo)+', '+r'$u_L=$'+str(uL),
             fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['#cf0234', 'deeppink', '#021bf9', '#0d75f8', '#82cafc', '#5ca904', '#0b4008']
labels = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHexp', 'HH']

imlist = []
im = ax.imshow(SocWelMatIgnCosts.T, vmin=np.nanmin(SocWelMatIgnCosts), vmax=np.nanmax(SocWelMatIgnCosts), aspect='auto',
                            extent=(0, Ctheta_max, 0, Cbeta_max), origin="lower", cmap='Oranges')
imlist.append(im)

ax.set_xbound(0, Ctheta_max)
ax.set_ybound(0, Cbeta_max)
ax.set_box_aspect(1)
plt.xlabel(r'$C_{\theta}^R$', fontsize=14)
plt.ylabel(r'$C_{\beta}^S$', fontsize=14, rotation=0, labelpad=14)
plt.show()

#######################
# SOCIAL WELFARE PLOT: IGNORE PENALTIES
#######################
b, cSup, supRateLo, supRateHi = 0.6, 0.2, 0.8, 1.0
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo, 'supRateHi': supRateHi}
numpts = 50
Ctheta_max, Cbeta_max = 1.2*CthetaHHFOCLBForNoCbeta(scDict), 1.2*CbetaHHFOCLB(scDict, 0)
uL = 0

SocWelMat = SocWelEqMatsForPlotIgnorePens(numpts, Ctheta_max, Cbeta_max, uH, uL, scDict)

fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$c_S=$'+str(cSup)+', '+r'$L=$'+str(supRateLo)+', '+r'$u_L=$'+str(uL),
             fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['#cf0234', 'deeppink', '#021bf9', '#0d75f8', '#82cafc', '#5ca904', '#0b4008']
labels = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHexp', 'HH']

imlist = []
im = ax.imshow(SocWelMat.T, vmin=np.min(SocWelMat), vmax=np.max(SocWelMat), aspect='auto',
                            extent=(0, Ctheta_max, 0, Cbeta_max), origin="lower", cmap='Oranges')
imlist.append(im)

ax.set_xbound(0, Ctheta_max)
ax.set_ybound(0, Cbeta_max)
ax.set_box_aspect(1)
plt.xlabel(r'$C_{\theta}^R$', fontsize=14)
plt.ylabel(r'$C_{\beta}^S$', fontsize=14, rotation=0, labelpad=14)
plt.show()





# TODO: OLD PLOTTING SECTIONS BELOW HERE

#######################
# WHOLESALE PRICE PLOTS
#######################
# for b=[0.6, 0.9]
b, cSup, supRateLo, supRateHi = 0.75, 0.3, 0.8, 1.0
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo, 'supRateHi': supRateHi}

CthLLUB, CthLHFOCLB, CthLLsqzUB = CthetaLLFOCUB(scDict), CthetaLHFOCLB(scDict), CthetaLLsqzUB(scDict)
CthHHLB, devS = CthetaHHFOCLB(scDict)
CthLHFOCUB, CthLHsqzUB, CthLHexpLB = CthetaLHFOCUB(scDict), CthetaLHsqzUB(scDict), CthetaLHexpLB(scDict)
CthHHexpAdjLB, CthHHexpLB = CthetaHHexpAdjLB(scDict), CthetaHHexpLB(scDict)
CthetaMax = 1.2*CthHHLB
CthetaVec = np.arange(0, CthetaMax, 0.001)
LLprices = np.empty((CthetaVec.shape[0], 2))
LLprices[:] = np.nan
HHprices, HHexpprices, LHexpprices, LHFOCprices = LLprices.copy(), LLprices.copy(), LLprices.copy(), LLprices.copy()
LHsqzprices, LHsqztwoprices, LLsqzprices = LLprices.copy(), LLprices.copy(), LLprices.copy()
# Store prices
for Cthetaind in range(CthetaVec.shape[0]):
    currCtheta = CthetaVec[Cthetaind]
    if currCtheta <= CthLLUB:  # LL
        LLprices[Cthetaind, :] = SupPriceLL(scDict, currCtheta)
    if currCtheta > CthLLUB and currCtheta <= CthLLsqzUB:  # LL sqz
        LLsqzprices[Cthetaind, :] = SupPriceLLsqz5(scDict, currCtheta)
    if currCtheta > CthLHexpLB and currCtheta < CthLHFOCLB:  # LHexp
        LHexpprices[Cthetaind, :] = SupPriceLHexp(scDict, currCtheta)
    if currCtheta >= CthLHFOCLB and currCtheta <= CthLHFOCUB:  # LHFOC
        LHFOCprices[Cthetaind, :] = SupPriceLHFOC(scDict, currCtheta)
    if currCtheta > CthLHFOCUB and currCtheta <= CthLHsqzUB:  # LHsqz
        LHsqzprices[Cthetaind, :] = SupPriceLHsqz(scDict, currCtheta)
    if currCtheta >= CthHHexpLB and currCtheta < CthHHexpAdjLB:  # HHexp
        HHexpprices[Cthetaind, :] = SupPriceHHexp(scDict, currCtheta)
    if currCtheta >= CthHHexpAdjLB and currCtheta < CthHHLB:  # HHexpAdj
        if devS == 1:
            HHexpprices[Cthetaind, :] = SupPriceHHexp1(scDict, currCtheta)
        elif devS == 2:
            HHexpprices[Cthetaind, :] = SupPriceHHexp2(scDict, currCtheta)
    if currCtheta >= CthHHLB:
        HHprices[Cthetaind, :] = SupPriceHH(scDict, currCtheta)

fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', ' +r'$c_1=$'+str(cSup1)+', '+r'$c_2=$'+str(cSup2)+', '+r'$L=$'+str(supRateLo)+
             ', '+r'$a=$'+str(a), fontsize=18, fontweight='bold')

al = 0.8
LLonecol, LLtwocol, HHonecol, HHtwocol = 'red', 'deeppink', 'indigo', 'mediumorchid'
LLsqzonecol, LLsqztwocol, HHexponecol, HHexptwocol = 'darkorange', 'bisque', 'sienna', 'sandybrown'
LHonecols = ['limegreen', 'seagreen', 'darkgreen']
LHtwocols = ['cornflowerblue', 'blue', 'midnightblue']
lnwd = 5

plt.plot(CthetaVec, LLprices[:, 0], linewidth=lnwd, linestyle='dashed', color=LLonecol, alpha=al)
plt.plot(CthetaVec, LLprices[:, 1], linewidth=lnwd, color=LLtwocol, alpha=al)
plt.plot(CthetaVec, LLsqzprices[:, 0], linewidth=lnwd, linestyle='dashed', color=LLsqzonecol, alpha=al)
plt.plot(CthetaVec, LLsqzprices[:, 1], linewidth=lnwd, color=LLsqztwocol, alpha=al)
plt.plot(CthetaVec, LHexpprices[:, 0], linewidth=lnwd, linestyle='dashed', color=LHtwocols[0], alpha=al)
plt.plot(CthetaVec, LHexpprices[:, 1], linewidth=lnwd, color=LHonecols[0], alpha=al)
plt.plot(CthetaVec, LHFOCprices[:, 0], linewidth=lnwd, linestyle='dashed', color=LHtwocols[1], alpha=al)
plt.plot(CthetaVec, LHFOCprices[:, 1], linewidth=lnwd, color=LHonecols[1], alpha=al)
plt.plot(CthetaVec, LHsqzprices[:, 0], linewidth=lnwd, linestyle='dashed', color=LHtwocols[2], alpha=al)
plt.plot(CthetaVec, LHsqzprices[:, 1], linewidth=lnwd, color=LHonecols[2], alpha=al)
plt.plot(CthetaVec, HHexpprices[:, 0], linewidth=lnwd, linestyle='dashed', color=HHexponecol, alpha=al)
plt.plot(CthetaVec, HHexpprices[:, 1], linewidth=lnwd, color=HHexptwocol, alpha=al)
plt.plot(CthetaVec, HHprices[:, 0], linewidth=lnwd, linestyle='dashed', color=HHonecol, alpha=al)
plt.plot(CthetaVec, HHprices[:, 1], linewidth=lnwd, color=HHtwocol, alpha=al)
plt.ylim(0, 1.0)
plt.xlim(0, CthetaMax)
plt.xlabel(r'$C_{\theta}$', fontsize=14)
plt.ylabel(r'$w$', fontsize=14, rotation=0, labelpad=14)
plt.show()

#####################
# Social welfare plot vs prices
#####################
# TODO: RANGE UL OVER [0, -5?]
# TODO: UL VS b SP OUTCOME PLOT
# TODO: UL VS cS SP OUTCOME PLOT, for 2 possible b's
# TODO: first-best plot where one single decision-maker
# TODO: proposition that first-best is LL or HH

b, cSup1, cSup2, supRateLo, a = 0.85, 0.15, 0.15, 0.8, 1
uH, uL, priceconst = 1, 0.5
scDict = {'b': b, 'cSup1': cSup1, 'cSup2': cSup2, 'supRateLo': supRateLo, 'a': a}

CthLLUB, CthLHFOCLB, CthLLsqzUB = CthetaLLFOCUB(scDict), CthetaLHFOCLB(scDict), CthetaLLsqzUB(scDict)
CthHHLB, devS = CthetaHHFOCLB(scDict)
CthLHFOCUB, CthLHsqzUB, CthLHexpLB = CthetaLHFOCUB(scDict), CthetaLHsqzUB(scDict), CthetaLHexpLB(scDict)
CthHHexpAdjLB, CthHHexpLB = CthetaHHexpAdjLB(scDict), CthetaHHexpLB(scDict)
CthetaMax = 1.2*CthHHLB
CthetaVec = np.arange(0, CthetaMax, 0.001)
LLprices = np.empty((CthetaVec.shape[0], 2))
LLprices[:] = np.nan
HHprices, HHexpprices, LHexpprices, LHFOCprices = LLprices.copy(), LLprices.copy(), LLprices.copy(), LLprices.copy()
LHsqzprices, LHsqztwoprices, LLsqzprices = LLprices.copy(), LLprices.copy(), LLprices.copy()
# Store prices
for Cthetaind in range(CthetaVec.shape[0]):
    currCtheta = CthetaVec[Cthetaind]
    if currCtheta <= CthLLUB:  # LL
        LLprices[Cthetaind, :] = SupPriceLL(scDict, currCtheta)
    if currCtheta > CthLLUB and currCtheta <= CthLLsqzUB:  # LL sqz
        LLsqzprices[Cthetaind, :] = SupPriceLLsqz5(scDict, currCtheta)
    if currCtheta > CthLHexpLB and currCtheta < CthLHFOCLB:  # LHexp
        LHexpprices[Cthetaind, :] = SupPriceLHexp(scDict, currCtheta)
    if currCtheta >= CthLHFOCLB and currCtheta <= CthLHFOCUB:  # LHFOC
        LHFOCprices[Cthetaind, :] = SupPriceLHFOC(scDict, currCtheta)
    if currCtheta > CthLHFOCUB and currCtheta <= CthLHsqzUB:  # LHsqz
        LHsqzprices[Cthetaind, :] = SupPriceLHsqz(scDict, currCtheta)
    if currCtheta >= CthHHexpLB and currCtheta < CthHHexpAdjLB:  # HHexp
        HHexpprices[Cthetaind, :] = SupPriceHHexp(scDict, currCtheta)
    if currCtheta >= CthHHexpAdjLB and currCtheta < CthHHLB:  # HHexpAdj
        if devS == 1:
            HHexpprices[Cthetaind, :] = SupPriceHHexp1(scDict, currCtheta)
        elif devS == 2:
            HHexpprices[Cthetaind, :] = SupPriceHHexp2(scDict, currCtheta)
    if currCtheta >= CthHHLB:
        HHprices[Cthetaind, :] = SupPriceHH(scDict, currCtheta)

# Store social welfare
socwelMat = np.empty((7, CthetaVec.shape[0]))
socwelMat[:] = np.nan
for Cthetaind in range(CthetaVec.shape[0]):
    currCtheta = CthetaVec[Cthetaind]
    if currCtheta <= CthLLUB:  # LL
        qual1, qual2 = supRateLo, supRateLo
        w1, w2 = SupPriceLL(scDict, currCtheta)
        q1, q2 = q1Opt(w1, w2, b, a), q2Opt(w1, w2, b, a)
        socwelMat[0, Cthetaind] = SocWel(uH, uL, q1, q2, 0, 0, qual1, qual2, priceconst)
    if currCtheta > CthLLUB and currCtheta <= CthLLsqzUB:  # LL sqz
        qual1, qual2 = supRateLo, supRateLo
        w1, w2 = SupPriceLLsqz5(scDict, currCtheta)
        q1, q2 = q1Opt(w1, w2, b, a), q2Opt(w1, w2, b, a)
        socwelMat[1, Cthetaind] = SocWel(uH, uL, q1, q2, 0, 0, qual1, qual2, priceconst)
    if currCtheta > CthLHexpLB and currCtheta < CthLHFOCLB:  # LHexp
        qual1, qual2 = supRateLo, 1
        w1, w2 = SupPriceLHexp(scDict, currCtheta)
        q1, q2 = q1Opt(w1, w2, b, a), q2Opt(w1, w2, b, a)
        socwelMat[2, Cthetaind] = SocWel(uH, uL, q1, q2, 0, cSup2, qual1, qual2, priceconst)
    if currCtheta >= CthLHFOCLB and currCtheta <= CthLHFOCUB:  # LHFOC
        qual1, qual2 = supRateLo, 1
        w1, w2 = SupPriceLHFOC(scDict, currCtheta)
        q1, q2 = q1Opt(w1, w2, b, a), q2Opt(w1, w2, b, a)
        socwelMat[3, Cthetaind] = SocWel(uH, uL, q1, q2, 0, cSup2, qual1, qual2, priceconst)
    if currCtheta > CthLHFOCUB and currCtheta <= CthLHsqzUB:  # LHsqz
        qual1, qual2 = supRateLo, 1
        w1, w2 = SupPriceLHsqz(scDict, currCtheta)
        q1, q2 = q1Opt(w1, w2, b, a), q2Opt(w1, w2, b, a)
        socwelMat[4, Cthetaind] = SocWel(uH, uL, q1, q2, 0, cSup2, qual1, qual2, priceconst)
    if currCtheta >= CthHHexpLB and currCtheta < CthHHexpAdjLB:  # HHexp
        qual1, qual2 = 1, 1
        w1, w2 = SupPriceHHexp(scDict, currCtheta)
        q1, q2 = q1Opt(w1, w2, b, a), q2Opt(w1, w2, b, a)
        socwelMat[5, Cthetaind] = SocWel(uH, uL, q1, q2, cSup1, cSup2, qual1, qual2, priceconst)
    if currCtheta >= CthHHexpAdjLB and currCtheta < CthHHLB:  # HHexpAdj
        if devS == 1:
            w1, w2 = SupPriceHHexp1(scDict, currCtheta)
        elif devS == 2:
            w1, w2 = SupPriceHHexp2(scDict, currCtheta)
        qual1, qual2 = 1, 1
        q1, q2 = q1Opt(w1, w2, b, a), q2Opt(w1, w2, b, a)
        socwelMat[5, Cthetaind] = SocWel(uH, uL, q1, q2, cSup1, cSup2, qual1, qual2, priceconst)
    if currCtheta >= CthHHLB:  # HHFOC
        qual1, qual2 = 1, 1
        w1, w2 = SupPriceHH(scDict, currCtheta)
        q1, q2 = q1Opt(w1, w2, b, a), q2Opt(w1, w2, b, a)
        socwelMat[6, Cthetaind] = SocWel(uH, uL, q1, q2, cSup1, cSup2, qual1, qual2, priceconst)


fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', ' +r'$c_1=$'+str(cSup1)+', '+r'$c_2=$'+str(cSup2)+', '+r'$L=$'+str(supRateLo)+
             ',\n'+r'$a=$'+str(a)+', '+r'$u^H=$'+str(uH)+', '+r'$u^L=$'+str(uL),
             fontsize=18, fontweight='bold')

al = 0.8
cm = plt.get_cmap('Greys')
cols = cm(np.linspace(0.4, 1, 7))
lnwd = 5

for i in range(7):
    plt.plot(CthetaVec, socwelMat[i, :], linewidth=lnwd, color=cols[i], alpha=al)
plt.ylim(0, np.nanmax(socwelMat)*1.1)
plt.xlim(0, CthetaMax)
plt.xlabel(r'$C_{\theta}$', fontsize=14)
plt.ylabel(r'$U^{SP}$', fontsize=14, rotation=0, labelpad=14)
plt.show()

#####################
# Social welfare eq plot, vs cS
#####################
b, cSup1, cSup2, supRateLo, a = 0.85, 0.25, 0.25, 0.8, 1
scDict = {'b': b, 'cSup1': cSup1, 'cSup2': cSup2, 'supRateLo': supRateLo, 'a': a}

numpts, cSupMax = 50, 0.5
uLmin, uLmax = -2, 0.9
alval = 0.7

eq_matList = SocWelEqMatsForPlot(numpts, uLmin, uLmax, cSupMax, scDict)

fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$L=$'+str(supRateLo), fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['deeppink',  '#82cafc',  '#0b4008']
labels = ['LLsqz', 'LHsqz',  'HH']

imlist = []
for eqind in range(len(labels)):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[eqind]], name='from_list', N=None)
    im = ax.imshow(eq_matList[eqind], vmin=0, vmax=1, aspect='auto',
                            extent=(uLmin, uLmax, 0, cSupMax),
                            origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)

legwidth = 20
wraplabels = ['\n'.join(textwrap.wrap(labels[i], width=legwidth)) for i in range(len(labels))]
patches = [mpatches.Patch(color=eqcolors[i], edgecolor='black', label=wraplabels[i], alpha=alval) for i in range(len(eqcolors))]
# put those patched as legend-handles into the legend
ax.legend(handles=patches, bbox_to_anchor=(1.3, 1.0), loc='upper right', borderaxespad=0.1, fontsize=8)
ax.set_xbound(uLmin, uLmax)
ax.set_ybound(0, cSupMax)
ax.set_box_aspect(1)
plt.xlabel(r'$u_{L}$', fontsize=14)
plt.ylabel(r'$c_S$', fontsize=14, rotation=0, labelpad=14)
plt.show()


#####################
# Equilibrium plots: Delta cS
#####################
b, cSup1, cSup2, supRateLo, a = 0.6, 0.151, 0.15, 0.75, 1.3
uH, uL, priceconst = 1, -1, 1.0
scDict = {'b': b, 'cSup1': cSup1, 'cSup2': cSup2, 'supRateLo': supRateLo, 'a': a}

numpts, CthetaMax, cSupDeltaMax = 50, 1.5, 0.5
alval = 0.7

eqStrat_matList = LthetaCsupEqMatsForPlot(numpts, CthetaMax, cSupDeltaMax, scDict)
# Fill holes
# for csupind in range(eqStrat_matList.shape[1]):
#     for cthetaind in range(eqStrat_matList.shape[2]):
#         if np.nansum(eqStrat_matList[:,csupind,cthetaind]) == 0:
#             eqStrat_matList[4, csupind, cthetaind] = 1

fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$L=$'+str(supRateLo)+', '+r'$c^S_2=$'+str(cSup2)+', '+r'$a=$'+str(a),
             fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['#cf0234', 'deeppink', '#021bf9', '#0d75f8', '#82cafc', '#5ca904', '#0b4008']
labels = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHsqz', 'HH']

imlist = []
for eqind in range(len(labels)):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[eqind]], name='from_list', N=None)
    im = ax.imshow(eqStrat_matList[eqind], vmin=0, vmax=1, aspect='auto',
                            extent=(0, CthetaMax, 0, cSupDeltaMax),
                            origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)

legwidth = 20
wraplabels = ['\n'.join(textwrap.wrap(labels[i], width=legwidth)) for i in range(len(labels))]
patches = [mpatches.Patch(color=eqcolors[i], edgecolor='black', label=wraplabels[i], alpha=alval) for i in range(len(eqcolors))]
# put those patched as legend-handles into the legend
ax.legend(handles=patches, bbox_to_anchor=(1.3, 1.0), loc='upper right', borderaxespad=0.1, fontsize=8)
ax.set_xbound(0, CthetaMax)
ax.set_ybound(0, cSupDeltaMax)
ax.set_box_aspect(1)
plt.xlabel(r'$C_{\theta}$', fontsize=14)
plt.ylabel(r'$\Delta c^S_1$', fontsize=14, rotation=0, labelpad=14)
plt.show()

#####################
# Equilibrium plots: Delta a
#####################
b, cSup1, cSup2, supRateLo, a = 0.6, 0.3, 0.1, 0.75, 1.3
uH, uL, priceconst = 1, -1, 1.0
scDict = {'b': b, 'cSup1': cSup1, 'cSup2': cSup2, 'supRateLo': supRateLo, 'a': a}

numpts, CthetaMax, aDeltaMax = 50, 1.5, 0.5
alval = 0.7

eqStrat_matList = LthetaAdemEqMatsForPlot(numpts, CthetaMax, aDeltaMax, scDict)
# Fill holes
# for csupind in range(eqStrat_matList.shape[1]):
#     for cthetaind in range(eqStrat_matList.shape[2]):
#         if np.nansum(eqStrat_matList[:,csupind,cthetaind]) == 0:
#             eqStrat_matList[4, csupind, cthetaind] = 1

fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$L=$'+str(supRateLo)+', '+r'$c^S_1=$'+str(cSup1)+', '+r'$c^S_2=$'+str(cSup2),
             fontsize=18, fontweight='bold')
ax = fig.add_subplot(111)

eqcolors = ['#cf0234', 'deeppink', '#021bf9', '#0d75f8', '#82cafc', '#5ca904', '#0b4008']
labels = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHsqz', 'HH']

imlist = []
for eqind in range(len(labels)):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[eqind]], name='from_list', N=None)
    im = ax.imshow(eqStrat_matList[eqind], vmin=0, vmax=1, aspect='auto',
                            extent=(0, CthetaMax, 0, aDeltaMax),
                            origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)

legwidth = 20
wraplabels = ['\n'.join(textwrap.wrap(labels[i], width=legwidth)) for i in range(len(labels))]
patches = [mpatches.Patch(color=eqcolors[i], edgecolor='black', label=wraplabels[i], alpha=alval) for i in range(len(eqcolors))]
# put those patched as legend-handles into the legend
ax.legend(handles=patches, bbox_to_anchor=(1.3, 1.0), loc='upper right', borderaxespad=0.1, fontsize=8)
ax.set_xbound(0, CthetaMax)
ax.set_ybound(0, aDeltaMax)
ax.set_box_aspect(1)
plt.xlabel(r'$C_{\theta}$', fontsize=14)
plt.ylabel(r'$\Delta a$', fontsize=14, rotation=0, labelpad=14)
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