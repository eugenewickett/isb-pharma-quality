# Examining robustness of  for "Strategic Role of Inspections in Pharmaceutical Supply Chains"
# We add potentially higher demand and a lower quality cost for S2
# These functions use the simpler version of the model
#   - perfect diagnostic, H=1, no retailer quality choice, retailer sources from both suppliers or neither
# 22-DEC-25

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

def SPUtil(q1, q2, lambsup1, lambsup2, alph):
    # Social planner's utility
    return q1*(alph+(lambsup1)-1) + q2*(alph+(lambsup2)-1)

def SocWel(uH, uL, q1, q2, cS1, cS2, lambsup1, lambsup2, priceconst=1):
    # Social welfare
    return q1*(uH*(lambsup1)+uL*(1-lambsup1)-priceconst*cS1) + q2*(uH*(lambsup2)+uL*(1-lambsup2)-priceconst*cS2)

def SupUtil(q, w, cSup):
    # Returns supplier utility
    return q*(w-cSup)

def invDemandPrice(qi, qj, b, ai):
    return ai - qi - b*qj

def q1Opt(w1, w2, b, a):
    # Returns optimal order quantities from S1 under dual sourcing
    return max(0, (1-a*b-w1+b*w2)/(2*(1-(b**2))))

def q2Opt(w1, w2, b, a):
    # Returns optimal order quantities from S1 under dual sourcing
    return max(0, (a-b+b*w1-w2)/(2*(1-(b**2))))

def RetUtil(b, w1, w2, Ctheta, lambsup1, lambsup2, a):
    # Returns retailer utility under pi=D
    q1, q2 = q1Opt(w1, w2, b, a), q2Opt(w1, w2, b, a)
    prof1, prof2 = q1*(invDemandPrice(q1, q2, b, 1)-w1), q2*(invDemandPrice(q2, q1, b, a)-w2)
    insppen = Ctheta * ((1 - lambsup1 * lambsup2))
    return prof1 + prof2 - insppen

def SupPriceLL(scDict, Ctheta):
    # Returns on-path LL prices
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    w1 = max(0, (2 - a*b - (b**2))/(4 - (b**2)))
    w2 = max(0, (2*a - b - a*(b ** 2)) / (4 - (b ** 2)))
    return w1, w2

def SupPriceHH(scDict, Ctheta):
    # Returns on-path HH prices
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    w1 = max(0, (2 - a*b - (b**2) + 2*cS1 + b*cS2)/(4 - (b**2)))
    w2 = max(0, (2*a - b - a*(b**2) + b*cS1 + 2*cS2)/(4 - (b**2)))
    return w1, w2

def SupPriceLHFOC(scDict, Ctheta):
    # Returns on-path LH prices
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    w1 = max(0, (2 - a*b - (b**2) + b*cS2)/(4 - (b**2)))
    w2 = max(0, (2*a - b - a*(b**2) + 2*cS2)/(4 - (b**2)))
    return w1, w2

def SupPriceLHsqz(scDict, Ctheta):
    # Returns on-path LHsqz prices
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    radterm = np.sqrt((-1 + (b**2))*(((a - cS2)**2) - 4*(-4 + 3*(b**2))*Ctheta*(-1 + supRateLo)))
    w1 = (b*(a + 3*b - cS2) + 2*(-2 + radterm))/(-4 + 3*(b**2))
    w2 = (-2*(a + cS2) + (b**2)*(2*a + cS2) + b*radterm)/(-4 +3*(b**2))
    if w2 < cS2:
        w2 = cS2
        w1 = 1 + b*(-a + cS2 + b*np.sqrt((((a - cS2)**2) + 4*Ctheta*(-1 + supRateLo))/(-1 +(b**2)))) -\
             np.sqrt((((a - cS2)**2)+4*Ctheta*(-1 + supRateLo))/(-1 + (b**2)))
    return w1, max(w2, 0)


def funcLHexpPrice(x, currCtheta, scDict):
    # When only S2 wants to move off-path; S1 chooses price s.t. S2 is indifferent, S2 chooses FOC
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # x = [w1HHexp, w2HHexp], the on-path HHsqz prices
    # radterm1 = np.sqrt((((-1 + x[0])**2) + 4*currCtheta*(-1 + supRateLo))/(-1 + (b**2)))
    # retval1 = (((cS2 - x[1])*(a + b*(-1 + x[0]) - x[1]))/(2*(-1 + (b**2)))) -\
    #           (0.5*(a +b*(-1 + x[0] + b* radterm1) - radterm1)* radterm1)
    # retval2 = (x[1]) - 0.5*(a + cS2 + b*(-1 + x[0]))
    radterm1 = np.sqrt((((-1 + x[0])**2) + 4*currCtheta*(-1 + (supRateLo**2)))/(-1 + (b**2)))
    retval1 = ((cS2 - x[1])*(a + b*(-1 + x[0]) - x[1]))/(2*(-1 + (b**2))) -\
                (0.5*radterm1*(a - radterm1 + b*(-1 + x[0] + b* radterm1)))
    retval2 = x[1] - (0.5*(a + cS2 + b*(-1 + x[0])))
    return [retval1, retval2]
def SupPriceLHexp(scDict, Ctheta):
    # Returns on-path LHexp prices
    xinit = SupPriceLHFOC(scDict, 0)
    root = fsolve(funcLHexpPrice, [xinit[0], xinit[1]], args=(Ctheta, scDict))
    return root[0], root[1]

def funcHHexpPrice(x, currCtheta, scDict):
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # x = [w1HHexp, w2HHexp], the on-path HHsqz prices
    retval1 = (-(((cS1 - x[0])*(-1 + a*b + x[0] - b*x[1]))/(2*(-1 + (b**2))))) - (0.5*(1 + b*(-a+x[1] +\
                b*np.sqrt((((a - x[1])**2)+ 4*currCtheta*(-1 + supRateLo))/(-1 + (b**2)))) -\
                np.sqrt((((a - x[1])**2) + 4*currCtheta* (-1 + supRateLo))/(-1 + (b**2))))*np.sqrt((((a -\
                x[1])**2) + 4* currCtheta* (-1 + supRateLo))/(-1 + (b**2))))
    radterm2 = np.sqrt((((-1 + x[0])**2) + 4*currCtheta* (-1 + supRateLo))/(-1 + (b**2)))
    retval2 = ((cS2 - x[1])*(a + b*(-1 + x[0]) - x[1]))/(2*(-1 + (b**2))) - \
              (0.5*(a + b*(-1 + x[0] + b*radterm2) - radterm2)*radterm2)
    return [retval1, retval2]

def funcHHexpPrice1(x, currCtheta, scDict):
    # When only S1 wants to move off-path; S2 chooses price s.t. S1 is indifferent, S1 chooses FOC
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # x = [w1HHexp, w2HHexp], the on-path HHsqz prices
    retval1 = x[0]- (0.5*(1 - a*b + cS1 + b* x[1]))
    radterm2 = np.sqrt((((a - x[1])**2) + 4* currCtheta*(-1 +supRateLo))/(-1 + (b**2)))
    retval2 = (-1*(((cS1 - x[0])*(-1 + a*b + x[0] - b* x[1]))/(2*(-1 + (b**2))))) - \
        (0.5*(1 +b*(-a + x[1] +b*radterm2) - radterm2) *radterm2)
    return [retval1, retval2]

def funcHHexpPrice2(x, currCtheta, scDict):
    # When only S2 wants to move off-path; S1 chooses price s.t. S2 is indifferent, S2 chooses FOC
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # x = [w1HHexp, w2HHexp], the on-path HHsqz prices
    radterm1 = np.sqrt((((-1 + x[0])**2) + 4*currCtheta*(-1 + supRateLo))/(-1 + (b**2)))
    retval1 = (((cS2 - x[1])*(a + b*(-1 + x[0]) - x[1]))/(2*(-1 + (b**2)))) -\
              (0.5*(a +b*(-1 + x[0] + b* radterm1) - radterm1)* radterm1)
    retval2 = (x[1]) - 0.5*(a + cS2 + b*(-1 + x[0]))
    return [retval1, retval2]

def SupPriceHHexp(scDict, Ctheta, tol=1E-8):
    # Returns on-path HHexp prices
    xinit = SupPriceHH(scDict, 0)
    root, infodict, _, _ = fsolve(funcHHexpPrice, [xinit[0], xinit[1]], args=(Ctheta, scDict), full_output=True)
    if np.sum(np.abs(infodict['fvec'])) > tol:
        root= [-1, -1]
    return root[0], root[1]

def SupPriceHHexp1(scDict, Ctheta):
    # Returns on-path HHexp prices; S1 wants to move but not S2
    xinit = SupPriceHH(scDict, 0)
    root = fsolve(funcHHexpPrice1, [xinit[0], xinit[1]], args=(Ctheta, scDict))
    return root[0], root[1]

def SupPriceHHexp2(scDict, Ctheta):
    # Returns on-path HHexp prices; S1 wants to move but not S2
    xinit = SupPriceHH(scDict, 0)
    root = fsolve(funcHHexpPrice2, [xinit[0], xinit[1]], args=(Ctheta, scDict))
    return root[0], root[1]

def SupPriceLLsqz(scDict, Ctheta):
    # Returns on-path LLsqz prices
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    w1 = 1-(2*np.sqrt((1+(a**2)-2*a*b)*(-1+(b**2))*Ctheta*(-1+(supRateLo**2))))/(1+(a**2)-2*a*b)
    w2 = a-(2*a*np.sqrt((1+(a**2)-2*a*b)*(-1+(b**2))*Ctheta*(-1 + (supRateLo**2))))/(1+(a**2)-2*a*b)
    return max(w1, 0), max(w2, 0)

def funcLLsqzPrice2(x, currCtheta, scDict):
    # Lagrangian using LLFOC utility ratios in objective
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # x = [w1LLsqz, w2LLsqz, lambLLsqz], the on-path HHsqz prices
    retval1 = (4 - 4*(b**2) + (b**4) - 8*x[0] + 8* (b**2)*x[0] - 2* (b**4)* x[0] + 4*b* x[1] - 3*(b**3)*x[1] +\
               (b**5)*x[1] + a*b*(-2 + (b**2))*(4 + 2*x[0]* (-2 + x[2]) - 2* b* x[1]* (-2 + x[2]) +\
              (b**2)* (-1 + x[2]) - 4* x[2]) +(a**3)* (b**3)* (-1 + x[2]) - ((-2 + (b**2))**2)*(1 - x[0] + b*x[1])*x[2]+\
              (a**2)*b*(4*x[1] + b* (5 - 2*x[0] + (b**3)* x[1] + 2* (b**2)* (-1 + x[2]) + (-5 + x[0])* x[2] -\
              b* x[1]*(3 + x[2]))))  # Derivative WRT w1
    retval2 = (b*(-4* x[0] + b* (b + 3* b* x[0] - (b**3)* x[0] + 2* x[1])) + a*b*(-8*x[1] + b*(-5 + 8* x[0] +\
              2*b*(b - 2* b* x[0] + 2* x[1]))) + ((-2 + (b**2))**2)*(b* (-1 + x[0]) - x[1])*x[2] +\
              a*(-2 + (b**2))* (-2 + (b**2)* (-1 + 2*x[0]) - 2* b* x[1])* x[2] + (a**3)*(-4 - (b**4) +\
              (b**2)* (4 + x[2])) + (a**2)*(-1*(b**5)*(-1 + x[0]) + 8*x[1] + 2* (b**4)* x[1] -\
              (b**2)* x[1] *(8 + x[2]) - 4* b *(-2 + x[0] + x[2]) + (b**3)*(-6 + x[2] + x[0]* (3 + x[2]))))  # Derivative WRT w2
    retval3 = (a**2) + 2* a* b* (-1 + x[0]) + ((-1 + x[0])**2) - 2*a *x[1] - 2* b* (-1 + x[0]) *x[1] + (x[1]**2) -\
              4*(-1 + (b**2))* currCtheta *(-1 + (supRateLo**2))
    return [retval1, retval2, retval3]

def SupPriceLLsqz2(scDict, Ctheta):  # USING RATIO OF LL UTILS DOESNT WORK
    # Returns on-path LLsqz prices
    xinit = SupPriceLL(scDict, 0)
    root = fsolve(funcLLsqzPrice2, [xinit[0], xinit[1], 1], args=(Ctheta, scDict))
    return root[0], root[1]

def funcLLsqzPrice3(x, currCtheta, scDict):
    # Lagrangian using 'a' as ratio in objective
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # x = [w1LLsqz, w2LLsqz, lambLLsqz], the on-path HHsqz prices
    retval1 = x[0]* (-2 + x[2]) - (1 + b* x[1])* (-1 + x[2]) + a* b* (-1 + x[1] + x[2])  # Derivative WRT w1
    retval2 = (a**2) + a* (b *(-1 + x[0]) - 2* x[1] - x[2]) + x[1]* x[2] + b* (x[0] + x[2] - x[0]* x[2]) # Derivative WRT w2
    retval3 = (a**2) + 2*a*b* (-1 + x[0]) + ((-1 + x[0])**2) - 2*a*x[1] - 2*b*(-1 + x[0])* x[1] +\
              (x[1]**2) - 4* (-1 + (b**2))*currCtheta* (-1 + (supRateLo**2))  # Derivative WRT lambda
    return [retval1, retval2, retval3]

def SupPriceLLsqz3(scDict, Ctheta):
    # Returns on-path LLsqz prices
    xinit = SupPriceLL(scDict, 0)
    root = fsolve(funcLLsqzPrice3, [xinit[0], xinit[1], 1], args=(Ctheta, scDict))
    return root[0], root[1]

def funcLLsqzPrice4(x, currCtheta, scDict):
    # Lagrangian using 'a' as ratio in objective
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # x = [w1LLsqz, w2LLsqz, lambLLsqz], the on-path HHsqz prices
    retval1 = (a**2)* b + b* x[1] *(-1 + x[2]) + x[2] - x[0]*x[2] + a* (-1 + 2* x[0] - b* (x[1] + x[2]))  # Derivative WRT w1
    retval2 = -1*x[1]* (-2 + x[2]) + b* (-1 + x[0])* (-1 + x[2]) + a* (-1 - b* x[0] + x[2]) # Derivative WRT w2
    retval3 = (a**2) + 2*a*b*(-1 + x[0]) + ((-1 + x[0])**2) - 2*a*x[1] - 2*b*(-1 + x[0])*x[1] + (x[1]**2) -\
              4*(-1 + (b**2))*currCtheta*(-1 + (supRateLo**2))  # Derivative WRT lambda
    return [retval1, retval2, retval3]

def SupPriceLLsqz4(scDict, Ctheta):
    # Returns on-path LLsqz prices
    xinit = SupPriceLL(scDict, 0)
    root = fsolve(funcLLsqzPrice4, [xinit[0], xinit[1], 1], args=(Ctheta, scDict))
    return root[0], root[1]

def funcLLsqzPrice5(x, currCtheta, scDict):
    # Lagrangian using 'a' as ratio in objective
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # x = [w1LLsqz, w2LLsqz, lambLLsqz], the on-path HHsqz prices
    retval1 = (-b*(4*x[1] + b*(1 - 2*x[0] + b* (-3 + (b**2))* x[1])) + ((-2 + (b**2))**2)*(1-x[0]+b*x[1])*x[2] +\
              a* b* (4 - (b**2) - 8*x[0] + 4* (b**2)*x[0] + 8*b*x[1]-4* (b**3)*x[1]-(-2 + (b**2))*(-4 + (b**2) +\
              2*x[0] - 2*b*x[1])*x[2]) + (a**3)*b*(4 + (b**4) - (b**2)*(4 + x[2])) + (a**2)*(-4 + 8*x[0]+b*(-4*x[1] +\
              b* (-1*(b**3)*x[1] + (b**2)*(1 + 2*x[0] - 2*x[2]) + 5* x[2]+b*x[1]*(3 +x[2]) - x[0]*(8 + x[2])))))  # Derivative WRT w1
    retval2 = (8*x[1] + b*(4 - 4*x[0] + b* (-8* x[1] + b* (-4 + 3*x[0] + b* (b - b* x[0] + 2* x[1])))) +\
                (a**3)* (b**2)* (-1 + x[2]) + ((-2 + (b**2))**2)* (b*(-1 + x[0]) - x[1])*x[2] +\
                a* (-2 + (b**2))* (2 - 2* x[2] + b* (b + 2* b* x[0]* (-2 + x[2]) - 2* x[1]* (-2 + x[2]) -\
                b* x[2])) - (a**2)* b* ((b**4)* x[0] + b* x[1]* (-2 + x[2]) + 4* (-1 + x[0] + x[2]) -\
                (b**2)* (-1 + x[2] + x[0]* (3 + x[2])))) # Derivative WRT w2
    retval3 = (a**2) + 2*a*b*(-1 + x[0]) + ((-1 + x[0])**2) - 2*a*x[1] - 2*b*(-1 + x[0])*x[1] + (x[1]**2) -\
              4*(-1 + (b**2))*currCtheta*(-1 + (supRateLo**2))  # Derivative WRT lambda
    return [retval1, retval2, retval3]

def SupPriceLLsqz5(scDict, Ctheta):
    # Returns on-path LLsqz prices
    xinit = SupPriceLL(scDict, 0)
    root = fsolve(funcLLsqzPrice5, [xinit[0], xinit[1], 1], args=(Ctheta, scDict))
    return root[0], root[1]

def CthetaLLFOCUB(scDict):
    # LL FOC UB in Ctheta
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    retval = (4*(1+(a**2))-3*(1+(a**2))*(b**2)-2*a*(b**3))/(4*((-4+(b**2))**2)*(-1+(b**2))*(-1+ (supRateLo**2)))
    return retval

def CthetaLHFOCUB(scDict):
    # LH FOC UB in Ctheta
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    retval = ((-4+3*(b**2)+(a - cS2)*(2*(b**3)+a*(-4 + 3*(b**2))+4*cS2-3*(b**2)*cS2))/(4*((-4+(b**2))**2)*(1-\
                        (b**2))*(-1+supRateLo)))
    return retval

def CthetaLHFOCLB(scDict, step=1/10000):
    # LH FOC LB in Ctheta
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    w1on, w2on = SupPriceLHFOC(scDict, 0)
    LHFOCUB = CthetaLHFOCUB(scDict)
    # Find Ctheta where off-path retIR price is preferred by S2
    s2utilOn = SupUtil(q2Opt(w1on, w2on, b, a), w2on, cS2)
    CthetaVec = np.arange(LHFOCUB, 0, -1 * step)
    currLB = CthetaVec[0]  # Move down from this
    found = False
    for Cthetaind, currCtheta in enumerate(CthetaVec):
        if not found:
            # Get off-path price
            radterm = np.sqrt((4 - 64*currCtheta +b*((a - cS2)*(4 + a*b - b*cS2) - 4*b*(-8 + (b**2))* currCtheta) +\
                      4*((-4 + (b**2))**2)* currCtheta* (supRateLo**2))/(((-4 + (b**2))**2)*(-1 +(b**2))))
            w2LHFOCoff = (2*a*(-2 + (b**2)) + 4*radterm + b*(2 - b*cS2 + b*(-5 + (b**2))* radterm))/(-4 + (b**2))
            s2utilOff = SupUtil(q2Opt(w1on, w2LHFOCoff, b, a), w2LHFOCoff, 0)
            if s2utilOff > s2utilOn:
                found = True
            if not found:
                currLB = currCtheta
    # retval = (1/(144*((-1+(supRateLo**2))**2)))*(-((6*((1+a-cS2)**2)*(-1+(supRateLo**2)))/((-2+b)**2))-\
    #           (6*((1-a+cS2)**2)*(-1+(supRateLo**2)))/((2+b)**2)-(2*(-1+a-cS2)*(-1+a+5*cS2)*(-1+(supRateLo**2)))/(2+\
    #           b)+(2*(1+a-cS2)*(1+a+5*cS2)*(-1+(supRateLo**2)))/(-2+b)+((-2*((1+a)**2)-8*(1+a)*cS2+(cS2**2))*(-1+\
    #           (supRateLo**2)))/(1+b) +((2 + 2*(a**2) - cS2*(8+cS2)+ a*(-4 + 8*cS2))*(-1+ (supRateLo**2)))/(-1+b)+\
    #           18*np.sqrt(-((cS2*(-4*a*(-2+ (b**2)) + 3*(b**2)*cS2- 4*(b + cS2))*((-2*a*(-2+ (b**2))+b*(-2+\
    #           b*cS2))**2)*((-1 + (supRateLo**2))**2))/(((-4+(b**2))**3)*((-1+(b**2))**2)))))
    return currLB

def CthetaLHexpLB(scDict, step=1/10000):
    # LH exp LB in Ctheta
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # We do three checks under each set of prices for given Ctheta: retailer IR, S2's off-path, S1's LL on-path
    LHFOCLB, LLFOCUB = CthetaLHFOCLB(scDict), CthetaLLFOCUB(scDict)
    w1LLFOC, w2LLFOC = SupPriceLL(scDict, 0)  # LL on-path prices/utility
    S1utilLLFOC = SupUtil(q1Opt(w1LLFOC,w2LLFOC,b,a),w1LLFOC,0)
    # S2utilLLFOC = SupUtil(q2Opt(w1LLFOC,w2LLFOC,b,a), w2LLFOC,0)
    CthetaVec = np.arange(LHFOCLB, 0, -1 * step)
    currLB = CthetaVec[0]  # Move down from this
    found = False
    for Cthetaind, currCtheta in enumerate(CthetaVec):
        if not found:
            w1on, w2on = SupPriceLHexp(scDict, currCtheta)
            currS1util, currS2util = SupUtil(q1Opt(w1on, w2on, b, a),w1on,0), SupUtil(q2Opt(w1on,w2on,b, a), w2on, cS2)
            # Check retailer IR
            if RetUtil(b, w1on, w2on, currCtheta, supRateLo, 1, a) < 0:
                found = True
            # Check S1 LL utility
            w1LLsqz, w2LLsqz = SupPriceLLsqz(scDict, currCtheta)
            currS1utilLLsqz = SupUtil(q1Opt(w1LLsqz,w2LLsqz,b,a),w1LLsqz,0)
            if currCtheta <= LLFOCUB and currS1util < S1utilLLFOC:  # Need to compare with LL
                found = True
            if currCtheta > LLFOCUB and currS1util < currS1utilLLsqz:  # Compare with LLsqz
                found = True
            # Update currLB if made it through checks
            if not found:
                currLB = currCtheta
    return currLB

def CthetaHHFOCLB(scDict, step=1/10000, UBstart=10):
    # HH FOC LB in Ctheta
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    w1on, w2on = SupPriceHH(scDict, 0)
    s1utilOn, s2utilOn = SupUtil(q1Opt(w1on, w2on, b, a), w1on, cS1), SupUtil(q2Opt(w1on, w2on, b, a), w2on, cS2)
    CthetaVec = np.arange(UBstart, 0, -1 * step)
    currLB = CthetaVec[0]  # Move down from this
    found = False
    for Cthetaind, currCtheta in enumerate(CthetaVec):
        if not found:
            w1offRad = np.sqrt((((a - w2on)**2) + 4* currCtheta*(-1 + supRateLo))/(-1 + (b**2)))
            w1off = 1 + b*(-a + w2on + b*w1offRad) - w1offRad
            w2offRad = np.sqrt((((-1 + w1on)**2) +4*currCtheta*(-1 +supRateLo))/(-1 + (b**2)))
            w2off = a + b*(-1 + w1on +b*w2offRad) - w2offRad
            # Check if off-path utility is higher for either supplier
            s1utilOff, s2utilOff = SupUtil(q1Opt(w1off, w2on, b, a), w1off, 0), SupUtil(q2Opt(w1on,w2off,b,a),w2off, 0)
            if s1utilOff > s1utilOn:
                found = True
                devSup = 1
            if s2utilOff > s2utilOn:
                found = True
                devSup = 2
            if not found:
                currLB = currCtheta
    return currLB, devSup

def CthetaLHsqzUB(scDict, step=1/1000, UBmax=10):
    # LH sqz UB in Ctheta
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # We do three checks under each set of prices for given Ctheta: retailer IR, S2's off-path, S1's LL on-path
    LHFOCUB, HHFOCLB = CthetaLHFOCUB(scDict), CthetaHHFOCLB(scDict)[0]
    w1HH, w2HH = SupPriceHH(scDict, 0)
    S1utilHH = SupUtil(q1Opt(w1HH, w2HH, b, a), w1HH, cS1)
    CthetaVec = np.arange(LHFOCUB, UBmax, step)
    currLB = CthetaVec[0]  # Move down from this
    found = False
    for Cthetaind, currCtheta in enumerate(CthetaVec):
        if not found:
            w1on, w2on = SupPriceLHsqz(scDict, currCtheta)
            currS1util = SupUtil(q1Opt(w1on,w2on,b, a), w1on, 0)
            # Check S1 price
            if w1on < 0:
                found = True
            # Check against S1 HH utility
            if currCtheta > HHFOCLB and currS1util < S1utilHH:
                found = True
            # Update currLB if made it through checks
            if not found:
                currLB = currCtheta
    return currLB

def CthetaHHexpAdjLB(scDict, step=1/10000):
    # HH exp adj LB in Ctheta; first checks which supplier wants to move in decreasing Ctheta
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # We do three checks under each set of prices for given Ctheta: retailer IR, S2's off-path, S1's LL on-path
    HHFOCLB, devS = CthetaHHFOCLB(scDict)
    CthetaVec = np.arange(HHFOCLB, 0, -1*step)
    currLB = CthetaVec[0]  # Move down from this
    found = False
    for Cthetaind, currCtheta in enumerate(CthetaVec):
        if not found:
            if devS==1:
                w1on, w2on = SupPriceHHexp1(scDict, currCtheta)
            elif devS==2:
                w1on, w2on = SupPriceHHexp2(scDict, currCtheta)
            else:
                print('No deviating supplier identified')
                return -1
            currS1util, currS2util = SupUtil(q1Opt(w1on,w2on,b,a),w1on,cS1), SupUtil(q2Opt(w1on,w2on,b, a), w2on, cS2)
            # Check against off-path for non-deviating supplier
            if devS==1:
                w2offRad = np.sqrt((((-1 + w1on) ** 2) + 4 * currCtheta * (-1 + supRateLo)) / (-1 + (b ** 2)))
                w2off = a + b * (-1 + w1on + b * w2offRad) - w2offRad
                s2utilOff = SupUtil(q2Opt(w1on, w2off, b, a), w2off, 0)
                if s2utilOff > currS2util:
                    found = True
            elif devS==2:
                w1offRad = np.sqrt((((a - w2on)**2)+4*currCtheta*(-1 + supRateLo)) / (-1 + (b**2)))
                w1off = 1 + b*(-1*a + w2on + b*w1offRad) - w1offRad
                s1utilOff = SupUtil(q1Opt(w1off, w2on, b, a), w1off, 0)
                if s1utilOff > currS1util:
                    found = True
            # Update currLB if made it through checks
            if not found:
                currLB = currCtheta
    return currLB

def CthetaHHexpLB(scDict, step=1/10000):
    # HH exp LB in Ctheta
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # We do three checks under each set of prices for given Ctheta: retailer IR, S2's off-path, S1's LL on-path
    HHexpAdjLB = CthetaHHexpAdjLB(scDict)
    CthetaVec = np.arange(HHexpAdjLB, 0, -1*step)
    currLB = CthetaVec[0]  # Move down from this
    found = False
    for Cthetaind, currCtheta in enumerate(CthetaVec):
        if not found:
            w1on, w2on = SupPriceHHexp(scDict, currCtheta)
            currS1util, currS2util = SupUtil(q1Opt(w1on,w2on,b,a),w1on,cS1), SupUtil(q2Opt(w1on,w2on,b, a), w2on, cS2)
            # Check against S1 HH utility
            if currS1util < 0 or currS2util < 0:
                found = True
            # Update currLB if made it through checks
            if not found:
                currLB = currCtheta
    return currLB

def CthetaLLsqzUB(scDict, step=1/10000, UBmax=10):
    # LL sqz UB in Ctheta
    b, cS1, cS2, supRateLo, a = scDict['b'], scDict['cSup1'], scDict['cSup2'], scDict['supRateLo'], scDict['a']
    # We do three checks under each set of prices for given Ctheta: retailer IR, S2's off-path, S1's LL on-path
    LLFOCUB = CthetaLLFOCUB(scDict)
    CthetaVec = np.arange(LLFOCUB, UBmax, step)
    currLB = CthetaVec[0]  # Move up from this
    found = False
    for Cthetaind, currCtheta in enumerate(CthetaVec):
        if not found:
            w1on, w2on = SupPriceLLsqz5(scDict, currCtheta)
            currS1util, currS2util = SupUtil(q1Opt(w1on, w2on, b, a),w1on,0), SupUtil(q2Opt(w1on,w2on,b, a), w2on, 0)
            # Get off-path prices
            w1off, w2off  = 0.5*(1 - a* b + cS1 + b* w2on), 0.5*(a + cS2 + b*(-1 + w1on))
            offS1util = SupUtil(q1Opt(w1off, w2on, b, a), w1off, cS1)
            offS2util = SupUtil(q2Opt(w1on, w2off, b, a), w2off, cS2)
            if currS1util < offS1util:
                found = True
            if currS2util < offS2util:
                found = True
            # Update currLB if made it through checks
            if not found:
                currLB = currCtheta
    return currLB

def LthetaCsupEqMatsForPlot(numpts, Ctheta_max, cSupDelta_max, scDict):
    # Generate list of equilibria matrices for plotting
    # cSupDelta ranges from 0 to cSupDelta_max, and denotes cSup2-cSup1
    CthetaVec = np.arange(0, Ctheta_max, (Ctheta_max)/numpts)
    cSupVec = np.arange(0.01, cSupDelta_max, cSupDelta_max/numpts)

    eq_list = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHsqz', 'HH']
    eqStrat_matList = np.zeros((len(eq_list), cSupVec.shape[0], CthetaVec.shape[0]))
    eqStrat_matList[:] = np.nan

    for currcSupind in range(cSupVec.shape[0]):
        currdict = scDict.copy()
        currdict['cSup1'] = scDict['cSup2'] + cSupVec[currcSupind]
        # Get bounds under current cSup
        CthLLUB, CthLHFOCLB, CthLLsqzUB = CthetaLLFOCUB(currdict), CthetaLHFOCLB(currdict), CthetaLLsqzUB(currdict)
        CthHHLB, devS = CthetaHHFOCLB(currdict)
        CthLHFOCUB, CthLHsqzUB, CthLHexpLB = CthetaLHFOCUB(currdict), CthetaLHsqzUB(currdict), CthetaLHexpLB(currdict)
        CthHHexpAdjLB, CthHHexpLB = CthetaHHexpAdjLB(currdict), CthetaHHexpLB(currdict)
        eqStrat_matList[0, currcSupind, np.where(CthetaVec < CthLLUB)] = 1  # LL
        eqStrat_matList[1, currcSupind, np.where((CthetaVec > CthLLUB) & (CthetaVec <= CthLLsqzUB))] = 1  # LLsqz
        eqStrat_matList[2, currcSupind, np.where((CthetaVec >= CthLHexpLB) & (CthetaVec < CthLHFOCLB))] = 1  # LHexp
        eqStrat_matList[3, currcSupind, np.where((CthetaVec >= CthLHFOCLB) & (CthetaVec <= CthLHFOCUB))] = 1  # LHFOC
        eqStrat_matList[4, currcSupind, np.where((CthetaVec > CthLHFOCUB) & (CthetaVec < CthLHsqzUB))] = 1  # LHsqz
        eqStrat_matList[5, currcSupind, np.where((CthetaVec >= CthHHexpLB) & (CthetaVec < CthHHLB))] = 1  # HHsqz
        eqStrat_matList[6, currcSupind, np.where(CthetaVec >= CthHHLB)] = 1  # HH

    return eqStrat_matList

def LthetaAdemEqMatsForPlot(numpts, Ctheta_max, a_max, scDict):
    # Generate list of equilibria matrices for plotting
    # cSupDelta ranges from 0 to cSupDelta_max, and denotes cSup2-cSup1
    CthetaVec = np.arange(0, Ctheta_max, (Ctheta_max)/numpts)
    aVec = np.arange(1.01, 1+a_max, a_max/numpts)

    eq_list = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHsqz', 'HH']
    eqStrat_matList = np.zeros((len(eq_list), cSupVec.shape[0], CthetaVec.shape[0]))
    eqStrat_matList[:] = np.nan

    for currcSupind in range(cSupVec.shape[0]):
        currdict = scDict.copy()
        currdict['a'] = aVec[currcSupind]
        # Get bounds under current cSup
        CthLLUB, CthLHFOCLB, CthLLsqzUB = CthetaLLFOCUB(currdict), CthetaLHFOCLB(currdict), CthetaLLsqzUB(currdict)
        CthHHLB, devS = CthetaHHFOCLB(currdict)
        CthLHFOCUB, CthLHsqzUB, CthLHexpLB = CthetaLHFOCUB(currdict), CthetaLHsqzUB(currdict), CthetaLHexpLB(currdict)
        CthHHexpAdjLB, CthHHexpLB = CthetaHHexpAdjLB(currdict), CthetaHHexpLB(currdict)
        eqStrat_matList[0, currcSupind, np.where(CthetaVec < CthLLUB)] = 1  # LL
        eqStrat_matList[1, currcSupind, np.where((CthetaVec > CthLLUB) & (CthetaVec <= CthLLsqzUB))] = 1  # LLsqz
        eqStrat_matList[2, currcSupind, np.where((CthetaVec >= CthLHexpLB) & (CthetaVec < CthLHFOCLB))] = 1  # LHexp
        eqStrat_matList[3, currcSupind, np.where((CthetaVec >= CthLHFOCLB) & (CthetaVec <= CthLHFOCUB))] = 1  # LHFOC
        eqStrat_matList[4, currcSupind, np.where((CthetaVec > CthLHFOCUB) & (CthetaVec < CthLHsqzUB))] = 1  # LHsqz
        eqStrat_matList[5, currcSupind, np.where((CthetaVec >= CthHHexpLB) & (CthetaVec < CthHHLB))] = 1  # HHsqz
        eqStrat_matList[6, currcSupind, np.where(CthetaVec >= CthHHLB)] = 1  # HH

    return eqStrat_matList

def SocWelEqMatsForPlot(numpts, uL_min, uL_max, cSup_max, scDict):
    # Generate list of equilibria matrices for plotting
    # Each point has a social-welfare maximizing equilibria
    b, supRateLo, a = scDict['b'], scDict['supRateLo'], scDict['a']
    # cSup to iterate over
    cSupVec = np.arange(0.01, cSup_max, cSup_max/numpts)
    # uL to iterate over
    uLVec = np.arange(uL_min, uL_max, (uL_max-uL_min)/numpts)

    eq_list = ['LLsqz','LHsqz', 'HH']
    eq_matList = np.zeros((len(eq_list), uLVec.shape[0], cSupVec.shape[0]))
    eq_matList[:] = np.nan

    for currcSupind in range(cSupVec.shape[0]):
        currdict = scDict.copy()
        currdict['cSup1'], currdict['cSup2'] = cSupVec[currcSupind], cSupVec[currcSupind]
        # Get bounds under current cSup
        CthLLsqzUB, CthLHsqzUB = CthetaLLsqzUB(currdict), CthetaLHsqzUB(currdict)
        # Get prices and welfare under each possible valid equilibrium
        w1LLsqz, w2LLsqz = SupPriceLLsqz5(currdict, CthLLsqzUB)
        q1LLsqz, q2LLsqz = q1Opt(w1LLsqz, w2LLsqz, b, a), q2Opt(w1LLsqz, w2LLsqz, b, a)
        w1LHsqz, w2LHsqz = SupPriceLHsqz(currdict, CthLHsqzUB)
        q1LHsqz, q2LHsqz = q1Opt(w1LHsqz, w2LHsqz, b, a), q2Opt(w1LHsqz, w2LHsqz, b, a)
        w1HH, w2HH = SupPriceHH(currdict, 0)
        q1HH, q2HH = q1Opt(w1HH, w2HH, b, a), q2Opt(w1HH, w2HH, b, a)
        for curruLind in range(uLVec.shape[0]):
            socwelLLsqz = SocWel(1, uLVec[curruLind], q1LLsqz, q2LLsqz, 0, 0, supRateLo, supRateLo, 1)
            socwelLHsqz = SocWel(1, uLVec[curruLind], q1LHsqz, q2LHsqz, 0, currdict['cSup2'], supRateLo, 1, 1)
            socwelHH = SocWel(1, uLVec[curruLind], q1HH, q2HH, currdict['cSup1'], currdict['cSup2'], 1, 1, 1)
            besteqind = np.argmax([socwelLLsqz, socwelLHsqz, socwelHH])
            eq_matList[besteqind, curruLind, currcSupind] = 1
    return eq_matList

#######################
# WHOLESALE PRICE PLOTS
#######################
# for b=[0.6, 0.9]
b, cSup1, cSup2, supRateLo, a = 0.75, 0.3, 0.3, 0.8, 1.00000001
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
uH, uL, priceconst = 1, 0.5, 1.0
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