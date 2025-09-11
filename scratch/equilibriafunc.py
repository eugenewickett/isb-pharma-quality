import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
import textwrap
import scipy.optimize as scipyOpt

matplotlib.use('qt5agg')  # pycharm backend doesn't support interactive plots, so we use qt here

np.set_printoptions(precision=3, suppress=True)

exoDict = {'b': 0.5, 'c': 0.07, 'cSup':0.1, 'sens':0.8, 'fpr':0.01, 'lambretlo':0.8,'lambrethi':0.95,'lambsuplo':0.55,
            'lambsuphi':0.95, 'LthetaR':0.4, 'LthetaS':0.1}

def SupUtil(q, w, exoDict, supQual='H', retQual='Y'):
    if supQual=='H':
        lambsup = exoDict['lambsuphi']
    elif supQual=='L':
        lambsup = exoDict['lambsuplo']
    else:
        print('Invalid supplier quality level')
    if retQual=='Y':
        lambret = exoDict['lambrethi']
    elif supQual=='N':
        lambret = exoDict['lambretlo']
    else:
        print('Invalid retailer quality level')
    # Returns supplier utility
    if q > 0:
        retVal = q*(w-exoDict['cSup']) - exoDict['LthetaS']*((exoDict['sens']-exoDict['fpr'])*
                                                             (1-lambret*lambsup)+exoDict['fpr'])
    else:
        retVal = 0
    return retVal

# For consideration of S1 deviation from LHY12
def Getw1A(b, c, cSup, sens, fpr, lambrethi, lambsuplo, lambsuphi, LthetaS, LthetaR, numWpts=30, numWmeshpts=4000):
    # Returns a vector of numWpts values corresponding to w2 values in [0,1], per Step 2 of AC4
    w2Vec = np.arange(1/numWpts,1,1/numWpts)
    wMeshVec = np.arange(-1,2,1/numWmeshpts)
    w1Vec = np.zeros(w2Vec.shape)
    for w2ind, w2 in enumerate(w2Vec):
        lhstermVec = [((1 - b - c + b*c + b*(currw1) - w2)/(2*(1 - b**2))*(w2-cSup) -
                      LthetaS*((sens - fpr)*(1 - (lambrethi*lambsuphi)) + fpr)) for currw1 in wMeshVec]
        radtermVec = [np.sqrt( (1+4*(c**2)+ (-2+w2)*w2 + 2*c*(-2+currw1+w2) - 2*b*(2*(c**2)+ (currw1-1)*(w2-1) + c*(-2+currw1+w2)) +
                               (b**2)*(1+ currw1*(-2+currw1) -4*LthetaR(lambrethi-lambretlo)*(lambsuplo**2)*(sens-fpr)) +
                               4*LthetaR*(lambrethi-lambretlo)*(lambsuplo**2)*(sens-fpr)) / ((-1+(b**2))**2)) for currw1 in wMeshVec]
        rhstermVec = [(1/(2*(1-b**2))) * (1-b-c-b*c+b*currw1 - (1+(b-1)*(-1+currw1+b*radtermVec[currw1ind]) ) ) -
                      LthetaS*((sens - fpr)*(1 - (lambrethi*lambsuplo)) + fpr) for currw1ind,currw1 in enumerate(wMeshVec)]

        secondterm = (((1 - b - c + b*c + b*w2 - (1 - c + b*(-1 + c + w2 + 2*b*np.sqrt((LthetaR* lambrethi *
                            (-1 + lambsuphi)*lambsuphi*(sens-fpr)) / (-1 + b ** 2)))-2*np.sqrt((LthetaR*lambrethi*
                                (-1 +lambsuphi)*lambsuphi*(sens - fpr)) / (-1 + b**2)))) / (2*(1 - b**2))) * ((1 - c +
                           b*(-1 + c + w2 + 2*b*np.sqrt((LthetaR* lambrethi* (-1 + lambsuphi)*lambsuphi *(sens- fpr)) /
                            (-1 + b**2)))-2*np.sqrt((LthetaR*lambrethi*(-1 +lambsuphi)*lambsuphi*(sens - fpr)) /
                                (-1 + b**2))) - cSup) - LthetaS*((sens-fpr)*(1-lambrethi*lambsuphi) + fpr))
        currGapVec = [np.abs(lhstermVec[i]-secondterm) for i in range(len(lhstermVec))]
        w1Vec[w2ind] = wMeshVec[np.argmin(currGapVec)]
        plt.plot(wMeshVec, currGapVec)
        plt.show(block=True)

    plt.plot(w2Vec,w1Vec)
    plt.show(block=True)

    return w1Vec


# For consideration of S2 deviation from LHY12
def Getw1A_S2Dev(exoDict, numWpts=30, numWmeshpts=4000):
    # Returns a vector of numWpts values corresponding to w2 values in [0,1], per Step 2 of AC4
    # These are the max w1 values before w2 wants to switch from H to L, assuming retailer switches to N12
    # todo: UPDATE SUCH THAT THE RETAILER MIGHT SWITCH TO ANY OTHER STRATEGY
    b, c, cSup, sens, fpr = exoDict['b'], exoDict['c'], exoDict['cSup'], exoDict['sens'], exoDict['fpr']
    lambretlo, lambrethi = exoDict['lambretlo'], exoDict['lambrethi']
    lambsuplo, lambsuphi = exoDict['lambsuplo'], exoDict['lambsuphi']
    LthetaR, LthetaS = exoDict['LthetaR'], exoDict['LthetaS']
    w2Vec = np.arange(1/numWpts, 1, 1/numWpts)
    wMeshVec = np.arange(1/numWmeshpts,1,1/numWmeshpts)
    w1Vec = np.zeros(w2Vec.shape)
    for w2ind, w2 in enumerate(w2Vec):
        # First get w2Bgivenw1Aw2A
        radtermVec = [np.sqrt((1 + 4 * (c ** 2) + (-2 + w2) * w2 + 2 * c * (-2 + currw1 + w2) - 2 * b * (
                    2 * (c ** 2) + (currw1 - 1) * (w2 - 1) + c * (-2 + currw1 + w2)) +
                               (b ** 2) * (1 + currw1 * (-2 + currw1) - 4 * LthetaR* (lambrethi - lambretlo) *
                                           (lambsuplo ** 2) * (sens - fpr)) +
                               4 * LthetaR * (lambrethi - lambretlo) * (lambsuplo ** 2) * (sens - fpr)) / (
                                          (-1 + (b ** 2)) ** 2)) for currw1 in wMeshVec]
        w2Bgivenw1Aw2AVec = [1 + b*(-1+currw1 + b*radtermVec[currw1ind]) - radtermVec[currw1ind]  for currw1ind, currw1 in enumerate(wMeshVec)]
        # Put into S2's utility to identify any intersection
        lhstermVec = [((1 - b - c + b*c + b*(currw1) - w2)/(2*(1 - b**2))*(w2-cSup) -
                      LthetaS*((sens - fpr)*(1 - (lambrethi*lambsuphi)) + fpr)) for currw1 in wMeshVec]

        rhstermVec = [((1 - b - c + b*c + b*(currw1) - w2Bgivenw1Aw2AVec[currw1ind])/(2*(1 - b**2))*(w2Bgivenw1Aw2AVec[currw1ind]) -
                      LthetaS*((sens - fpr)*(1 - (lambrethi*lambsuplo)) + fpr)) for currw1ind, currw1 in enumerate(wMeshVec)]
        # Do we have an intersection?
        w1priceInds = [lhstermVec[i]>rhstermVec[i] for i in range(len(lhstermVec))]
        if all(lhstermVec[i]>rhstermVec[i] for i in range(len(lhstermVec))):
            # S2 has no incentive to switch, regardless of w1A
            print('S2 NEVER switches, regardless of w1')
            w1Vec[w2ind] = -1
        elif all(lhstermVec[i]<rhstermVec[i] for i in range(len(lhstermVec))):
            # S2 always has incentive to switch, regardless of w1A
            print('S2 ALWAYS switches, regardless of w1')
            w1Vec[w2ind] = -2
        else:
            # Whether S2 switches depends on S1; S1 will
            print('S2 switch depends on w1')
            w1Vec[w2ind] = np.max(wMeshVec[w1priceInds])

        plt.plot(lhstermVec)
        plt.plot(rhstermVec)
        plt.show(block=True)

    return w1Vec

# S2 will select w2 s.t. the corresponding w1|w2 yields the highest utility
def GetEqPrice_S2Dev(w1A_S2DevVec, exoDict):
    # Check S2 utility under each pair of prices
    numWpts = w1A_S2DevVec.shape[0]+1
    w2Vec = np.arange(1/numWpts, 1, 1/numWpts)
    b, c, cSup, sens, fpr = exoDict['b'], exoDict['c'], exoDict['cSup'], exoDict['sens'], exoDict['fpr']
    lambretlo, lambrethi = exoDict['lambretlo'], exoDict['lambrethi']
    lambsuplo, lambsuphi = exoDict['lambsuplo'], exoDict['lambsuphi']
    LthetaR, LthetaS = exoDict['LthetaR'], exoDict['LthetaS']
    # Initialize return values
    w1star, w2star, S2utilstar = 0, 0, 0
    for w2ind, w2 in enumerate(w2Vec):
        w1 = w1A_S2DevVec[w2ind]
        if w1 == -1:  # S2 always stays with H; evaluate on-path LHY12
            w1curr, w2curr = ((2-b-(b**2))*(1-c)+b*cSup)/(4-(b**2)), ((2-b-(b**2))*(1-c)+2*cSup)/(4-(b**2))
            q = (1 - b - c + b * c - w2curr + b * w1curr) / (2 * (1 - (b ** 2)))
            S2util = SupUtil(q, w2curr, exoDict, supQual='H', retQual='Y')
        elif w1 == -2:  # S2 always chooses L; evaluate on-path LLY12
            w1curr, w2curr = (1-b)*(1-c) / (2-b), (1-b)*(1-c) / (2-b)
            q = (1 - b - c + b * c - w2curr + b * w1curr) / (2 * (1 - (b ** 2)))
            S2util = SupUtil(q, w2curr, exoDict, supQual='L', retQual='Y')
        else:
            w1curr, w2curr = w1, w2
            q = (1-b-c+b*c-w2curr+b*w1curr)/(2*(1-(b**2)))
            S2util = SupUtil(q, w2curr, exoDict, supQual='H', retQual='Y')

        print(w1curr, w2curr, S2util)
        # Check if improved S2 utility
        if S2util > S2utilstar:
            w1star, w2star, S2utilstar =  w1curr, w2curr, S2util

    return w1star, w2star, S2utilstar



w1A_S2DevVec = Getw1A_S2Dev(exoDict)

GetEqPrice_S2Dev(w1A_S2DevVec, exoDict)



