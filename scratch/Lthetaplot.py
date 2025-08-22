import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
import textwrap

matplotlib.use('qt5agg')  # pycharm backend doesn't support interactive plots, so we use qt here

np.set_printoptions(precision=3, suppress=True)

def SPUtil(q1, q2, lambret, lambsup1, lambsup2, alph):
    # Social planner's utility
    return q1*(alph+(lambret*lambsup1)-1) + q2*(alph+(lambret*lambsup2)-1)
# Function returning retailer utilities under each of 7 possible policies
def UtilsRet_Scen5(lambsup1, lambsup2, Ltheta, b, c, w1, w2, lambretlo, lambrethi, sens, fpr, sourceBothSups=True,
                   onlyDoubleSource=False, onlyQualInvest=False, const=-0.9):
    # sourceBothSups indicates if both suppliers are eligible to be sourced; if False then only S1 can be sourced
    # onlyDoubleSource restricts retailer to choosing 0, 1, or 2 suppliers while making the quality investment
    util_list = []
    # What is retailer's utility as a function of different policies?
    # Returns a list of utilities for each policy under the given inputs
    # 0={Y12}, 1={N12}, 2={Y1}, 3={N1}, 4={Y2}, 5={N2}, 6={N}
    # Policy {Y12}

    q1 = max((1 - b - c + b*c - w1 + b*w2) / (2 * (1-(b**2))), 0)
    q2 = max((1 - b - c + b*c - w2 + b*w1) / (2 * (1-(b**2))), 0)
    util_list.append((1 - b - c + b*c - w1 + b*w2)*(1 - c - w1 - b*q2 - q1) / (2 * (1-b**2)) + \
                     (1 - b - c + b*c + b*w1 - w2)*(1 - c - w2 - b*q1 - q2) / (2 * (1-b**2)) - \
                     Ltheta * (fpr + (sens - fpr) * (1 - lambrethi * lambsup1 * lambsup2)) )

    # Policy {N12}
    q1 = max((1 - b - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
    q2 = max((1 - b - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
    util_list.append((1 - b - w1 + b*w2) * (1 - w1 - b*q2 - q1) / (2 * (1 - b ** 2)) + \
                     (1 - b + b*w1 - w2) * (1 - w2 - b*q1 - q2) / (2 * (1 - b ** 2)) - \
                     Ltheta * (fpr + (sens - fpr) * (1 - lambretlo * lambsup1 * lambsup2)) )
    # Policy {Y1}
    util_list.append(0.5 * (1 - c - w1) * (1 - c - w1 + 0.5 * (-1 + c + w1)) - Ltheta * (
                fpr + (sens - fpr) * (1 - lambrethi * lambsup1)) )
    # Policy {N1}
    util_list.append(0.5 * (1 - w1) * (1 - w1 + 0.5 * (-1 + w1)) - Ltheta * (
            fpr + (sens - fpr) * (1 - lambretlo * lambsup1)) )
    # Policy {Y2}
    util_list.append(0.5 * (1 - c - w2) * (1 - c - w2 + 0.5 * (-1 + c + w2)) - Ltheta * (
            fpr + (sens - fpr) * (1 - lambrethi * lambsup2)) )
    # Policy {N2}
    util_list.append(0.5 * (1 - w2) * (1 - w2 + 0.5 * (-1 + w2)) - Ltheta * (
            fpr + (sens - fpr) * (1 - lambretlo * lambsup2)) )
    # Policy {N}
    util_list.append(0)
    if sourceBothSups is False:  # S2 is not possibly sourced; set negative so that {N} is preferred
        util_list[0], util_list[1], util_list[4], util_list[5] = -1, -1, -1, -1
    if onlyDoubleSource is True:  # Either both suppliers are sourced, or none
        util_list[2], util_list[3], util_list[4], util_list[5] = -1, -1, -1, -1
    if onlyQualInvest is True:  # Only 'Y' strategies considered
        util_list[1], util_list[3], util_list[5] = -1, -1, -1
    return util_list

def RetOrderQuantsFromStrat(stratind, b, c, w1, w2):
    # Returns order quantities under inputs. If w2=-1, single sourcing quantity given
    strat = int(stratind)
    if strat == 0:
        q1 = max((1 - b - c + b * c - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
        q2 = max((1 - b - c + b * c - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
    elif strat == 1:
        q1 = max((1 - b - w1 + b * w2) / (2 * (1 - (b ** 2))), 0)
        q2 = max((1 - b - w2 + b * w1) / (2 * (1 - (b ** 2))), 0)
    elif strat == 2:
        q1 = max((1 - c - w1) / 2, 0)
        q2 = 0
    elif strat == 3:
        q1 = max((1 - w1) / 2, 0)
        q2 = 0
    elif strat == 4:
        q1 = 0
        q2 = max((1 - c - w2) / 2, 0)
    elif strat == 5:
        q1 = 0
        q2 = max((1 - w2) / 2, 0)
    elif strat == 6:
        q1 = 0
        q2 = 0
    return q1, q2

def retStratToStr(stratInt):
    # Returns string for integer retailer strategy
    if stratInt == 0:
        retStr = ', {Y12}'
    elif stratInt == 1:
        retStr = ', {N12}'
    elif stratInt == 2:
        retStr = ', {Y1}'
    elif stratInt == 3:
        retStr = ', {N1}'
    elif stratInt == 4:
        retStr = ', {Y2}'
    elif stratInt == 5:
        retStr = ', {N2}'
    elif stratInt == 6:
        retStr = ', {N}'
    return retStr

def SupSymPrices(b, c, cSup):
    # Returns wholesale prices under symmetric supplier quality levels
    w = max((1 - b - c + b * c + cSup) / (2 - b), 0)
    return w, w

def SupAsymPrices(b, c, cSup):
    # Returns wholesale prices under asymmetric supplier quality levels, assuming S1 is low-quality
    w1 = max(((b-1) * (b+2) * (c-1) + (b*cSup)) / (4-b**2), 0)
    w2 = max(((b-1) * (b+2) * (c-1) + (2*cSup)) / (4-b**2), 0)
    return w1, w2

def SupExclPrices(b, c, cSup, s1QualInvest, lambsuphi, lambsuplo, lambrethi, lambretlo, LthetaR, LthetaS, sens, fpr,
                  numWpts=50, printUpdate=False):
    # Returns wholesale limit price under single-supplier sourcing
    # Needs to be sufficiently low such that the non-sourced supplier does not have an incentive to enter the market
    wVec = np.arange(1 / numWpts, 1, 1 / numWpts)
    if s1QualInvest is True:  # S1 makes quality investment
        lambsup1 = lambsuplo
    else:
        lambsup1 = lambsuphi
    bestw1 = 0.0
    if printUpdate is True:
        print('Seeking limit price...')
    for w1 in wVec:
        validLimit = True
        for w2 in wVec:
            for lambsup2 in [lambsuphi, lambsuplo]:
                currRetUtilVec = UtilsRet_Scen5(lambsup1, lambsup2, LthetaR, b, c, w1, w2, lambretlo, lambrethi, sens,
                                                fpr)
                retBestStrat = np.argmax(currRetUtilVec)
                _, q2 = RetOrderQuantsFromStrat(currRetUtilVec[retBestStrat], b, c, w1, w2)
                if lambsup2 == lambsuphi:
                    cSup2 = cSup
                else:
                    cSup2 = 0
                if retBestStrat in [0, 2, 4]:
                    lambret = lambrethi
                else:
                    lambret = lambretlo
                if SupUtil(q2, w2, cSup2, LthetaS, lambret, lambsup2, sens, fpr) > 0:
                    validLimit = False
        if validLimit is True:
            bestw1 = w1
            if printUpdate is True:
                print('Higher limit price found: ' + str(w1))

    return bestw1, bestw1

def WholesalePricesFromEq(eqind, b, c, cSup, lambsuplo, lambsuphi, lambretlo, lambrethi, LthetaR, LthetaS, sens, fpr,
                          printUpdate=False, numWpts=50):
    # Returns wholesale prices under given equilibrium
    eq = int(eqind)
    if eq == 0:  # {HHY12}
        w1, w2 = SupSymPrices(b, c, cSup)
    elif eq == 1:  # {HHN12}
        w1, w2 = SupSymPrices(b, 0, cSup)
    elif eq == 2:  # {LLY12}
        w1, w2 = SupSymPrices(b, c, 0)
    elif eq == 3:  # {LLN12}
        w1, w2 = SupSymPrices(b, 0, 0)
    elif eq == 4:  # {LHY12}
        w1, w2 = SupAsymPrices(b, c, cSup)
    elif eq == 5:  # {LHN12}
        w1, w2 = SupAsymPrices(b, 0, cSup)
    elif eq == 6:  # {HHY1}
        w1, w2 = SupExclPrices(b, c, cSup, True, lambsuphi, lambsuplo, lambrethi, lambretlo, LthetaR, LthetaS, sens,
                               fpr, printUpdate=printUpdate, numWpts=numWpts)
    elif eq == 7:  # {HHN1}
        w1, w2 = SupExclPrices(b, c, cSup, True, lambsuphi, lambsuplo, lambrethi, lambretlo, LthetaR, LthetaS, sens,
                               fpr, printUpdate=printUpdate, numWpts=numWpts)
    elif eq == 8:  # {LLY1}
        w1, w2 = SupExclPrices(b, c, cSup, False, lambsuphi, lambsuplo, lambrethi, lambretlo, LthetaR, LthetaS, sens,
                               fpr, printUpdate=printUpdate, numWpts=numWpts)
    elif eq == 9:  # {LLN1}
        w1, w2 = SupExclPrices(b, c, cSup, False, lambsuphi, lambsuplo, lambrethi, lambretlo, LthetaR, LthetaS, sens,
                               fpr, printUpdate=printUpdate, numWpts=numWpts)
    elif eq == 10:  # {HLY1}
        w1, w2 = SupExclPrices(b, c, cSup, True, lambsuphi, lambsuplo, lambrethi, lambretlo, LthetaR, LthetaS, sens,
                               fpr, printUpdate=printUpdate, numWpts=numWpts)
    elif eq == 11:  # {HLN1}
        w1, w2 = SupExclPrices(b, c, cSup, True, lambsuphi, lambsuplo, lambrethi, lambretlo, LthetaR, LthetaS, sens,
                               fpr, printUpdate=printUpdate, numWpts=numWpts)
    elif eq == 12:  # {LHY1}
        w1, w2 = SupExclPrices(b, c, cSup, False, lambsuphi, lambsuplo, lambrethi, lambretlo, LthetaR, LthetaS, sens,
                               fpr, printUpdate=printUpdate, numWpts=numWpts)
    elif eq == 13:  # {LHN1}
        w1, w2 = SupExclPrices(b, c, cSup, False, lambsuphi, lambsuplo, lambrethi, lambretlo, LthetaR, LthetaS, sens,
                               fpr, printUpdate=printUpdate, numWpts=numWpts)
    else:
        print('Invalid equilibrium')
    return w1, w2

def SupUtil(q, w, cSup, Ls, lambret, lambsup, sens, fpr):
    # Returns supplier utility
    if q > 0:
        retVal = q*(w-cSup) - Ls*((sens-fpr)*(1-lambret*lambsup)+fpr)
    else:
        retVal = 0
    return retVal

def GetTransInd(eq, retPref = -1, sup1Pref = -1, sup2Pref = -1):
    # Return transition strategy state as function of current state and deviating player
    eq_list = ['HHY12', 'HHN12', 'LLY12', 'LLN12', 'LHY12', 'LHN12', 'HHY1', 'HHN1', 'LLY1', 'LLN1', 'HLY1', 'HLN1',
               'LHY1', 'LHN1', 'N']
    ret_strat_list = ['Y12', 'N12', 'Y1', 'N1', 'Y2', 'N2', 'N']
    eqInd = eq_list.index(eq)
    if retPref >= 0 and sup1Pref < 0 and sup2Pref < 0:  # Retailer deviates
        if sup1Pref >= 0 or sup2Pref >= 0:
            print('ERROR: More than 1 deviating player')
            return
        if retPref < 4:
            eq_trans = eq[:2] + ret_strat_list[retPref]
        elif retPref < 6:  # Y2 or N2; need to switch which retailer is L or H
            eq_trans = eq[1]+eq[0] + ret_strat_list[retPref-2]
        elif retPref == 6:  # Retailer doesn't participate
            eq_trans = 'N'
        else:
            print('ERROR: Retailer transition not accounted for')
            return
        # Account for equivalent states
        if eq_trans == 'HLY12':
            eq_trans = 'LHY12'
        elif eq_trans == 'HLN12':
            eq_trans = 'LHN12'
    elif sup1Pref >= 0:  # Supplier 1 deviates
        # What is the retailer doing?
        if retPref < 0:  # Stays the same
            retpol = eq[2:]
        else:  # Changes
            retpol = ret_strat_list[retPref]

        if sup2Pref >= 0:
            print('ERROR: More than 1 deviating player')
            return
        if eq[0] == 'H':
            eq_trans = 'L' + eq[1] + retpol
        elif eq[0] == 'L':
            eq_trans = 'H' + eq[1] + retpol
        # Account for equivalent states
        if eq_trans == 'HLY12':
            eq_trans = 'LHY12'
        elif eq_trans == 'HLN12':
            eq_trans = 'LHN12'
        # Account for Y2/N2 policies
        if retPref in [4, 5]:
            eq_trans = eq_trans[1] + eq_trans[0] + ret_strat_list[retPref - 2]
    elif sup2Pref >= 0:  # Supplier 2 deviates
        # What is the retailer doing?
        if retPref < 0:  # Stays the same
            retpol = eq[2:]
        else:  # Changes
            retpol = ret_strat_list[retPref]

        if eq[1] == 'H':
            eq_trans = eq[0] + 'L' + retpol
        elif eq[1] == 'L':
            eq_trans = eq[0] + 'H' + retpol
        # Account for equivalent states
        if eq_trans == 'HLY12':
            eq_trans = 'LHY12'
        elif eq_trans == 'HLN12':
            eq_trans = 'LHN12'
        # Account for Y2/N2 policies
        if retPref in [4, 5]:
            eq_trans = eq_trans[1]+eq_trans[0] + ret_strat_list[retPref-2]
    return eq_list.index(eq_trans)

def GetQuantMatFromEqMats(eqStrat_matList, eqQuant_matList):
    eqQuant_matList_plot = []
    numLpts = eqStrat_matList[0].shape[0]
    for i in range(len(eqStrat_matList)):
        temp = eqStrat_matList[i].copy()
        temp.flatten()
        eqStrat_matList_dup = np.repeat(temp, 2)
        temp2 = eqStrat_matList_dup.reshape((numLpts, numLpts, 2))
        qMat = temp2 * eqQuant_matList[i]
        eqQuant_matList_plot.append(qMat)
    eqQuant_plot = np.sum(np.nanmean(np.array(eqQuant_matList_plot), axis=0), axis=2)
    return eqQuant_plot

def GetQualMatsFromEqMat(eqStrat_matList, eqStrat_vec):
    # Produces matrix of retailer and supplier quality investments using equilibria
    # Presence of HL or multiple equilibria produces value of 0.5
    eqStratArr = np.array(eqStrat_matList)
    retQual_plot = np.empty(eqStratArr.shape[1:])
    supQual_plot = np.empty(eqStratArr.shape[1:])
    for lr in range(eqStratArr.shape[1]):
        for ls in range(eqStratArr.shape[2]):
            # Check indices of equilibria
            currVec = eqStratArr[:, lr, ls]
            currEqInds = np.where(currVec==1)[0].tolist()
            currEq = [eqStrat_vec[x] for x in currEqInds]
            # Retailer quality levels
            if len(currEq) == 0:
                retQual_plot[lr, ls] = np.nan
            elif set(currEq).issubset(['{HHY12}', '{LLY12}',  '{LHY12}',  '{HHY1}',  '{LLY1}', '{HLY1}', '{LHY1}']):  # All Y
                retQual_plot[lr, ls] = 1
            elif set(currEq).issubset(['{HHN12}', '{LLN12}',  '{LHN12}',  '{HHN1}',  '{LLN1}', '{HLN1}', '{LHN1}']):  # All N
                retQual_plot[lr, ls] = 0
            else:  # Mixture
                retQual_plot[lr, ls] = 0.5
            # Supplier quality levels
            if len(currEq) == 0:
                supQual_plot[lr, ls] = np.nan
            elif set(currEq).issubset(['{LLY12}', '{LLN12}', '{LLY1}', '{LLN1}']):  # All LL
                supQual_plot[lr, ls] = 0
            elif set(currEq).issubset(['{HHY12}', '{HHN12}', '{HHY1}', '{HHN1}']):  # All HH
                supQual_plot[lr, ls] = 1
            else:  # Mixture
                supQual_plot[lr, ls] = 0.5

    return retQual_plot, supQual_plot

def GetQualMatsFromMixedEqMat(eqStrat_matList, eqStrat_vec, eqMix_matList, eqMixName_List):
    eqStratArr = np.array(eqStrat_matList)
    retQual_plot = np.empty(eqStratArr.shape[1:])
    supQual_plot = np.empty(eqStratArr.shape[1:])

    eqMixArr = np.array(eqMix_matList)

    for lr in range(eqStratArr.shape[1]):
        for ls in range(eqStratArr.shape[2]):
            # Check indices of equilibria
            currVec = eqStratArr[:, lr, ls]
            currEqInds = np.where(currVec==1)[0].tolist()
            currEq = [eqStrat_vec[x] for x in currEqInds]
            # Retailer quality levels
            if len(currEq) == 0:  # Either mixed or {N}
                # retQual_plot[lr, ls] = np.nan
                currMixVec = eqMixArr[:, lr, ls]
                currMixInds = np.where(currMixVec==1)[0].tolist()
                if currMixInds == [0]:  # {N}
                    retQual_plot[lr, ls] = 0
                    supQual_plot[lr, ls] = 0
                else:
                    mixList = [j for i in currMixInds for j in eqMixName_List[i]]
                    if set(mixList).issubset(['HHY12', 'LLY12',  'LHY12',  'HHY1',  'LLY1', 'HLY1', 'LHY1']):  # All Y
                        retQual_plot[lr, ls] = 1
                    elif set(mixList).issubset(['HHN12', 'LLN12',  'LHN12',  'HHN1',  'LLN1', 'HLN1', 'LHN1']):  # All N
                        retQual_plot[lr, ls] = 0
                    else:
                        retQual_plot[lr, ls] = 0.5
                    # Suppliers
                    if set(mixList).issubset(['LLY12', 'LLN12', 'LLY1', 'LLN1']):  # All LL
                        supQual_plot[lr, ls] = 0
                    elif set(mixList).issubset(['HHY12', 'HHN12', 'HHY1', 'HHN1']):  # All HH
                        supQual_plot[lr, ls] = 1
                    else:
                        supQual_plot[lr, ls] = 0.5

            elif set(currEq).issubset(['{HHY12}', '{LLY12}',  '{LHY12}',  '{HHY1}',  '{LLY1}', '{HLY1}', '{LHY1}']):  # All Y
                retQual_plot[lr, ls] = 1
            elif set(currEq).issubset(['{HHN12}', '{LLN12}',  '{LHN12}',  '{HHN1}',  '{LLN1}', '{HLN1}', '{LHN1}']):  # All N
                retQual_plot[lr, ls] = 0
            else:  # Mixture
                retQual_plot[lr, ls] = 0.5
            # Supplier quality levels
            if len(currEq) == 0:
                # supQual_plot[lr, ls] = np.nan
                pass
            elif set(currEq).issubset(['{LLY12}', '{LLN12}', '{LLY1}', '{LLN1}']):  # All LL
                supQual_plot[lr, ls] = 0
            elif set(currEq).issubset(['{HHY12}', '{HHN12}', '{HHY1}', '{HHN1}']):  # All HH
                supQual_plot[lr, ls] = 1
            else:  # Mixture
                supQual_plot[lr, ls] = 0.5

    return retQual_plot, supQual_plot

def GetSPUtilMatFromQuantQualMats(al, retQual_plot, supQual_plot, eqQuant_plot, lambretlo, lambrethi, lambsuplo,
                                  lambsuphi):
    SPUtil_plot = np.empty(retQual_plot.shape)
    for lr in range(SPUtil_plot.shape[0]):
        for ls in range(SPUtil_plot.shape[1]):
            if not np.isnan(retQual_plot[lr, ls]):
                currRetQual = ((1-retQual_plot[lr, ls]) * lambretlo) + ((retQual_plot[lr, ls]) * lambrethi)
                currSupQual = ((1 - supQual_plot[lr, ls]) * lambsuplo) + ((supQual_plot[lr, ls]) * lambsuphi)
                SPUtil_plot[lr, ls] = SPUtil(eqQuant_plot[lr, ls], 0, currRetQual, currSupQual, 0, al)
            else:
                SPUtil_plot[lr, ls] = np.nan
    return SPUtil_plot

def steadyStateMat(P, printUpdate = False):
    # Returns steady-state vector for transition matrix P
    # If singulartiy issues, break into smaller matrices
    if P.sum() == 0:
        retVec = np.zeros(P.shape[0])
    elif np.diag(P)[:-1].sum() > 0:  # We have some equilibria
        retVec = np.diag(P)
    else:
        try:
            A = np.transpose(P) - np.eye(P.shape[0])
            A[-1] = np.ones(P.shape[0])  # Constraint: sum(pi) = 1
            b = np.zeros(P.shape[0])
            b[-1] = 1
            retVec = np.linalg.solve(A, b)
        except:
            if printUpdate is True:
                print('Singular transition matrix identified')
            # Break into 2 sub matrices: those connected to 0 and those that are not
            mainlist = [i for i in range(P.shape[0])]
            list1 = [0]
            poplist = [0]
            while len(poplist) > 0:
                popind = poplist.pop()  # grab next index
                # get all rows and columns tied to this index not already in list1
                colinds = np.where(P[:, popind] == 1)[0].tolist()
                rowinds = np.where(P[popind, :] == 1)[0].tolist()
                for ind in colinds:
                    if ind not in list1 and ind not in poplist:
                        list1.append(ind)
                        poplist.append(ind)
                for ind in rowinds:
                    if ind not in list1 and ind not in poplist:
                        list1.append(ind)
                        poplist.append(ind)
            list1.sort()
            list2 = [mainlist[i] for i in range(len(mainlist)) if (i not in list1 and P[i].sum()>0)]
            ixgrid1 = np.ix_(list1, list1)
            ixgrid2 = np.ix_(list2, list2)
            P1 = P[ixgrid1]
            P2 = P[ixgrid2]
            P1vec = steadyStateMat(P1)
            P2vec = steadyStateMat(P2)
            # Zip together these 2 vectors according to the indices of list1 and list2
            retVec = np.zeros(len(mainlist))
            for ind in range(len(mainlist)):
                if ind in list1:
                    listind = list1.index(ind)
                    retVec[ind] = P1vec[listind]
                elif ind in list2:
                    listind = list2.index(ind)
                    retVec[ind] = P2vec[listind]

    return retVec

def LthetaEqMatsForPlot(numLpts, Ltheta_max, numWpts, lambsuplo, lambsuphi, b, c, cSup, lambretlo, lambrethi, sens, fpr,
                        printUpdates = False, Lr_insp_min = 0.0, Lr_insp_max = 0.0, Ls_insp_min = 0.0,
                        Ls_insp_max = 0.0, Lr_ind_Tmat = -1, Ls_ind_Tmat = -1, onlyDoubleSource=False,
                        onlyQualInvest=False, eps=0.001):
    # Generate list of equilibria matrices for plotting, including quality levels and order quantities
    # Loop through possible equilibria and generate matrices demarcating where they are Nash equilibria
    # 0:{HHY12}, 1:{HHN12}, 2:{LLY12}, 3:{LLN12}, 4:{LHY12}, 5:{LHN12}, 6:{HHY1}, 7:{HHN1}, 8:{LLY1}, 9:{LLN1},
    #   10:{HLY1}, 11:{HLN1}, 12:{LHY1}, 13:{LHN1}
    w_vec = np.arange(1 / numWpts, 1, 1 / numWpts)

    eq_list = ['HHY12', 'HHN12', 'LLY12', 'LLN12', 'LHY12', 'LHN12', 'HHY1', 'HHN1', 'LLY1', 'LLN1', 'HLY1', 'HLN1',
               'LHY1', 'LHN1']
    eqStrat_matList = []
    eqQuant_matList = []
    Ltheta_vec = np.arange(0, Ltheta_max + 0.00001, Ltheta_max / numLpts)

    Lr_insp_vec = Ltheta_vec[Ltheta_vec > Lr_insp_min]
    Lr_insp_vec = Lr_insp_vec[Lr_insp_vec < Lr_insp_max]
    Ls_insp_vec = Ltheta_vec[Ltheta_vec > Ls_insp_min]
    Ls_insp_vec = Ls_insp_vec[Ls_insp_vec < Ls_insp_max]

    # Initialize transition matrix
    Tmat = np.zeros((len(eq_list)+1, len(eq_list)+1))

    for eqind, eq_curr in enumerate(eq_list):
        # lambsup_eq
        if eq_curr in ['HHY12', 'HHN12', 'HHY1', 'HHN1']:
            lambsup1_eq = lambsuphi
            lambsup2_eq = lambsuphi
        elif eq_curr in ['LLY12', 'LLN12', 'LLY1', 'LLN1']:
            lambsup1_eq = lambsuplo
            lambsup2_eq = lambsuplo
        elif eq_curr in ['LHY12', 'LHN12', 'LHY1', 'LHN1']:
            lambsup1_eq = lambsuplo
            lambsup2_eq = lambsuphi
        elif eq_curr in ['HLY1', 'HLN1']:
            lambsup1_eq = lambsuphi
            lambsup2_eq = lambsuplo
        # supplier sourcing
        if eq_curr in ['HHY12', 'HHN12', 'LLY12', 'LLN12', 'LHY12', 'LHN12']:
            bothSupsSourced = True
        else:
            bothSupsSourced = False
        # symmetric suppliers
        if eq_curr in ['HHY12', 'HHN12', 'LLY12', 'LLN12', 'HHY1', 'HHN1', 'LLY1', 'LLN1']:
            bothSupsSym = True
        elif eq_curr in ['LHY12', 'LHN12', 'HLY1', 'HLN1', 'LHY1', 'LHN1']:
            bothSupsSym = False
        # retailer strategy
        if eq_curr in ['HHY12', 'LLY12', 'LHY12']:
            retStrat_eq = 0
            lambret_eq = lambrethi
        elif eq_curr in ['HHN12', 'LLN12', 'LHN12']:
            retStrat_eq = 1
            lambret_eq = lambretlo
        elif eq_curr in ['HHY1', 'LLY1', 'HLY1', 'LHY1']:
            retStrat_eq = 2
            lambret_eq = lambrethi
        elif eq_curr in ['HHN1', 'LLN1', 'HLN1', 'LHN1']:
            retStrat_eq = 3
            lambret_eq = lambretlo

        if eqind < 6:  # Double-supplier sourcing
            # wholesale prices
            w1_eq, w2_eq = WholesalePricesFromEq(eqind, b, c, cSup, lambsuplo, lambsuphi, lambretlo, lambrethi, 0,
                                                 0, sens, fpr, printUpdates)
            # Retailer order quantities
            q1_eq, q2_eq = RetOrderQuantsFromStrat(retStrat_eq, b, c, w1_eq, w2_eq)
            if printUpdates is True:
                print('Current eq point: ' + eq_curr)
                print('q1, q2: '+str(round(q1_eq, 3))+', '+str(round(q2_eq, 3)))
                print('w1, w2: ' + str(round(w1_eq, 3)) + ', ' + str(round(w2_eq, 3)))
            limitPrice = False
        # Loop through Ltheta; rows are Lr, cols are Ls
        eq_mat = np.zeros((numLpts + 1, numLpts + 1))
        eqQuant_mat = np.zeros((numLpts + 1, numLpts + 1, 2))  # S1, then S2
        for Lrind, Lr in enumerate(Ltheta_vec):
            for Lsind, Ls in enumerate(Ltheta_vec):
                if Lr in Lr_insp_vec and Ls in Ls_insp_vec and printUpdates is True:
                    print('Lr: ' + str(round(Lr, 3)))
                    print('Ls: ' + str(round(Ls, 3)))
                # Proceed if no transition matrix L indices specified or if we are at our focus point
                if (Lr_ind_Tmat < 0 and Ls_ind_Tmat < 0) or (Lrind == Lr_ind_Tmat and Lsind == Ls_ind_Tmat):
                    if eqind >= 6:  # Single-supplier sourcing
                        # wholesale prices
                        w1_eq, w2_eq = WholesalePricesFromEq(eqind, b, c, cSup, lambsuplo, lambsuphi, lambretlo,
                                                             lambrethi, Lr, Ls, sens, fpr, printUpdates, numWpts=numWpts)
                        # Retailer order quantities
                        q1_eq, q2_eq = RetOrderQuantsFromStrat(retStrat_eq, b, c, w1_eq, w2_eq)
                        if printUpdates is True:
                            print('Current eq point: ' + eq_curr)
                            print('q1, q2: ' + str(round(q1_eq, 3)) + ', ' + str(round(q2_eq, 3)))
                            print('w1, w2: ' + str(round(w1_eq, 3)) + ', ' + str(round(w2_eq, 3)))
                        limitPrice = True

                    if Lrind == Lr_ind_Tmat and Lsind == Ls_ind_Tmat:
                        storeTmat = True
                    else:
                        storeTmat = False
                    # Initialize matrix storage value to 1, and update to 0 if any equilibrium conditions violated
                    matVal = 1

                    # CHECK 1: Suppliers have non-negative utility
                    if eq_curr in ['HHY12', 'HHN12', 'HHY1', 'HHN1']:
                        s1util_eq = SupUtil(q1_eq, w1_eq, cSup, Ls, lambret_eq, lambsup1_eq, sens, fpr)
                        s2util_eq = SupUtil(q2_eq, w2_eq, cSup, Ls, lambret_eq, lambsup2_eq, sens, fpr)
                    elif eq_curr in ['LLY12', 'LLN12', 'LLY1', 'LLN1']:
                        s1util_eq = SupUtil(q1_eq, w1_eq, 0, Ls, lambret_eq, lambsup1_eq, sens, fpr)
                        s2util_eq = SupUtil(q2_eq, w2_eq, 0, Ls, lambret_eq, lambsup2_eq, sens, fpr)
                    elif eq_curr in ['LHY12', 'LHN12', 'LHY1', 'LHN1']:
                        s1util_eq = SupUtil(q1_eq, w1_eq, 0, Ls, lambret_eq, lambsup1_eq, sens, fpr)
                        s2util_eq = SupUtil(q2_eq, w2_eq, cSup, Ls, lambret_eq, lambsup2_eq, sens, fpr)
                    elif eq_curr in ['HLY1', 'HLN1']:
                        s1util_eq = SupUtil(q1_eq, w1_eq, cSup, Ls, lambret_eq, lambsup1_eq, sens, fpr)
                        s2util_eq = SupUtil(q2_eq, w2_eq, 0, Ls, lambret_eq, lambsup2_eq, sens, fpr)

                    if s1util_eq < 0 or s2util_eq < 0:
                        matVal = np.nan
                        if printUpdates is True:
                            print('A supplier utility is negative')
                            print('S1 util: ' + str(round(s1util_eq, 3)))
                            print('S2 util: ' + str(round(s2util_eq, 3)))
                        if storeTmat is True:
                            # Store transition from current equilibrium
                            Tmat[eqind, len(eq_list)] = 1

                    # CHECK 2: Suppliers want to deviate
                    if lambsup1_eq == lambsuphi:
                        lambsup1_dev = lambsuplo
                    elif lambsup1_eq == lambsuplo:
                        lambsup1_dev = lambsuphi
                    # Iterate through possible deviation prices for S1
                    if matVal == 1 and limitPrice is False:
                        breakloop = False
                        for w1_dev in w_vec:
                            if w1_dev != w1_eq and breakloop is False:  # Deviation price cannot be the equilibrium price
                                # What is retailer policy under deviation?
                                if bothSupsSourced is True:
                                    utilRetVec_eq = UtilsRet_Scen5(lambsup1_dev, lambsup2_eq, Lr, b, c, w1_dev, w2_eq,
                                                                   lambretlo, lambrethi, sens, fpr,
                                                                   onlyDoubleSource=onlyDoubleSource,
                                                                   onlyQualInvest=onlyQualInvest)
                                else:
                                    utilRetVec_eq = UtilsRet_Scen5(lambsup1_dev, lambsup2_eq, Lr, b, c, w1_dev, w2_eq,
                                                                   lambretlo, lambrethi, sens, fpr,
                                                                   sourceBothSups=False,
                                                                   onlyDoubleSource=onlyDoubleSource,
                                                                   onlyQualInvest=onlyQualInvest)
                                retStrat_dev = np.argmax(utilRetVec_eq)
                                if retStrat_dev in [0, 2, 4]:
                                    lambret_dev = lambrethi
                                else:
                                    lambret_dev = lambretlo

                                # IF SINGLE-SUPPLIER, IS THIS A LIMIT PRICE PREVENTING S2 FROM ENTERING MARKET
                                validLimit = True
                                if retStrat_dev in [2, 3]:
                                    if printUpdates is True:
                                        print('Verifying if limit price deviation identified...')
                                    for tempw2 in w_vec:
                                        for templambsup2 in [lambsuphi, lambsuplo]:
                                            currRetUtilVec = UtilsRet_Scen5(lambsup1_dev, templambsup2, Lr, b, c,
                                                                            w1_dev, tempw2, lambretlo, lambrethi, sens,
                                                                            fpr)
                                            retBestStrat = np.argmax(currRetUtilVec)
                                            _, tempq2 = RetOrderQuantsFromStrat(currRetUtilVec[retBestStrat], b, c,
                                                                                w1_dev, tempw2)
                                            if templambsup2 == lambsuphi:
                                                cSup2 = cSup
                                            else:
                                                cSup2 = 0
                                            if retBestStrat in [0, 2, 4]:
                                                templambret = lambrethi
                                            else:
                                                templambret = lambretlo
                                            if SupUtil(tempq2, tempw2, cSup2, Ls, templambret, templambsup2, sens, fpr) > 0:
                                                validLimit = False

                                # Get q1 from retailer's preferred strategy
                                q1_dev, q2_dev = RetOrderQuantsFromStrat(retStrat_dev, b, c, w1_dev, w2_eq)
                                if lambsup1_dev == lambsuphi:
                                    utilSup1_dev = SupUtil(q1_dev, w1_dev, cSup, Ls, lambret_dev, lambsup1_dev, sens, fpr)
                                elif lambsup1_dev == lambsuplo:
                                    utilSup1_dev = SupUtil(q1_dev, w1_dev, 0, Ls, lambret_dev, lambsup1_dev, sens, fpr)
                                if utilSup1_dev > s1util_eq + eps and validLimit is True:
                                    if printUpdates is True:
                                        print('S1 does better with w1=' + str(round(w1_dev, 3)) + ', '
                                              + str(round(utilSup1_dev, 3)) + ' over ' + str(round(s1util_eq, 3)))
                                        print('where retailer opts for ' + str(retStrat_dev))
                                    matVal = np.nan
                                    if storeTmat is True and Tmat[eqind, :].sum() == 0:
                                        # Store transition from current equilibrium
                                        Tmat[eqind, GetTransInd(eq_curr, sup1Pref=1, retPref=retStrat_dev)] = 1
                                    breakloop = True
                    # Check S2 if asymmetric or single-sourcing
                    if (eq_curr in ['LHY12', 'LHN12', 'HHY1', 'HHN1', 'LLY1', 'LLN1', 'HLY1', 'HLN1', 'LHY1', 'LHN1']
                        and matVal == 1 and limitPrice is False):
                        if lambsup2_eq == lambsuphi:
                            lambsup2_dev = lambsuplo
                        elif lambsup2_eq == lambsuplo:
                            lambsup2_dev = lambsuphi
                        breakloop = False
                        for w2_dev in w_vec:
                            if w2_dev != w2_eq and breakloop is False:  # Deviation price cannot be the equilibrium price
                                # What is retailer policy under deviation?
                                utilRetVec_eq = UtilsRet_Scen5(lambsup1_eq, lambsup2_dev, Lr, b, c, w1_eq, w2_dev,
                                                               lambretlo, lambrethi, sens, fpr,
                                                                   onlyDoubleSource=onlyDoubleSource,
                                                                   onlyQualInvest=onlyQualInvest)
                                retStrat_dev = np.argmax(utilRetVec_eq)
                                if retStrat_dev in [0, 2, 4]:
                                    lambret_dev = lambrethi
                                else:
                                    lambret_dev = lambretlo

                                # IF SINGLE-SUPPLIER, IS THIS A LIMIT PRICE PREVENTING S2 FROM ENTERING MARKET
                                validLimit = True
                                if retStrat_dev in [4, 5]:
                                    if printUpdates is True:
                                        print('Verifying if limit price deviation identified...')
                                    for tempw1 in w_vec:
                                        for templambsup1 in [lambsuphi, lambsuplo]:
                                            currRetUtilVec = UtilsRet_Scen5(templambsup1, lambsup2_dev, Lr, b, c,
                                                                            tempw1, w2_dev, lambretlo, lambrethi,
                                                                            sens, fpr)
                                            retBestStrat = np.argmax(currRetUtilVec)
                                            tempq1, _ = RetOrderQuantsFromStrat(currRetUtilVec[retBestStrat], b, c,
                                                                                tempw1, w2_dev)
                                            if templambsup1 == lambsuphi:
                                                cSup1 = cSup
                                            else:
                                                cSup1 = 0
                                            if retBestStrat in [0, 2, 4]:
                                                templambret = lambrethi
                                            else:
                                                templambret = lambretlo
                                            if SupUtil(tempq1, tempw1, cSup1, Ls, templambret, templambsup1, sens,
                                                       fpr) > 0:
                                                validLimit = False

                                # Get q2 from retailer's preferred strategy
                                q1_dev, q2_dev = RetOrderQuantsFromStrat(retStrat_dev, b, c, w1_eq, w2_dev)
                                if lambsup2_dev == lambsuphi:
                                    utilSup2_dev = SupUtil(q2_dev, w2_dev, cSup, Ls, lambret_dev, lambsup2_dev, sens, fpr)
                                elif lambsup2_dev == lambsuplo:
                                    utilSup2_dev = SupUtil(q2_dev, w2_dev, 0, Ls, lambret_dev, lambsup2_dev, sens, fpr)
                                if utilSup2_dev > s2util_eq + eps and validLimit is True:
                                    if printUpdates is True:
                                        print('S2 does better with w2=' + str(round(w2_dev, 3))+ ', '
                                              + str(round(utilSup2_dev, 3)) + ' over ' + str(round(s2util_eq, 3)))
                                        print('where retailer opts for ' + str(retStrat_dev))
                                    matVal = np.nan
                                    if storeTmat is True and Tmat[eqind, :].sum() == 0:
                                        # Store transition from current equilibrium
                                        Tmat[eqind, GetTransInd(eq_curr, sup2Pref=1, retPref=retStrat_dev)] = 1
                                    breakloop = True

                    # CHECK 3: Retailer prefers the equilibrium strategy
                    # If currently in exclusive sourcing, retailer cannot order from S2
                    if bothSupsSourced is True:
                        utilRetVec_eq = UtilsRet_Scen5(lambsup1_eq, lambsup2_eq, Lr, b, c, w1_eq, w2_eq, lambretlo,
                                                   lambrethi, sens, fpr,
                                                   onlyDoubleSource=onlyDoubleSource,
                                                   onlyQualInvest=onlyQualInvest)
                    else:
                        utilRetVec_eq = UtilsRet_Scen5(lambsup1_eq, lambsup2_eq, Lr, b, c, w1_eq, w2_eq, lambretlo,
                                                       lambrethi, sens, fpr, sourceBothSups=False,
                                                       onlyDoubleSource=onlyDoubleSource,
                                                       onlyQualInvest=onlyQualInvest)
                    if utilRetVec_eq[np.argmax(utilRetVec_eq)] > utilRetVec_eq[retStrat_eq] + eps and matVal == 1:
                        matVal = np.nan
                        if printUpdates is True:
                            print('The retailer prefers strategy ' + str(np.argmax(utilRetVec_eq)) + retStratToStr(
                                np.argmax(utilRetVec_eq)) + ', '+str(round(utilRetVec_eq[np.argmax(utilRetVec_eq)],3)) +
                                  ' util instead of ' + str(round(utilRetVec_eq[retStrat_eq], 3)))
                        if storeTmat is True and Tmat[eqind, :].sum() == 0:
                            # Store transition from current equilibrium
                            Tmat[eqind, GetTransInd(eq_curr, retPref=np.argmax(utilRetVec_eq))] = 1
                    # If {HHY1} is eq_curr and retailer still prefers to leave, then all players must exit
                    # if eq_curr == 'HHY1' and np.argmax(utilRetVec_eq) == 6:
                    #     print('No non-negative utility for all players possible')
                    #     noeq_mat[Lrind, Lsind] = 1

                    # CHECK 4: POSITIVE Q and w
                    if q1_eq + q2_eq == 0 or w1_eq + w2_eq == 0:
                        matVal = np.nan

                    eq_mat[Lrind, Lsind] = matVal
                    if matVal == 1:  # Store quantities
                        eqQuant_mat[Lrind, Lsind, :] = np.array([q1_eq, q2_eq])
                        Tmat[eqind, eqind] = 1
                    else:
                        eqQuant_mat[Lrind, Lsind, :] = np.array([0, 0])
                    # Update the printing
                    if Lr in Lr_insp_vec and Ls in Ls_insp_vec:
                        printUpdates = False
        eqStrat_matList.append(eq_mat)
        eqQuant_matList.append(eqQuant_mat)
    Tmat[-1, -1] = 1
    return eqStrat_matList, eqQuant_matList, Tmat

# Define policy function when all parameters are fixed
alph_0 = 0.8
b_0 = 0.5
c_0 = 0.08
cSup_0 = 0.12
w_0 = 0.075
lambretlo_0, lambrethi_0 = 0.8, 0.95
sens_0, fpr_0 = 0.8, 0.01
lambsuplo_0, lambsuphi_0 = 0.75, 0.95
lambsup, lambsup1, lambsup2 = 0.9, 0.9, 0.9


# HIGH LR, LOW LS REGION
# Lr_insp_min, Lr_insp_max = 0.24, 0.3
# Ls_insp_min, Ls_insp_max = 0.05, 0.1
# LOW LR, HIGH LS REGION
# Lr_insp_min, Lr_insp_max = 0.01, 0.05
# Ls_insp_min, Ls_insp_max = 0.27, 0.31
# SECOND EQ GAP
Lr_insp_min, Lr_insp_max = 0.1, 0.11
Ls_insp_min, Ls_insp_max = 0.19, 0.2
Lr_ind_Tmat, Ls_ind_Tmat = 14, 26
# SECOND EQ GAP
Lr_insp_min, Lr_insp_max = 0.1415, 0.143
Ls_insp_min, Ls_insp_max = 0.145, 0.151
Lr_ind_Tmat, Ls_ind_Tmat = 19, 20



eq_list = ['HHY12', 'HHN12', 'LLY12', 'LLN12', 'LHY12', 'LHN12', 'HHY1', 'HHN1', 'LLY1', 'LLN1', 'HLY1', 'HLN1', 'LHY1',
           'LHN1']

numLpts = 7  # Refinement along each axis for plotting
numWpts = 50  # Refinement for deviation prices
Ltheta_max = 1.5
Ltheta_vec = np.arange(0, Ltheta_max + 0.00001, Ltheta_max / numLpts)
eps = 0.001  # tolerance for switching strategies
doubleSource = False
retQualInvest = False

eqStrat_matList, eqQuant_matList, _ = LthetaEqMatsForPlot(numLpts, Ltheta_max, numWpts, lambsuplo_0, lambsuphi_0, b_0,
                                                          c_0, cSup_0, lambretlo_0, lambrethi_0, sens_0, fpr_0,
                                                          printUpdates=False, Lr_insp_min=Lr_insp_min,
                                                          Lr_insp_max=Lr_insp_max, Ls_insp_min=Ls_insp_min,
                                                          Ls_insp_max=Ls_insp_max, onlyDoubleSource=doubleSource,
                                                          onlyQualInvest=retQualInvest)

eqStrat_matList, eqQuant_matList, Tmat = LthetaEqMatsForPlot(numLpts, Ltheta_max, numWpts, lambsuplo_0, lambsuphi_0,
                                                             b_0, c_0, cSup_0, lambretlo_0, lambrethi_0, sens_0, fpr_0,
                                                             printUpdates=True, Lr_insp_min=Lr_insp_min,
                                                             Lr_insp_max=Lr_insp_max, Ls_insp_min=Ls_insp_min,
                                                             Ls_insp_max=Ls_insp_max, Lr_ind_Tmat=8, Ls_ind_Tmat=2,
                                                             onlyDoubleSource=doubleSource,
                                                             onlyQualInvest=retQualInvest)

# Fill in gaps with no equilibria
# Initialize with 'no profit' equilibrium
eqMixName_List = [['N']]
eqNoprofit = np.empty((numLpts + 1, numLpts + 1))
eqNoprofit[:] = np.nan
eqMix_matList = [eqNoprofit]
for Lr_ind, Lr in enumerate(Ltheta_vec):
    for Ls_ind, Ls in enumerate(Ltheta_vec):
        if np.nansum([eqStrat_matList[i][Lr_ind, Ls_ind] for i in range(len(eqStrat_matList))]) == 0:
            # print('Lr:' + str(Lr_ind))
            # print('Ls:' + str(Ls_ind))
            # Need to identify equilibria mixture
            _, _, currTmat = LthetaEqMatsForPlot(numLpts, Ltheta_max, numWpts, lambsuplo_0, lambsuphi_0, b_0, c_0,
                                                 cSup_0, lambretlo_0, lambrethi_0, sens_0, fpr_0, printUpdates=False,
                                                 Lr_insp_min=Lr_insp_min, Lr_insp_max=Lr_insp_max,
                                                 Ls_insp_min=Ls_insp_min, Ls_insp_max=Ls_insp_max, Lr_ind_Tmat=Lr_ind,
                                                 Ls_ind_Tmat=Ls_ind, onlyDoubleSource=doubleSource,
                                                 onlyQualInvest=retQualInvest)
            currTvec = steadyStateMat(currTmat)
            if currTvec[:-1].sum() == 0:  # Only {N} possible
                # print('{N} value found')
                eqMix_matList[0][Lr_ind, Ls_ind] = 1
            else:  # Mixture
                tempVec = currTvec[:-1]
                currTnames = [eq_list[i] for i in range(len(eq_list)) if tempVec[i] > 0]
                currTnames.sort()
                if currTnames not in eqMixName_List:  # Add a new mixture
                    eqMixName_List.append(currTnames)
                    newMat = np.empty((numLpts+1, numLpts+1))
                    newMat[:] = np.nan
                    newMat[Lr_ind, Ls_ind] = 1
                    eqMix_matList.append(newMat)
                else:  # Add to previous mixture
                    mixInd = eqMixName_List.index(currTnames)
                    eqMix_matList[mixInd][Lr_ind, Ls_ind] = 1

# Drop overlapping equilibria if equivalent
equivPairs_list = [[6, 10], [7, 11], [8, 12], [9, 13]]
for Lr_ind, Lr in enumerate(Ltheta_vec):
    for Ls_ind, Ls in enumerate(Ltheta_vec):
        for pairlist in equivPairs_list:
            if np.nansum([eqStrat_matList[i][Lr_ind, Ls_ind] for i in pairlist]) > 1:
                eqStrat_matList[pairlist[1]][Lr_ind, Ls_ind] = np.nan

# Now do plotting
d = np.linspace(0, Ltheta_max, numLpts+1)

alval = 0.6  # Transparency for plots
values = [0, 1]
labels = ['{HHY12}', '{HHN12}', '{LLY12}', '{LLN12}', '{LHY12}', '{LHN12}', '{HHY1}', '{HHN1}', '{LLY1}', '{LLN1}',
          '{HLY1}', '{HLN1}', '{LHY1}', '{LHN1}']
eqcolors = ['royalblue', 'deepskyblue', 'red', 'pink', 'darkorange', 'bisque', 'darkgreen', 'greenyellow',
            'gold', 'lemonchiffon', 'purple', 'plum', 'silver', 'lightgray']
mixcolors = ['lime', 'magenta', 'yellow', 'orangered', 'cyan'] * 10
# Add in {N} and mixed equilibria
for mixeqlistind, mixeqlist in enumerate(eqMixName_List):
    if mixeqlist == ['N']:
        labels.append('{N}')
        eqcolors.append('black')
    else:  # Mixed list
        tempstr = ''
        for mixitem in mixeqlist:
            tempstr = tempstr + '{' + mixitem + '}, '
        tempstr = tempstr[:-2]  # Drop last comma and space
        labels.append(tempstr)
        eqcolors.append(mixcolors[mixeqlistind-1])
# Add to equilibria strategy list
eqStrat_matList = eqStrat_matList + eqMix_matList




fig = plt.figure()
fig.suptitle('Nash equilibria vs. '+r'$L^{R}_{\theta}$'+', '+r'$L^{S}_{\theta}$', fontsize=18, fontweight='bold')
axStrat = fig.add_subplot(321)
axQuant = fig.add_subplot(322)
axSupQual = fig.add_subplot(323)
axRetQual = fig.add_subplot(324)
axSPUtil = fig.add_subplot(313)

figrt = 0.6
fig.subplots_adjust(right=figrt)

imlist = []
for eqind in range(len(labels)):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[eqind]], name='from_list', N=None)
    if eqcolors[eqind] == 'black':  # No alpha transparency
        im = axStrat.imshow(eqStrat_matList[eqind].T, vmin=0, vmax=1,
                            extent=(Ltheta_vec.min(), Ltheta_vec.max(), Ltheta_vec.min(), Ltheta_vec.max()),
                            origin="lower", cmap=mycmap, alpha=1)
    else:
        im = axStrat.imshow(eqStrat_matList[eqind].T, vmin=0, vmax=1,
                    extent=(Ltheta_vec.min(), Ltheta_vec.max(), Ltheta_vec.min(), Ltheta_vec.max()),
                    origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)

# Equilibria
axStrat.set_xlabel(r'$L^{R}_{\theta}$', fontsize=12)
axStrat.set_ylabel(r'$L^{S}_{\theta}$', rotation=0, fontsize=12, labelpad=3)
axStrat.set_title('Equilibria strategies')
# create a patch for every color
legwidth = 20
wraplabels = ['\n'.join(textwrap.wrap(labels[i], width=legwidth)) for i in range(len(labels))]
patches = [mpatches.Patch(color=eqcolors[i], edgecolor='black', label=wraplabels[i], alpha=alval) for i in range(len(eqcolors))]
# put those patched as legend-handles into the legend
axStrat.legend(handles=patches, bbox_to_anchor=(-0.3, 1.0), loc='upper right', borderaxespad=0.1, fontsize=8)

# Quantities
eqQuant_plot = GetQuantMatFromEqMats(eqStrat_matList[:-len(eqMix_matList)], eqQuant_matList)
imQuant = axQuant.imshow(eqQuant_plot.T, vmin=0, vmax=1,
                    extent=(Ltheta_vec.min(), Ltheta_vec.max(), Ltheta_vec.min(), Ltheta_vec.max()),
                    origin="lower", cmap='Blues')
imQuant_N = axQuant.imshow(eqStrat_matList[14].T, vmin=0, vmax=1,
                    extent=(Ltheta_vec.min(), Ltheta_vec.max(), Ltheta_vec.min(), Ltheta_vec.max()),
                    origin="lower",
                    cmap=matplotlib.colors.ListedColormap(['white', 'black'], name='from_list', N=None))
axQuant.set_xlabel(r'$L^{R}_{\theta}$', fontsize=12)
axQuant.set_ylabel(r'$L^{S}_{\theta}$', rotation=0, fontsize=12, labelpad=3)
axQuant.set_title('Equilibria quantities')

# Quality levels
# retQual_plot, supQual_plot = GetQualMatsFromEqMat(eqStrat_matList[:-len(eqMix_matList)], labels[:-len(eqMix_matList)])
retQual_plot, supQual_plot = GetQualMatsFromMixedEqMat(eqStrat_matList[:-len(eqMix_matList)],
                                                       labels[:-len(eqMix_matList)], eqMix_matList, eqMixName_List)
imSupQual = axSupQual.imshow(supQual_plot.T, vmin=-0.5, vmax=1.5,
                    extent=(Ltheta_vec.min(), Ltheta_vec.max(), Ltheta_vec.min(), Ltheta_vec.max()),
                    origin="lower", cmap='Grays')
imSupQual_N = axSupQual.imshow(eqStrat_matList[14].T, vmin=0, vmax=1,
                    extent=(Ltheta_vec.min(), Ltheta_vec.max(), Ltheta_vec.min(), Ltheta_vec.max()),
                    origin="lower",
                    cmap=matplotlib.colors.ListedColormap(['white', 'black'], name='from_list', N=None))
imRetQual = axRetQual.imshow(retQual_plot.T, vmin=-0.5, vmax=1.5,
                    extent=(Ltheta_vec.min(), Ltheta_vec.max(), Ltheta_vec.min(), Ltheta_vec.max()),
                    origin="lower", cmap='Grays')
imRetQual_N = axRetQual.imshow(eqStrat_matList[14].T, vmin=0, vmax=1,
                    extent=(Ltheta_vec.min(), Ltheta_vec.max(), Ltheta_vec.min(), Ltheta_vec.max()),
                    origin="lower",
                    cmap=matplotlib.colors.ListedColormap(['white', 'black'], name='from_list', N=None))
axSupQual.set_xlabel(r'$L^{R}_{\theta}$', fontsize=12)
axSupQual.set_ylabel(r'$L^{S}_{\theta}$', rotation=0, fontsize=12, labelpad=3)
axSupQual.set_title('Supplier quality levels')
axRetQual.set_xlabel(r'$L^{R}_{\theta}$', fontsize=12)
axRetQual.set_ylabel(r'$L^{S}_{\theta}$', rotation=0, fontsize=12, labelpad=3)
axRetQual.set_title('Retailer quality levels')

# SP utility
SPUtil_plot = GetSPUtilMatFromQuantQualMats(alph_0, retQual_plot, supQual_plot, eqQuant_plot, lambretlo_0, lambrethi_0,
                                            lambsuplo_0, lambsuphi_0)
imSPUtil = axSPUtil.imshow(SPUtil_plot.T, vmin=np.nanmin(SPUtil_plot)-0.2*(np.nanmax(SPUtil_plot)-np.nanmin(SPUtil_plot)),
                                          vmax=np.nanmax(SPUtil_plot)+0.2*(np.nanmax(SPUtil_plot)-np.nanmin(SPUtil_plot)),
                                extent=(Ltheta_vec.min(), Ltheta_vec.max(), Ltheta_vec.min(), Ltheta_vec.max()),
                                origin="lower", cmap='Greens')
imSPUtil_N = axSPUtil.imshow(eqStrat_matList[14].T, vmin=0, vmax=1,
                    extent=(Ltheta_vec.min(), Ltheta_vec.max(), Ltheta_vec.min(), Ltheta_vec.max()),
                    origin="lower",
                    cmap=matplotlib.colors.ListedColormap(['white', 'black'], name='from_list', N=None))
axSPUtil.set_xlabel(r'$L^{R}_{\theta}$', fontsize=12)
axSPUtil.set_ylabel(r'$L^{S}_{\theta}$', rotation=0, fontsize=12, labelpad=3)
axSPUtil.set_title('Social planner utility')

# Add sliders for changing the parameters
slstrtrightval = figrt + 0.08
sltopval = 0.75
slgap = 0.04
slwidth, slht = 0.2, 0.02
b_slider_ax = fig.add_axes([slstrtrightval, sltopval, slwidth, slht])
b_slider = Slider(b_slider_ax, r'$b$', 0.01, 0.99, valinit=b_0)
c_slider_ax = fig.add_axes([slstrtrightval, sltopval-slgap, slwidth, slht])
c_slider = Slider(c_slider_ax, r'$c_R$', 0.01, 0.99, valinit=c_0)
cSup_slider_ax = fig.add_axes([slstrtrightval, sltopval-slgap*2, slwidth, slht])
cSup_slider = Slider(cSup_slider_ax, r'$c_S$', 0.01, 0.99, valinit=cSup_0)
lambretlo_slider_ax = fig.add_axes([slstrtrightval, sltopval-slgap*3, slwidth, slht])
lambretlo_slider = Slider(lambretlo_slider_ax, r'$\lambda^{lo}$', 0.01, 0.99, valinit=lambretlo_0)
lambrethi_slider_ax = fig.add_axes([slstrtrightval, sltopval-slgap*4, slwidth, slht])
lambrethi_slider = Slider(lambrethi_slider_ax, r'$\lambda^{hi}$', 0.01, 0.99, valinit=lambrethi_0)
lambsuplo_slider_ax = fig.add_axes([slstrtrightval, sltopval-slgap*5, slwidth, slht])
lambsuplo_slider = Slider(lambsuplo_slider_ax, r'$\Lambda^{lo}$', 0.01, 0.99, valinit=lambsuplo_0)
lambsuphi_slider_ax = fig.add_axes([slstrtrightval, sltopval-slgap*6, slwidth, slht])
lambsuphi_slider = Slider(lambsuphi_slider_ax, r'$\Lambda^{hi}$', 0.01, 0.99, valinit=lambsuphi_0)
sens_slider_ax = fig.add_axes([slstrtrightval, sltopval-slgap*7, slwidth, slht])
sens_slider = Slider(sens_slider_ax, r'$\rho$', 0.5, 0.99, valinit=sens_0)
fpr_slider_ax = fig.add_axes([slstrtrightval, sltopval-slgap*8, slwidth, slht])
fpr_slider = Slider(fpr_slider_ax, r'$\phi$', 0.01, 0.2, valinit=fpr_0)

alpha_slider_ax = fig.add_axes([slstrtrightval, sltopval-slgap*10, slwidth, slht])
alpha_slider = Slider(alpha_slider_ax, r'$\alpha$', 0.01, 0.99, valinit=alph_0)

def sliders_on_changed(val):
    eqStrat_matList, eqQuant_matList, _ = LthetaEqMatsForPlot(numLpts, Ltheta_max, numWpts, lambsuplo_slider.val,
                                                              lambsuphi_slider.val, b_slider.val, c_slider.val,
                                                              cSup_slider.val, lambretlo_slider.val,
                                                              lambrethi_slider.val, sens_slider.val, fpr_slider.val)
    for imind, im in enumerate(imlist):
        im.set_data(eqStrat_matList[imind].T)

    eqQuant_plot = GetQuantMatFromEqMats(eqStrat_matList, eqQuant_matList)
    imQuant.set_data(eqQuant_plot.T)

    retQual_plot, supQual_plot = GetQualMatsFromEqMat(eqStrat_matList, labels)
    imSupQual.set_data(supQual_plot.T)
    imRetQual.set_data(retQual_plot.T)

    SPUtil_plot = GetSPUtilMatFromQuantQualMats(alpha_slider.val, retQual_plot, supQual_plot, eqQuant_plot,
                                                lambretlo_slider.val, lambrethi_slider.val,
                                                lambsuplo_slider.val, lambsuphi_slider.val)
    imSPUtil.set_data(SPUtil_plot.T)
    imSPUtil.set_clim(vmin=np.nanmin(SPUtil_plot)-0.2*(np.nanmax(SPUtil_plot)-np.nanmin(SPUtil_plot)),
                        vmax=np.nanmax(SPUtil_plot)+0.2*(np.nanmax(SPUtil_plot)-np.nanmin(SPUtil_plot)))

    fig.canvas.draw_idle()
b_slider.on_changed(sliders_on_changed)
c_slider.on_changed(sliders_on_changed)
cSup_slider.on_changed(sliders_on_changed)
lambretlo_slider.on_changed(sliders_on_changed)
lambrethi_slider.on_changed(sliders_on_changed)
lambsuplo_slider.on_changed(sliders_on_changed)
lambsuphi_slider.on_changed(sliders_on_changed)
sens_slider.on_changed(sliders_on_changed)
fpr_slider.on_changed(sliders_on_changed)
alpha_slider.on_changed(sliders_on_changed)

plt.subplots_adjust(left=None, bottom=0.07, right=None, top=0.85, wspace=0.5, hspace=0.4)
plt.show(block=True)


