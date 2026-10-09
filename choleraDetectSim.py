# Used for simulating WES and clinical detection times for cholera in some city

# IMPORTANT: For cholera, asymptomatic patients are a majority (~80%), and only
# product about 10^3 copies/10 mL per stool, which is below the LOD at the lab.
# Thus, we assume that asymptomatic patients do not impact WES calculations directly;
# these patients DO NOT infect others, DO NOT show up at the clinic, and CANNOT
# generally be measured by standard WES analysis.

# TODO: Biswajit, please check the logic of shedding and symptom onset; it seems odd that one could have symptoms prior to any shedding?
#       This might have to do with our choice of the South Africa distribution

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.stats as sps

# Sim settings
N_SIM = 10000
np.random.seed(44)
M = 1e3 # Big M value for when detection time doesn't occur

# CITY-BASD INPUTS
h = 0.53  # healthcare coverage
lamb = 14  # frequency of testing; every lamb^th day a test occurs at every wastewater site
v = 0.90  # wastewater population coverage
h_not_v = 0.00   # pop. proportion with HC coverage but no wastewater coverage
beta_segment = 0.2  # concentration parameter of within-segement infection spread
sitePopProps = np.random.beta(3,4,size=30)
sitePopProps = sitePopProps/np.sum(sitePopProps)
cityPop = 10440000
sitePopList = [x*cityPop for x in sitePopProps]  # list of covered populations for each WES site  # TODO: Biswajit, please fill out with STP values


seg0_prop = 1 - v - h_not_v   # propor. NO healthcare, NO wastewater coverage
seg1_prop = v - h + h_not_v   # propor. NO healthcare, YES wastewater coverage
seg2_prop = h_not_v           # propor. YES healthcare, NO wastewater coverage
seg3_prop = h - h_not_v       # propor. YES healthcare, YES wastewater coverage

segprop_vec = [seg0_prop, seg1_prop, seg2_prop, seg3_prop]

# PATHOGEN INPUTS
sympt_prob = 0.2  # probability of symptomatic patient
clin_pres_prob = 0.5  # prob of sympt patient going to clinic
k_decay = -0.014  # decay rate in wastewater system

def concentratProb(a, b, C):
    return 1/(1+np.exp(-1*(a+b*np.log10(C))))

LabA_chol, LabB_chol = -3.7, 1.93

# Define a class for all newly generated infected patients
class Patient:
    def __init__(self, infectedTime, segment, beta_segment, parent='NA', verbose=False):
        # if verbose:
        #     print('Adding patient')
        self.segment = int(segment)  # Assigned segment
        self.parent = parent  # 'Index' or other patient object
        currSegProb = segprop_vec[segment] + beta_segment*(1 - segprop_vec[segment])
        othersegs = segprop_vec.copy()
        othersegs.pop(segment)
        templist = [float((1-currSegProb)*(x/(np.sum(othersegs)))) for x in othersegs]
        templist.insert(segment, currSegProb)
        self.segmentSpreadProbs = templist
        self.infectedTime = infectedTime  # Time of initial infection
        self.beginShedTime = infectedTime + np.random.randint(1, 10 + 1)  # high value is exclusive; # TODO: REMIND TO TRY DIFFERENT DISTRIBUTION HERE
        # if verbose:
        #     print('Adding shedding times')
        num_shed_days = np.random.randint(7, 14 + 1)  # Number of shedding/infectious days # TODO: REMIND TO TRY DIFFERENT DISTRIBUTION HERE
        self.shedLevelList = [10**11]*num_shed_days  # Shedding level each day of shedding window # TODO: REMIND TO TRY DIFFERENT DISTRIBUTION HERE, USING KNOWN SHEDDING LEVEL RANGE
        self.sym_b_time = infectedTime + np.random.randint(1, 5 + 1) # Symptom onset day # TODO: REMIND TO TRY DIFFERENT DISTRIBUTION HERE
        if segment in [2, 3]:  # Patient has healthcare coverage; find clinical diagnosis time
            clinPresVal = np.random.binomial(n=1, p=clin_pres_prob)
            if clinPresVal == 1:  # Patient shows up to clinic
                self.clinPresTime = self.sym_b_time + np.random.randint(1, 7)  # Days after symptom onset when goes to clinic  # TODO: REMIND TO TRY DIFF DIST HERE
                clinPresShedAmt = self.shedLevelList[self.clinPresTime - self.sym_b_time]  # Assume this concentration is in sample to be tested
                diagnosisVal = np.random.binomial(n=1, p=concentratProb(LabA_chol, LabB_chol, clinPresShedAmt))
                if diagnosisVal == 1:  # Cholera detected in clinic; add in lab turnaround time
                    clinTAT = np.random.choice([1, 2, 3], size=1, p=[0.35, 0.50, 0.15])[0]  # TODO: TRY DIFF DIST
                    self.clinDiagTime = self.clinPresTime + clinTAT
                else:   # Cholera NOT detected by lab at clinic
                    self.clinDiagTime = M
            else:  # No presentation
                self.clinPresTime = M
                self.clinDiagTime = M
        else:   # No healthcare access
            self.clinPresTime = M
            self.clinDiagTime = M
        if verbose:
            print('Finding simexit')
        temp = [self.beginShedTime+num_shed_days, self.sym_b_time, self.clinPresTime, self.clinDiagTime]
        temp2 = [x for x in temp if x != M]
        self.simexittime = int(np.max(temp2))

def GetWESResult(patList, t, sitePop, avgsitetraveltime=0.25):
    # Returns WES testing result on day t from set of patients shedding at SAME site on t
    # avgsitetraveltime is average patient distance from sampling site in WES flow time in days
    WESresult = False  # Initialize
    shedtotal = 0  # For summing up shedding level
    for pat in patList:
        # Check if patient in WES zone sheds on this day
        if (pat.beginShedTime <= t) and (pat.beginShedTime + len(pat.shedLevelList) - 1 >= t) and \
                pat.segment in [1, 3]:
            shedtotal += (pat.shedLevelList[curr_t - pat.beginShedTime])
    flowtotal = 100*sitePop  # TODO: Biswajit, verify this aligns with Burnor et al. approach; if sitePop is large then CLT applies, which I'm using here
    # TODO: Add dilution flow somewhere?
    concentration_t = shedtotal / flowtotal
    # Next add in transport decay; for now assume that each patient is some average distance away from the sampling site
    # TODO: Can incorporate variance in the distance per patient from the sampling site later
    concentration_t_decay = float(concentration_t * np.exp(k_decay * avgsitetraveltime))
    # Assume wastewater is well dispersed by the time it reaches the sampling location
    # Sample then has equivalent concentration as decayed concentration
    detectProb = concentratProb(LabA_chol, LabB_chol, concentration_t_decay)
    if np.random.binomial(n=1, p=detectProb) == 1:
        WESresult = True

    return WESresult

# Storage of detection times across simulations
simDetect_WES, simDetect_clin = [], []

# For printing output
verbose = False

for sim in range(N_SIM):  # Main simulation loop
    if verbose:
        print('Starting sim '+str(sim)+'...')
    R_0 = np.random.uniform(1.7, 2.6)  # NON-integer mean reproduction number; unknown range;
    #   For cholera, considered the number of SYMPTOMATIC patients created
    k_0 = 4.5  # Dispersion parameter

    infectSitePop = int(np.random.choice(sitePopList, size=1, p=sitePopList/np.sum(sitePopList))[0])

    # For python functions
    nbinom_n, nbinom_p = k_0, R_0 / (R_0 + (R_0**2)/k_0)

    # Index patient; choose initial segment purely randomly
    pat0 = Patient(0, np.random.choice([0, 1, 2, 3], size=1, p=segprop_vec)[0],
                   beta_segment, parent='Index')
    patList = [pat0]

    # Intialize list of clinical detection times
    clinDetectTimes = [int(pat0.clinDiagTime)]

    # Initialize first day of WES testing POST-infection of index case
    WESTestTime = np.random.randint(0, lamb)

    # Infect other patients; first, how many, according to (R_0, k_0)
    numOthersInfect = sps.nbinom.rvs(nbinom_n, nbinom_p)
    # Which segments are these patients a part of? Depends on segment spread
    newInfectSegments = np.random.choice([0, 1, 2, 3], size=numOthersInfect, p=pat0.segmentSpreadProbs)
    newInfectSegments = newInfectSegments.tolist()
    # Which times do these infections occur? Sample uniformly across shedding time
    temp = np.random.choice(np.arange(len(pat0.shedLevelList)),
                                      p = pat0.shedLevelList/np.sum(pat0.shedLevelList),
                                      size = numOthersInfect)
    newInfectTimes = [int(pat0.beginShedTime + x) for x in temp]

    for newpatind in range(numOthersInfect):
        if verbose:
            print('Adding patient at infect time '+str(newInfectTimes[newpatind])+', segment '+
                  str(newInfectSegments[newpatind]))
        patList.append(Patient(newInfectTimes[newpatind], newInfectSegments[newpatind], beta_segment,
                               parent=pat0, verbose=verbose))
        clinDetectTimes.append(int(patList[-1].clinDiagTime))

    patFinishedInfectingList = [pat0]   # List of patients whose secondary infections have been added to patList

    # We now step through each time step until one of two things happens:
    #   1) We have a clinical detection AND a WES detection, or
    #   2) The outbreak has died out on its own before any detection; big M is used for non-detection

    # Initialize a list of patients whose exit time has been exceeded; stop once the length of this list matches
    #   the number of patients we've generated
    patExitList = []
    WESDetect, clinDetect = False, False  # Booleans for tracking if we've detected in each surveillance system
    curr_t = 0  # initialize the start day of the outbreak
    # Continue until all patients have exited the simulation OR both detection times have been identified
    # OR until max time reached
    while (len(patList) > len(patExitList)) and (WESDetect is False or clinDetect is False) and (curr_t < M):
        if verbose:
            print('Day ' + str(curr_t) + ' starting...')
        # Scan if we've reached a clinical detection time
        if (min(clinDetectTimes) == curr_t) and (clinDetect is False):
            simDetect_clin.append(curr_t)
            clinDetect = True
        if verbose:
            print('Clinical detection scanned')
        # Do WES measurement if on a scheduled WES day
        if np.mod(curr_t, lamb) == WESTestTime and (WESDetect is False):
            WESresult = GetWESResult(patList, curr_t, infectSitePop)
            if WESresult is True:
                simDetect_WES.append(curr_t)
                WESDetect = True
        if verbose:
            print('WES detection scanned')
        # Add new infections if we've reached the beginning of any patient's shedding time
        for pat in patList:
            if (pat.beginShedTime == curr_t) and (pat not in patFinishedInfectingList):
                # Infect other patients; first, how many, according to (R_0, k_0)
                numOthersInfect = sps.nbinom.rvs(nbinom_n, nbinom_p)
                # Which segments are these patients a part of? Depends on segment spread
                newInfectSegments = np.random.choice([0, 1, 2, 3], size=numOthersInfect, p=pat.segmentSpreadProbs)
                newInfectSegments = newInfectSegments.tolist()
                # Which times do these infections occur? Sample uniformly across shedding time
                temp = np.random.choice(np.arange(len(pat.shedLevelList)),
                                        p=pat.shedLevelList / np.sum(pat.shedLevelList),
                                        size=numOthersInfect)
                newInfectTimes = [int(pat.beginShedTime + x) for x in temp]
                # Add new infections
                for newpatind in range(numOthersInfect):
                    if verbose:
                        print('Adding patient at infect time ' + str(newInfectTimes[newpatind]) + ', segment ' +
                              str(newInfectSegments[newpatind]))
                    patList.append(Patient(newInfectTimes[newpatind], newInfectSegments[newpatind], beta_segment,
                                           parent=pat, verbose=verbose))
                    clinDetectTimes.append(int(patList[-1].clinDiagTime))
                # Add current patient to finished list
                patFinishedInfectingList.append(pat)
        if verbose:
            print('New infections added')
        # Add patients to exit list if simexittime exceeded
        for pat in patList:
            if (pat not in patExitList) and (pat.simexittime < curr_t):
                patExitList.append(pat)
                # Put default detection times if all patients have exited
                if len(patList) == len(patExitList):
                    if WESDetect is False:
                        simDetect_WES.append(M)
                    if clinDetect is False:
                        simDetect_clin.append(M)
        if verbose:
            print('Exit patients compiled')
        # Increment time step
        curr_t += 1
        if curr_t == M:  # Put default values for detection times
            if WESDetect is False:
                simDetect_WES.append(M)
            if clinDetect is False:
                simDetect_clin.append(M)

def printHists(simDetect_WES, simDetect_clin, M):
    # Print distributions of detection times
    # Grab sub-M values into temp arrays
    tempWES, tempclin = [x for x in simDetect_WES if x < M], [x for x in simDetect_clin if x < M]
    numNoWES, numNoClin = len(simDetect_WES) - len(tempWES), len(simDetect_clin) - len(tempclin)
    distMax = max(max(tempWES), max(tempclin))
    binstouse = np.arange(distMax)
    alval = 0.6

    counts1, bins1, patches1 = plt.hist([tempWES, tempclin], bins=binstouse, density=True, alpha=alval,
                                        color=['blue','red'],
                                        label=['WES detection time', 'Clinical detection time'])
    # Add counts of non-detection
    topbarval = np.max(counts1)
    WESnoDetBar, WESnoDetBarht = bins1[-1] +10, (topbarval)*numNoWES/len(simDetect_WES)
    clinnoDetBar, clinnoDetBarht = bins1[-1] +12, (topbarval)*numNoClin/len(simDetect_clin)
    plt.bar(
        x=WESnoDetBar, height=WESnoDetBarht, width=bins1[2] - bins1[0], align='edge',
        color='darkgray', edgecolor='black', label='No WES detection')
    plt.bar(
        x=clinnoDetBar, height=clinnoDetBarht, width=bins1[2] - bins1[0], align='edge',
        color='lightgray', edgecolor='black', label='No clinical detection')
    text_x, text_y = WESnoDetBar + bins1[1] - bins1[0], WESnoDetBarht + (np.max(counts1) * 0.02)
    plt.text(x=text_x, y=text_y, s=f"{numNoWES/len(simDetect_WES):.1%}",
        ha='center',  va='bottom', fontsize=10,fontweight='bold')
    text_x, text_y = clinnoDetBar + bins1[1] - bins1[0], clinnoDetBarht + (np.max(counts1) * 0.02)
    plt.text(x=text_x, y=text_y, s=f"{numNoClin / len(simDetect_clin):.1%}",
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.legend()
    # Add density lines
    kdeWES = gaussian_kde(tempWES)
    x_rangeWES = np.linspace(min(tempWES), max(tempWES), 1000)
    plt.plot(x_rangeWES, kdeWES(x_rangeWES), color='skyblue', linewidth=4, label="WES density")
    kdeclin = gaussian_kde(tempclin)
    x_rangeclin = np.linspace(min(tempclin), max(tempclin), 1000)
    plt.plot(x_rangeclin, kdeclin(x_rangeclin), color='pink', linewidth=4, label="Clin density")
    plt.xlabel('Time until outbreak detection (days)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Detection times under WES and clinical surveillance\nHyderabad - Cholera',
              fontsize=14)
    plt.tight_layout()
    plt.show()

    # Histogram of WES less clinical detection times
    tempWESclindiff = [simDetect_clin[i] - simDetect_WES[i] for i in range(len(simDetect_clin))
                       if (simDetect_clin[i] < M and simDetect_WES[i] < M)]
    numyesWESnoClin = len([i for i in range(len(simDetect_clin)) if (simDetect_clin[i] == M and simDetect_WES[i] < M)])
    numnoWESyesClin = len([i for i in range(len(simDetect_clin)) if (simDetect_clin[i] < M and simDetect_WES[i] == M)])
    numnoWESnoClin = len([i for i in range(len(simDetect_clin)) if (simDetect_clin[i] == M and simDetect_WES[i] == M)])
    distMin, distMax = min(tempWESclindiff) - 1, max(tempWESclindiff)+1
    binstouse = np.arange(distMin, distMax)
    posDiff, negDiff = [x for x in tempWESclindiff if x>0], [x for x in tempWESclindiff if x<=0]
    counts2, bins2, patches2 = plt.hist([negDiff, posDiff], bins=binstouse, density=True, alpha=alval,
                                        color=['red', 'green'],
                                        label=['', ''])
    topbarval = np.max(counts2)
    yesWESnoClinBar, yesWESnoClinBarht = bins2[-1] + 10, (topbarval) * numyesWESnoClin / len(simDetect_WES)
    noWESyesClinBar, noWESyesClinBarht = bins2[-1] + 12, (topbarval) * numnoWESyesClin / len(simDetect_WES)
    noWESnoClinBar, noWESnoClinBarht = bins2[-1] + 14, (topbarval) * numnoWESnoClin / len(simDetect_WES)
    plt.bar(x=yesWESnoClinBar, height=yesWESnoClinBarht, width=bins2[2] - bins2[0], align='edge',
        color='darkgray', edgecolor='black', label='WES, no clinical detection')
    plt.bar(x=noWESyesClinBar, height=noWESyesClinBarht, width=bins1[2] - bins1[0], align='edge',
        color='lightgray', edgecolor='black', label='Clinical, no WES detection')
    plt.bar(x=noWESnoClinBar, height=noWESnoClinBarht, width=bins1[2] - bins1[0], align='edge',
            color='dimgray', edgecolor='black', label='No clinical or WES detection')
    text_x, text_y = yesWESnoClinBar + bins1[1] - bins1[0], yesWESnoClinBarht + (np.max(counts2) * 0.02)
    plt.text(x=text_x, y=text_y, s=f"{numyesWESnoClin / len(simDetect_WES):.1%}",
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    text_x, text_y = noWESyesClinBar + bins1[1] - bins1[0], noWESyesClinBarht + (np.max(counts2) * 0.02)
    plt.text(x=text_x, y=text_y, s=f"{numnoWESyesClin / len(simDetect_WES):.1%}",
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    text_x, text_y = noWESnoClinBar + bins1[1] - bins1[0], noWESnoClinBarht + (np.max(counts2) * 0.02)
    plt.text(x=text_x, y=text_y, s=f"{numnoWESnoClin / len(simDetect_WES):.1%}",
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.legend()
    # Add density lines
    kde = gaussian_kde(tempWESclindiff)
    x_range = np.linspace(min(tempWESclindiff), max(tempWESclindiff), 1000)
    plt.plot(x_range, kde(x_range), color='gray', linewidth=4, label="Difference density")
    plt.xlabel('Clinical less WES detection time (days)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Difference in WES and clinical surveillance detection\nHyderabad - Cholera',
              fontsize=14)
    plt.tight_layout()
    plt.show()

    return np.quantile(tempWESclindiff, 0.5)

medianDiff = printHists(simDetect_WES, simDetect_clin, M)


###################
# Plot of median detection times vs beta intensity
###################
import scipy.stats as stats

beta_segmentVec = np.arange(0.0, 0.8, 0.05)
betaMedianList_WES, betaMedianList_clin = [], []
betaMedianList_WES_lo, betaMedianList_clin_lo = [], []
betaMedianList_WES_hi, betaMedianList_clin_hi = [], []
N_SIM = 1000

for betaSeg in beta_segmentVec:
    beta_segment = betaSeg

    # Storage of detection times across simulations
    simDetect_WES, simDetect_clin = [], []

    # For printing output
    verbose = False

    for sim in range(N_SIM):  # Main simulation loop
        if verbose:
            print('Starting sim '+str(sim)+'...')
        R_0 = np.random.uniform(1.7, 2.6)  # NON-integer mean reproduction number; unknown range;
        #   For cholera, considered the number of SYMPTOMATIC patients created
        k_0 = 4.5  # Dispersion parameter

        infectSitePop = int(np.random.choice(sitePopList, size=1, p=sitePopList/np.sum(sitePopList))[0])

        # For python functions
        nbinom_n, nbinom_p = k_0, R_0 / (R_0 + (R_0**2)/k_0)

        # Index patient; choose initial segment purely randomly
        pat0 = Patient(0, np.random.choice([0, 1, 2, 3], size=1, p=segprop_vec)[0],
                       beta_segment, parent='Index')
        patList = [pat0]

        # Intialize list of clinical detection times
        clinDetectTimes = [int(pat0.clinDiagTime)]

        # Initialize first day of WES testing POST-infection of index case
        WESTestTime = np.random.randint(0, lamb)

        # Infect other patients; first, how many, according to (R_0, k_0)
        numOthersInfect = sps.nbinom.rvs(nbinom_n, nbinom_p)
        # Which segments are these patients a part of? Depends on segment spread
        newInfectSegments = np.random.choice([0, 1, 2, 3], size=numOthersInfect, p=pat0.segmentSpreadProbs)
        newInfectSegments = newInfectSegments.tolist()
        # Which times do these infections occur? Sample uniformly across shedding time
        temp = np.random.choice(np.arange(len(pat0.shedLevelList)),
                                          p = pat0.shedLevelList/np.sum(pat0.shedLevelList),
                                          size = numOthersInfect)
        newInfectTimes = [int(pat0.beginShedTime + x) for x in temp]

        for newpatind in range(numOthersInfect):
            if verbose:
                print('Adding patient at infect time '+str(newInfectTimes[newpatind])+', segment '+
                      str(newInfectSegments[newpatind]))
            patList.append(Patient(newInfectTimes[newpatind], newInfectSegments[newpatind],
                                   beta_segment, parent=pat0, verbose=verbose))
            clinDetectTimes.append(int(patList[-1].clinDiagTime))

        patFinishedInfectingList = [pat0]   # List of patients whose secondary infections have been added to patList

        # We now step through each time step until one of two things happens:
        #   1) We have a clinical detection AND a WES detection, or
        #   2) The outbreak has died out on its own before any detection; big M is used for non-detection

        # Initialize a list of patients whose exit time has been exceeded; stop once the length of this list matches
        #   the number of patients we've generated
        patExitList = []
        WESDetect, clinDetect = False, False  # Booleans for tracking if we've detected in each surveillance system
        curr_t = 0  # initialize the start day of the outbreak
        # Continue until all patients have exited the simulation OR both detection times have been identified
        # OR until max time reached
        while (len(patList) > len(patExitList)) and (WESDetect is False or clinDetect is False) and (curr_t < M):
            if verbose:
                print('Day ' + str(curr_t) + ' starting...')
            # Scan if we've reached a clinical detection time
            if (min(clinDetectTimes) == curr_t) and (clinDetect is False):
                simDetect_clin.append(curr_t)
                clinDetect = True
            if verbose:
                print('Clinical detection scanned')
            # Do WES measurement if on a scheduled WES day
            if np.mod(curr_t, lamb) == WESTestTime and (WESDetect is False):
                WESresult = GetWESResult(patList, curr_t, infectSitePop)
                if WESresult is True:
                    simDetect_WES.append(curr_t)
                    WESDetect = True
            if verbose:
                print('WES detection scanned')
            # Add new infections if we've reached the beginning of any patient's shedding time
            for pat in patList:
                if (pat.beginShedTime == curr_t) and (pat not in patFinishedInfectingList):
                    # Infect other patients; first, how many, according to (R_0, k_0)
                    numOthersInfect = sps.nbinom.rvs(nbinom_n, nbinom_p)
                    # Which segments are these patients a part of? Depends on segment spread
                    newInfectSegments = np.random.choice([0, 1, 2, 3], size=numOthersInfect, p=pat.segmentSpreadProbs)
                    newInfectSegments = newInfectSegments.tolist()
                    # Which times do these infections occur? Sample uniformly across shedding time
                    temp = np.random.choice(np.arange(len(pat.shedLevelList)),
                                            p=pat.shedLevelList / np.sum(pat.shedLevelList),
                                            size=numOthersInfect)
                    newInfectTimes = [int(pat.beginShedTime + x) for x in temp]
                    # Add new infections
                    for newpatind in range(numOthersInfect):
                        if verbose:
                            print('Adding patient at infect time ' + str(newInfectTimes[newpatind]) + ', segment ' +
                                  str(newInfectSegments[newpatind]))
                        patList.append(Patient(newInfectTimes[newpatind], newInfectSegments[newpatind],
                                               beta_segment, parent=pat, verbose=verbose))
                        clinDetectTimes.append(int(patList[-1].clinDiagTime))
                    # Add current patient to finished list
                    patFinishedInfectingList.append(pat)
            if verbose:
                print('New infections added')
            # Add patients to exit list if simexittime exceeded
            for pat in patList:
                if (pat not in patExitList) and (pat.simexittime < curr_t):
                    patExitList.append(pat)
                    # Put default detection times if all patients have exited
                    if len(patList) == len(patExitList):
                        if WESDetect is False:
                            simDetect_WES.append(M)
                        if clinDetect is False:
                            simDetect_clin.append(M)
            if verbose:
                print('Exit patients compiled')
            # Increment time step
            curr_t += 1
            if curr_t == M:  # Put default values for detection times
                if WESDetect is False:
                    simDetect_WES.append(M)
                if clinDetect is False:
                    simDetect_clin.append(M)

    betaMedianList_WES.append(np.quantile(simDetect_WES, 0.5))
    betaMedianList_clin.append(np.quantile(simDetect_clin, 0.5))
    res_WES = stats.bootstrap((simDetect_WES,), np.median, confidence_level=0.95,
                          method='percentile', n_resamples=2000)
    res_clin = stats.bootstrap((simDetect_clin,), np.median, confidence_level=0.95,
                              method='percentile', n_resamples=2000)

    lower_ci_WES, upper_ci_WES = res_WES.confidence_interval.low, res_WES.confidence_interval.high
    lower_ci_clin, upper_ci_clin = res_clin.confidence_interval.low, res_clin.confidence_interval.high
    betaMedianList_WES_lo.append(lower_ci_WES)
    betaMedianList_clin_lo.append(lower_ci_clin)
    betaMedianList_WES_hi.append(upper_ci_WES)
    betaMedianList_clin_hi.append(upper_ci_clin)

# Plot
# plt.plot(beta_segmentVec, betaMedianList_WES, color='blue', linewidth=3)
# plt.plot(beta_segmentVec, betaMedianList_clin, color='red', linewidth=3)
lo_err_WES = [betaMedianList_WES[i] - betaMedianList_WES_lo[i] for i in range(len(betaMedianList_WES))]
hi_err_WES = [betaMedianList_WES_hi[i] - betaMedianList_WES[i]  for i in range(len(betaMedianList_WES))]
lo_err_clin = [betaMedianList_clin[i] - betaMedianList_clin_lo[i] for i in range(len(betaMedianList_clin))]
hi_err_clin = [betaMedianList_clin_hi[i] - betaMedianList_clin[i]  for i in range(len(betaMedianList_clin))]
asym_err_WES, asym_err_clin = [lo_err_WES, hi_err_WES], [lo_err_clin, hi_err_clin]
# mid_WES = [(betaMedianList_WES_hi[i]+betaMedianList_WES_lo[i])/2 for i in range(len(hi_err_WES))]
# mid_clin = [(betaMedianList_clin_hi[i]+betaMedianList_clin_lo[i])/2 for i in range(len(hi_err_clin))]
plt.errorbar(x=beta_segmentVec, y=betaMedianList_WES, yerr=asym_err_WES, fmt='o',
             color='darkblue', ecolor='cornflowerblue', elinewidth=2, capsize=8,
             capthick=2)
plt.errorbar(x=beta_segmentVec, y=betaMedianList_clin, yerr=asym_err_clin, fmt='o',
             color='darkred', ecolor='tomato', elinewidth=2, capsize=8,
             capthick=2)
plt.xlabel(r'Segment-infection intensity, $\beta$', fontsize=12)
plt.ylabel('Median detection time', fontsize=12)
plt.title(r'Median detection time vs. $\beta$'+'\nHyderabad - Cholera',
          fontsize=14)
plt.ylim([0, 1.1*np.max((betaMedianList_WES, betaMedianList_clin))])
plt.tight_layout()
plt.show()

###################
# Plot of median detection times vs unknown pop segment xi
###################
xiVec = np.arange(0.0, 1-v, 0.01)
xiMedianList_WES, xiMedianList_clin = [], []
xiMedianList_WES_lo, xiMedianList_clin_lo = [], []
xiMedianList_WES_hi, xiMedianList_clin_hi = [], []
N_SIM = 1000
beta_segment = 0.2 # Revert

for currxi in xiVec:
    h_not_v = currxi
    seg0_prop = 1 - v - h_not_v  # propor. NO healthcare, NO wastewater coverage
    seg1_prop = v - h + h_not_v  # propor. NO healthcare, YES wastewater coverage
    seg2_prop = h_not_v  # propor. YES healthcare, NO wastewater coverage
    seg3_prop = h - h_not_v  # propor. YES healthcare, YES wastewater coverage
    segprop_vec = [seg0_prop, seg1_prop, seg2_prop, seg3_prop]

    # Storage of detection times across simulations
    simDetect_WES, simDetect_clin = [], []

    # For printing output
    verbose = False

    for sim in range(N_SIM):  # Main simulation loop
        if verbose:
            print('Starting sim '+str(sim)+'...')
        R_0 = np.random.uniform(1.7, 2.6)  # NON-integer mean reproduction number; unknown range;
        #   For cholera, considered the number of SYMPTOMATIC patients created
        k_0 = 4.5  # Dispersion parameter

        infectSitePop = int(np.random.choice(sitePopList, size=1, p=sitePopList/np.sum(sitePopList))[0])

        # For python functions
        nbinom_n, nbinom_p = k_0, R_0 / (R_0 + (R_0**2)/k_0)

        # Index patient; choose initial segment purely randomly
        pat0 = Patient(0, np.random.choice([0, 1, 2, 3], size=1, p=segprop_vec)[0],
                       beta_segment, parent='Index')
        patList = [pat0]

        # Intialize list of clinical detection times
        clinDetectTimes = [int(pat0.clinDiagTime)]

        # Initialize first day of WES testing POST-infection of index case
        WESTestTime = np.random.randint(0, lamb)

        # Infect other patients; first, how many, according to (R_0, k_0)
        numOthersInfect = sps.nbinom.rvs(nbinom_n, nbinom_p)
        # Which segments are these patients a part of? Depends on segment spread
        newInfectSegments = np.random.choice([0, 1, 2, 3], size=numOthersInfect, p=pat0.segmentSpreadProbs)
        newInfectSegments = newInfectSegments.tolist()
        # Which times do these infections occur? Sample uniformly across shedding time
        temp = np.random.choice(np.arange(len(pat0.shedLevelList)),
                                          p = pat0.shedLevelList/np.sum(pat0.shedLevelList),
                                          size = numOthersInfect)
        newInfectTimes = [int(pat0.beginShedTime + x) for x in temp]

        for newpatind in range(numOthersInfect):
            if verbose:
                print('Adding patient at infect time '+str(newInfectTimes[newpatind])+', segment '+
                      str(newInfectSegments[newpatind]))
            patList.append(Patient(newInfectTimes[newpatind], newInfectSegments[newpatind],
                                   beta_segment, parent=pat0, verbose=verbose))
            clinDetectTimes.append(int(patList[-1].clinDiagTime))

        patFinishedInfectingList = [pat0]   # List of patients whose secondary infections have been added to patList

        # We now step through each time step until one of two things happens:
        #   1) We have a clinical detection AND a WES detection, or
        #   2) The outbreak has died out on its own before any detection; big M is used for non-detection

        # Initialize a list of patients whose exit time has been exceeded; stop once the length of this list matches
        #   the number of patients we've generated
        patExitList = []
        WESDetect, clinDetect = False, False  # Booleans for tracking if we've detected in each surveillance system
        curr_t = 0  # initialize the start day of the outbreak
        # Continue until all patients have exited the simulation OR both detection times have been identified
        # OR until max time reached
        while (len(patList) > len(patExitList)) and (WESDetect is False or clinDetect is False) and (curr_t < M):
            if verbose:
                print('Day ' + str(curr_t) + ' starting...')
            # Scan if we've reached a clinical detection time
            if (min(clinDetectTimes) == curr_t) and (clinDetect is False):
                simDetect_clin.append(curr_t)
                clinDetect = True
            if verbose:
                print('Clinical detection scanned')
            # Do WES measurement if on a scheduled WES day
            if np.mod(curr_t, lamb) == WESTestTime and (WESDetect is False):
                WESresult = GetWESResult(patList, curr_t, infectSitePop)
                if WESresult is True:
                    simDetect_WES.append(curr_t)
                    WESDetect = True
            if verbose:
                print('WES detection scanned')
            # Add new infections if we've reached the beginning of any patient's shedding time
            for pat in patList:
                if (pat.beginShedTime == curr_t) and (pat not in patFinishedInfectingList):
                    # Infect other patients; first, how many, according to (R_0, k_0)
                    numOthersInfect = sps.nbinom.rvs(nbinom_n, nbinom_p)
                    # Which segments are these patients a part of? Depends on segment spread
                    newInfectSegments = np.random.choice([0, 1, 2, 3], size=numOthersInfect, p=pat.segmentSpreadProbs)
                    newInfectSegments = newInfectSegments.tolist()
                    # Which times do these infections occur? Sample uniformly across shedding time
                    temp = np.random.choice(np.arange(len(pat.shedLevelList)),
                                            p=pat.shedLevelList / np.sum(pat.shedLevelList),
                                            size=numOthersInfect)
                    newInfectTimes = [int(pat.beginShedTime + x) for x in temp]
                    # Add new infections
                    for newpatind in range(numOthersInfect):
                        if verbose:
                            print('Adding patient at infect time ' + str(newInfectTimes[newpatind]) + ', segment ' +
                                  str(newInfectSegments[newpatind]))
                        patList.append(Patient(newInfectTimes[newpatind], newInfectSegments[newpatind],
                                               beta_segment, parent=pat, verbose=verbose))
                        clinDetectTimes.append(int(patList[-1].clinDiagTime))
                    # Add current patient to finished list
                    patFinishedInfectingList.append(pat)
            if verbose:
                print('New infections added')
            # Add patients to exit list if simexittime exceeded
            for pat in patList:
                if (pat not in patExitList) and (pat.simexittime < curr_t):
                    patExitList.append(pat)
                    # Put default detection times if all patients have exited
                    if len(patList) == len(patExitList):
                        if WESDetect is False:
                            simDetect_WES.append(M)
                        if clinDetect is False:
                            simDetect_clin.append(M)
            if verbose:
                print('Exit patients compiled')
            # Increment time step
            curr_t += 1
            if curr_t == M:  # Put default values for detection times
                if WESDetect is False:
                    simDetect_WES.append(M)
                if clinDetect is False:
                    simDetect_clin.append(M)

    xiMedianList_WES.append(np.quantile(simDetect_WES, 0.5))
    xiMedianList_clin.append(np.quantile(simDetect_clin, 0.5))
    res_WES = stats.bootstrap((simDetect_WES,), np.median, confidence_level=0.95,
                          method='percentile', n_resamples=2000)
    res_clin = stats.bootstrap((simDetect_clin,), np.median, confidence_level=0.95,
                              method='percentile', n_resamples=2000)

    lower_ci_WES, upper_ci_WES = res_WES.confidence_interval.low, res_WES.confidence_interval.high
    lower_ci_clin, upper_ci_clin = res_clin.confidence_interval.low, res_clin.confidence_interval.high
    xiMedianList_WES_lo.append(lower_ci_WES)
    xiMedianList_clin_lo.append(lower_ci_clin)
    xiMedianList_WES_hi.append(upper_ci_WES)
    xiMedianList_clin_hi.append(upper_ci_clin)

# Plot
# plt.plot(beta_segmentVec, betaMedianList_WES, color='blue', linewidth=3)
# plt.plot(beta_segmentVec, betaMedianList_clin, color='red', linewidth=3)
lo_err_WES = [xiMedianList_WES[i] - xiMedianList_WES_lo[i] for i in range(len(xiMedianList_WES))]
hi_err_WES = [xiMedianList_WES_hi[i] - xiMedianList_WES[i]  for i in range(len(xiMedianList_WES))]
lo_err_clin = [xiMedianList_clin[i] - xiMedianList_clin_lo[i] for i in range(len(xiMedianList_clin))]
hi_err_clin = [xiMedianList_clin_hi[i] - xiMedianList_clin[i]  for i in range(len(xiMedianList_clin))]
asym_err_WES, asym_err_clin = [lo_err_WES, hi_err_WES], [lo_err_clin, hi_err_clin]
# mid_WES = [(betaMedianList_WES_hi[i]+betaMedianList_WES_lo[i])/2 for i in range(len(hi_err_WES))]
# mid_clin = [(betaMedianList_clin_hi[i]+betaMedianList_clin_lo[i])/2 for i in range(len(hi_err_clin))]
plt.errorbar(x=xiVec, y=xiMedianList_WES, yerr=asym_err_WES, fmt='o',
             color='darkblue', ecolor='cornflowerblue', elinewidth=2, capsize=8,
             capthick=2)
plt.errorbar(x=xiVec, y=xiMedianList_clin, yerr=asym_err_clin, fmt='o',
             color='darkred', ecolor='tomato', elinewidth=2, capsize=8,
             capthick=2)
plt.xlabel(r'Pop. proportion with HC but no WES access, $\xi$', fontsize=12)
plt.ylabel('Median detection time', fontsize=12)
plt.title(r'Median detection time vs. $\xi$'+'\nHyderabad - Cholera',
          fontsize=14)
plt.ylim([0, 1.1*np.max((betaMedianList_WES, betaMedianList_clin))])
plt.tight_layout()
plt.show()

# TODO: Distribution of branches before WES/clinical detection
# TODO: Distribution of infected patients upon '' detection
# TODO: Median WES detection vs WES coverage by sites added
# TODO: Median WES detection vs WES frequency

###################
# Plot of median WES detection times vs number of sites
###################
h_not_v = 0.0 # Revert

# Site population proportions affect the coverage as well
# Form lists of corresponding v values under different sets of sites, largest to smallest
numSitesArr = np.arange(len(sitePopProps)).tolist()
sitePopProps_sort = sitePopProps.tolist().copy()
sitePopProps_sort.sort(reverse=True)
sitePops_sort = np.round(np.array(sitePopList),decimals=0).tolist().copy()
sitePops_sort = [int(x) for x in sitePops_sort]
sitePops_sort.sort(reverse=True)
# Use these for the ascending v values
sitePropsCumulCov = [sum(sitePopProps_sort[:(i+1)])*v for i in range(len(sitePopProps_sort))]
# Generate list of lists of props
sitePopPropLists = [[x for x in sitePopProps_sort[:(i+1)]] for i in range(len(sitePopProps_sort))]

numSitesMedianList = []
numSitesMedianList_lo = []
numSitesMedianList_hi = []
N_SIM = 1000

for numsitesind in numSitesArr:
    print('Evaluating using variable number of sites: ' + str(numsitesind+1))
    currv = sitePropsCumulCov[numsitesind]
    # For this exercise, ASSUME non-/covered WES populations are equally likely to have HC
    seg0_prop = (1 - currv)*(1-h)  # propor. NO healthcare, NO wastewater coverage
    seg1_prop = currv*(1-h)  # propor. NO healthcare, YES wastewater coverage
    seg2_prop = (1-currv)*h  # propor. YES healthcare, NO wastewater coverage
    seg3_prop = currv * h  # propor. YES healthcare, YES wastewater coverage
    segprop_vec = [seg0_prop, seg1_prop, seg2_prop, seg3_prop]

    # Storage of detection times across simulations
    simDetect_WES, simDetect_clin = [], []

    # For printing output
    verbose = False

    for sim in range(N_SIM):  # Main simulation loop
        if verbose:
            print('Starting sim '+str(sim)+'...')
        R_0 = np.random.uniform(1.7, 2.6)  # NON-integer mean reproduction number; unknown range;
        #   For cholera, considered the number of SYMPTOMATIC patients created
        k_0 = 4.5  # Dispersion parameter

        infectSitePop = int(np.random.choice(sitePops_sort[:(numsitesind+1)], size=1,
                                             p=sitePopPropLists[numsitesind]/np.sum(sitePopPropLists[numsitesind]))[0])

        # For python functions
        nbinom_n, nbinom_p = k_0, R_0 / (R_0 + (R_0**2)/k_0)

        # Index patient; choose initial segment purely randomly
        pat0 = Patient(0, np.random.choice([0, 1, 2, 3], size=1, p=segprop_vec)[0],
                       beta_segment, parent='Index')
        patList = [pat0]

        # Intialize list of clinical detection times
        clinDetectTimes = [int(pat0.clinDiagTime)]

        # Initialize first day of WES testing POST-infection of index case
        WESTestTime = np.random.randint(0, lamb)

        # Infect other patients; first, how many, according to (R_0, k_0)
        numOthersInfect = sps.nbinom.rvs(nbinom_n, nbinom_p)
        # Which segments are these patients a part of? Depends on segment spread
        newInfectSegments = np.random.choice([0, 1, 2, 3], size=numOthersInfect, p=pat0.segmentSpreadProbs)
        newInfectSegments = newInfectSegments.tolist()
        # Which times do these infections occur? Sample uniformly across shedding time
        temp = np.random.choice(np.arange(len(pat0.shedLevelList)),
                                          p = pat0.shedLevelList/np.sum(pat0.shedLevelList),
                                          size = numOthersInfect)
        newInfectTimes = [int(pat0.beginShedTime + x) for x in temp]

        for newpatind in range(numOthersInfect):
            if verbose:
                print('Adding patient at infect time '+str(newInfectTimes[newpatind])+', segment '+
                      str(newInfectSegments[newpatind]))
            patList.append(Patient(newInfectTimes[newpatind], newInfectSegments[newpatind],
                                   beta_segment, parent=pat0, verbose=verbose))
            clinDetectTimes.append(int(patList[-1].clinDiagTime))

        patFinishedInfectingList = [pat0]   # List of patients whose secondary infections have been added to patList

        # We now step through each time step until one of two things happens:
        #   1) We have a clinical detection AND a WES detection, or
        #   2) The outbreak has died out on its own before any detection; big M is used for non-detection

        # Initialize a list of patients whose exit time has been exceeded; stop once the length of this list matches
        #   the number of patients we've generated
        patExitList = []
        WESDetect, clinDetect = False, False  # Booleans for tracking if we've detected in each surveillance system
        curr_t = 0  # initialize the start day of the outbreak
        # Continue until all patients have exited the simulation OR both detection times have been identified
        # OR until max time reached
        while (len(patList) > len(patExitList)) and (WESDetect is False or clinDetect is False) and (curr_t < M):
            if verbose:
                print('Day ' + str(curr_t) + ' starting...')
            # Scan if we've reached a clinical detection time
            if (min(clinDetectTimes) == curr_t) and (clinDetect is False):
                simDetect_clin.append(curr_t)
                clinDetect = True
            if verbose:
                print('Clinical detection scanned')
            # Do WES measurement if on a scheduled WES day
            if np.mod(curr_t, lamb) == WESTestTime and (WESDetect is False):
                WESresult = GetWESResult(patList, curr_t, infectSitePop)
                if WESresult is True:
                    simDetect_WES.append(curr_t)
                    WESDetect = True
            if verbose:
                print('WES detection scanned')
            # Add new infections if we've reached the beginning of any patient's shedding time
            for pat in patList:
                if (pat.beginShedTime == curr_t) and (pat not in patFinishedInfectingList):
                    # Infect other patients; first, how many, according to (R_0, k_0)
                    numOthersInfect = sps.nbinom.rvs(nbinom_n, nbinom_p)
                    # Which segments are these patients a part of? Depends on segment spread
                    newInfectSegments = np.random.choice([0, 1, 2, 3], size=numOthersInfect, p=pat.segmentSpreadProbs)
                    newInfectSegments = newInfectSegments.tolist()
                    # Which times do these infections occur? Sample uniformly across shedding time
                    temp = np.random.choice(np.arange(len(pat.shedLevelList)),
                                            p=pat.shedLevelList / np.sum(pat.shedLevelList),
                                            size=numOthersInfect)
                    newInfectTimes = [int(pat.beginShedTime + x) for x in temp]
                    # Add new infections
                    for newpatind in range(numOthersInfect):
                        if verbose:
                            print('Adding patient at infect time ' + str(newInfectTimes[newpatind]) + ', segment ' +
                                  str(newInfectSegments[newpatind]))
                        patList.append(Patient(newInfectTimes[newpatind], newInfectSegments[newpatind],
                                               beta_segment, parent=pat, verbose=verbose))
                        clinDetectTimes.append(int(patList[-1].clinDiagTime))
                    # Add current patient to finished list
                    patFinishedInfectingList.append(pat)
            if verbose:
                print('New infections added')
            # Add patients to exit list if simexittime exceeded
            for pat in patList:
                if (pat not in patExitList) and (pat.simexittime < curr_t):
                    patExitList.append(pat)
                    # Put default detection times if all patients have exited
                    if len(patList) == len(patExitList):
                        if WESDetect is False:
                            simDetect_WES.append(M)
                        if clinDetect is False:
                            simDetect_clin.append(M)
            if verbose:
                print('Exit patients compiled')
            # Increment time step
            curr_t += 1
            if curr_t == M:  # Put default values for detection times
                if WESDetect is False:
                    simDetect_WES.append(M)
                if clinDetect is False:
                    simDetect_clin.append(M)

    numSitesMedianList.append(np.quantile(simDetect_WES, 0.5))
    res_WES = stats.bootstrap((simDetect_WES,), np.median, confidence_level=0.95,
                          method='percentile', n_resamples=2000)

    lower_ci_WES, upper_ci_WES = res_WES.confidence_interval.low, res_WES.confidence_interval.high
    numSitesMedianList_lo.append(lower_ci_WES)
    numSitesMedianList_hi.append(upper_ci_WES)

# Plot
lo_err_WES = [numSitesMedianList[i] - numSitesMedianList_lo[i] for i in range(len(numSitesMedianList))]
hi_err_WES = [numSitesMedianList_hi[i] - numSitesMedianList[i]  for i in range(len(numSitesMedianList))]
asym_err_WES = [lo_err_WES, hi_err_WES]
# mid_WES = [(betaMedianList_WES_hi[i]+betaMedianList_WES_lo[i])/2 for i in range(len(hi_err_WES))]
# mid_clin = [(betaMedianList_clin_hi[i]+betaMedianList_clin_lo[i])/2 for i in range(len(hi_err_clin))]
plt.errorbar(x=sitePropsCumulCov, y=numSitesMedianList, yerr=asym_err_WES, fmt='o',
             color='darkblue', ecolor='cornflowerblue', elinewidth=2, capsize=8,
             capthick=2)
plt.xlabel(r'Pop. proportion covered under WES', fontsize=12)
plt.ylabel('Median detection time', fontsize=12)
plt.title(r'Median detection time vs. number of WES sites'+'\nHyderabad - Cholera',
          fontsize=14)
plt.ylim([0, 1.1*np.max(numSitesMedianList)])
plt.tight_layout()
plt.show()