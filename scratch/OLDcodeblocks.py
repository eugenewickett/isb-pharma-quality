#######################
# WHOLESALE PRICE PLOTS - OLD
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


#############################
# Social welfare plot
#############################
b, cSup, supRateLo = 0.95, 0.2, 0.8
scDict = {'b': b, 'cSup': cSup, 'supRateLo': supRateLo}
uL_min, uL_max = -1.5, 0.75
cSup_max = 0.5
numpts, alval = 6, 0.9

eqMat, CthMat, SWMat = SocWelEqMatsForPlot(numpts, uL_min, uL_max, cSup_max, scDict, printUpdate=True)

fig = plt.figure()
fig.suptitle(r'$b=$'+str(b)+', '+r'$L=$'+str(supRateLo),
             fontsize=18, fontweight='bold')
ax1 = fig.add_subplot(221)

eqcolors = ['#cf0234', 'deeppink', '#021bf9', '#0d75f8', '#82cafc', '#5ca904', '#0b4008', 'black']
labels = ['LL', 'LLsqz', 'LHexp', 'LHFOC', 'LHsqz', 'HHsqz', 'HH', 'N']

imlist = []
for eqind in range(len(labels)):
    mycmap = matplotlib.colors.ListedColormap(['white', eqcolors[eqind]], name='from_list', N=None)
    if eqcolors[eqind] == 'black':  # No alpha transparency
        im = ax1.imshow(eqMat[eqind].T, vmin=0, vmax=1, aspect='auto',
                            extent=(uL_min, uL_max, 0, cSup_max),
                            origin="lower", cmap=mycmap, alpha=1)
    else:
        im = ax1.imshow(eqMat[eqind].T, vmin=0, vmax=1, aspect='auto',
                            extent=(uL_min, uL_max, 0, cSup_max),
                            origin="lower", cmap=mycmap, alpha=alval)
    imlist.append(im)

legwidth = 20
wraplabels = ['\n'.join(textwrap.wrap(labels[i], width=legwidth)) for i in range(len(labels))]
patches = [mpatches.Patch(color=eqcolors[i], edgecolor='black', label=wraplabels[i], alpha=alval) for i in range(len(eqcolors))]
# put those patched as legend-handles into the legend
ax1.legend(handles=patches, bbox_to_anchor=(1.3, 1.0), loc='upper right', borderaxespad=0.1, fontsize=8)
ax1.set_xbound(uL_min, uL_max)
ax1.set_ybound(0, cSup_max)
ax1.set_box_aspect(1)
plt.xlabel(r'$u_{L}$', fontsize=14)
plt.ylabel(r'$c_S$', fontsize=14, rotation=0, labelpad=14)
ax2 = fig.add_subplot(222)
im = ax2.imshow(CthMat.T, vmin=np.min(CthMat), vmax=np.max(CthMat), aspect='auto', extent=(uL_min, uL_max, 0, cSup_max),
                            origin="lower", cmap='Reds', alpha=1)
ax3 = fig.add_subplot(223)
im = ax3.imshow(SWMat.T, vmin=np.min(SWMat), vmax=np.max(SWMat), aspect='auto', extent=(uL_min, uL_max, 0, cSup_max),
                            origin="lower", cmap='Blues', alpha=1)

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