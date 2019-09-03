import os
import re
import numpy as np

import matplotlib as mlp
import matplotlib.pyplot as plt
from matplotlib import rc

import util
import log

def pltSetUp():
    plt.close('all')
    plt.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
        

def coreFigure3_1(data):
    #Set up subplots
    f, ax = plt.subplots(2, 5, figsize=(6, 6))
    f.suptitle('Bayesian Core Figure 3.1', fontsize=14)
    log.info("Creating Bayesian Core Figure 3.1")

    n = 0
    for (i,j), ax0 in np.ndenumerate(ax):
        ax0.plot(data[:,n],data[:,-1],'o',markersize=3.5)
        ax0.set_yscale('log')
        # Get rid of the ticks                          
        ax0.set_xticks([])                               
        ax0.set_yticks([])
        ax0.minorticks_off() 
        #Axis label
        ax0.set_xlabel(r'$x_'+str(n+1)+'$')
        n+=1
    
    plt.tight_layout(rect=[0,0, 1.0, 0.93])
    log.log("Saving Figure...")
    plt.savefig('Figure3_1.png')
    log.sucess("Figure created and sucessfully saved")

def coreFigure3_2(beta_hat, std_err, t, p):
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(6, 6))
    f.suptitle('Bayesian Core Figure 3.2', fontsize=14)
    log.info("Creating Bayesian Core Figure 3.2")

    # hide axes
    f.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    cell_data = np.array([beta_hat, std_err, t, p]).T
    cell_text = util.npArrayToStrList(cell_data, '{0:.6f}')

    col_labels = ['Estimate','Std. Error','t-value',r'$Pr(>|t|)$']
    row_labels = ['intercept']
    for i, val in enumerate(beta_hat[1:]):
        row_labels.append('XV'+str(i))

    tab = ax.table(cellText=cell_text, 
                    rowLabels=row_labels, 
                    colLabels=col_labels, 
                    cellLoc='center', 
                    loc='center',
                    bbox=[0.15, 0.2, 0.9, 0.7])

    tab.set_fontsize(16)
    tab.scale(1, 2)
    log.log("Saving Figure...")
    plt.savefig('Figure3_2.png')
    log.sucess("Figure created and sucessfully saved")

def coreTable3_1(c, exp_sig2, exp_beta, var_beta):
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    f.suptitle('Bayesian Core Table 3.1', fontsize=14)
    log.info("Creating Bayesian Core Table 3.2")

    # hide axes
    f.patch.set_visible(False)
    ax.axis('off')

    cell_data = np.array([c, exp_sig2, exp_beta, var_beta]).T
    cell_text = util.npArrayToStrList(cell_data, '{0:.4f}')
    col_labels = [r'c',r'$E^{\pi}(\sigma^{2}|Y,X)$',r'$E^{\pi}(\beta_{0}|Y,X)$',r'$V^{\pi}(\beta_{0}|Y,X)$']

    tab = ax.table(cellText=cell_text,
                    colLabels=col_labels, 
                    cellLoc='center', 
                    loc='center')

    tab.set_fontsize(16)
    tab.scale(1, 2)
    log.log("Saving Figure...")
    plt.savefig('Table3_1.png')
    log.sucess("Figure created and sucessfully saved")

def coreTableExpVar(exp_beta, var_beta, fignum_str, c=None):
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(6, 5))
    if(c != None):
        f.suptitle('Bayesian Core Table '+fignum_str+', C = %d'%(c), fontsize=14)
    else:
        f.suptitle('Bayesian Core Table '+fignum_str, fontsize=14)
    log.info("Creating Bayesian Core Table "+fignum_str)

    # hide axes
    f.patch.set_visible(False)
    ax.axis('off')

    cell_data = np.array([exp_beta, var_beta]).T
    cell_text = util.npArrayToStrList(cell_data, '{0:.4f}')
    col_labels = [r'$\beta_{i}$',r'$E^{\pi}(\beta_{i}|Y,X)$',r'$V^{\pi}(\beta_{i}|Y,X)$']
    beta_labels = []
    for i, val in enumerate(exp_beta):
        beta_labels.append(r'$\beta_{'+str(i)+'}$')
    cell_text = util.appendListColumn(cell_text, beta_labels, 0)

    tab = ax.table(cellText=cell_text, 
                    colLabels=col_labels, 
                    colWidths=[0.2,0.3,0.3],
                    cellLoc='center', 
                    loc='center')

    tab.set_fontsize(16)
    tab.scale(1, 2)
    log.log("Saving Figure...")
    plt.savefig('Table' +fignum_str.replace('.','_')+ '.png')
    log.sucess("Figure created and sucessfully saved")

def coreTable_B10(exp_beta, var_beta, log_b10, c):
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(6, 5))
    f.suptitle('Bayes\' Factor, C='+str(c), fontsize=14)
    log.info("Creating Bayes Factor Table")

    # hide axes
    f.patch.set_visible(False)
    ax.axis('off')

    cell_data = np.array([exp_beta, var_beta, log_b10]).T
    cell_text = util.npArrayToStrList(cell_data, '{0:.4f}')
    col_labels = [r'$\beta_{i}$',r'$E^{\pi}(\beta_{i}|Y,X)$',r'$V^{\pi}(\beta_{i}|Y,X)$',r'$log_{10}(BF)$']
    beta_labels = []
    for i, val in enumerate(exp_beta):
        beta_labels.append(r'$\beta_{'+str(i)+'}$')
    cell_text = util.appendListColumn(cell_text, beta_labels, 0)

    tab = ax.table(cellText=cell_text, 
                    colLabels=col_labels, 
                    colWidths=[0.2,0.3,0.3,0.3],
                    cellLoc='center', 
                    loc='center')

    tab.set_fontsize(16)
    tab.scale(1, 2)
    log.log("Saving Figure...")
    plt.savefig('Table_BayesFactor.png')
    log.sucess("Figure created and sucessfully saved")

def coreTable3_5(hpd):
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(6, 5))
    f.suptitle('Bayesian Core Table 3.5', fontsize=14)
    log.info("Creating Bayesian Core Table 3.5")

    # hide axes
    f.patch.set_visible(False)
    ax.axis('off')

    cell_text0 = util.npArrayToStrList(hpd, '{0:.4f}')
    cell_text1 = [str(row).replace('\'','') for i, row in enumerate(cell_text0)]
    cell_text = map(list, zip(cell_text1))

    col_labels = [r'$\beta_{i}$','HPD Interval']
    beta_labels = []
    for i, val in enumerate(cell_text):
        beta_labels.append(r'$\beta_{'+str(i)+'}$')
    cell_text = util.appendListColumn(cell_text, beta_labels, 0)
    

    tab = ax.table(cellText=cell_text, 
                    colLabels=col_labels, 
                    colWidths=[0.2,0.3],
                    cellLoc='center', 
                    loc='center')

    tab.set_fontsize(16)
    tab.scale(1, 2)
    log.log("Saving Figure...")
    plt.savefig('Table3_5.png')
    log.sucess("Figure created and sucessfully saved")

def coreVarSelTable(evidence, fignum_str, K, c=None):
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(6, 7))
    if(c != None):
        f.suptitle('Bayesian Core Table '+fignum_str+', C = %d'%(c), fontsize=14)
    else:
        f.suptitle('Bayesian Core Table '+fignum_str, fontsize=14)
    log.info("Creating Bayesian Core Table "+fignum_str)

    t_gamma = []
    for e0 in evidence:
        #e0 contains [model id, evidence]
        t_gamma_np, q = util.getGammaIndexes(K, int(e0[0]))
        model_label = str(t_gamma_np.tolist())
        t_gamma.append(re.sub('[^0-9 ,]+', '', model_label))
    
    # hide axes
    f.patch.set_visible(False)
    ax.axis('off')

    cell_data = np.array([evidence[:,1]]).T
    cell_text = util.npArrayToStrList(cell_data, '{0:.5f}')
    
    col_labels = [r'$t_{1}(\gamma)$',r'$\pi(\gamma|Y,X)$']
    cell_text = util.appendListColumn(cell_text, t_gamma, 0)
    

    tab = ax.table(cellText=cell_text, 
                    colLabels=col_labels, 
                    colWidths=[0.4, 0.3],
                    cellLoc='center', 
                    loc='center')

    tab.set_fontsize(12)
    tab.scale(1, 1.75)
    log.log("Saving Figure...")
    plt.savefig('Table' +fignum_str.replace('.','_')+ '.png')
    log.sucess("Figure created and sucessfully saved")

def coreGibbsModelEvidenceTable(model_evid, gibbs_evid, fignum_str, K, c=None):
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(6, 7))
    if(c != None):
        f.suptitle('Bayesian Core Table '+fignum_str+', C = %d'%(c), fontsize=14)
    else:
        f.suptitle('Bayesian Core Table '+fignum_str, fontsize=14)
    log.info("Creating Bayesian Core Table "+fignum_str)

    t_gamma = []
    for e0 in gibbs_evid:
        #e0 contains [model id, evidence]
        t_gamma_np, q = util.getGammaIndexes(K, int(e0[0]))
        model_label = str(t_gamma_np.tolist())
        t_gamma.append(re.sub('[^0-9 ,]+', '', model_label))
    
    # hide axes
    f.patch.set_visible(False)
    ax.axis('off')

    cell_data = np.array([model_evid[:,1], gibbs_evid[:,1]]).T
    cell_text = util.npArrayToStrList(cell_data, '{0:.5f}')
    
    col_labels = [r'$t_{1}(\gamma)$',r'$\pi(\gamma|Y,X)$',r'$\hat{\pi}(\gamma|Y,X)$']
    cell_text = util.appendListColumn(cell_text, t_gamma, 0)
    

    tab = ax.table(cellText=cell_text, 
                    colLabels=col_labels, 
                    colWidths=[0.4, 0.3, 0.3],
                    cellLoc='center', 
                    loc='center')

    tab.set_fontsize(12)
    tab.scale(1, 1.75)
    log.log("Saving Figure...")
    plt.savefig('Table' +fignum_str.replace('.','_')+ '.png')
    log.sucess("Figure created and sucessfully saved")

def coreTable3_11(gibs_info_beta, gibs_noninfo_beta, c):
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(6, 5))
    f.suptitle('Bayesian Core Table 3.11, C=%d'%(c), fontsize=14)
    ax.set_title('Left: Informative, Right: Non-Informative')
    log.info("Creating Bayesian Core Table 3.11")
    
    # hide axes
    f.patch.set_visible(False)
    ax.axis('off')

    cell_data = np.array([gibs_info_beta, gibs_noninfo_beta]).T
    cell_text = util.npArrayToStrList(cell_data, '{0:.5f}')
    
    col_labels = [r'$\gamma_{i}$',r'$\hat{P}^{\pi}(\gamma_{i}=1|Y,X)$',r'$\hat{P}^{\pi}(\gamma_{i}=1|Y,X)$']
    gamma_labels = []
    for i, val in enumerate(cell_text):
        gamma_labels.append(r'$\gamma_{'+str(i)+'}$')
    cell_text = util.appendListColumn(cell_text, gamma_labels, 0)

    tab = ax.table(cellText=cell_text, 
                    colLabels=col_labels, 
                    colWidths=[0.2, 0.3, 0.3],
                    cellLoc='center', 
                    loc='center')

    tab.set_fontsize(12)
    tab.scale(1, 1.75)
    log.log("Saving Figure...")
    plt.savefig('Table3_11.png')
    log.sucess("Figure created and sucessfully saved")

def showFigures():
    log.info("Displaying all tables and plots")
    log.warning("May take a bit to display all...")
    plt.show()