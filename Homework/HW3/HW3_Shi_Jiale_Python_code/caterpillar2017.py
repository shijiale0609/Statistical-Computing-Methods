'''
Nicholas Geneva
ngeneva@nd.edu
September 23, 2017
'''
import operator as op
import numpy as np
from scipy import special

import util
import log
import plotFigures as plot
import regression as re
import variableSelection as vs

if __name__ == '__main__':
    plot.pltSetUp()

    #======= Problem 1 =======
    data = util.readFileData('caterpillar.data')
    plot.coreFigure3_1(data)

    #======= Problem 2 =======
    x_data = data[:,:-1] #training inputs
    t_data = np.log(data[:,-1]) #training outputs
    (N,K) = x_data.shape
    (beta_hat, std_err, t, p) = re.mleRegression(x_data,t_data,0)
    plot.coreFigure3_2(beta_hat, std_err, t, p)

    #======= Problem 3 =======
    beta_tilde = np.zeros(beta_hat.shape)
    b0_data = np.zeros((5,4))
    for i, c0 in enumerate([0.1, 1, 10, 100, 1000]):
        b0_data[i,0] = c0
        (exp_sig2, exp_beta, var_beta) = re.priorExpectations(x_data, t_data, beta_hat, beta_tilde, 2.1, 2.0, 100)
        b0_data[i,1:] = [exp_sig2, exp_beta[0], var_beta[0]]
        if(c0 == 100):
            plot.coreTableExpVar(exp_beta, var_beta, '3.2')
    
    plot.coreTable3_1(b0_data[:,0], b0_data[:,1], b0_data[:,2],  b0_data[:,3])

    #======= Problem 4 =======
    (exp_beta, var_beta, b_10) = re.gPriorExpectations(x_data, t_data, beta_hat, beta_tilde, 100, 100)
    plot.coreTableExpVar(exp_beta, var_beta, '3.3', 100)
    
    (exp_beta, var_beta, b_10) = re.gPriorExpectations(x_data, t_data, beta_hat, beta_tilde, 1000, 1000)
    plot.coreTableExpVar(exp_beta, var_beta, '3.4', 1000)
    plot.coreTable_B10(exp_beta, var_beta, np.log10(b_10), 1000)

    #======= Problem 5 =======
    hpd = re.getHPD(x_data, t_data, beta_hat, beta_tilde, 0.1)
    plot.coreTable3_5(hpd)

    #======= Problem 6 =======
    (exp_beta, var_beta) = re.zellnerNonInfoGPrior(x_data, t_data, beta_hat, beta_tilde)
    plot.coreTableExpVar(exp_beta, var_beta, '3.6')

    #======= Problem 7 =======
    (evid_info, evid_noninfo) = vs.variableSelection(x_data, t_data, beta_tilde, 100, 1e3)
    plot.coreVarSelTable(evid_info[:20], '3.7', K, 100)
    plot.coreVarSelTable(evid_noninfo[:20], '3.8', K)
    
    (gibbs_info, gibs_info_beta) = vs.gibbsSamplingInformative(x_data, t_data, beta_tilde, 100, 1e4, 1e3)
    (gibbs_noninfo, gibs_noninfo_beta) = vs.gibbsSamplingNonInformative(x_data, t_data, beta_tilde, 1000, 5e3, 1e3)

    #unsort gibbs variable selection (redundant I know)
    gibbs_info0 = gibbs_info[ gibbs_info[:,0].argsort()]
    gibbs_noninfo0 = gibbs_noninfo[ gibbs_noninfo[:,0].argsort()]
    #Now get the gibb's models in order of our variable selection (Incase they don't match precisely)
    gibbs_info0 = gibbs_info0[evid_info[:20,0].astype(int), :]
    gibbs_noninfo0 = gibbs_noninfo0[evid_noninfo[:20,0].astype(int), :]

    plot.coreGibbsModelEvidenceTable(evid_info[:20], gibbs_info0, '3.9', K)
    plot.coreGibbsModelEvidenceTable(evid_noninfo[:20], gibbs_noninfo0, '3.10', K)
    plot.coreTable3_11(gibs_info_beta, gibs_noninfo_beta, 100)
    
    #plot.showFigures()
