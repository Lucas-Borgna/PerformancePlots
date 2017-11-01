#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:43:11 2017

@author: Santi
"""
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt


def get_mat_data(filename):
    # Assuming sample_matlab_file.mat has 2 matrices X and Y
    matData = scipy.io.loadmat(filename)
    return matData

#LOW PT
V_fpr = np.load("Data/fpr_2017.08.18_21.56.57.npy")
V_tpr = np.load("Data/tpr_2017.08.18_21.56.57.npy")
V_auc = np.load("Data/auc_2017.08.18_21.56.57.npy")

L_fpr = np.load("Data/fpr_2017.08.18_21.46.24.npy")
L_tpr = np.load("Data/tpr_2017.08.18_21.46.24.npy")
L_auc = np.load("Data/auc_2017.08.18_21.46.24.npy")

V16_fpr = np.load('Data/fpr_2017.08.22_12.34.03.npy')
V16_tpr = np.load('Data/tpr_2017.08.22_12.34.03.npy')
V16_auc = np.load('Data/auc_2017.08.22_12.34.03.npy')

#High PT
LH_fpr = np.load('Data/fpr_2017.08.25_13.03.43.npy')
LH_tpr = np.load('Data/tpr_2017.08.25_13.03.43.npy')
LH_auc = np.load('Data/auc_2017.08.25_13.03.43.npy')

VH_fpr = np.load('Data/fpr_2017.08.25_15.41.37.npy')
VH_tpr = np.load('Data/tpr_2017.08.25_15.41.37.npy')
VH_auc = np.load('Data/auc_2017.08.25_15.41.37.npy')

V16H_fpr = np.load('Data/fpr_2017.08.25_13.22.27.npy')
V16H_tpr = np.load('Data/tpr_2017.08.25_13.22.27.npy')
V16H_auc = np.load('Data/auc_2017.08.25_13.22.27.npy')


#LOAD LOW PT LITERATURE DATA FOR D2, DNN, BDT

filename = "Data/D2.mat"
D2 = get_mat_data(filename)
D2_X = np.array(D2['X'])
D2_Y = np.array(D2['Y'])

filename = "Data/DNN.mat"
DNN = get_mat_data(filename)
DNN_X = np.array(DNN['X'])
DNN_Y = np.array(DNN['Y'])

filename = "Data/BDT.mat"
BDT = get_mat_data(filename)
BDT_X = np.array(BDT['X'])
BDT_Y = np.array(BDT['Y'])


#LOAD HIGH PT LITERATURE DATA FOR D2, DNN, BDT

D2_High = get_mat_data('Data/D2High2.mat')
D2H_X = np.array(D2_High['X'])
D2H_Y = np.array(D2_High['Y'])

DNN_High = get_mat_data('Data/DNN_HighPt.mat')
DNNH_X = np.array(DNN_High['X'])
DNNH_Y = np.array(DNN_High['Y'])

BDT_High = get_mat_data('Data/BDT_HighPt.mat')
BDTH_X = np.array(BDT_High['X'])
BDTH_Y = np.array(BDT_High['Y'])

#PLOTTING ROC CURVES IN THE HEP (W-TAGGING) STYLE
#MEANS WE PLOT (1/FPR) VS TPR 

eps = 0.0000000001 #To avoid division by zero error

#LOW PT PLOT
plt.clf()
plt.figure()
plt.semilogy(BDT_X, BDT_Y,  '--',label = 'BDT')
plt.semilogy(D2_X, D2_Y, '-.', label = r'$\mathbf{D_2}$ $m^{comb}$ [60, 100] GeV')
plt.semilogy(DNN_X, DNN_Y, label = 'DNN')
plt.semilogy(L_tpr, 1/(L_fpr+eps), label = 'LeNet5 AUC: %0.2f' %L_auc )
plt.semilogy(V_tpr, 1/(V_fpr+eps), label = 'VGG19 AUC: %0.2f' %V_auc)
plt.semilogy(V16_tpr, 1/(V16_fpr+eps),'-.', label = 'VGG16 AUC: %0.2f' %V16_auc)
plt.ylabel(r'Background Rejection (1/${\varepsilon_{bkg}}$)')
plt.xlabel(r'Signal Efficiency ($\varepsilon_{sig}$)')
plt.title(r'Performance Comparison: Low $p^{truth}_T$')
plt.text(0.35, 850,r'$ s = \sqrt{13}$ TeV')
plt.text(0.35, 600, r'anti-$k_t$ R = 1.0 jets, $|\eta|^{truth} < 2.0$')
plt.text(0.35, 400, r'Trimmed ($f_{cut} = 0.05$ $R_{sub}$ = 0.2)')
plt.text(0.35, 270, r'$p_T^{truth} = [500, 1000]$ GeV') 
plt.xlim([0.3, 1.0])
plt.ylim([1.0, 1.2*10**3])
plt.legend(frameon=False)
#plt.grid(True, which ='both', ls='-')
plt.grid(False)
plt.savefig('performanceLowPt.png', format='png', dpi= 600)
plt.show()



#HIGH PT PLOT
plt.clf()
plt.figure()
plt.semilogy(BDTH_X, BDTH_Y, '--', label = 'BDT')
plt.semilogy(D2H_X, D2H_Y, label = r'$\mathbf{D_2}$ $m^{comb}$ [60, 100] GeV')
plt.semilogy(DNNH_X, DNNH_Y, label = 'DNN')
plt.semilogy(VH_tpr, 1/(VH_fpr+eps), '-.', label = 'VGG19 AUC: %0.2f' %VH_auc)
plt.semilogy(LH_tpr, 1/(LH_fpr+eps), '--', label = 'LeNet5 AUC: %0.2f' %LH_auc)
plt.semilogy(V16H_tpr, 1/(V16H_fpr+eps), label = 'VGG16 AUC: %0.2f' %V16H_auc)
plt.ylabel(r'Background Rejection (1/${\varepsilon_{bkg}}$)')
plt.xlabel(r'Signal Efficiency ($\varepsilon_{sig}$)')
plt.title(r'Performance Comparison: High $p^{truth}_T$')
plt.text(0.35, 850,r'$ s = \sqrt{13}$ TeV')
plt.text(0.35, 600, r'anti-$k_t$ R = 1.0 jets, $|\eta|^{truth} < 2.0$')
plt.text(0.35, 400, r'Trimmed ($f_{cut} = 0.05$ $R_{sub}$ = 0.2)')
plt.text(0.35, 270, r'$p_T^{truth} = [1500, 2000]$ GeV')
plt.xlim([0.3, 1.0])
plt.ylim([1.0, 1.2*10**3])
plt.legend(frameon = False)
plt.grid(False)
plt.savefig('performanceHighPt.png', format='png', dpi = 600)
plt.show()