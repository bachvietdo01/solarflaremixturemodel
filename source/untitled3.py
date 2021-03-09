#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:10:43 2021

@author: vietdo
"""

from SolarFlareMM0EM import SolarFlareMM0EM

K = 2
em_run = run_mm0_em(20, X, y, K, X_test, y_test, debug_sigma2 = mm0_sim.sigma2,
                    debug_Sigma = mm0_sim.Sigma, debug_pi = mm0_sim.pi, 
                    debug_mu = mm0_sim.mu, debug_beta = mm0_sim.beta)

em_run['z_test'][19] - mm0_sim.z[800:,]
z