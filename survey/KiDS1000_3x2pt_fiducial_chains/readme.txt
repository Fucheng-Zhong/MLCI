This archive contains the samples for the main results presented in 
Heymans, Tröster et al. 2020 (arXiv:2007.15632).

Files
-----
cosmology/samples_multinest_blindC_EE.txt
    KiDS-1000 bandpower cosmic shear.
cosmology/samples_multinest_blindC_w.txt
    BOSS DR12 galaxy clustering.
cosmology/samples_multinest_blindC_EE_nE.txt
    Joint analysis of cosmic shear + galaxy-galaxy lensing (GGL).
cosmology/samples_multinest_blindC_EE_w.txt
    Joint analysis of cosmic shear + galaxy clustering.
cosmology/samples_multinest_blindC_EE_nE_w.txt
    Joint analysis of cosmic shear + GGL + galaxy clustering. 
    These contain the fiducial 3x2pt results.

The sample files are in CosmoSIS format. The parameter names are listed on the 
first line. The last four columns correspond to log prior, log likelihood, 
log posterior, and the multinest weight. Not all files contain all parameters. 
Below the parameters of the 3x2pt are listed.

Sampled parameters
------------------

Cosmological parameters:

1. cosmological_parameters--omch2
    Dark matter density \Omega_c h^2
2. cosmological_parameters--ombh2	
    Baryon density \Omega_b h^2
3. cosmological_parameters--h0
    Hubble constant h
4. cosmological_parameters--n_s
    Tilt of the primordial power spectrum
5. cosmological_parameters--s_8_input
    Approximate value S_8 = \sigma_8 \sqrt{\Omega_m/0.3}. This is converted into
    A_s in an initial, fast run of CAMB. 

Astrophysical lensing nuisance parameters:

6. halo_model_parameters--a
    HMCode baryon parameter A_bary.
7. intrinsic_alignment_parameters--a
    Intrinsic alignment amplitude A_IA.

n(z) nuisance parameters:
These parameters are the uncorrelated shifts, which get correlated and shifted
with the covariance of the SOM n(z) shifts.

8. nofz_shifts--p_1
9. nofz_shifts--p_2	
10. nofz_shifts--p_3	
11. nofz_shifts--p_4	
12. nofz_shifts--p_5

Galaxy bias and RSD parameters:
The two BOSS redshift bins have independent set of parameters.

13. bias_parameters--b1_bin_1
    Linear bias.
14. bias_parameters--b2_bin_1	
15. bias_parameters--gamma3_bin_1	
16. bias_parameters--a_vir_bin_1
    Non-Gaussianty of galaxy velocity distribution.

17. bias_parameters--b1_bin_2
18. bias_parameters--b2_bin_2	
19. bias_parameters--gamma3_bin_2	
20. bias_parameters--a_vir_bin_2	

Derived parameters
------------------
21. cosmological_parameters--s_8
    Accurate estimate of S_8.
22. cosmological_parameters--sigma_8
    Clustering amplitude \sigma_8.
23. cosmological_parameters--sigma_12
    Analogous to sigma_8 but measuring the variance of matter fluctuations in
    spheres of 12 Mpc (no factor of h). See Sanchez 2020 (arXiv:2002.07829).
24. cosmological_parameters--a_s
    Primordial power spectrum amplitude A_s.
25. cosmological_parameters--omega_m
    Matter density \Omega_m.
26. cosmological_parameters--omega_nu
    Neutrino density \Omega_\nu.
27. cosmological_parameters--omega_lambda
    Dark energy density \Omega_\Lambda.
28. cosmological_parameters--cosmomc_theta
    Parameter \theta_MC

29. nofz_shifts--bias_1
    Shift of n(z) in the first tomographic bin.
30. nofz_shifts--bias_2	
31. nofz_shifts--bias_3	
32. nofz_shifts--bias_4	
33. nofz_shifts--bias_5	

34. delta_z_out--bin_1
    Mean of the shifted n(z) of the first tomographic bin.
35. delta_z_out--bin_2	
36. delta_z_out--bin_3	
37. delta_z_out--bin_4	
38. delta_z_out--bin_5	

Other parameters
----------------
39. prior
    Logarithm of the prior at the sample position.
40. like
    Logarithm of the likelihood at the sample position.
41. post
    Logarithm of the poster at the sample position.
42. weight
    MultiNest sample weight.
