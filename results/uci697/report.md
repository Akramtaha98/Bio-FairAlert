# UCI-697 external validation report

## Macro-F1 (3-class), mean over folds
```
checkpoint              C0     C1     C2
model        filter                     
Hybrid-Stack mi      0.550  0.677  0.705
             none    0.574  0.694  0.724
LightGBM     mi      0.546  0.665  0.695
             none    0.569  0.683  0.719
RandomForest mi      0.537  0.664  0.698
             none    0.547  0.668  0.707
XGBoost      mi      0.554  0.672  0.699
             none    0.575  0.690  0.726
```

## Paired tests (Nadeau-Bengio, Holm)
```
checkpoint                                                   test  mean_diff   ci_lo   ci_hi       t   p_nb      dz  p_holm
        C0               filter effect (MI - none) | RandomForest    -0.0104 -0.0294  0.0085 -1.1340 0.2680 -0.6107  1.0000
        C0                    filter effect (MI - none) | XGBoost    -0.0216 -0.0417 -0.0014 -2.2124 0.0367 -1.1914  0.3670
        C0                   filter effect (MI - none) | LightGBM    -0.0225 -0.0430 -0.0020 -2.2612 0.0331 -1.2177  0.3639
        C0               filter effect (MI - none) | Hybrid-Stack    -0.0241 -0.0376 -0.0106 -3.6775 0.0012 -1.9804  0.0190
        C0 Hybrid-Stack - best individual (XGBoost) | filter=none    -0.0011 -0.0179  0.0158 -0.1299 0.8977 -0.0700  1.0000
        C0   Hybrid-Stack - best individual (XGBoost) | filter=mi    -0.0036 -0.0189  0.0117 -0.4827 0.6337 -0.2600  1.0000
        C1               filter effect (MI - none) | RandomForest    -0.0042 -0.0171  0.0088 -0.6639 0.5131 -0.3575  1.0000
        C1                    filter effect (MI - none) | XGBoost    -0.0174 -0.0290 -0.0058 -3.1020 0.0049 -1.6705  0.0632
        C1                   filter effect (MI - none) | LightGBM    -0.0173 -0.0274 -0.0073 -3.5555 0.0016 -1.9147  0.0241
        C1               filter effect (MI - none) | Hybrid-Stack    -0.0169 -0.0263 -0.0075 -3.7026 0.0011 -1.9939  0.0189
        C1 Hybrid-Stack - best individual (XGBoost) | filter=none     0.0043 -0.0051  0.0137  0.9462 0.3535  0.5095  1.0000
        C1   Hybrid-Stack - best individual (XGBoost) | filter=mi     0.0048 -0.0066  0.0162  0.8726 0.3915  0.4699  1.0000
        C2               filter effect (MI - none) | RandomForest    -0.0090 -0.0237  0.0056 -1.2713 0.2158 -0.6846  1.0000
        C2                    filter effect (MI - none) | XGBoost    -0.0270 -0.0405 -0.0134 -4.1110 0.0004 -2.2138  0.0072
        C2                   filter effect (MI - none) | LightGBM    -0.0239 -0.0414 -0.0065 -2.8260 0.0093 -1.5219  0.1122
        C2               filter effect (MI - none) | Hybrid-Stack    -0.0181 -0.0289 -0.0074 -3.4799 0.0019 -1.8740  0.0271
        C2 Hybrid-Stack - best individual (XGBoost) | filter=none    -0.0021 -0.0116  0.0074 -0.4551 0.6531 -0.2451  1.0000
        C2   Hybrid-Stack - best individual (XGBoost) | filter=mi     0.0067 -0.0057  0.0192  1.1158 0.2756  0.6009  1.0000
```

## Side effects on DIR (C2)
```
checkpoint   mitigation      target       audited metric  before  after  delta  ci_lo  ci_hi  worsened  significant
        C2   expgrad_eo  age_mature        gender    dir   0.545  0.551  0.006 -0.000  0.013     False        False
        C2   expgrad_eo  age_mature    age_mature    dir   0.398  0.402  0.005  0.000  0.009     False         True
        C2   expgrad_eo  age_mature   scholarship    dir   0.263  0.264  0.001 -0.005  0.007     False        False
        C2   expgrad_eo  age_mature special_needs    dir   0.933  0.966  0.006 -0.046  0.070     False        False
        C2   expgrad_eo  age_mature international    dir   0.896  0.914  0.005 -0.035  0.039     False        False
        C2   expgrad_eo      gender        gender    dir   0.545  0.556  0.011  0.003  0.018     False         True
        C2   expgrad_eo      gender    age_mature    dir   0.398  0.399  0.001 -0.004  0.006     False        False
        C2   expgrad_eo      gender   scholarship    dir   0.263  0.266  0.003 -0.004  0.009     False        False
        C2   expgrad_eo      gender special_needs    dir   0.933  0.931 -0.004 -0.067  0.053      True        False
        C2   expgrad_eo      gender international    dir   0.896  0.906  0.001 -0.037  0.035     False        False
        C2   expgrad_eo       joint        gender    dir   0.545  0.558  0.014  0.007  0.021     False         True
        C2   expgrad_eo       joint    age_mature    dir   0.398  0.399  0.002 -0.004  0.007     False        False
        C2   expgrad_eo       joint   scholarship    dir   0.263  0.284  0.021  0.013  0.029     False         True
        C2   expgrad_eo       joint special_needs    dir   0.933  0.957  0.005 -0.039  0.064     False        False
        C2   expgrad_eo       joint international    dir   0.896  0.891 -0.012 -0.062  0.031      True        False
        C2   expgrad_eo scholarship        gender    dir   0.545  0.549  0.004 -0.003  0.010     False        False
        C2   expgrad_eo scholarship    age_mature    dir   0.398  0.401  0.003 -0.002  0.008     False        False
        C2   expgrad_eo scholarship   scholarship    dir   0.263  0.279  0.016  0.009  0.023     False         True
        C2   expgrad_eo scholarship special_needs    dir   0.933  0.899 -0.005 -0.092  0.089      True        False
        C2   expgrad_eo scholarship international    dir   0.896  0.905 -0.002 -0.040  0.025      True        False
        C2      reweigh  age_mature        gender    dir   0.545  0.561  0.016  0.007  0.027     False         True
        C2      reweigh  age_mature    age_mature    dir   0.398  0.465  0.067  0.058  0.076     False         True
        C2      reweigh  age_mature   scholarship    dir   0.263  0.285  0.022  0.013  0.032     False         True
        C2      reweigh  age_mature special_needs    dir   0.933  0.944  0.004 -0.085  0.107     False        False
        C2      reweigh  age_mature international    dir   0.896  0.891 -0.005 -0.046  0.035      True        False
        C2      reweigh      gender        gender    dir   0.545  0.562  0.018  0.011  0.026     False         True
        C2      reweigh      gender    age_mature    dir   0.398  0.396 -0.002 -0.007  0.004      True        False
        C2      reweigh      gender   scholarship    dir   0.263  0.271  0.008  0.001  0.015     False         True
        C2      reweigh      gender special_needs    dir   0.933  0.928 -0.006 -0.069  0.049      True        False
        C2      reweigh      gender international    dir   0.896  0.894 -0.005 -0.044  0.027      True        False
        C2      reweigh       joint        gender    dir   0.545  0.565  0.020  0.012  0.030     False         True
        C2      reweigh       joint    age_mature    dir   0.398  0.401  0.003 -0.003  0.010     False        False
        C2      reweigh       joint   scholarship    dir   0.263  0.317  0.054  0.041  0.065     False         True
        C2      reweigh       joint special_needs    dir   0.933  0.913 -0.008 -0.080  0.067      True        False
        C2      reweigh       joint international    dir   0.896  0.900 -0.010 -0.062  0.034      True        False
        C2      reweigh scholarship        gender    dir   0.545  0.555  0.010  0.001  0.018     False         True
        C2      reweigh scholarship    age_mature    dir   0.398  0.405  0.007  0.002  0.013     False         True
        C2      reweigh scholarship   scholarship    dir   0.263  0.318  0.055  0.043  0.068     False         True
        C2      reweigh scholarship special_needs    dir   0.933  0.941  0.001 -0.065  0.073     False        False
        C2      reweigh scholarship international    dir   0.896  0.884 -0.015 -0.060  0.030      True        False
        C2 threshold_eo  age_mature        gender    dir   0.545  0.595  0.051  0.031  0.071     False         True
        C2 threshold_eo  age_mature    age_mature    dir   0.398  0.638  0.240  0.224  0.260     False         True
        C2 threshold_eo  age_mature   scholarship    dir   0.263  0.350  0.088  0.068  0.107     False         True
        C2 threshold_eo  age_mature special_needs    dir   0.933  0.903  0.011 -0.123  0.176     False        False
        C2 threshold_eo  age_mature international    dir   0.896  0.885 -0.007 -0.069  0.060      True        False
        C2 threshold_eo      gender        gender    dir   0.545  0.709  0.165  0.148  0.183     False         True
        C2 threshold_eo      gender    age_mature    dir   0.398  0.452  0.054  0.043  0.065     False         True
        C2 threshold_eo      gender   scholarship    dir   0.263  0.317  0.053  0.037  0.070     False         True
        C2 threshold_eo      gender special_needs    dir   0.933  0.941  0.013 -0.097  0.144     False        False
        C2 threshold_eo      gender international    dir   0.896  0.848 -0.034 -0.101  0.054      True        False
        C2 threshold_eo       joint        gender    dir   0.545  0.719  0.175  0.154  0.198     False         True
        C2 threshold_eo       joint    age_mature    dir   0.398  0.479  0.082  0.067  0.097     False         True
        C2 threshold_eo       joint   scholarship    dir   0.263  0.558  0.295  0.263  0.326     False         True
        C2 threshold_eo       joint special_needs    dir   0.933  0.904 -0.016 -0.137  0.149      True        False
        C2 threshold_eo       joint international    dir   0.896  0.863 -0.026 -0.106  0.056      True        False
        C2 threshold_eo scholarship        gender    dir   0.545  0.594  0.049  0.032  0.065     False         True
        C2 threshold_eo scholarship    age_mature    dir   0.398  0.457  0.059  0.047  0.071     False         True
        C2 threshold_eo scholarship   scholarship    dir   0.263  0.529  0.266  0.237  0.296     False         True
        C2 threshold_eo scholarship special_needs    dir   0.933  0.903 -0.005 -0.122  0.167      True        False
        C2 threshold_eo scholarship international    dir   0.896  0.873 -0.015 -0.082  0.056      True        False
```

## Intersectional DIR (baseline)
```
checkpoint   method                          pair  dir_pair  dir_a  dir_b  reliable_cells  repeats_worse_than_both  repeats
        C0 baseline    age_mature x international     0.299  0.319  0.818               4                        5        5
        C0 baseline      age_mature x scholarship     0.078  0.319  0.170               4                        5        5
        C0 baseline    age_mature x special_needs     0.292  0.319  0.707               3                        5        5
        C0 baseline           gender x age_mature     0.187  0.461  0.319               4                        5        5
        C0 baseline        gender x international     0.436  0.461  0.818               4                        5        5
        C0 baseline          gender x scholarship     0.096  0.461  0.170               4                        5        5
        C0 baseline        gender x special_needs     0.296  0.461  0.707               3                        5        5
        C0 baseline   scholarship x international     0.150  0.170  0.818               3                        5        5
        C0 baseline   scholarship x special_needs     0.167  0.170  0.707               3                        5        5
        C0 baseline special_needs x international     0.571  0.707  0.818               3                        5        5
        C1 baseline    age_mature x international     0.363  0.384  0.827               4                        5        5
        C1 baseline      age_mature x scholarship     0.121  0.384  0.240               4                        5        5
        C1 baseline    age_mature x special_needs     0.382  0.384  0.897               3                        5        5
        C1 baseline           gender x age_mature     0.264  0.550  0.384               4                        5        5
        C1 baseline        gender x international     0.520  0.550  0.827               4                        5        5
        C1 baseline          gender x scholarship     0.148  0.550  0.240               4                        5        5
        C1 baseline        gender x special_needs     0.503  0.550  0.897               3                        4        5
        C1 baseline   scholarship x international     0.213  0.240  0.827               3                        5        5
        C1 baseline   scholarship x special_needs     0.235  0.240  0.897               3                        5        5
        C1 baseline special_needs x international     0.761  0.897  0.827               3                        4        5
        C2 baseline    age_mature x international     0.394  0.398  0.896               4                        5        5
        C2 baseline      age_mature x scholarship     0.135  0.398  0.263               4                        5        5
        C2 baseline    age_mature x special_needs     0.396  0.398  0.933               3                        5        5
        C2 baseline           gender x age_mature     0.270  0.545  0.398               4                        5        5
        C2 baseline        gender x international     0.527  0.545  0.896               4                        5        5
        C2 baseline          gender x scholarship     0.164  0.545  0.263               4                        5        5
        C2 baseline        gender x special_needs     0.523  0.545  0.933               3                        5        5
        C2 baseline   scholarship x international     0.252  0.263  0.896               3                        5        5
        C2 baseline   scholarship x special_needs     0.247  0.263  0.933               3                        5        5
        C2 baseline special_needs x international     0.867  0.933  0.896               3                        5        5
```
