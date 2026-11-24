# UCI-697 external validation report

## Macro-F1 (3-class), mean over folds
```
checkpoint          C0     C1     C2
model    filter                     
LightGBM none    0.522  0.648  0.692
```

## Side effects on DIR (C2)
```
checkpoint   mitigation      target       audited metric  before  after  delta  ci_lo  ci_hi  worsened  significant
        C2      reweigh  age_mature        gender    dir   0.531  0.529 -0.003 -0.013  0.009      True        False
        C2      reweigh  age_mature    age_mature    dir   0.393  0.467  0.074  0.062  0.086     False         True
        C2      reweigh  age_mature   scholarship    dir   0.251  0.269  0.019  0.009  0.031     False         True
        C2      reweigh  age_mature special_needs    dir   0.923  0.936  0.006 -0.045  0.077     False        False
        C2      reweigh  age_mature international    dir   0.974  0.902 -0.031 -0.122  0.075      True        False
        C2      reweigh      gender        gender    dir   0.531  0.542  0.012  0.001  0.022     False         True
        C2      reweigh      gender    age_mature    dir   0.393  0.392 -0.002 -0.010  0.006      True        False
        C2      reweigh      gender   scholarship    dir   0.251  0.249 -0.002 -0.012  0.008      True        False
        C2      reweigh      gender special_needs    dir   0.923  0.923  0.003 -0.068  0.079     False        False
        C2      reweigh      gender international    dir   0.974  0.944 -0.007 -0.057  0.055      True        False
        C2      reweigh       joint        gender    dir   0.531  0.546  0.015  0.005  0.026     False         True
        C2      reweigh       joint    age_mature    dir   0.393  0.394  0.001 -0.009  0.010     False        False
        C2      reweigh       joint   scholarship    dir   0.251  0.323  0.073  0.055  0.092     False         True
        C2      reweigh       joint special_needs    dir   0.923  0.930  0.001 -0.094  0.096     False        False
        C2      reweigh       joint international    dir   0.974  0.875 -0.047 -0.144  0.076      True        False
        C2      reweigh scholarship        gender    dir   0.531  0.525 -0.006 -0.015  0.005      True        False
        C2      reweigh scholarship    age_mature    dir   0.393  0.399  0.006 -0.002  0.015     False        False
        C2      reweigh scholarship   scholarship    dir   0.251  0.310  0.059  0.040  0.075     False         True
        C2      reweigh scholarship special_needs    dir   0.923  0.928  0.003 -0.070  0.076     False        False
        C2      reweigh scholarship international    dir   0.974  0.918 -0.020 -0.098  0.045      True        False
        C2 threshold_eo  age_mature        gender    dir   0.531  0.593  0.061  0.041  0.079     False         True
        C2 threshold_eo  age_mature    age_mature    dir   0.393  0.619  0.226  0.207  0.248     False         True
        C2 threshold_eo  age_mature   scholarship    dir   0.251  0.331  0.080  0.056  0.102     False         True
        C2 threshold_eo  age_mature special_needs    dir   0.923  0.917  0.014 -0.086  0.172     False        False
        C2 threshold_eo  age_mature international    dir   0.974  0.935  0.004 -0.087  0.094     False        False
        C2 threshold_eo      gender        gender    dir   0.531  0.722  0.192  0.174  0.213     False         True
        C2 threshold_eo      gender    age_mature    dir   0.393  0.444  0.050  0.039  0.065     False         True
        C2 threshold_eo      gender   scholarship    dir   0.251  0.302  0.051  0.034  0.070     False         True
        C2 threshold_eo      gender special_needs    dir   0.923  0.936  0.016 -0.095  0.163     False        False
        C2 threshold_eo      gender international    dir   0.974  0.903 -0.018 -0.101  0.106      True        False
        C2 threshold_eo       joint        gender    dir   0.531  0.781  0.251  0.222  0.279     False         True
        C2 threshold_eo       joint    age_mature    dir   0.393  0.561  0.167  0.147  0.188     False         True
        C2 threshold_eo       joint   scholarship    dir   0.251  0.646  0.397  0.362  0.441     False         True
        C2 threshold_eo       joint special_needs    dir   0.923  0.856  0.006 -0.152  0.229     False        False
        C2 threshold_eo       joint international    dir   0.974  0.948  0.019 -0.080  0.173     False        False
        C2 threshold_eo scholarship        gender    dir   0.531  0.615  0.086  0.065  0.104     False         True
        C2 threshold_eo scholarship    age_mature    dir   0.393  0.512  0.119  0.103  0.135     False         True
        C2 threshold_eo scholarship   scholarship    dir   0.251  0.607  0.358  0.318  0.400     False         True
        C2 threshold_eo scholarship special_needs    dir   0.923  0.953  0.021 -0.129  0.186     False        False
        C2 threshold_eo scholarship international    dir   0.974  0.984  0.023 -0.067  0.166     False        False
```

## Intersectional DIR (baseline)
```
checkpoint   method                          pair  dir_pair  dir_a  dir_b  reliable_cells  repeats_worse_than_both  repeats
        C0 baseline    age_mature x international     0.274  0.292  0.947               4                        3        3
        C0 baseline      age_mature x scholarship     0.042  0.292  0.116               4                        3        3
        C0 baseline    age_mature x special_needs     0.200  0.292  0.624               3                        3        3
        C0 baseline           gender x age_mature     0.171  0.462  0.292               4                        3        3
        C0 baseline        gender x international     0.457  0.462  0.947               4                        3        3
        C0 baseline          gender x scholarship     0.056  0.462  0.116               4                        3        3
        C0 baseline        gender x special_needs     0.339  0.462  0.624               3                        3        3
        C0 baseline   scholarship x international     0.116  0.116  0.947               3                        2        3
        C0 baseline   scholarship x special_needs     0.112  0.116  0.624               3                        3        3
        C0 baseline special_needs x international     0.608  0.624  0.947               3                        2        3
        C1 baseline    age_mature x international     0.338  0.382  0.902               4                        3        3
        C1 baseline      age_mature x scholarship     0.100  0.382  0.207               4                        3        3
        C1 baseline    age_mature x special_needs     0.374  0.382  0.825               3                        3        3
        C1 baseline           gender x age_mature     0.257  0.537  0.382               4                        3        3
        C1 baseline        gender x international     0.534  0.537  0.902               4                        3        3
        C1 baseline          gender x scholarship     0.120  0.537  0.207               4                        3        3
        C1 baseline        gender x special_needs     0.518  0.537  0.825               3                        3        3
        C1 baseline   scholarship x international     0.207  0.207  0.902               3                        1        3
        C1 baseline   scholarship x special_needs     0.183  0.207  0.825               3                        3        3
        C1 baseline special_needs x international     0.778  0.825  0.902               3                        2        3
        C2 baseline    age_mature x international     0.364  0.393  0.974               4                        3        3
        C2 baseline      age_mature x scholarship     0.128  0.393  0.251               4                        3        3
        C2 baseline    age_mature x special_needs     0.392  0.393  0.923               3                        3        3
        C2 baseline           gender x age_mature     0.261  0.531  0.393               4                        3        3
        C2 baseline        gender x international     0.527  0.531  0.974               4                        3        3
        C2 baseline          gender x scholarship     0.147  0.531  0.251               4                        3        3
        C2 baseline        gender x special_needs     0.529  0.531  0.923               3                        3        3
        C2 baseline   scholarship x international     0.251  0.251  0.974               3                        1        3
        C2 baseline   scholarship x special_needs     0.221  0.251  0.923               3                        3        3
        C2 baseline special_needs x international     0.898  0.923  0.974               3                        2        3
```
