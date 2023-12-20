### Water molecule from CrawfordGroup/Programming Projects
h2o = """
O  0.000000000000  -0.143225816552   0.000000000000
H  1.638036840407   1.136548822547  -0.000000000000
H -1.638036840407   1.136548822547  -0.000000000000
symmetry c1
no_com
no_reorient
units bohr
"""

# Chiral molecules
# H2 dimer
H2_2 = """
H
H 1 0.75
H 2 1.5 1 90.0
H 3 0.75 2 90.0 1 60.0
symmetry c1
"""

# Hydrogen peroxide
H2O2 = """
    O       0.0000000000        1.3192641900       -0.0952542913
    O      -0.0000000000       -1.3192641900       -0.0952542913
    H       1.6464858700        1.6841036400        0.7620343300
    H      -1.6464858700       -1.6841036400        0.7620343300
symmetry c1
units bohr
no_reorient
nocom
"""

# Ethylene oxide: Amos et al., Chem. Phys. Lett. 133, 21-26 (1987)
etho = """
O       0.0000000000        0.0000000000        1.6119363900
C       0.0000000000       -1.3813890400       -0.7062143040
C       0.0000000000        1.3813905700       -0.7062514120
H      -1.7489765900       -2.3794725300       -1.1019539000
H       1.7489765900       -2.3794725300       -1.1019539000
H       1.7489765900        2.3794634300       -1.1020178200
H      -1.7489765900        2.3794634300       -1.1020178200
symmetry c1
no_reorient
no_com
units bohr
"""

# (S)-1,3-dimethylallene
sdma = """
C  0.000000  0.000000  0.414669
C  0.000000  1.309935  0.412904
C  0.000000 -1.309935  0.412904
H -0.677576  1.833097  1.091213
H  0.677576 -1.833097  1.091213
C  0.875978  2.177784 -0.460206
C -0.875978 -2.177784 -0.460206
H  1.519837  1.571995 -1.104163
H -1.519837 -1.571995 -1.104163
H  0.267559  2.832375 -1.098077
H -0.267559 -2.832375 -1.098077
H  1.514280  2.829499  0.150828
H -1.514280 -2.829499  0.150828
symmetry c1
"""

# (S)-2-chloropropionitrile
s2cpn = """
C  -0.074850  0.536809  0.165803
C  -0.131322  0.463480  1.692375
H   0.576771  1.182613  2.118184
C   1.260194  0.235581 -0.354931
N   2.332284  0.017558 -0.742030
CL -1.285147 -0.612976 -0.578450
H  -0.370843  1.528058 -0.186624
H  -1.138978  0.710124  2.035672
H   0.130424 -0.538322  2.041140
symmetry c1
"""

# (R)-methylthiirane
rmt = """
C  0.364937  1.156173 -0.153047
C -0.521930  0.166108  0.502066
C -1.785526 -0.309717 -0.178180
H -1.645392 -0.376658 -1.261546
S  1.101404 -0.526707 -0.054082
H  0.106264  1.485766 -1.157365
H  0.850994  1.912823  0.458508
H -0.587772  0.237291  1.586932
H -2.083855 -1.297173  0.188561
H -2.607593  0.389884  0.025191
symmetry c1
"""

moldict = {}
moldict["H2O"] = h2o
moldict["(H2)_2"] = H2_2
moldict["H2O2"] = H2O2
moldict["Ethylene Oxide"] = etho
moldict["(S)-dimethylallene"] = sdma
moldict["(S)-2-chloropropionitrile"] = s2cpn
moldict["(R)-methylthiirane"] = rmt