ALPHA = 0.75

class Nuclide:
    pass

class Be10Qtz(Nuclide):
    def __init__(self):
        self.Natoms = 2.005e22 # atoms of O / g quartz, from John's program
        
        # probability factors from Heisinger et al. 2002b
        # for Be-10 production in O in quartz:
        self.fC = 0.704
        self.fD = 0.1828
        self.fstar = 0.0043
        
        # be10stopped_mu_yield = fC10 * fD10 * fstar10
        # superseded value for be10 yield, is ~ 0.000553
        self.k_neg = (fC * fD * fstar) / 1.096 
        # 1.096 factor is normalization to 07KNSTD from Balco's AlBe_changes_v221
        self.sigma190 = 8.6e-29 # from Balco AlBe_changes_v221
        self.sigma0 = sigma190 / 190 ** ALPHA
    
        self.fC = fc
    
    
class Al26Qtz(Nuclide):
    def __init__(self):
        self.Natoms = 1.0025e22
        
        self.k_neg = 0.296 * 0.6559 * 0.022
        
        self.sigma190 = 1.41e-27
        self.sigma0 = sigma190 / 190 ** ALPHA