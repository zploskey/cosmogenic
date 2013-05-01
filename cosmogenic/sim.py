import numpy as np

import production
import muon

import scipy.integrate

def multiglaciate(dz, t_gl, t_intergl, t_postgl, z, n, p, n_gl=None,
                  postgl_shielding=0):
    """Find the resulting concentration profile for a glacial history and site.
    
    This function predicts the concentration profile for a glacial history. The
    glacial history of the site is described in such a way that the parameters
    are easy to vary for the Monte Carlo simulation--i.e. the times of
    glacial and interglacial periods are in lengths rather than absolute ages.
    Depths of the sample and the depths eroded during each glaciation are both
    in units of g/cm2, avoiding tying our results to a rock density. Presumably 
    this could be adapted to handle variable densities but that might get
    really complicated.
    
    Parameters:
    z: vector of samples depths beneath the modern surface (g/cm2)
    t_gl: vector of lengths of time spent ice covered in each glaciation (yr)
    t_intergl: vector, time spent exposed for each interglaciation  period (yr)
    dz: vector of the depths eroded during each glaciation (g/cm2)
    t_postgl: time the sample has been exposed since deglaciation (yr)
    n: nuclide object
    p: production rate function p(z)
    
    Optional Parameters:
    n_gl: If supplied, this is the number of glaciations to simulate
    assuming that t_gl and t_intergl are scalars, not vectors.
    """
    if n_gl is not None:
        ngl = n_gl
        t_gl = np.ones(ngl) * t_gl
        t_intergl = np.ones(ngl) * t_intergl
        dz = np.atleast_1d(dz)
        if dz.size == 1:
            dz = np.ones(ngl) * dz
    else:
        ngl = dz.size
    
    # add the atoms created as we go back in time
    # recent interglacial first
    conc = simple_expose(z + postgl_shielding, t_postgl, n, p) 
    z_cur = np.atleast_1d(z).copy()    # start at current depths
    t_begint = t_postgl # the time when the current interglacial began
    t_endint = 0.0      # time (now) when current interglacial ended
    for i in range(ngl):
        z_cur += dz[i] # go back to depth and time before glacial erosion
        t_endint = t_begint + t_gl[i]
        t_begint = t_endint + t_intergl[i]
        conc += expose(z_cur, t_begint, t_endint, n, p)
    return conc

def depth_v_time(gl, intergl, postgl, dz, n_gl=None):
    """ Returns a tuple of times and depths of a surface sample.
    
    gl: vector of lengths of each glaciation (yr)
    intergl: vector of lengths of interglacial periods (yr)
    postgl: time since last deglaciation (yr)
    dz: vector of glacial erosion depths during each glaciation 
    """
    if n_gl != None:
        if isinstance(gl, (int, long, float)):
            gl = np.ones(n_gl) * gl
            intergl = np.ones(n_gl) * intergl
            dz = np.ones(n_gl) * dz
    assert gl.size == intergl.size == dz.size
    # interleave the two arrays
    tmp = np.column_stack((gl, intergl)).reshape(1, gl.size * 2).flatten()
    t = np.add.accumulate(np.concatenate(([0, postgl], tmp)))
    tmp = np.column_stack((dz, np.zeros(dz.size))).reshape(1, dz.size * 2).flatten()
    z = np.add.accumulate(np.concatenate(([0, 0], tmp)))
    return (t, z)

def expose(z, t_init, t_final, n, p):
    L = n.LAMBDA
    conc = (p(z) / L) * (np.exp(-L * t_final) - np.exp(-L * t_init))
    return conc

def expose_from_site_data(z, t_init, t_final, n, h, lat):
    p = production.P_tot(z, h, lat, n)
    return (expose(z, t_init, t_final, n, p), p)

def simple_expose_slow(z, t_exp, n, h, lat):
    # calculate the production rate
    p = production.P_tot(z, h, lat, n)
    return (p / n.LAMBDA) * (1 - np.exp(-n.LAMBDA * t_exp))

def simple_expose(z, t_exp, n, p):
    return (p(z) / n.LAMBDA) * (1 - np.exp(-n.LAMBDA * t_exp))

def fwd_profile(z0, z_removed, t, n, p):
    """
    Calculates the nuclide concentration profile resulting from repeated
    glaciation of a bedrock surface.

    In all parameters that reference time, time is zero starting at modern day
    and increases into the past.

    z0: modern depths at which we want predicted concentrations (g/cm2)
    z_removed: list of depths of rock removed in successive glaciations (g/cm2)
    t: ages of switching between glacial/interglacial (array of times in years)
    exposed to cosmic rays in the recent past (in years). The first element of
    this array should be the exposure time since deglaciation, increasing after.
    n: nuclide object    
    p: production rate function of depth in g/cm2
    """
    L = n.LAMBDA # decay constant
    N = np.zeros(z0.size) # nuclide concentration
    t_beg = t[2::2]
    t_end = t[1::2]

    # Add nuclides formed postglacially
    N += simple_expose(z0, t[0], n, p)
    
    z_cur = z0.copy()
    for i, dz in enumerate(z_removed):
        z_cur += dz
        N += (p(z_cur) / L) * (np.exp(-L * t_end[i]) - np.exp(-L * t_beg[i]))
        
    return N

def fwd_profile_slow(z0, z_removed, t, n, h, lat):
    """
    Calculates the nuclide concentration profile resulting from repeated
    glaciation of a bedrock surface.

    In all parameters that reference time, time is zero starting at modern day
    and increases into the past.

    z0: modern depths at which we want predicted concentrations (g/cm2)
    z_removed: list of depths of rock removed in successive glaciations (g/cm2)
    t: ages of switching between glacial/interglacial (array of times in years)
    exposed to cosmic rays in the recent past (in years). The first element of
    this array should be the exposure time since deglaciation, increasing after.
    n: the nuclide being produced (nuclide object)
    h: elevation of the site (m)
    lat: latitude of the site (degrees) 
    """
    L = n.LAMBDA
    N = np.zeros(z0.size)
    t_beg = t[2::2]
    t_end = t[1::2]

    # Add nuclides formed postglacially
    N += simple_expose_slow(z0, t[0], n, h, lat)
    
    z_cur = z0.copy()
    for i, dz in enumerate(z_removed):
        z_cur += dz
        p = production.P_tot(z_cur, h, lat, n)
        N += (p / L) * (np.exp(-L * t_end[i]) - np.exp(-L * t_beg[i]))
        
    return N

#def fwd_profile_steps(z0, z_removed, t, n, h, lat):
#    """
#    The profile at end of each interglacial period.
#    """
#    L = n.LAMBDA
#    N = np.zeros(len(z0))
#    t_beg = t[2::2]
#    t_end = t[1::2]
#    Ns = np.zeros( (len(t_end) + 1, len(z0)) )
#    z_cur = z0.copy() + z_removed.sum()
#    idepth = np.add.accumulate(z_removed)
#    npts = 200
#    
#    for i in range(1, z_removed.size):
#        # begin interglacial period
#        depths = np.linspace(0, z0[-1] + idepth[-i], npts)
#        N *= np.exp(-L * tg) # radioactive decay
#        Nexp = sim.simple_expose(depths, ti, be10, p)
#        N += Nexp
#        # fnc = sp.interpolate.interp1d(depths, N)
#        
#        depthsm = depths / rho / 100.0
#        ln = plt.semilogx(N, depthsm, lw=2, color='lightslategray')
#    #    leg.append('Ig ' + str(i))
#        lines = np.append(lines, ln)
#
#        # begin glaciation
#        N *= np.exp(-be10.LAMBDA * tg) # isotopic decay
#        # erode the top of the profile away
#        depths -= z_removed[-i]
#        nofz = interpolate.interp1d(depths, N) # interpolate conc. curve
#        depths = np.linspace(0, depths[-1], 500)
#        N = nofz(depths)
#
#        depthsm = depths / rho / 100.0
#        ln = plt.semilogx(N, depthsm, color='lightslategray', lw=2)
#        lines = np.append(lines, ln)
#    #    leg.append('Gl ' + str(i))
#
#        # account for recent cosmic ray exposure
#        Nhol = sim.simple_expose(depths, tH, be10, h, lat) 
#        N *= np.exp(-be10.LAMBDA * tH)
#        N += Nhol
#        ln = plt.semilogx(N, depthsm, color='r', lw=2)
#        lines = np.append(lines, ln)
#    
#    # Add nuclides formed postglacially
#    for i, dz in enumerate(z_removed):
#        ind = -(i + 1)
#        dt = t_beg[ind] - t_end[ind]
#        # dt += t_beg[ind - 1]
#        buildup = simple_expose_slow(z0, dt, n, h, lat)
#        Ndec = N * np.exp(-L * dt)
#        Ns[i] = N
#        z_cur -= dz
#        p = production.P_tot(z_cur, h, lat, n)
#        buildup =  (p / L) * (np.exp(-L * t_end[i]) - np.exp(-L * t_beg[i]))
#        N += buildup
#        Ns[-i + 1, :] = N.copy().T
#    Nhol = simple_expose_slow(z0, t[0], n, h, lat)
#    N += Nhol
#
#    Ns[0, :] = N + Nhol
#    return Ns

def steady_multiglaciate(model, constraints, bottom_depth_error, z_vs_t=False):
    """
    Model should have: [erosion_rate, t_gl, t_int]
    in units g / cm^2 / yr, yr and yr respectively
    This is currently unused and might take some cleanup
    *UNTESTED*!
    """
    con = constraints
    eros, t_gl, t_int = tuple(model.tolist())
    t_cycle = t_gl + t_int
    eros_depth = con['eros_rate'] * t_cycle
    z_cur = con['sample_depths'].copy()
    bottom_idx = np.argmin(z_cur)
    t_beg = con['t_postgl']
    t_end = 0
    
    conc = np.zeros(z_cur.size)
    while True:
        p = production.P_tot(z_cur, con['sample_h'], con['lat'],
                             con['nuclide'])
        added_conc = expose(z_cur, t_beg, t_end, con['nuclide'], 
                            con['sample_h'], con['lat'])
        conc += added_conc

        if added_conc[bottom_idx] < bottom_depth_error:
            break
        t_beg += t_cycle
        t_end += t_cycle
        dz += eros_depth
    
#    if z_vs_t:
#        return (conc_true, t_true, z_true)
#    else:
    return conc

def rand_erosion_hist(avg, sigma, n):
    """
    Generates a sequence of n numbers randomly distributed about avg_dz and
    standard deviation approximately equal to sigma.
    """
    return np.random.normal(avg, sigma, n)

def steady_erosion(P, z0, eros, nuc, T, T_stop=0):
    
    def int_eq(t):
        return P(z(t)) * np.exp(-nuc.LAMBDA * t)

    z0 = np.atleast_1d(z0)
    N = np.zeros_like(z0)
    for i, depth in enumerate(z0):
        z = lambda t: eros * t + depth
        res, _ = scipy.integrate.quad(int_eq, T_stop, T)
        N[i] = res
    
    return N
    
