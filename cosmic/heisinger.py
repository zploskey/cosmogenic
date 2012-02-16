from numpy import *
from matplotlib.pyplot import *

import muon

fig1 = figure(1)
subplot(2, 2, 1)
h = exp(arange(0, log(1e5), log(1e5) / 250.0)) # hg cm-2
z = h * 100.0 # convert to g cm-2 from hg cm-2
phi_v = muon.phi_vert_slhl(z)
loglog(h, phi_v)
xlim((1, 1e5))
ylim((1e-14, 1e-1))
xlabel('lithospheric depth h [hg/cm2]')
ylabel('$\Phi_\mathrm{v}$(h) [cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')

subplot(2, 2, 2)
h = arange(1, 1e4, 1)
z = h * 100.0
ebar = muon.ebar(z)
loglog(h, ebar)
xlim((1, 1e4))
ylim((1, 1e3))
ylabel('Mean E(h)')
xlabel('lithospheric depth h [hg/cm2]')

subplot(2, 2, 3)
z = arange(0, 100000, 1) # g / cm2
beta = muon.beta(z)
h = z / 100.0 # convert to hg / cm2 ... Heisinger paper has typo
semilogx(h, beta) 
xlim((1, 1000))
xlabel('lithospheric depth [hg/cm2]')
ylim((0.82, 0.96))
ylabel('Beta')

show()

