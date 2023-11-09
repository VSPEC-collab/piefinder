"""
JAX Configurations
------------------
"""
from astropy import units as u


wl_unit:u.Unit = u.um
"""
The default wavelength unit.

:type: astropy.units.Unit
"""
time_unit:u.Unit = u.day
"""
The default time unit.

:type: astropy.units.Unit
"""
flux_unit:u.Unit = u.Unit('W m-2 s-1 um-1')
"""
The default flux unit.

:type: astropy.units.Unit
"""
