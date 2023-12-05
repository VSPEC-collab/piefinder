from GridPolator.grid import GridSpectra
from astropy import units as u
import numpy as np
from astropy.io import fits


class StarShot:
    """
    StarShot object to model the snapshot of a star containing faculae and spots.

    Parameters
    ----------
    t_phot : astropy.units.Quantity
        Temperature of the star's atmosphere.
    t_fac : astropy.units.Quantity
        Temperature of the faculae.
    f_fac : float
        Fraction of the star's surface covered by faculae.
    t_spot : astropy.units.Quantity
        Temperature of the spots.
    f_spot : float
        Fraction of the star's surface covered by spots.

    Attributes
    ----------
    t_phot : astropy.units.Quantity
        Temperature of the star's atmosphere.
    f_phot : astropy.units.Quantity
        Fraction of the star's surface not covered by faculae or spots.
    t_fac : astropy.units.Quantity
        Effective temperature of the faculae.
    f_fac : float
        Fraction of the star's surface covered by faculae.
    t_spot : astropy.units.Quantity
        Effective temperature of the spots.
    f_spot : float
        Fraction of the star's surface covered by spots.
    """

    def __init__(self, t_phot: u. Quantity,
                t_fac: u. Quantity,
                f_fac: float,
                t_spot: u. Quantity,
                f_spot: float):
        
        if f_fac + f_spot > 1:
            raise ValueError("The sum of the fractions of the stellar surface covered by spots and faculae cannot be larger than 1.")
        
        self.t_phot = t_phot
        self.f_phot = 1 - f_fac - f_spot
        self.t_fac = t_fac
        self.f_fac = f_fac
        self.t_spot = t_spot
        self.f_spot = f_spot
        self.grid = None
    
    def genPhxGrid(self):
        """
        Generate a grid of PHOENIX models from https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix/

        Parameters
        ----------

        Returns
        -------
        wl : astropy.units.Quantity
            The wavelength axis of the model.
        fl : astropy.units.Quantity
            The flux values of the model.
        """
        
        phoeModel = fits.open('phoenix/phoenixm00/phoenixm00_2000.fits')
        phoeDat = phoeModel[1].data
        n = len(phoeDat)
        wl: u.Quantity = np.array([phoeDat[i][0] for i in range(n)]) * u.Angstrom
        phoeModel.close()

        params = [
            # tuple(np.linspace(2000, 70000, 341, dtype = int)), # teff
            tuple(np.linspace(2000, 2400, 3, dtype = int)), # teff
            tuple(np.linspace(0, 60, 13, dtype = int))   # log_g
        ]

        fl = []
        for teff in np.linspace(2000, 2400, 3, dtype = int):
            phoeModel = fits.open(f'phoenix/phoenixm00/phoenixm00_{teff}.fits')
            phoeDat = phoeModel[1].data
            n = len(phoeDat)
            flt = []
            for j in range(13): 
                flt += [list(phoeDat[i][j+1] for i in range(n))]
            fl += [flt]
            phoeModel.close()

        self.wl = wl
        self.grid = GridSpectra(wl, fl, *params)

        return wl, fl
    
    def genSpec(self, *args)->np.ndarray:
        """
            Generates a stellar spectrum based on the star temperatures and their fractions
        """
        spec = self.grid.evaluate(self.wl, *args)
        return spec

x = StarShot(5000, 5800, 0.2, 3800, 0.1)
x.genPhxGrid()
print(x.genSpec([2000, 0]))