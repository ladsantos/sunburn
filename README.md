# sunburn
Data analysis of HST/COS far-UV targeted towards exoplanet transit spectroscopy.

Dependencies
------------

* `numpy` >= 1.12
* `scipy` >= 0.19
* `matplotlib` >= 2.0
* `astropy` >= 2.0.2
* `astroquery` >= 0.3.7.dev4234
* `astroplan`

**Note**: The development version of `astroquery` is necessary because of a specific implementation of queries to the NASA Exoplanet Archive. In order to install this development version, you will have to [build it from source](http://astroquery.readthedocs.io/en/latest/#building-from-source). In the near future this may not be necessary anymore because `astroquery` will eventually consolidate the development version into the stable version.

Installation
------------
Clone the repository:

    git clone https://github.com/ladsantos/sunburn.git

Navigate to the source code and install it in your Python environment:

    cd sunburn
    python setup.py install