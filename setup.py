from setuptools import setup, find_packages
from os.path import exists


setup(name='trendfit',
      #version=versioneer.get_version(),
      #cmdclass=versioneer.get_cmdclass(),
      description=('Trend analysis and fitting for (atmospheric) time series'),
      url='https://github.com/girpas-ulg/trendfit',
      maintainer='Benoit Bovy',
      maintainer_email='benbovy@gmail.com',
      license='BSD-Clause3',
      keywords='python time-series trend fitting analysis atmosphere',
      packages=find_packages(),
      long_description=(open('README.md').read() if exists('README.md')
                        else ''),
      python_requires='>=3.5',
      install_requires=['numpy'],
      # tests_require=['pytest >= 3.3.0'],
      zip_safe=False)
