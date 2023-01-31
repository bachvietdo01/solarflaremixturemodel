<!-- ABOUT THE PROJECT -->
## About Solar Mixture Model Project

The physics of solar flares on the surface of the Sun is highly complex and not yet fully understood. However, observations show that solar eruptions are associated with the intense kilogauss fields of active regions (ARs), where free energies are stored with field-aligned electric currents. With the advent of high-quality data sources such as the Geostationary Operational Environmental Satellites (GOES) and Solar Dynamics Observatory (SDO)/Helioseismic and Magnetic Imager (HMI), recent works on solar flare forecasting have been focusing on data driven methods. In particular, black box machine learning and deep learning models are increasingly being adopted in which underlying data structures are not modeled explicitly. If the active regions indeed follow the same laws of physics, there should be similar patterns shared among them, reflected by the observations. Yet, these black box models currently used in the space weather literature do not explicitly characterize the heterogeneous nature of the solar flare data, within and between active regions. In this paper, we propose two finite mixture models designed to capture the heterogeneous patterns of active regions and their associated solar flare events. With extensive numerical studies, we demonstrate the usefulness of our proposed method for both resolving the sample imbalance issue and modeling the heterogeneity for solar flare events, which are strong and rare. This is the GitHub repo for the codes to reproduce the metrics presented in the paper.


### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```sh
   git clone https://github.com/bachvietdo01/solarflaremixturemodel.git
   ```
2. Install scikit learn packages
   ```sh
   pip install sk-learn
   ```
3. Install numpy and pandas package
   ```
   pip install numpy
   pip install pandas
   ```


<!-- USAGE EXAMPLES -->
## Usage

In the source directory, there are a few important soruce files.

* **SolarFlareMM2REM.py** is the source codes mixture model MM-2R where the heterogeneity pattern is shared among Active Regions. 
* **SolarFlareMM2AdEM.py** is the source codes mixture model MM-2H where the heterogeneity pattern is extended to locations within Active Regions.
* **trainutilities.py** contains many useful helper functions.
* **run_solar_flare_mm2RH_parallelv3_xxh.py** to run both models MM-2R and MM-2H and produce performance metrics.


To run the projects and reproduce metric for xx hours simply execute the following

```
python run_solar_flare_mm2RH_parallelv3_xxh.py
```




<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Bach Do - vietdo@umich.edu

Project Link: [https://github.com/bachvietdo01/solarflaremixturemodel](https://github.com/bachvietdo01/solarflaremixturemodel)




<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project is funded by U of M.
