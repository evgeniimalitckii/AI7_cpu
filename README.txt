
Use requirements.txt file to create the environment to launch the web-application.

############## Model description ################

HSPmodel_ff_43.h5 file contains the deep learning model designed to predict the hydrogen sensitivity parameter of steels.

The input shape of the Model is (17, ). Input data consists of five Gaussian distribution parameters that can fit the original hydrogen thermal desorption spectra properly, forming a descriptor of TDS data.
The Gaussian parameters must be ordered as 1 - amplitude of the Gaussian peak; 2 - temperature position (Kelvin) of the Gaussian peak; 3 - width of the Gaussian peak. This order of Gaussian parameters should repeat for each Gaussian peak from lower to higher temperature peak position, successively.
The last two values with an index of 16 and 17 must be filed with zeros.

Before feeding the obtained array to the model the normalization is needed. Divide your array of data (cons_descriptor) by normalization array as follow:

# normalization
    norm = numpy.array([[3.12280000e-01, 5.98813002e+03, 7.40635820e+02, 1.80230000e-01,
            6.93425373e+03, 6.21512470e+02, 7.37760000e-01, 7.57653689e+03,
            7.10082360e+02, 5.70730000e-01, 9.36837481e+03, 1.06340591e+03,
            7.99100000e-02, 1.07656541e+04, 2.58237395e+03, 1.00000000e+00,
            1.00000000e+00]])
    cons = numpy.divide(cons_descriptor, norm)

The example file containing the Gaussian parameters and appropriate for the analysis is available in this folder.

############## Running the application with the model deployed #############

There are two files attached that can be used to run the web-application at your localhost.
- HSP_app_v1.ipynb
- HSP_server_v1.ipynb

Web-application can be launched from Jupyter Notebook running the HSP_app_v1.ipynb.
Other option to launch the web-application is through the terminal or anaconda prompt typing the following:

bokeh serve --show HSP_server_v1.py
