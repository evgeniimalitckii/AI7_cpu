#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bokeh.plotting import *
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.layouts import row, column, layout
from bokeh.models import Slider, ColumnDataSource, Paragraph, LinearAxis, Range1d, Legend, Paragraph
from bokeh.io import curdoc, output_notebook, output_file, show, push_notebook
import numpy as np
from bokeh.server.server import Server


# In[2]:


import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model('HSPmodel_ff_43.h5')


# In[3]:


# Set up data
# propose the imput parameters of the TDS spectra
cons_descriptor = np.array([[8.78787148e-03, 5.05248470e+02, 1.72279298e+01, 6.45007124e-03,
        5.28826992e+02, 2.71719037e+01, 4.41696912e-03, 6.50468422e+02,
        3.40690416e+01, 3.58418440e-03, 7.70951668e+02, 2.25282542e+01,
        4.25201110e-03, 8.14798528e+02, 1.88453904e+02, 0.00000000e+00,
        0.00000000e+00]])
# normalisation
norm = np.array([[3.12280000e-01, 5.98813002e+03, 7.40635820e+02, 1.80230000e-01,
        6.93425373e+03, 6.21512470e+02, 7.37760000e-01, 7.57653689e+03,
        7.10082360e+02, 5.70730000e-01, 9.36837481e+03, 1.06340591e+03,
        7.99100000e-02, 1.07656541e+04, 2.58237395e+03, 1.00000000e+00,
        1.00000000e+00]])
cons = np.divide(cons_descriptor, norm)

x_min = np.array([2.64001512e-03, 4.58097935e+02, 1.35084567e+01, 8.13918680e-04,
       4.72610997e+02, 9.78633535e+00, 2.83963824e-03, 6.48733395e+02,
       2.47541812e+01, 2.82225985e-03, 7.50819031e+02, 1.26715448e+01,
       1.22102480e-03, 7.50366089e+02, 2.32878483e+01])
x_max = np.array([3.12280000e-02, 5.98813002e+02, 7.40635820e+01, 1.80230000e-02,
       6.93425373e+02, 6.21512470e+01, 7.37760000e-02, 7.57653689e+02,
       7.10082360e+01, 5.70730000e-02, 9.36837481e+02, 1.06340591e+02,
       7.99100000e-03, 1.07656541e+03, 2.58237395e+02])

x = np.arange(308, 1070, 1)
mu, sigma = 0, 0.0005 # mean and standard deviation
rand = np.random.normal(mu, sigma, x.shape[0]) 
rand = np.absolute(rand)
gaus1 = cons_descriptor[0,0] * np.exp( - (x - cons_descriptor[0,1])**2 / (cons_descriptor[0,2]**2))
gaus2 = cons_descriptor[0,3] * np.exp( - (x - cons_descriptor[0,4])**2 / (cons_descriptor[0,5]**2))
gaus3 = cons_descriptor[0,6] * np.exp( - (x - cons_descriptor[0,7])**2 / (cons_descriptor[0,8]**2))
gaus4 = cons_descriptor[0,9] * np.exp( - (x - cons_descriptor[0,10])**2 / (cons_descriptor[0,11]**2))
gaus5 = cons_descriptor[0,12] * np.exp( - (x - cons_descriptor[0,13])**2 / (cons_descriptor[0,14]**2))
y = gaus1 + gaus2 + gaus3 + gaus4 + gaus5 + rand

HSP = model.predict(cons)
HSP_text = 'HSP = ' + str(format(HSP[0,0]*100, '.1f')) + '%'
HSP = np.array([0, HSP[0,0]*100])
y2 = np.array([1,1])

source = ColumnDataSource(data=dict(x=x, y=y, g1=gaus1, g2=gaus2, g3=gaus3, g4=gaus4, g5=gaus5))
source2 = ColumnDataSource(data=dict(hsp=HSP, y2=y2))

output_file("div.html")
t1 = Paragraph(text="""The application is designed to predict the hydrogen sensitivity parameter (HSP) of hydrogen embrittlement
 (HE) of steels. HSP defines the percent of reduction of elongation to fracture relating to the total elongation of the as-supplied
  steel specimen [1]. Follow the steps to test your spectra:""")
t2 = Paragraph(text="""1. Measure thermal desorption spectra of hydrogen release from as-supplied steel specimen with the heating rate of 10 K/min and
  a temperature range from room temperature to 1070 K;""")
t3 = Paragraph(text="""2. Fit the spectra with five Gaussian peaks and exponential background as shown on the graph;""")
t4 = Paragraph(text="""3. Provide the fitting parameters to the application by the change of the corresponding slider position;""")
t5 = Paragraph(text="""4. HSP-value is automatically predicted for the input data.""")
t6 = Paragraph(text="""[1] Malitckii, E., Fangnon, E. & Vilaça, P. Study of correlation between the steels susceptibility to hydrogen embrittlement 
  and hydrogen thermal desorption spectroscopy using artificial neural network. Neural Comput & Applic (2020). 
  https://doi.org/10.1007/s00521-020-04853-3 """)


# In[4]:


#Set up plot

left = figure(title = 'Termal Desorption Spectra', plot_height=370,
       x_axis_label = 'Temperature, K', y_axis_label = 'Hydrogen Desorption Rate, at.ppm / s')
left.line('x', 'y', source=source, legend_label='TDS', line_width=2, color='black', alpha=0.8)
left.varea('x', y1=0, y2='g1', source=source, legend_label='gaus1', color='blue', alpha=0.2)
left.varea('x', y1=0, y2='g2', source=source, legend_label='gaus2', color='orange', alpha=0.2)
left.varea('x', y1=0, y2='g3', source=source, legend_label='gaus3', color='green', alpha=0.2)
left.varea('x', y1=0, y2='g4', source=source, legend_label='gaus4', color='red', alpha=0.2)
left.varea('x', y1=0, y2='g5', source=source, legend_label='gaus5', color='gray', alpha=0.2)

right = figure(plot_height=100,
            x_axis_label = 'Hydrogen Sensitivity Parameter, %')
right.title.text = HSP_text

right.x_range = Range1d(start=0, end=100)
right.yaxis.visible = False
right.line('hsp', 'y2', source=source2, color='red', line_width=10, alpha=0.5)

# Set up widgets

peak1_amplitude = Slider(title='Gaussian 1, amplitude', value=cons_descriptor[0,0], start=x_min[0], end=x_max[0], step=0.001, bar_color='blue')
peak1_temperature = Slider(title='Gaussian 1, temperature', value=cons_descriptor[0,1], start=x_min[1], end=x_max[1], step=1, bar_color='blue')
peak1_width = Slider(title='Gaussian 1, width', value=cons_descriptor[0,2], start=x_min[2], end=x_max[2], step=1, bar_color='blue')

peak2_amplitude = Slider(title='Gaussian 2, amplitude', value=cons_descriptor[0,3], start=x_min[3], end=x_max[3], step=0.001, bar_color='orange')
peak2_temperature = Slider(title='Gaussian 2, temperature', value=cons_descriptor[0,4], start=x_min[4], end=x_max[4], step=1, bar_color='orange')
peak2_width = Slider(title='Gaussian 2, width', value=cons_descriptor[0,5], start=x_min[5], end=x_max[5], step=1, bar_color='orange')

peak3_amplitude = Slider(title='Gaussian 3, amplitude', value=cons_descriptor[0,6], start=x_min[6], end=x_max[6], step=0.001, bar_color='green')
peak3_temperature = Slider(title='Gaussian 3, temperature', value=cons_descriptor[0,7], start=x_min[7], end=x_max[7], step=1, bar_color='green')
peak3_width = Slider(title='Gaussian 3, width', value=cons_descriptor[0,8], start=x_min[8], end=x_max[8], step=1, bar_color='green')

peak4_amplitude = Slider(title='Gaussian 4, amplitude', value=cons_descriptor[0,9], start=x_min[9], end=x_max[9], step=0.001, bar_color='red')
peak4_temperature = Slider(title='Gaussian 4, temperature', value=cons_descriptor[0,10], start=x_min[10], end=x_max[10], step=1, bar_color='red')
peak4_width = Slider(title='Gaussian 4, width', value=cons_descriptor[0,11], start=x_min[11], end=x_max[11], step=1, bar_color='red')

peak5_amplitude = Slider(title='Gaussian 5, amplitude', value=cons_descriptor[0,12], start=x_min[12], end=x_max[12], step=0.001, bar_color='gray')
peak5_temperature = Slider(title='Gaussian 5, temperature', value=cons_descriptor[0,13], start=x_min[13], end=x_max[13], step=1, bar_color='gray')
peak5_width = Slider(title='Gaussian 5, width', value=cons_descriptor[0,14], start=x_min[14], end=x_max[14], step=1, bar_color='gray')

# Set up callbacks

def update_data(attrname, old, new):
    # get the current slider values
    A1 = peak1_amplitude.value
    T1 = peak1_temperature.value
    W1 = peak1_width.value
    A2 = peak2_amplitude.value
    T2 = peak2_temperature.value
    W2 = peak2_width.value
    A3 = peak3_amplitude.value
    T3 = peak3_temperature.value
    W3 = peak3_width.value
    A4 = peak4_amplitude.value
    T4 = peak4_temperature.value
    W4 = peak4_width.value
    A5 = peak5_amplitude.value
    T5 = peak5_temperature.value
    W5 = peak5_width.value
    # generate the new curve
    x = np.arange(308, 1070, 1)
    cons_descriptor = np.array([[A1, T1, W1, A2,
        T2, W2, A3, T3,
        W3, A4, T4, W4,
        A5, T5, W5, 0.00000000e+00,
        0.00000000e+00]])
    # normalisation
    norm = np.array([[3.12280000e-01, 5.98813002e+03, 7.40635820e+02, 1.80230000e-01,
        6.93425373e+03, 6.21512470e+02, 7.37760000e-01, 7.57653689e+03,
        7.10082360e+02, 5.70730000e-01, 9.36837481e+03, 1.06340591e+03,
        7.99100000e-02, 1.07656541e+04, 2.58237395e+03, 1.00000000e+00,
        1.00000000e+00]])
    cons = np.divide(cons_descriptor, norm)

    gaus1 = A1 * np.exp( - (x - T1)**2 / (W1**2))
    gaus2 = A2 * np.exp( - (x - T2)**2 / (W2**2))
    gaus3 = A3 * np.exp( - (x - T3)**2 / (W3**2))
    gaus4 = A4 * np.exp( - (x - T4)**2 / (W4**2))
    gaus5 = A5 * np.exp( - (x - T5)**2 / (W5**2))
    y = gaus1 + gaus2 + gaus3 + gaus4 + gaus5 + rand

    HSP = model.predict(cons)
    HSP_text = 'HSP = ' + str(format(HSP[0,0]*100, '.1f')) + '%'
    right.title.text = HSP_text
    HSP = np.array([0, HSP[0,0]*100])

    source.data = dict(x=x, y=y, g1=gaus1, g2=gaus2, g3=gaus3, g4=gaus4, g5=gaus5)
    source2.data = dict(hsp=HSP, y2=y2)

for i in [peak1_amplitude, peak1_temperature, peak1_width,
         peak2_amplitude, peak2_temperature, peak2_width,
         peak3_amplitude, peak3_temperature, peak3_width,
         peak4_amplitude, peak4_temperature, peak4_width,
         peak5_amplitude, peak5_temperature, peak5_width]:
    i.on_change('value', update_data)


# Set up layouts and add to document
results = column(left)
description = column(t1,t2,t3,t4,t5,t6,right, height=370)

input1 = row(peak1_amplitude, peak1_temperature, peak1_width, height=40, sizing_mode = "stretch_width")
input2 = row(peak2_amplitude, peak2_temperature, peak2_width, height=40, sizing_mode = "stretch_width")
input3 = row(peak3_amplitude, peak3_temperature, peak3_width, height=40, sizing_mode = "stretch_width")
input4 = row(peak4_amplitude, peak4_temperature, peak4_width, height=40, sizing_mode = "stretch_width")
input5 = row(peak5_amplitude, peak5_temperature, peak5_width, height=40, sizing_mode = "stretch_width")
inputs = column(input1, input2, input3, input4, input5)

curdoc().add_root(layout([[results, description],
                    [inputs]], sizing_mode = "scale_both"))
curdoc().title = 'Interaction'


# In[ ]:




