# need a function that turns the 6 attributes to a spectrum
# could be that we have 0-1 for each attribute (scaled) and each has it's own sigma?

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from spectrogram import true_distribution

def get_spell_colour(level,school,dtype,aoe,ran,duration,concentration = False,ritual = False):
    spell_df = pd.read_csv("SpellAttributes.csv")
    level_list = spell_df["Level"]
    school_list = spell_df["School"]
    dtype_list = spell_df["Damage Type"]
    aoe_list = spell_df["Area of Effect"]
    ran_list = spell_df["Range"]
    dur_list = spell_df["Duration"]
    heights = [1,1,1,1,1,1]
    if concentration:
        heights[0] += 1
    if ritual:
        heights[-1] += 1
    if dtype == "-":
        heights[2] = 0.25
    if aoe == "-":
        heights[3] = 0.25
    level_mu,level_sigma = get_spectral_line(level,level_list)
    school_mu,school_sigma = get_spectral_line(school,school_list)
    dtype_mu,dtype_sigma = get_spectral_line(dtype,dtype_list)
    aoe_mu,aoe_sigma = get_spectral_line(aoe,aoe_list)
    ran_mu,ran_sigma = get_spectral_line(ran,ran_list)
    dur_mu,dur_sigma = get_spectral_line(duration,dur_list)
    spell_spectrum = true_distribution([level_mu,school_mu,dtype_mu,aoe_mu,ran_mu,dur_mu],
                                       [level_sigma,school_sigma,dtype_sigma,aoe_sigma,ran_sigma,dur_sigma],
                                       heights)
    return(spell_spectrum)

def get_spectral_line(attr,attr_list):
    attr_list = list(attr_list.dropna())
    
    n_att = len(attr_list)
    att_sigma = 1/(25*n_att)
    att_mean = attr_list.index(attr)/n_att
    return(att_mean,att_sigma)

#fireball
test_spectro = get_spell_colour(3.,"Evocation","Fire","sphere (30)","150 feet","Instantaneous")
#wish
#test_spectro = get_spell_colour(9.,"Conjuration","-","-","Self","Instantaneous")
#magic missile
#test_spectro = get_spell_colour(1.,"Evocation","Force","-","120 feet","Instantaneous")
test_spectro.plot_ideal_spectrum()

