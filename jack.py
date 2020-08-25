import scipy.integrate as integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import multiprocessing as mp
import time

'''
  Script for performing jackknife on the data from the random walk.
'''
# ---------------General case-to-case information---------------
t = np.linspace( 0, 3600, num = 4*3600+1 )

cov_full = np.zeros( (len( t ), 100), dtype = float )

obst = ("none", "central", "two_sym", "three_sym")

name = "chaos_" + obst[0] + "_520_25_"#"_55_18_"

#name = "chaos_three_sym_00_"

f = open( "Chaos/Final/" + name + "final.dat", 'w' )
f.write( "#t\tcov\terr\ttop\tbot" )

# Get all the data
for j in range( 100 ):
  with open( "Chaos/" + name + str( j ) + ".dat", 'r' ) as fin:
    cov = np.genfromtxt( fin, delimiter = '\t', dtype = 'float', usecols = 1 )
    cov_full[:,j] = cov

top = np.max( cov_full, axis = 1 )
bot = np.min( cov_full, axis = 1 )

# ---------------Perform jackknife---------------
# Take the average from n-1 samples for all n samples as well as the total average
cov_ave = np.asarray( [np.sum( np.delete( cov_full, i, 1 ), axis = 1 ) / 99 
                     for i in range( 100 )] + [np.sum( cov_full, axis = 1 )/100] )
print( cov_ave.shape )

# Calculate the jackknifed average
cov_jack = np.array( [100*cov_ave[-1] - 99*cov_ave[i] for i in range( 100 )] )
cov_true = np.mean( cov_jack, axis = 0 )
print( cov_true )

# Calculate the error in the jackknife estimate
err = np.sqrt( np.var( cov_jack, axis = 0 ) / 99 )
print( err )

#Save the data
for i in range( len( t ) ):
  f.write( "\n" + str( t[i] ) + "\t" + str( cov_true[i] ) + "\t" + str( err[i] )
           + "\t" + str( top[i] ) + "\t" + str( bot[i] ) )
f.close()
