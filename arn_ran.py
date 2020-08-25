import scipy.integrate as integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import multiprocessing as mp
import time

'''
  Script for Arnold flow in two "real" dimensions, random trajectory in two
  dimensions, and mirror mapping trajectories onto 2D surfaces.
'''

'''
  Global constants.
'''
X = 0
Y = 1

# Full surface (rectangle)
xr, yr = 25, 25 # Right x and top y boundaries
xl, yl = 0, 0 # Left x and bottom y boundaries
# Starting position
xStart, yStart = 5, 20

'''
  Obstacle realization
'''
# ------------Generate rectangular obstacles in the map------------
def Obs( l, w, start ):
  r1 = [start[0], start[1] + w]
  r2 = [start[0] + l, start[1] + w]
  r3 = [start[0] + l, start[1]]
  vert = np.array( [start, r1, r2, r3, start] )
  
  return vert

obstacle = ("none", "central", "two_sym", "three_sym")

no    = 0
x_len = 5
y_len = 5
ob    = False
if( no > 0 ): ob = True
obst  = []
obst.append( Obs( x_len, y_len, np.asarray( [10, 10] ) ) )
obst.append( Obs( x_len, y_len, np.asarray( [5, 5] ) ) )
obst.append( Obs( x_len, y_len, np.asarray( [15, 5] ) ) )
obst.append( Obs( x_len, y_len, np.asarray( [10, 15] ) ) )

# Chaos in Arnold
v = 1 # m/s
A = 1 # 1/s
B = 0.5 # 1/s
C = 0.5 # 1/s
# C = 0 results in periodic motion (very clear without mirror mapping).
# Large C results in chaos.
# Initial parameters
x0   = np.asarray( [4, 3.5, 0, xStart, yStart] ) #x1, x2, x3, x, y
t0   = 0
hour = 3600
minu = 60
tEnd = hour
res  = 4*tEnd+1
t    = np.linspace( t0, tEnd, num = res )


'''
  Mirror mapping
'''
# ------------Mirror the boundary and obstacles------------
def Mirror( m, times ):
  new = []
  s = 0
  n = np.copy( m ) # invariant
  o = np.copy( m )
  
  # Time change
  t_ex = np.copy( times )
  
  for i in range( len( n ) ):
    cont = True
    
    # ------------Boundary mirroring------------
    if( n[i][X] ) > xr:
      r = i + s # Position of where the new points should be inserted.
      if ( n[i-1][X] ) != xr:
        a = (n[i][Y]-n[i-1][Y])/(n[i][X]-n[i-1][X]) # Slope
        b = n[i][Y] - a * n[i][X] # Intercept
        yy = xr * a + b
        o = np.insert(o, r, np.array((xr, yy)), 0)
        
        # Time added
        t_ex = np.insert( t_ex, r, t_ex[r] )
        
        new.append(np.array((xr, yy)))
        s = s + 1 # The number of new points at the boundaries.
      n[i:][:,X] = 2*xr - n[i:][:,X]
      o[r:][:,X] = 2*xr - o[r:][:,X]
      
    elif( n[i][X] ) < xl:
      r = i + s
      if ( n[i-1][X] ) != xl:
        a = (n[i][Y]-n[i-1][Y])/(n[i][X]-n[i-1][X])
        b = n[i][Y] - a * n[i][X]
        yy = xl * a + b
        
        o = np.insert(o, r, np.array((xl, yy)), 0)
        
        t_ex = np.insert( t_ex, r, t_ex[r] )

        new.append(np.array((xl, yy)))
        s = s + 1
      n[i:][:,X] = 2*xl - n[i:][:,X]
      o[r:][:,X] = 2*xl - o[r:][:,X]
      
    if( n[i][Y] ) > yr:
      r = i + s
      if ( n[i-1][Y] ) != yr:
        a = (n[i][Y]-n[i-1][Y])/(n[i][X]-n[i-1][X]) 
        b = n[i][Y] - a * n[i][X] 
        xx = (yr - b) / a
        
        o = np.insert(o, r, np.array((xx, yr)), 0)

        t_ex = np.insert( t_ex, r, t_ex[r] )
        
        new.append(np.array((xx, yr)))
        s = s + 1 
      n[i:][:,Y] = 2*yr - n[i:][:,Y]
      o[r:][:,Y] = 2*yr - o[r:][:,Y]
      
    elif( n[i][Y] ) < yl:
      r = i + s
      if ( n[i-1][Y] ) != yl:
        a = (n[i][Y]-n[i-1][Y])/(n[i][X]-n[i-1][X])
        b = n[i][Y] - a * n[i][X]
        xx = (yl - b) / a
        
        o = np.insert(o, r, np.array((xx, yl)), 0)
        
        t_ex = np.insert( t_ex, r, t_ex[r] )
        
        new.append(np.array((xx, yl)))
        s = s + 1
      n[i:][:,Y] = 2*yl - n[i:][:,Y]
      o[r:][:,Y] = 2*yl - o[r:][:,Y]
    
    # ------------Obstacle mirroring------------
    if( ob ):
      for j in range( no ):
        if( not cont ): break
        
        x1 = obst[j][0, 0]
        x2 = obst[j][2, 0]
        y1 = obst[j][0, 1]
        y2 = obst[j][2, 1]
        
        if( n[i][X] > x1 and n[i][X] < x2 and n[i][Y] > y1 and n[i][Y] < y2 ):
          cont = False
          
          r = i + s
          
          a = (n[i][Y]-n[i-1][Y]) / (n[i][X]-n[i-1][X]) # Slope
          b = n[i][Y] - a * n[i][X] # Intercept
          
          y_L = x1 * a + b   # Left y-bounce
          y_R = x2 * a + b   # Right y-bounce
          x_L = (y1 - b) / a # Low x-bounce
          x_R = (y2 - b) / a # High x-bounce
                  
          # If it is moving to the right => left side of obstacle.
          # Else => right side of the obstacle.
          if( n[i-1][X] < n[i][X] ):
            # Left bounce!
            if( y_L < y2 and y_L > y1 ):
              if ( n[i-1][X] ) != x1:
                o = np.insert( o, r, np.array( (x1, y_L) ), 0 )
                
                t_ex = np.insert( t_ex, r, t_ex[r] )
                
                s = s + 1
              
              n[i:][:,X] = 2*x1 - n[i:][:,X]
              o[r:][:,X] = 2*x1 - o[r:][:,X]
            
            # Negative slope => top bounce.
            # Top bounce!
            elif( a < 0 ):
              if ( n[i-1][Y] ) != y2:
                o = np.insert( o, r, np.array( (x_R, y2) ), 0 )
                
                t_ex = np.insert( t_ex, r, t_ex[r] )
                
                s = s + 1
              
              n[i:][:,Y] = 2*y2 - n[i:][:,Y]
              o[r:][:,Y] = 2*y2 - o[r:][:,Y]
            
            # Bottom bounce!
            else:
              if ( n[i-1][Y] ) != y1:
                o = np.insert( o, r, np.array( (x_L, y1) ), 0 )
                
                t_ex = np.insert( t_ex, r, t_ex[r] )
                
                s = s + 1
              
              n[i:][:,Y] = 2*y1 - n[i:][:,Y]
              o[r:][:,Y] = 2*y1 - o[r:][:,Y]
          
          else:
            # Right bounce!
            if( y_R > y1 and y_R < y2 ):
              if ( n[i-1][X] ) != x2:
                o = np.insert( o, r, np.array( (x2, y_R) ), 0 )
                
                t_ex = np.insert( t_ex, r, t_ex[r] )
                
                s = s + 1
              
              n[i:][:,X] = 2*x2 - n[i:][:,X]
              o[r:][:,X] = 2*x2 - o[r:][:,X]
            
            # Bottom bounce!
            elif( a < 0 ):
              if ( n[i-1][Y] ) != y1:
                o = np.insert( o, r, np.array( (x_L, y1) ), 0 )
                
                t_ex = np.insert( t_ex, r, t_ex[r] )
                
                s = s + 1
              
              n[i:][:,Y] = 2*y1 - n[i:][:,Y]
              o[r:][:,Y] = 2*y1 - o[r:][:,Y]
            # Top bounce!
            else:
              if ( n[i-1][Y] ) != y2:
                o = np.insert( o, r, np.array( (x_R, y2) ), 0 )
                
                t_ex = np.insert( t_ex, r, t_ex[r] )
                
                s = s + 1
              
              n[i:][:,Y] = 2*y2 - n[i:][:,Y]
              o[r:][:,Y] = 2*y2 - o[r:][:,Y]
  
  return np.copy( o ), np.copy( t_ex )

# ------------Scanning of the surface------------
def scanned( m, times ):
  # How fine of a scan. Divided up into squares of [resolution, 2*resolution).
  resolution = 1
  
  area = np.zeros( (int( (xl+xr)/resolution )+1, 
                    int( (yl+yr)/resolution )+1), dtype = int )

  coverage = np.zeros( len( m ), dtype = float )
  
  area[int( m[0, 0]/resolution ), int( m[0, 1]/resolution )] = 1
  
  l = -1
  true_cov = np.zeros( len( t ) )
  
  n = np.copy( m )
  for i in range( len( n ) ):
    area[int( n[i, 0]/resolution ), int( m[i, 1]/resolution )] = 1
    
    coverage[i] = np.sum( area[:-1, :-1] )
    if( times[i] != times[i-1] ): l += 1
    
    true_cov[l] = coverage[i]
  
  tot_area = np.prod( np.asarray( area.shape )-1 )
  
  if( ob ):
    for i in range( no ):
      tot_area -= (obst[i][2, 0] - obst[i][0, 0]-1)*(obst[i][2, 1] - obst[i][0, 1]-1)
  
  true_cov /= tot_area
  
  return np.copy( true_cov )

# ------------Deterministic chaos, Arnold equation---------------
def detChaos( dxs, t ):
  dx1 = A*np.sin( dxs[2] ) + C*np.cos( dxs[1] )
  dx2 = B*np.sin( dxs[0] ) + A*np.cos( dxs[2] )
  dx3 = C*np.sin( dxs[1] ) + B*np.cos( dxs[0] )
  dx  = v*np.cos( dxs[2] )
  dy  = v*np.sin( dxs[2] )
  
  return np.asarray( [dx1, dx2, dx3, dx, dy] )

# ---------------Random walk---------------
def ranWalk():
  xyRan = np.zeros( (len( t ), 2), dtype = float )
  rng = np.random.default_rng()
  xyRan[0] = xStart, yStart # start from the middle
  for i in range( (tEnd)//2 ):
    rans = rng.uniform( 0.0, 2*np.pi )
    # runs straight for two seconds before turning
    i *= int( 2/t[1] )
    for j in range( int( 2/t[1] ) ):
      xyRan[i+j+1] = xyRan[i+j][X] + t[1]*v*np.cos( rans ), xyRan[i+j][Y] + t[1]*v*np.sin( rans )

  return np.copy( xyRan )

# ---------------Perform the simulation---------------
def run():

  # Chaos
  l = integrate.odeint( detChaos, x0, t )
  # Only the X and Y coordinates.
  xy = np.copy( l[:,[3,4]] )
  test, t_new = Mirror( xy, t )
  test, t_new = Mirror( test, t_new )
  cov = scanned( test, t_new )

  # Save the data.
  with open( "Chaos/true_chaos_" + obstacle[no] + "_" + str( xStart ) 
             + str( yStart ) + "_" + str( xr ) + ".dat", 'w' ) as f:
    f.write( str( t[0] ) + "\t" + str( cov[0] ) )
    for k in range( 1, len( cov ) ):
      f.write( "\n" + str( t[k] ) + "\t" + str( cov[k] ) )

  # ---------------Do several random walks for statistics---------------
  for j in range( 100 ):
    g = ranWalk()
    tist, time_new = Mirror( g, t )
    tist, time_new = Mirror( tist, time_new )
    cov_ran = scanned( tist, time_new )
    with open( "Chaos/chaos_" + obstacle[no] + "_" + str( xStart ) 
               + str( yStart ) + "_" + str( xr ) + "_" + str( j ) + ".dat", 'w' 
             ) as f:
      f.write( str( t[0] ) + "\t" + str( cov_ran[0] ) )
      for k in range( 1, len( cov_ran ) ):
        f.write( "\n" + str( t[k] ) + "\t" + str( cov_ran[k] ) )
    print( j )

run()
