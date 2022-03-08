#! /usr/bin/env python3

import sys, os
import pyhull.convex_hull
import numpy
import tfm

import scipy.optimize

try:
    import igraph
except:
    igraph = None


def initpoints(sizes):
    """Plant seed points.  Returns 3D points on unit sphere, roughly equally spaced"""
    npts = len(sizes)
    gr = igraph.Graph.Full(npts)
    la = gr.layout_circle(dim=3)
    seedpts = numpy.array( la.coords )

    # turn the points so that the biggest cluster goes at the -z pole
    ppole = seedpts[-1]
    T = tfm.basis( 3, 2, -ppole )  # transformation to rotate ppole => 0,0,1
    seedpts = numpy.dot( seedpts, T.transpose() )
    # Check: seedpts[-1] should be close to [0,0,-1]
    return seedpts[0:-1]  # return all the points *except* the pole


# transform between stereographic projection and 
def sph2ster(xyzpts):
    xypts = xyzpts[:,0:2] / (1 + xyzpts[:,2]).reshape(-1,1)
    return xypts

def ster2sph(xypts):
    # X,Y = x,y/(1+z)
    # R_XY = r_xy/(1+z)
    # z = (1-R^2)/(1+R^2)
    # r = sqrt(1-z^2)
    # x = X * r/R
    xypts = xypts.reshape(-1,2)
    R2 = numpy.sum( numpy.square(xypts), axis=1 )
    z = (1 - R2) / (1 + R2)
    r = numpy.sqrt( 1 - numpy.square(z) )
    xyzpts = numpy.empty( (len(xypts), 3 ) )
    xyzpts[:, 0:2] = xypts * ( r / numpy.sqrt(R2) ).reshape(-1,1)
    xyzpts[:,2] = z
    return xyzpts


def check(npts=10):
    xyzpts = numpy.random.random( 3*npts ).reshape(-1,3) - 0.5
    
    xyzpts /= numpy.sqrt( numpy.sum( numpy.square(xyzpts), axis=1 ) ).reshape(-1,1)
    # points should be on the unit sphere now
    print("R=1 check: ", " ".join(["%g" % tfm.mag(v) for v in xyzpts]))
    xypts = sph2ster(xyzpts)
    newxyzpts = ster2sph(xypts)
    delta = newxyzpts - xyzpts
    print("eqcheck: ", " ".join(["%g" % tfm.mag(dv) for dv in delta]))

#check()

ecount = 0

def minme( xypts, *args ):
    """minimize this energy"""
    global weights

    xyzpts = ster2sph( xypts )

    #nearthresh = -0.2 # points for which dot-product < thresh are too far away to be counted
    nearthresh = -1 # points for which dot-product < thresh are too far away to be counted
    
    def term( w, p0, otherp, otherw ):
        e = 0
        for p, w in zip(otherp, otherw):
            near = numpy.dot(p0,p)
            if near > nearthresh:
                e += w * (near - nearthresh) / tfm.mag( p - p0 )
        return e*w

    ppole = numpy.array([0,0,-1]) 
    etotal = term( weights[-1], ppole, xyzpts, weights[:-1] )  # ppole corresponds to last weight.  There's one more weights[] entry than in xyzpts[]

    for k in range( len(xyzpts)-1 ):
        etotal += term( weights[k], xyzpts[k], xyzpts[k+1:], weights[k+1:-1] )

    global ecount
    ecount += 1
    if ecount % 100 == 0:
        print("# eval %d energy %g" % (ecount, etotal))
    return etotal



def readsizepoints(fname):
    sizes = []
    points = []
    with open(fname) as inf:
        for line in inf.readlines():
            ss = line.split('#')[0].split()
            if len(ss) >= 4:
                sizes.append( float(ss[0]) )
                points.append( [ float(ss[1]), float(ss[2]), float(ss[3]) ] )

    if len(sizes) == 0:
        raise ValueError("Didn't find any cluster-sizes in " + fname)

    sizes = numpy.array(sizes)
    points = numpy.array(points)
    points /= numpy.sqrt( numpy.sum( numpy.square(points), axis=1 ) ).reshape(-1,1)
    return sizes, points[:-1]

sizes, points = readsizepoints( sys.argv[1] )
#weights = 10*tfm.normalize( numpy.sqrt( sizes ) )
weights = 5*tfm.normalize( sizes )

if False:
    xyzpts = initpoints( weights )

    xyzpts += ( numpy.random.random( 3*len(xyzpts) ).reshape(len(xyzpts),3) - 0.5 ) * 0.001
    xyzpts /= numpy.sqrt( numpy.sum( numpy.square( xyzpts ), axis=1 ) ).reshape(-1,1)
else:
    xyzpts = points

sterxypts = sph2ster(xyzpts)
energy = minme( sterxypts, weights )
print(f"# seed energy {energy}")

something = scipy.optimize.fmin( minme, sterxypts, args=(weights,), xtol=0.0001, ftol=0.0001, maxiter=20, full_output=1 )
#something = scipy.optimize.basinhopping( minme, sterxypts, minimizer_kwargs=dict(args=(weights,)), niter=300 )
##print(something)

new_xypts = something[0]
new_xyzpts = numpy.concatenate( [ ster2sph( new_xypts ), [[0,0,-1]] ] )  # add back the missing pole

neardist = []
for i, pi in enumerate( new_xyzpts ):
    neardist.append( min([ tfm.mag( pi - pj ) for j, pj in enumerate( new_xyzpts ) if i != j ]) )

for p, w, nd, sz in zip(new_xyzpts, weights, neardist, sizes):
    print("%10g %10g %10g" % tuple(p), " %10g %10g %10g" % (w, nd, sz))

# fmin(func, x0, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None, initial_simplex=None)
