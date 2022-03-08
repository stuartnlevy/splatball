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
    print("# R=1 check: ", " ".join(["%g" % tfm.mag(v) for v in xyzpts]))
    xypts = sph2ster(xyzpts)
    newxyzpts = ster2sph(xypts)
    delta = newxyzpts - xyzpts
    print("# eqcheck: ", " ".join(["%g" % tfm.mag(dv) for dv in delta]))

#check()

ecount = 0
bestxy = None
beste = 1e10
bestew = 1e10

def snappoints(xyzpts, msg=''):
    global weights, sizes

    if len(xyzpts) < len(weights):
        xyzpts = numpy.concatenate( [ xyzpts, [[0,0,-1]] ] )  # add back the missing pole

    with open(sys.argv[1] + '.speck', 'w') as outf:

        print("#! /usr/bin/env partiview", file=outf)
        print("datavar 0 w", file=outf)
        print("datavar 1 nd", file=outf)
        print("datavar 2 sz", file=outf)
        print("", file=outf)
        print("eval polylumvar point-size area", file=outf)
        print("eval lum sz 0 1", file=outf)
        print("eval polysize 0.001", file=outf)
        print("eval poly on", file=outf)
        print("eval alpha 0.9", file=outf)
        print("", file=outf)
        print('# ', msg, file=outf)

        neardist = []
        for i, pi in enumerate( xyzpts ):
            neardist.append( min([ tfm.mag( pi - pj ) for j, pj in enumerate( xyzpts ) if i != j ]) )

        print("#%11s %12s %12s  %10s %10s %10s" % ("x", "y", "z", "weight", "neardist", "size_in"), file=outf)
        for p, w, nd, sz in zip(xyzpts, weights, neardist, sizes):
            print("%12g %12g %12g" % tuple(p), " %10g %10g %10g" % (w, nd, sz), file=outf)

        print("### ### ###", file=outf, flush=True)


def minme( xypts, *args ):
    """minimize this energy"""
    global weights

    xyzpts = ster2sph( xypts )

    #nearthresh = -0.2 # points for which dot-product < thresh are too far away to be counted
    nearthresh = -1.0 # points for which dot-product < thresh are too far away to be counted
    
    def term( w, p0, otherp, otherw ):
        e = 0
        for p, ww in zip(otherp, otherw):
            near = numpy.dot(p0,p)
            if near > nearthresh:
                e += ww * (near - nearthresh) / tfm.mag( p - p0 )
        return e*w

    ppole = numpy.array([0,0,-1]) 
    etotal = term( weights[-1], ppole, xyzpts, weights[:-1] )  # ppole corresponds to last weight.  There's one more weights[] entry than in xyzpts[]

    for k in range( len(xyzpts)-1 ):
        etotal += term( weights[k], xyzpts[k], xyzpts[k+1:], weights[k+1:-1] )

    global ecount
    global bestxy, beste, bestew
    ecount += 1
    if ecount % 100 == 0:
        print("# eval %d energy %g" % (ecount, etotal), flush=True)
        if etotal < bestew:
            snappoints( xyzpts, 'energy %g iter %d' % (etotal, ecount) )


    if etotal < beste:
        bestxy = xyzpts
        bestf = etotal
    return etotal



def readsizes(fname):
    sizes = []
    with open(fname) as inf:
        for line in inf.readlines():
            ss = line.split('#')
            if len(ss) > 0 and ss[0].strip() != '':
                sizes.append( float(ss[0]) )
    if len(sizes) == 0:
        raise ValueError("Didn't find any cluster-sizes in " + fname)

    sizes = numpy.array(sizes)
    return sizes

sizes = readsizes( sys.argv[1] )
#weights = 10*tfm.normalize( numpy.sqrt( sizes ) )
weights = 5*tfm.normalize( sizes )

xyzpts = initpoints( weights )

xyzpts += ( numpy.random.random( 3*len(xyzpts) ).reshape(len(xyzpts),3) - 0.5 ) * 0.001
xyzpts /= numpy.sqrt( numpy.sum( numpy.square( xyzpts ), axis=1 ) ).reshape(-1,1)

print("#! /usr/bin/env partiview")
print("datavar 0 weight")
print("datavar 1 neardist")
print("datavar 2 size_in")

sterxypts = sph2ster(xyzpts)
energy = minme( sterxypts, weights )
print(f"# seed energy {energy}")

def basin_cb( x, fval, accept ):

    new_xyzpts = numpy.concatenate( [ ster2sph( x ), [[0,0,-1]] ] )  # add back the missing pole

    neardist = []
    for i, pi in enumerate( new_xyzpts ):
        neardist.append( min([ tfm.mag( pi - pj ) for j, pj in enumerate( new_xyzpts ) if i != j ]) )

    print("#%11s %12s %12s  %10s %10s %10s" % ("x", "y", "z", "weight", "neardist", "size_in"))
    for p, w, nd, sz in zip(x, weights, neardist, sizes):
        print("%12g %12g %12g" % tuple(p), " %10g %10g %10g" % (w, nd, sz))

    print("### ### ###", flush=True)

    return False


#something = scipy.optimize.fmin( minme, sterxypts, args=(weights,), xtol=0.0001, ftol=0.0005, maxiter=6800, full_output=1 )
#something = scipy.optimize.fmin_cg( minme, sterxypts, args=(weights,), gtol=0.01, epsilon=1e-5 )
something = scipy.optimize.basinhopping( minme, sterxypts, callback=basin_cb, minimizer_kwargs=dict(method='bfgs', args=(weights,)), niter=1 )
##print(something)

new_xypts = something[0]
new_xyzpts = numpy.concatenate( [ ster2sph( new_xypts ), [[0,0,-1]] ] )  # add back the missing pole

neardist = []
for i, pi in enumerate( new_xyzpts ):
    neardist.append( min([ tfm.mag( pi - pj ) for j, pj in enumerate( new_xyzpts ) if i != j ]) )

print("#%11s %12s %12s  %10s %10s %10s" % ("x", "y", "z", "weight", "neardist", "size_in"))
for p, w, nd, sz in zip(new_xyzpts, weights, neardist, sizes):
    print("%12g %12g %12g" % tuple(p), " %10g %10g %10g" % (w, nd, sz))

# fmin(func, x0, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None, initial_simplex=None)
