#! /usr/bin/env python3

import sys, os
import pyhull.convex_hull
import numpy
import tfm

import scipy.optimize

import igraph


class GoodEnoughException(Exception):
    pass

class SpacedPoints(object):

    maxfuncalls = 10000

    def __init__(self):

        self.ecount = 0
        self.bestxy = None
        self.beste = 1e10

        self.curxy = None

        self.nearthresh = 0.4

        self.reportrate = 0

    def bestxyz(self):
        xy = self.curxy if self.bestxy is None else self.bestxy
        xyzpts = SpacedPoints.ster2sph( xy )
        xyzpts = numpy.concatenate( [ xyzpts, [[0,0,-1]] ] )  # add back the missing pole
        return xyzpts

    def nudgepoints(self):
        try:
            # I can't get the minimizer (global or local) to stop when I want it to, so the minme() function keeps a count and raises a GoodEnoughException.
            results = scipy.optimize.basinhopping( SpacedPoints.minme, self.curxy, minimizer_kwargs=dict(method='bfgs', args=(self,)), niter=2 )
            #results = scipy.optimize.minimize( SpacedPoints.minme, self.curxy, args=(self,) )
            #results = scipy.optimize.fmin( SpacedPoints.minme, self.curxy, xtol=0.005, ftol=0.005, args=(self,) )

        except GoodEnoughException:
            # timed out - use the best we've found so far
            return self.bestxyz()

        # didn't time out - use the optimizer's final result.  basinhopper() returns a dict, fmin() returns the xy vector itself.
        self.bestxy = results['x'] if isinstance(results, dict) else results
        return self.bestxyz()


    def seedpoints(self, sizes):
        """Plant seed points.  Sets up 3D points on unit sphere, roughly equally spaced"""
        npts = len(sizes)
        gr = igraph.Graph.Full(npts)
        la = gr.layout_circle(dim=3)
        seedpts = numpy.array( la.coords )

        # turn the points so that the biggest cluster goes at the -z pole
        ppole = seedpts[-1]
        T = tfm.basis( 3, 2, -ppole )  # transformation to rotate ppole => 0,0,1
        seedpts = numpy.dot( seedpts, T.transpose() )
        # Check: seedpts[-1] should be close to [0,0,-1]

        # Cook up weights for optimizer, based on sizes
        self.weights = 5*tfm.normalize( sizes )

        # xy points are everything *except* the final point, which is fixed (non-optimizable) at the south pole
        self.curxy = self.sph2ster( seedpts[0:-1] )

    # transform between stereographic projection and 3D points on unit sphere.
    @staticmethod
    def sph2ster(xyzpts):
        """2D xy points to 3D unit-sphere points  (point at infinity would map to [0,0,-1] south pole)"""
        xypts = xyzpts[:,0:2] / (1 + xyzpts[:,2]).reshape(-1,1)
        return xypts

    @staticmethod
    def ster2sph(xypts):
        """3D unit-sphere points to 2D points in the infinite plane by stereographic projection"""
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

    @staticmethod
    def minme( xypts, self ):
        """minimize this energy.  Note funny calling sequence to be compatible with optimizers."""

        xyzpts = self.ster2sph( xypts )

        #nearthresh = -0.2 # points for which dot-product < thresh are too far away to be counted
        # nearthresh = 0.5 # points for which dot-product < thresh are too far away to be counted
        
        def term( w, p0, otherp, otherw ):
            e = 0
            for p, ww in zip(otherp, otherw):
                near = numpy.dot(p0,p)
                if near > self.nearthresh:
                    e += ww * (near - self.nearthresh) / tfm.mag( p - p0 )
            return e*w

        ppole = numpy.array([0,0,-1]) 
        etotal = term( self.weights[-1], ppole, xyzpts, self.weights[:-1] )  # ppole corresponds to last weight.  There's one more weights[] entry than in xyzpts[]

        for k in range( len(xyzpts)-1 ):
            etotal += term( self.weights[k], xyzpts[k], xyzpts[k+1:], self.weights[k+1:-1] )

        self.ecount += 1
        if etotal < self.beste:
            if self.bestxy is None:
                self.bestxy = xypts.copy()
            else:
                self.bestxy[:] = xypts

        if self.ecount > self.maxfuncalls:
            raise GoodEnoughException

        if self.reportrate > 0 and self.ecount % self.reportrate == 0:
            print("# eval %d energy %g" % (self.ecount, etotal), flush=True)

        return etotal


if __name__ == "__main__":

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

    def snappoints(outname, xyzpts, msg=''):

        print("# Writing to ", outname)
        with open(outname, 'w') as outf:

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

            weights = spt.weights

            print("#%11s %12s %12s  %10s %10s %10s" % ("x", "y", "z", "weight", "neardist", "size_in"), file=outf)
            for p, w, nd, sz in zip(xyzpts, weights, neardist, sizes):
                print("%12g %12g %12g" % tuple(p), " %10g %10g %10g" % (w, nd, sz), file=outf)

            print("### ### ###", file=outf, flush=True)



    ## Read table of sizes
    sizes = readsizes( sys.argv[1] )
    
    ### Do the work
    spt = SpacedPoints()
    spt.seedpoints( sizes )

    spt.reportrate = 100    # report energy at every 100th step during optimization

    xyzpts = spt.nudgepoints()

    ### Emit points
    snappoints( sys.argv[1]+'.speck', xyzpts )
