#! /usr/bin/env python3

import sys, os
import pyhull.convex_hull
import numpy

import spacedpoints

try:
    import igraph
except:
    igraph = None

def randS2(npts):

    # randomly distribute npts points on the unit sphere.
    pts = numpy.empty( (npts,3) )
    theta = (2 * numpy.pi) * numpy.random.random(npts)
    z = numpy.random.random(npts) * 2 - 1
    r = numpy.sqrt( 1 - z*z )
    pts[:,0] = r*numpy.cos(theta)                     # x
    pts[:,1] = r*numpy.sin(theta)                     # y
    pts[:,2] = z                                      # z
    return pts

def basis( dim, *args ):
    """basis( dim, ax0,vec0, ax1,vec1, ... )
    Construct a {dim}x{dim} orthonormal basis matrix given some input vectors along particular axes.   The output's row ax0 = vec0, ax1 = vec1 with vec0 projected out, etc."""

    # M is the list of vectors assigned so far
    # T is the final output arra
    # done is an array with zeros for unassigned rows, ones elsewhere.
    o = 0
    M = [ ]
    T = numpy.eye( dim )
    undone = set( range(dim) )

    iargs = 0
    for idone in range(dim):
        if iargs+2 <= len(args):
            row = args[iargs]
            vec = vunit( numpy.array(args[iargs+1]) )
            iargs += 2
        else:
            # choose an unused row, and invent a vector
            row = undone.pop()
            undone.add(row) # we'll remove it later
            vec = numpy.zeros( dim )
            vec[row] = 1.0

        # orthogonalize against all preceding rows
        for j in range(idone):
            for prevvec in M:
                vec -= numpy.dot( prevvec, vec ) * prevvec
            vdot = numpy.dot( vec, vec )
            if vdot > 1e-7: # if not degenerate, we're done
                vec /= numpy.sqrt( vdot )
                break
            # recover from nearly-degenerate case: perturb one coordinate (the j'th one) and retry.
            vec[(j + row)%dim] += 1.0

        M.append( vec )
        T[row] = vec
        undone.remove( row )

    if numpy.linalg.det(T) < 0:
        # flip the last-assigned row to ensure T has a non-negative determinant
        T[row] *= -1
    return T

def vdist(pa,pb):
    return numpy.sqrt( numpy.square(pa-pb).sum() )

def vunit(p):
    r = numpy.sqrt( numpy.square(p).sum() )
    return p / (1 if r==0 else r)

def vmag2(vec):
    return numpy.square(vec).sum()

def vmag(vec):
    return numpy.sqrt( numpy.square(vec).sum() )

def linedistance( p, pa, pb ):
    """How far is point p from the pa-pb line?"""
    va = p - pa
    uline = vunit(pb - pa)
    # p projected onto ab
    fromline = va - numpy.dot(va,uline) * uline
    return vmag(fromline)

class HullNet(object):

    class OriEdge(object):
        def __init__(self, v0, v1, vother, facet01):
            self.v0 = v0
            self.v1 = v1
            self.vother = vother
            self.facet01 = facet01 # facet containing v0->v1
            self.circumcenter = None

    def addedge(self, v0, v1, vother, f01):
        self.edges[(v0,v1)] = self.OriEdge(v0, v1, vother, f01)
        if v0 not in self.neighborset:
            self.neighborset[v0] = set()
        self.neighborset[v0].add(v1)

    def findcenters(self):
        pts = self.pts
        for ed in self.edges.values():
            # circumcenter of a triangle in R3, from formula in
            #   https://gamedev.stackexchange.com/questions/60630/how-do-i-find-the-circumcenter-of-a-triangle-in-3d
            pa = pts[ed.v0]
            vba = pts[ed.v1] - pa
            vca = pts[ed.vother] - pa
            l2ba = vmag2(vba)
            l2ca = vmag2(vca)
            xbca = numpy.cross( vba, vca )     # (b-a) cross (c-a)
            l2xbca = vmag2(xbca)  # |(b-a) cross (c-a)|^2
            ed.circumcenter = pa + (l2ca * numpy.cross(xbca, vba) - l2ba * numpy.cross(xbca, vca)) / (2 * l2xbca)

    def findcentertfm(self, vno, on_facet=True):
        """Return 4x4 transformation matrix to put a 2-D picture on the Voronoi cell around the vno'th point.
    It's scaled so that a picture that lies within a unit circle in the XY plane should fit.
    With on_facet=True, puts the picture on the flat(ish) facet spanned by voronoiverts(vno),
        centered on a representative point in the middle of the facet.
    With on_facet=False, puts the picture on a plane that's tangent to the sphere,
        centered on the vno'th point."""

        pcen = self.pts[vno]

        # how far (half-chord-length) to the nearest neighbor vertex?
        halfnear = 0.5 * min( [ vdist(pcen, self.pts[nb]) for nb in self.neighborset[vno] ] )
        # how far away from the origin is this point?

        vorverts = self.voronoiverts( vno )

        normmid, normspan = self.voronoinormal( vorverts )
        # over the span of normal-vectors on this facet,
        # normmid is the middle of the range, normalized to a unit vector.
        vnormal = normmid

        # How far from the sphere's center, along that normal direction, is the farthest voronoi vert?
        howhigh = max( [numpy.dot(vnormal, vert) for vert in vorverts] )

        if on_facet:
            ## centerpoint = howhigh * normal
            ## radius = halfnear
            nvverts = len(vorverts)
            # rough center.  Use below to compute more accurate centroid.
            centerpoint = numpy.mean( vorverts, axis=0 ) # howhigh * vnormal
            wsum = numpy.zeros(3)
            wcensum = numpy.zeros(3)
            for i in range(nvverts):
                pa, pb = vorverts[i], vorverts[(i+1)%nvverts]
                w = vmag( numpy.cross(pa-centerpoint, pb-centerpoint) )
                wcensum += w * (centerpoint + pa + pb) / 3.0
                wsum += w

            # centroid -- area-weighted mean of centroids of individual triangles
            # thanks to: https://math.stackexchange.com/questions/90463/how-can-i-calculate-the-centroid-of-polygon
            centerpoint = wcensum / wsum
                
            radius = min( [ linedistance(centerpoint, vorverts[i], vorverts[(i+1)%nvverts]) for i in range(nvverts) ] )

        else:
            centerpoint = pcen
            vnormal = pcen
            radius = halfnear / howhigh

        ## print("vno %d halfnear %.3f howhigh %g radius %.3f" % (vno, halfnear, howhigh, radius))

        tfm3 = basis( 3, 2,vnormal, 1,[0,0,1] )

        tfm3 *= radius  # cheapo scaling

        tfm4 = numpy.eye(4)
        tfm4[0:3, 0:3] = tfm3
        tfm4[3, 0:3] = centerpoint
        
        return tfm4

        

    def allneighbors(self, vno):
        nblist = list( self.neighborset[vno] )
        # pick the lowest-numbered neighbor and traverse triangles
        nbfirst = min( nblist )
        neighbors = [ ]
        nb = nbfirst
        while True:
            neighbors.append( nb )
            try:
                oriedge = self.edges[(vno,nb)]
            except:
                # safety check
                print("Trouble: expected vno %d as neighbor of %d, but no edge (%d,%d).  Only have: %s" % (nb, vno, vno,edge, " ".join([str(vv) for vv in sorted(self.edges.keys()) if vv[0]==vno])))
            nb = oriedge.vother
            if nb == nbfirst:
                break
            # safety check
            if nb in neighbors:
                print("Trouble: surprise loop while scanning neighbors of %d: %s already includes %d" % (vno, " ".join("%d"%v for v in neighbors), nbnext))
            

        return neighbors

    def voronoiverts(self, vno):
        """Return list of [x,y,z] positions of vertex # vno"""
        nbs = self.allneighbors(vno)
        verts = [ self.edges[(vno,nb)].circumcenter for nb in nbs ]

        return verts


    def voronoinormal(self, verts):
        """Return 2-tuple of facet normal stuff: midnorm, normspanptp"""
        v0 = numpy.mean( verts, axis=0 )
        norms = []
        for i in range(0, len(verts)-1):
            vxab = vunit( numpy.cross( verts[i+1]-v0, verts[i]-v0 ) )
            norms.append( vxab )
        normspan = numpy.ptp( norms, axis=0 )
        normmid = vunit( numpy.min( norms, axis=0 ) + numpy.max(norms, axis=0) )

        return normmid, normspan
        

    def __init__(self, points):

        # Use joggle=True to ensure that every convex-hull facet is a triangle.
        # Otherwise, if >=4 points appeared to be coplanar, we might get some quadrilateral-or-bigger polys.

        self.hull = pyhull.convex_hull.ConvexHull( points, joggle=True )

        self.edges = {}
        self.neighborset = {}

        self.pts = numpy.array( points )

        for fno, (va,vb,vc) in enumerate( self.hull.vertices ):
            self.addedge(va, vb, vc, fno)
            self.addedge(vb, vc, va, fno)
            self.addedge(vc, va, vb, fno)

        self.findcenters()


###
# Example usage
        
def voronoifaces(pts, diagfile=None):

    # hull.vertices is a list of (probably) 3-element lists, giving vertex indices of each facet in the convex hull. e.g.
    # [[2, 0, 3], [0, 4, 3], [4, 0, 2], [1, 2, 3], [4, 1, 3], [1, 4, 2]]

    # so what are the neighbors of vertex 0, in order?
    # 0 has neighbors 3  (0->3 on facet 0, 3->0 on facet 1), 2 (2->0 on facet 0, 0->2 on facet 2), 
    # 0->3 (from facet 0), 0->4 (from facet 1), 0->2 (from facet 2)
    # neighbors of vertex 2?
    # 2->0 (facet 0), 2->4 (facet 1), 2->3 (facet 


    hnet = HullNet( pts )

    apts = hnet.pts


    def dist(va,vb):
        return numpy.sqrt( numpy.square(apts[va]-apts[vb]).sum() )

    print("centers and neighbors")
    for vno in range(len(pts)):
        nbs = hnet.allneighbors(vno)
        dists = [dist(vno, nb) for nb in nbs]
        vverts = hnet.voronoiverts( vno )
        normmid, normspan = hnet.voronoinormal( vverts )
        print("%3d [%5.3f .. %5.3f] ~%6.5f %2d:  %s" % (vno, min(dists), max(dists), max(normspan), len(nbs), " ".join(["%d" % nb for nb in nbs])))

    if diagfile is not None:
        with open(diagfile, 'w') as speckf:

            maxskip = 0.04
            def puthalfedge(va,vb):
                n = max( int(0.25*dist(va,vb) / maxskip)+2, 3 )
                pa, pb = apts[va], apts[vb]
                for frac in numpy.linspace(0, 0.25, num=n, endpoint=False):
                    p = ( pa + (pb-pa)*frac )
                    print("%g %g %g %d %d" % (*tuple(p), va, vb), file=speckf)

            def putvoronoiedge(va, nb0, nb1):
                pa = apts[va]
                cen0 = hnet.edges[(va,nb0)].circumcenter
                cen1 = hnet.edges[(va,nb1)].circumcenter

                n = max( int(vdist(cen0,cen1) / maxskip)+2, 3 )

                for frac in numpy.linspace(0, 1, num=n, endpoint=False):
                    ## p = vunit( 0.05*pa + 0.95*(cen0 + (cen1-cen0)*frac) )
                    p = ( 0.05*pa + 0.95*(cen0 + (cen1-cen0)*frac) )   # flat facet
                    print("%g %g %g %d" % (*tuple(p), va), file=speckf)

            print("#! /usr/bin/env partiview", file=speckf)
            print("object g1=hull", file=speckf)
            print("datavar 0 va", file=speckf)
            print("datavar 1 vb", file=speckf)
            print("eval lum const 100", file=speckf)
            print("eval color va", file=speckf)
            for vno in range(len(pts)):
                for nb in hnet.allneighbors(vno):
                    puthalfedge( vno, nb )

            print("", file=speckf)
            print("object g2=voronoi", file=speckf)
            print("datavar 0 vcell", file=speckf)
            print("eval lum const 100", file=speckf)
            print("eval color vcell", file=speckf)
            for vno in range(len(pts)):
                nbs = hnet.allneighbors(vno)
                for i in range(len(nbs)):
                    putvoronoiedge( vno, nbs[i], nbs[(i+1)%len(nbs)] )

            print("", file=speckf)
            print("object g3=facetpictures", file=speckf)
            for vno in range(len(pts)):
                tfm4 = hnet.findcentertfm(vno, on_facet=True)
                # Here's the 4x4 transformation matrix for this point: tfm
                stfm = "%g %g %g %g  %g %g %g %g  %g %g %g %g  %g %g %g %g" % tuple(tfm4.ravel())

                print("0 0 0 ellipsoid -c %d -r 1,1,0.05 -s wire %s" % (vno, stfm), file=speckf)
            print("eval alpha=1", file=speckf)

            # The on_facet=False case isn't working right yet...
            ## print("", file=speckf)
            ## print("object g4=sphpictures", file=speckf)
            ## for vno in range(len(pts)):
            ##     tfm4 = hnet.findcentertfm(vno, on_facet=False)
            ##     stfm = "%g %g %g %g  %g %g %g %g  %g %g %g %g  %g %g %g %g" % tuple(tfm4.ravel())
            ##     print("0 0 0 ellipsoid -c %d -r 1,1,0.05 -s wire %s" % (vno, stfm), file=speckf)
            ## print("eval alpha=1", file=speckf)


            print("", file=speckf)
            print("eval jump 0 0 7  0 0 0", file=speckf)


def readpts( infile ):
    pts = []
    for l in infile.readlines():
        ss = l.split('#')[0].split()
        if len(ss) >= 3:
            pts.append( [float(s) for s in ss[0:3]] )
    return numpy.array(pts)

###
# main

def How_To_Use( points ):
    
    hnet = HullNet( points )

    npoints = len(points)

    for vno in range(npoints):

        tfm4 = hnet.findcentertfm( vno, on_facet=True )

        # tfm4 is a 4x4 numpy array representing a transformation matrix
        #  from the unit disk in the XY plane (-1 <= x <= 1, -1 <= y <= 1, z near zero)
        #  to the facet
        # intended to be applied as in, if you have a mostly-XY point in the form
        #   picturepoint = [x, y, z, 1]
        # then the 3D-on-spherical-facet location is
        #   p3d = numpy.dot( picturepoint, tfm4 )
        # where p3d will look like
        #   [X3D, Y3D, Z3D, 1] 

        # or easier, with just plain 3-component points:
        ppoint = [x, y, z]

        ppoint_3d = numpy.dot( ppoint, tfm4[0:3, 0:3] ) +  tfm4[3, 0:3]

        verts_for_i = hullnet.voronoiverts( i )
    

if __name__ == "__main__":

    ii = 1

    outname = "voronoi.speck"  # unless overridden below

    while ii < len(sys.argv) and sys.argv[ii][0] == '-' and len(sys.argv[ii])>1:
        opt = sys.argv[ii]; ii += 1
        if opt == '-o':
            outname = sys.argv[ii]; ii += 1


    if ii+1 == len(sys.argv) and sys.argv[ii].startswith("ico"):

        # default: vertices of a regular icosahedron
        spts = """   0                        0                        1.                  # 0
       0.89442719099991587856   0                        0.44721359549995793 # 1
       0.27639320225002104342   0.85065080835203993366   0.44721359549995793 # 2
      -0.72360679774997893378   0.52573111211913365982   0.44721359549995793 # 3
      -0.72360679774997893378  -0.52573111211913365982   0.44721359549995793 # 4
       0.27639320225002104342  -0.85065080835203993366   0.44721359549995793 # 5
       0.72360679774997893378   0.52573111211913365982  -0.44721359549995793 # 6
      -0.27639320225002104342   0.85065080835203993366  -0.44721359549995793 # 7
      -0.89442719099991587856   0                       -0.44721359549995793 # 8
      -0.27639320225002104342  -0.85065080835203993366  -0.44721359549995793 # 9
       0.72360679774997893378  -0.52573111211913365982  -0.44721359549995793 # 10
       0                        0                       -1.                  # 11"""

        pts = [  [float(s) for s in l.split('#')[0].split()]  for l in spts.split('\n') ]
        # now pts is a list of 3-element lists

    elif ii+1 == len(sys.argv) and sys.argv[ii] == '-':
        pts = readpts( sys.stdin )

    elif ii+2 == len(sys.argv) and sys.argv[ii] == 'igraph':

        npts = int(sys.argv[ii+1])
        
        gr = igraph.Graph.Full(npts)
        la = gr.layout_circle(dim=3)
        pts = la.coords

    elif ii+2 == len(sys.argv) and sys.argv[ii] == 'points':

        with open(sys.argv[ii+1]) as inf:
            pts = readpts( inf )

    elif ii+2 == len(sys.argv) and sys.argv[ii] == 'spaced':

        sizes = []

        sizefile = sys.argv[ii+1]
        with open(sizefile) as szf:
            for line in szf.readlines():
                ss = line.split('#')[0].split()
                if len(ss) > 0:
                    sizes.append( float(ss[0]) )

        spt = spacedpoints.SpacedPoints()
        spt.seedpoints( sizes )
        pts = spt.nudgepoints()

    elif ii < len(sys.argv):
        # sphvoronoi.py 
        npts = int(sys.argv[ii])
        seed = int(sys.argv[ii+1]) if ii+1<=len(sys.argv) else 314159

        numpy.random.seed( seed )
        pts = randS2(npts)


    else:
        print("""Usage: %s [-o outname.speck] <npoints> [<randseed>]
    or  %s ico
    or  %s igraph <npoints>
    or  %s spaced <file_of_sizes_per_cluster>
    or  %s points pointsfile.dat
    or  ... stream of X Y Z values, one per line ... | %s -
    Constructs a set of points on the unit sphere,
    computes their Voronoi tessellation,
    writes "voronoi.speck", a partiview file with clumps of points for the vertices
    and arcs of points for the voronoi cell surrounding each vertex.
    With "points", reads the points from a file: one x y z point per line, space-separated.""" % ((sys.argv[0],)*6))
        sys.exit(1)


    voronoifaces( pts, diagfile=outname )
