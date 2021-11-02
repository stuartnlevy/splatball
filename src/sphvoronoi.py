#! /usr/bin/env python3

import sys, os
import pyhull.convex_hull
import numpy

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

class EdgeNet(object):

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
        pts = numpy.array( self.hull.points )
        for ed in self.edges.values():
            # circumcenter of a triangle in R3, from formula in
            #   https://gamedev.stackexchange.com/questions/60630/how-do-i-find-the-circumcenter-of-a-triangle-in-3d
            pa = pts[ed.v0]
            vba = pts[ed.v1] - pa
            vca = pts[ed.vother] - pa
            l2ba = numpy.square(vba).sum()
            l2ca = numpy.square(vca).sum()
            xbca = numpy.cross( vba, vca )     # (b-a) cross (c-a)
            l2xbca = numpy.square(xbca).sum()  # |(b-a) cross (c-a)|^2
            ed.circumcenter = pa + (l2ca * numpy.cross(xbca, vba) - l2ba * numpy.cross(xbca, vca)) / (2 * l2xbca)

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

    
    def __init__(self, hull):
        self.edges = {}
        self.neighborset = {}

        self.hull = hull

        for fno, (va,vb,vc) in enumerate( hull.vertices ):
            self.addedge(va, vb, vc, fno)
            self.addedge(vb, vc, va, fno)
            self.addedge(vc, va, vb, fno)

        self.findcenters()
        
def voronoifaces(pts, diagfile=None):

    # Use joggle=True to ensure that every convex-hull facet is a triangle.
    # Otherwise, if >=4 points appeared to be coplanar, we might get some quadrilateral-or-bigger polys.

    hull = pyhull.convex_hull.ConvexHull( pts, joggle=True )

    # hull.vertices is a list of (probably) 3-element lists, giving vertex indices of each facet in the convex hull. e.g.
    # [[2, 0, 3], [0, 4, 3], [4, 0, 2], [1, 2, 3], [4, 1, 3], [1, 4, 2]]

    # so what are the neighbors of vertex 0, in order?
    # 0 has neighbors 3  (0->3 on facet 0, 3->0 on facet 1), 2 (2->0 on facet 0, 0->2 on facet 2), 
    # 0->3 (from facet 0), 0->4 (from facet 1), 0->2 (from facet 2)
    # neighbors of vertex 2?
    # 2->0 (facet 0), 2->4 (facet 1), 2->3 (facet 


    enet = EdgeNet( hull )

    apts = numpy.array(pts)


    def dist(va,vb):
        return numpy.sqrt( numpy.square(apts[va]-apts[vb]).sum() )

    def vlen(pa,pb):
        return numpy.sqrt( numpy.square(pa-pb).sum() )

    def unit(p):
        r = numpy.sqrt( numpy.square(p).sum() )
        return p / (1 if r==0 else r)

    print("centers and neighbors")
    for vno in range(len(pts)):
        nbs = enet.allneighbors(vno)
        dists = [dist(vno, nb) for nb in nbs]
        print("%3d [%5.3f .. %5.3f] %2d:  %s" % (vno, min(dists), max(dists), len(nbs), " ".join(["%d" % nb for nb in nbs])))

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
                cen0 = enet.edges[(va,nb0)].circumcenter
                cen1 = enet.edges[(va,nb1)].circumcenter

                n = max( int(vlen(cen0,cen1) / maxskip)+2, 3 )

                for frac in numpy.linspace(0, 1, num=n, endpoint=False):
                    p = unit( 0.05*pa + 0.95*(cen0 + (cen1-cen0)*frac) )
                    print("%g %g %g %d" % (*tuple(p), va), file=speckf)

            print("#! /usr/bin/env partiview", file=speckf)
            print("object g1=hull", file=speckf)
            print("datavar 0 va", file=speckf)
            print("datavar 1 vb", file=speckf)
            print("eval lum const 100", file=speckf)
            print("eval color va", file=speckf)
            for vno in range(len(pts)):
                for nb in enet.allneighbors(vno):
                    puthalfedge( vno, nb )

            print("", file=speckf)
            print("object g2=voronoi", file=speckf)
            print("datavar 0 vcell", file=speckf)
            print("eval lum const 100", file=speckf)
            print("eval color vcell", file=speckf)
            for vno in range(len(pts)):
                nbs = enet.allneighbors(vno)
                for i in range(len(nbs)):
                    putvoronoiedge( vno, nbs[i], nbs[(i+1)%len(nbs)] )


            print("", file=speckf)
            print("eval jump 0 0 7  0 0 0", file=speckf)


def readpts( infile ):
    pts = []
    for l in infile.readlines():
        ss = l.split('#')[0].split()
        if len(ss) == 3:
            pts.append( [float(s) for s in ss] )
    return numpy.array(pts)

###
# main
            

if len(sys.argv)==2 and sys.argv[1].startswith("ico"):

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

elif len(sys.argv) == 2 and sys.argv[1] == '-':
    pts = readpts( sys.stdin )

elif len(sys.argv) == 3 and sys.argv[1] == 'igraph':

    npts = int(sys.argv[2])
    
    gr = igraph.Graph.Full(npts)
    la = gr.layout_circle(dim=3)
    pts = la.coords

elif len(sys.argv) > 1:
    # sphvoronoi.py 
    npts = int(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv)>2 else 314159

    numpy.random.seed( seed )
    pts = randS2(npts)


else:
    print("""Usage: %s <npoints> [<randseed>]
or  %s ico
or  %s igraph <npoints>
or  ... stream of X Y Z values, one per line ... | %s -
Constructs a set of points on the unit sphere,
computes their Voronoi tessellation,
writes "voronoi.speck", a partiview file with clumps of points for the vertices
and arcs of points for the voronoi cell surrounding each vertex.""" % (sys.argv[0], sys.argv[0]))
    sys.exit(1)


voronoifaces( pts, diagfile="voronoi.speck" )

    
