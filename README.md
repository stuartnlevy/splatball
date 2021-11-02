splatball -- sample code for splatting 2-D (or 2.5-D) figures onto Voronoi facets on a sphere.

Utility functions for that live here.   See sphvoronoi.py.

Dependencies: pyhull, numpy, optionally igraph.

sphvoronoi.py takes a set of points on the unit sphere, uses pyhull to construct spherical Voronoi cells, writes some summary information and a graphical model in a partiview-format voronoi.speck file.

Usage:
    python3 sphvoronoi.py 77
        (uses 77 points randomly placed on the sphere)
or
    python3 sphvoronoi.py igraph 53
        (uses igraph's spherical layout to sprinkle 53 points roughly uniformly on the sphere)
or other forms.

