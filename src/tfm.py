#! /usr/bin/env python

from __future__ import print_function

# Python analog of (parts of) ../scripts/tfm.pl

import sys
import numpy
import re

__version__ = "1.3"

debug = False

def tfm( *args ):
    """tfm(...) => 4x4 transformation matrix as numpy array
    tfm(s) => scaling
    tfm("scale",sx,sy,sz) => nonuniform scaling
    tfm("scale",s, xcen,ycen,zcen) => scaling about given point
    tfm(tx,ty,tz) => translation
    tfm('x',degrees) => rotation about X (or Y or Z) axis
    tfm(ax,ay,az, degrees) => rotation about given 3-D axis vector
    tfm(ax,ay,az, degrees, cenx,y,z) => rotation about given 3-D axis vector which fixes given point
    tfm( ... list of 16 numbers ... ) -> just make an array of them
    """

    t = numpy.identity(4)

    if len(args) == 1:	# tfm(s) (scale)
        t[0,0] = t[1,1] = t[2,2] = args[0]

    elif len(args) == 4 and args[0] == "scale":	# tfm('scale',sx,sy,sz)
        t[0,0], t[1,1], t[2,2] = args[1:4]

    elif len(args) == 5 and args[0] == "scale": # tfm('scale', s, fixedx,y,z)
        t = tmul( tfm(-args[2], -args[3], -args[4]), \
                  tfm(args[1]), \
                  tfm(*args[2:5]) )

    elif len(args) == 3:		# tfm(tx,ty,tz)
        t[3, 0:3] = args[:]

    elif len(args) == 2:		# tfm('x',degrees)   (named axis, angle)
        au,av = {'x':(1,2), 'y':(2,0), 'z':(0,1)}[ args[0] ]
        s = numpy.sin( numpy.radians(args[1]) )
        c = numpy.cos( numpy.radians(args[1]) )
        t[au,au], t[au,av] = c, s
        t[av,au], t[av,av] = -s, c

    elif len(args) == 4:		# tfm(ax,ay,az, degrees)  (vector axis, angle)
        ax,ay,az = normalize( args[0:3] )
        s = numpy.sin( numpy.radians(args[3]) )
        c = numpy.cos( numpy.radians(args[3]) )
        v = 1-c

        t[0,0], t[0,1], t[0,2] = ax*ax*v + c,    ax*ay*v + az*s,  ax*az*v - ay*s
        t[1,0], t[1,1], t[1,2] = ax*ay*v - az*s, ay*ay*v + c,     az*ay*v + ax*s
        t[2,0], t[2,1], t[2,2] = ax*az*v + ay*s, ay*az*v - ax*s,  az*az*v + c

    elif len(args) == 7:		# tfm(ax,ay,az, degrees, fixedx,y,z)
        t = tmul( tfm(-args[4],-args[5],-args[6]), \
                  tfm(*args[0:4]), \
                  tfm(*args[4:7]) )

    elif len(args) == 16:
        t = numpy.ndarray( args, shape=(4,4) )

    else:
        print("tfm(", args, "): expected 1, 2, 3, 4 or 7 arguments", file=sys.stderr)
        raise ValueError(args)

    return t

def list(str):
    """list(str) => turns text string of numbers, separated by space or commas, into a list [] of floats
        or None if not all are parse-able
    """ 
    try:
        return numpy.array([float(s) for s in str.replace(',',' ').split()])
    except:
        return None

def tmul(*args):
    """tmul( T1, T2, ... ) => T1*T2*...  for each Ti a 4x4 matrix"""
    t = args[0]
    if not isinstance(t, numpy.ndarray):
        t = numpy.array(t)
    for i in range(1, len(args)):
        a = args[i]
        if not isinstance(a, numpy.ndarray):
            a = numpy.array(a)
        t = numpy.dot( t, a )
    return t

def eucinv(t):
    """eucinv(T) => quick inverse of 4x4 matrix T, assuming it's a Euclidean isometry (rotation, translation) with uniform scaling"""
    if not isinstance(t, numpy.ndarray):
        t = numpy.array(t)
    invt = numpy.identity(4)
    s = numpy.dot( t[0,0:3], t[0,0:3] )
    invs = s and 1.0/s or 0
    invt[0:3,0:3] = invs * t[0:3,0:3].transpose()
    invt[3,0:3] = - numpy.dot( t[3, 0:3], invt[0:3,0:3] )
    return invt

#t2euler(\"xzy\",T) => X,Z,Y (deg) so rotY*rotZ*rotX = T.  t2euler(\"yxz\",T) = t2aer(T).
# t2meuler(\"yzx\",T) = X,Y,Z, using t2euler(\"xzy\",T).  meuler2t(\"yzx\",X,Y,Z)
# euler2quat(\"xzy\",X,Z,Y) => q;  quat2euler(\"xzy\", q) => angleX,Z,Y, so rotY*rotZ*rotX = q.
# meuler2quat(\"yzx\",X,Y,Z) => q; quat2meuler(\"yzx\",q) => angleX,Y,Z

def zyxperm( *args ):	# zyxperm("yxz", X,Y,Z) => [ "zxy", Z,X,Y ]
    if len(args) == 1:
        args = args[0]

    perm = args[0]
    if hasattr(args[1],'__getitem__') and len(args[1]) == len(perm):
        ABC = args[1]
    else:
        ABC = args[1:]

    prev = perm[::-1]
    qrev = xyz2ax( prev )
    a = [ prev ]
    a.extend( [ ABC[q] for q in qrev ] )
    return a

def xyzmrep( *args ):  # xyzmrep( "yxz",Y,X,Z ) => [ X,Y,Z ]  or xyzmrep("yxz",[Y,X,Z]) => [ X,Y,Z ]
    if len(args) == 1:
        args = args[0]
    perm = args[0]
    if len(args[1]) == len(perm):
        ABC = args[1]
    else:
        ABC = args[1:]

    qerm = xyz2ax( perm )

    return [ ABC[ q ] for q in qerm ]

def meuler2quat( *args ):  # meuler2quat("yxz", X,Y,Z) = euler2quat("zxy", Z,X,Y)
    revved = zyxperm( *args )
    return euler2quat( *revved )

def quat2meuler( perm, q ): # quat2meuler("yxz",q) = quat2euler("zxy", q) returned in X,Y,Z order
    prev = perm[2]+perm[1]+perm[0]
    return xyzmrep( prev, quat2euler( prev, q ) )

def xyz2ax( str ):
    return [ {'x':0, 'y':1, 'z':2, '0':0, '1':1, '2':2}[ s.lower() ] for s in str ]

def euler2quat( perm, ra,rb,rc ): # euler2quat( "yxz", Ydeg,Xdeg,Zdeg ) => quaternion zrot(Z)*xrot(X)*yrot(Y)
  """euler2quat( axABC, angleA,angleB,angleC ) => quat
        e.g. euler2quat( "yxz", Ydeg,Xdeg,Zdeg ) => quaternion zrot(Z)*xrot(X)*yrot(Y)"""
  A,B,C = map( numpy.radians, (ra,rb,rc) )
  try:
    ia, ib, ic = xyz2ax( perm )
  except:
    raise ValueError( "euler2quat: permutation '%s' must have exactly 3 characters - some permutation of x y z" % perm )
  # sines and cosines of half-angles

  qA = numpy.array( (numpy.cos(.5*A), 0,0,0) );  qA[ia+1] = numpy.sin(.5*A);
  qB = numpy.array( (numpy.cos(.5*B), 0,0,0) );  qB[ib+1] = numpy.sin(.5*B);
  qC = numpy.array( (numpy.cos(.5*C), 0,0,0) );  qC[ic+1] = numpy.sin(.5*C);
  
  return quatmul( qC, quatmul( qB, qA ) )

def quat2euler( perm, qu,qi,qj,qk ):	# "xzy", u,i,j,k => ( angleX, Z, Y )  (in degrees)
    """quat2euler( axABC, qr,qi,qj,qk ) => [ angleA, angleB, angleC ] in degrees
        e.g. quat2euler( "zxy", qr,qi,qj,qk ) => [ angleZ, angleX, angleY ]"""
    return t2euler( perm, quat2t(qu,qi,qj,qk) )

def mag( vec ):
    """mag(vec) => magnitude of vec"""
    return numpy.sqrt( numpy.dot( vec,vec ) )

def dist( p0, p1 ):
    return mag( p0-p1 )

def normalize( vec ):
    """normalize(vec) => unit-length copy of vec (or all zeros if it was that)"""
    if not isinstance(vec, numpy.ndarray):
        vec = numpy.array( vec )
    s = numpy.sqrt( numpy.dot( vec,vec ) )
    invs = s and 1.0/s or 0
    return invs * vec


         


def meuler2t( axABC, rx,ry,rz ):
    """Turn Euler angles (in rx,ry,rz order) into 4x4 matrix.
    meuler2t('zxy',10,20,30) == tmul( tfm('z',30), tfm('x',10), tfm('y',20) )"""
    return euler2t( *zyxperm( axABC, rx,ry,rz ) )

def euler2t( axABC, ra,rb,rc ):
    """Turn Euler angles (in reverse of multiplication order) into 4x4 matrix.
    euler2t('yxz',20,10,30) == tmul( tfm('z',30), tfm('x',10), tfm('y',20) )"""
    return quat2t( euler2quat( axABC, ra,rb,rc ) )

def t2meuler( axABC, T ):
    """t2meuler( axABC, T) => angleX,angleY,angleZ in degrees
        such that p * rotC(angleC) * rotB(angleB) * rotA(angleA) = p * T
        e.g. t2meuler( "zxy", T ) => angleX, angleY, angleZ such that
            p * rotZ(angleZ) * rotX(angleX) * rotY(angleY) = p * T
        See also t2euler() which returns always in ABC order."""
    revp = axABC[::-1]  # reversed
    return xyzmrep( revp, t2euler( revp, T ) )

def t2euler( axABC, T, nearABC=None ):  #  => angleA,angleB,angleC
                        # such that p * rotC(angleC) * rotB(angleB) * rotA(angleA) = p * T
    """t2euler( axABC, T) => angleA,angleB,angleC in degrees
        such that p * rotC(angleC) * rotB(angleB) * rotA(angleA) = p * T
        e.g. t2euler( "zxy", T ) => angleY, angleX, angleZ.
        See also t2meuler() which returns always in angleX,angleY,angleZ order."""
    a,b,c = xyz2ax( axABC )
    t = T[0:3,0:3].copy()
    s = mag( t[2,:] )
    t *= 1/s

    perm = [1, -1] [ ( (a>b) + (b>c) + (a>c) ) % 2 ]

    Ta = t[a,:]
    Tb = t[b,:]
    Tc = t[c,:]
    Tca = Tc[a]		# Tca * perm = sin(B)
    cosB = numpy.sqrt( Ta[a]*Ta[a] + Tb[a]*Tb[a] ) # hypot( Taa, Tba ) = hypot( Tcb, Tcc ) = cos(B)
    B = numpy.arctan2( perm * Tca, cosB )

    A = numpy.arctan2( -perm*Tc[b], Tc[c] )	# A = atan2( -perm*Tcb, Tcc )
    C = numpy.arctan2( -perm*Tb[a], Ta[a] )	# C = atan2( -perm*Tba, Taa )
    if Tca > 0.9:
        # perm*sin(B) is near +1, cos(B) near zero, so Taa/Tba/Tcb/Tcc terms inaccurate.
        # $Tca is near +1, use A+C terms, which should be quite accurate
        # -perm*(Tab+Tbc) = sin(A+C) * (1 + Tca)
        #       (Tbb-Tac) = cos(A+C) * (1 + Tca)
        ApC = numpy.arctan2( perm * (Ta[b] + Tb[c]), Tb[b] - Ta[c] )
        # Tweak A and C so that they sum to this.
        delta = 0.5 * (ApC - (A + C));
        if delta > 2: delta -= numpy.pi
        elif delta < -2: delta += numpy.pi
        if debug:
            print("# t2euler( '%s', T ): A %g C %g A+C %g => A+%g C+%g" % (axABC, A, C, ApC, delta,delta), file=sys.stderr)
        # print STDERR "A $A C $C A+C $ApC => A+$delta C+$delta\n" if $debug;
        A += delta;
        C += delta;

    elif Tca < -0.9:
        # perm*sin(B) is near -1, cos(B) near zero, so Taa/Tba/Tcb/Tcc terms inaccurate
        # Tca is near -1, use A-C terms, which should be quite accurate
        # perm*(Tab-Tbc) = sin(A-C) * (1 - Tca)
        #      (Tac+Tbb) = cos(A-C) * (1 - Tca)
        AmC = numpy.arctan2( -perm * (Ta[b] - Tb[c]), Ta[c] + Tb[b] );
        # Tweak A and C so that their difference is this.
        delta = 0.5 * (AmC - (A - C));
        if delta > 2: delta -= numpy.pi;
        elif delta < -2: delta += numpy.pi
        if debug:
            print("# t2euler( '%s', T ): A %g C %g A-C %g => A+%g C-%g" % (axABC, A, C, AmC, delta,delta), file=sys.stderr)
        # print STDERR "A $A C $C A-C $AmC => A+$delta C-$delta\n" if $debug;
        A += delta;
        C -= delta;
    return numpy.degrees( [A,B,C] )

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
            vec = normalize( args[iargs+1] )
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

def dms2rad( dms ):
    return numpy.radians( dms2d(dms) )

def hms2d( hms ):
    return 15 * dms2d(hms)

def dms2d( dms ):
    """dd:mm:ss to floating-point degrees"""
    if isinstance(dms, str):
        sign = 1
        if dms[0] == '-':
            sign = -1
            dms = dms[1:]
        vdms = map(float, dms.split(':'))
        v = vdms[0]
        if len(vdms) > 1:
            v += vdms[1]/60.0
        if len(vdms) > 2:
            v += vdms[2]/3600.0
        return sign*v
    else:
        return float(dms)

def ra2radians( ra ):
    """Accepts RA as h:m[:s] string or as decimal degrees!  Returns RA in radians."""
    if isinstance(ra, str) and ra.find(':') >= 0:
        return numpy.radians( 15*dms2d( ra ) )
    else:
        return numpy.radians( float( ra ) )


def radec2eqbasis( ra, dec ):
    """Returns a 3x3 orthonormal basis matrix.   Third row is J2000 unit vector pointing toward (ra,dec).   First two rows are tangent to sky plane there, with Y (second row) pointing northward"""
    rra = ra2radians( ra )
    rdec = numpy.radians( dms2d(dec) )
    zv = numpy.array( [numpy.cos(rra)*numpy.cos(rdec), numpy.sin(rra)*numpy.cos(rdec), numpy.sin(rdec)] )
    xv = normalize( [-zv[0]*zv[2], -zv[1]*zv[2], 1-zv[2]**2] )
    yv = numpy.cross( zv, xv )
    return numpy.array( [xv, yv, zv] )

# @quataxb = &quatmul( @quata, @quatb )
# Quaternion multiplication
def quatmul( qa, qb ):
    """quatmul(qa,qb) => qa*qb  quaternion multiplication"""
    # rr-ii-jj-kk
    # ri+ir-jk+kj
    # rj+jr-ki+ik
    # rk+kr-ij+ji
    return numpy.array( ( \
        qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3], \
        qa[0]*qb[1] + qa[1]*qb[0] - qa[2]*qb[3] + qa[3]*qb[2], \
        qa[0]*qb[2] + qa[2]*qb[0] - qa[3]*qb[1] + qa[1]*qb[3], \
        qa[0]*qb[3] + qa[3]*qb[0] - qa[1]*qb[2] + qa[2]*qb[1] ) );
        

def quatdiv( qa, qb ):
    """quatdiv(qa, qb) => qa/qb  quaternion division"""
    #  rr-ii-jj-kk
    # -ri+ir+jk-kj
    # -rj+jr+ki-ik
    # -rk+kr+ij-ji
    return numpy.array( ( \
          qa[0]*qb[0] + qa[1]*qb[1] + qa[2]*qb[2] + qa[3]*qb[3], \
        - qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2], \
        - qa[0]*qb[2] + qa[2]*qb[0] + qa[3]*qb[1] - qa[1]*qb[3], \
        - qa[0]*qb[3] + qa[3]*qb[0] + qa[1]*qb[2] - qa[2]*qb[1] ) );

def quatinv(q):
    """quatinv(q) => qinv   inverse of quaternion"""
    return numpy.array( ( -q[0], q[1],q[2],q[3] ) )

# x y z rx ry rz (multiplied in the virdir order, scl*rz*rx*ry*transl(x,y,z)) => Tc2world
def vds2tfm( tx,ty,tz, rx,ry,rz, scale=1 ):
    """vd2tfm( tx,ty,tz, rx,ry,rz[, scale] ) => T as 4x4 matrix"""
    t = tmul( tfm('z', rz), tfm('x', rx), tfm('y', ry), tfm(tx,ty,tz) )
    if scale != 1:
        return tmul( tfm(scale), t )
    else:
        return t

# x y z rx ry rz (multiplied in the virdir order, scl*rz*rx*ry*transl(x,y,z)) => Tc2world
def vd2tfm( tx,ty,tz, rx,ry,rz, scale=1 ):
    """vd2tfm( tx,ty,tz, rx,ry,rz ) => T as 4x4 matrix"""
    return tmul( tfm('z', rz), tfm('x', rx), tfm('y', ry), tfm(tx,ty,tz) )

# T4x4 => ( tx,ty,tz, rx,ry,rz, scale ) with rx,ry,rz applied in virdir order
def tfm2vds( T ):
    """tfm2vds(T) => ( tx,ty,tz, rx,ry,rz, scale ) with rx,ry,rz applied in virdir order, given 4x4 matrix T"""
    xyz = t2meuler( "zxy", T )
    vds = numpy.empty( 7 )
    vds[0:3] = T[3,0:3]
    vds[3:6] = xyz
    vds[7] = mag( T[0, 0:3] )
    return vds

# T4x4 => ( tx,ty,tz, rx,ry,rz ) with rx,ry,rz applied in virdir order
def tfm2vd( T ):
    """tfm2vd(T) => ( tx,ty,tz, rx,ry,rz ) with rx,ry,rz applied in virdir order, given 4x4 matrix T"""
    xyz = t2meuler( "zxy", T )
    vd = numpy.empty( 6 )
    vd[0:3] = T[3, 0:3]
    vd[3:6] = xyz
    return vd


# VirDir2 T=<tx,ty,tz>,R=[qu,<qx,qy,qz>],S=scale => 4x4 object-to-world matrix
# Input is just the numbers from the above,
#   vd22tfm( tx,ty,tz,qu,qx,qy,qz[,scale] ) => @To2w
# vd22tfm accepts either seven numbers, or a T=<x,y,z>,R=[q,<qx,qy,qz>],S=s string.
# Returns object-to-world transform.
def vd22tfm( *args ):
    """VirDir2 'T=<tx,ty,tz>,R=[qu,<qx,qy,qz>],S=scale' => 4x4 object-to-world matrix
    Input is either a string of the above form,
    or else just the numbers from the above, as in:
       vd22tfm( tx,ty,tz,qu,qx,qy,qz[,scale] )
    Returns object-to-world transform T as a 4x4 numpy array.
    """
    try:
        if len(args) == 1:
            # assume string
            pat = re.compile('T=<([^<>,]+),([^<>,]+),([^<>,]+)>,R=\[([^<>,]+),<([^<>,]+),([^<>,]+),([^<>,]+)>\],S=([^<>,]+)')
            m = pat.match( args[0] )
            tx,ty,tz, qr,qx,qy,qz, scl = [ float(m.group(i)) for i in range(1,9) ]
        elif len(args) == 7:
            tx,ty,tz, qr,qx,qy,qz = args
            scl = 1
        else:
            tx,ty,tz, qr,qx,qy,qz, scl = args
    except:
        raise ValueError("vd22tfm() expects 7 or 8 numbers, or a virdir2 'T=[tx,ty,tz],R=[qr,<qx,qy,qz>],S=scl' string, not " + str(args))

    return tmul( tfm(scl), quat2t(qr,qx,qy,qz), tfm(tx,ty,tz) )

def tfm2vd2trs( T ):
    """tfm2vd2trs(T) => string 'T=<x,y,z>,R=[qr,<qi,qj,qk>],S=scale' given 4x4 object2world matrix T"""
    scl = mag(T[0,0:3])
    quat = t2quat( T )
    return "T=<%.17g,%.17g,%.17g>,R=[%.12g,<%.12g,%.12g,%.12g>],S=%.12g" % ( T[3,0],T[3,1],T[3,2], quat[0],quat[1],quat[2],quat[3], scl )

def quat2t( qq ):
    """quat2t( [qr,qi,qj,qk] ) => 4x4 matrix T"""
    q = normalize(qq)
    x2 = q[1]*q[1]; xy = q[1]*q[2]; xz = q[1]*q[3]; xw = q[1]*q[0];
    y2 = q[2]*q[2]; yz = q[2]*q[3]; yw = q[2]*q[0];
    z2 = q[3]*q[3]; zw = q[3]*q[0];
    
    return numpy.array( [
        [ 1-2*(y2+z2),	2*(xy+zw),	2*(xz-yw),	0  ],
        [ 2*(xy-zw),	1-2*(x2+z2),	2*(yz+xw),	0  ],
        [ 2*(xz+yw),	2*(yz-xw),	1-2*(x2+y2),	0  ],
        [ 0,		0,		0,		1  ]
      ] )

def m3( T4 ):
    """m3(T) => T3 -- extracts 3x3 sub-matrix from 4x4 matrix T"""
    return T4[0:3,0:3]

def m4( T3 ):
    """m4(T3) => T -- pads 3x3 matrix to 4x4"""
    if not isinstance(T3, numpy.ndarray):
        return tfm(T3)
    if T3.shape == (3,3):
        T4 = numpy.identity(4)
        T4[0:3, 0:3] = T3
        return T4
    if T3.shape == (16,):
        return numpy.ndarray( T3, shape=(4,4) )
    return T3

def t2quat( T ):
    """t2quat(T) => quaternion [qr,qi,qj,qk] representing the rotation part of 4x4 matrix T"""
    t = m4( T )

    s = mag( t[0, 0:3] );

# A rotation matrix is
#  ww+xx-yy-zz    2(xy-wz)  2(xz+wy)
#  2(xy+wz)    ww-xx+yy-zz  2(yz-wx)
#  2(xz-wy)       2(yz+wx)  ww-xx-yy+zz
  
  # ww+xx+yy+zz = ss
    ww = (s + t[0,0] + t[1,1] + t[2,2]);	# 4 * w^2
    xx = (s + t[0,0] - t[1,1] - t[2,2]);
    yy = (s - t[0,0] + t[1,1] - t[2,2]);
    zz = (s - t[0,0] - t[1,1] + t[2,2]);

    mm = max( ww, xx, yy, zz )

    if ww == mm:
        w = numpy.sqrt(ww) * 2;	# 4w
        x = (t[2,1] - t[1,2]) / w;	# 4wx/4w
        y = (t[0,2] - t[2,0]) / w;	# 4wy/4w
        z = (t[1,0] - t[0,1]) / w;	# 4wz/4w
        w *= .25;			# w

    elif xx == mm:
        x = numpy.sqrt(xx) * 2;	# 4x
        w = (t[2,1] - t[1,2]) / x;	# 4wx/4x
        y = (t[1,0] + t[0,1]) / x;	# 4xy/4x
        z = (t[0,2] + t[2,0]) / x;	# 4xz/4x
        x *= .25;			# x

    elif yy == mm:
        y = numpy.sqrt(yy) * 2;	# 4y
        w = (t[0,2] - t[2,0]) / y;	# 4wy/4y
        x = (t[1,0] + t[0,1]) / y;	# 4xy/4y
        z = (t[2,1] + t[1,2]) / y;	# 4yz/4y
        y *= .25;			# y

    else:
        z = numpy.sqrt(zz) * 2;	# 4z
        w = (t[1,0] - t[0,1]) / z;	# 4wz/4z
        x = (t[0,2] + t[2,0]) / z;	# 4xz/4z
        y = (t[2,1] + t[1,2]) / z;	# 4yz/4z
        z *= .25;

    invs = numpy.sqrt(s);
    return (-w*invs, x*invs,y*invs,z*invs);
        
    
def ax2quat( ax, deg ):
    """ax2quat(axisname, angle_in_degrees) => quaternion  -- e.g. ax2quat('x',30)"""
    iax, = xyz2ax( ax )
    halfang = 0.5 * numpy.radians( deg )
    q = numpy.array( ( numpy.cos(halfang), 0,0,0 ) )
    q[iax+1] = numpy.sin(halfang)
    return q

def vd2quat( tx,ty,tz, rx,ry,rz, *args ):
    """vd2quat( tx,ty,tz, rx,ry,rz, ... ) => quaternion representing rx,ry,rz rotation applied in virdir's zxy order"""
    return quatmul( ax2quat('z', rz), quatmul( ax2quat('x',rx), ax2quat('y',ry) ) )

def qrotbtwn( v1, v2 ):
    """qrotbtwn( vec1, vec2 ) => quaternion which rotates vec1 into vec2 in the plane spanned by those two vectors"""
    ijk = normalize( numpy.cross( v1, v2 ) )
    cost = numpy.dot( v1, v2 ) / (mag(v1) * mag(v2))
    sinhalf = numpy.sqrt( (1-cost) * 0.5 )
    return numpy.array( ( numpy.sqrt((1+cost)*.5), sinhalf*ijk[0], sinhalf*ijk[1], sinhalf*ijk[2] ) )

def v3mmul( p3, T ):
    """v3mmul(p3, T4) returns point p3 transformed by 4x4 matrix T4"""
    return numpy.dot(p3, T[:3,:3]) + T[3,0:3]

def vmmul(p4, T):
    return numpy.dot(p4, T)

def __main__(*args):
    if 0:
        print( meuler2t('zyx', -305,-118.1,-42.8) )
        print( tfm('x',90 ) )
        print( tmul( tfm('x',90), meuler2t('zyx', -305,-118.1,-42.8) ) )

        print( t2euler( 'xyz',  euler2t('xyz', -305,-118.1,-42.8) ) )


    tfm = sys.modules[__name__]  # allows us to use "tfm.functionname()" as if we had done "import tfm".

    print("demo of tfm.py for virdir camera transformations")

    vdkey = [ 0, 0, 0, 19.57309736065, -133.8691243723, -80.39628544259 ]

    print("Original:")
    print(vdkey)

    T = tfm.vd2tfm( *vdkey ) # turn virdir TxTyTzRxRyRz into 4x4 matrix (numpy array)
                             # Note "*vdkey" expands list into separate arguments

    print("Converted back to virdir form:")
    print(tfm.tfm2vd( T ))  # turn matrix back into virdir form

    print("Just virdir-style euler angles, with t2meuler('zxy', T):")
    print(tfm.t2meuler( "zxy", T ))  # equivalent to tfm2vd() to get euler angles, applied in zxy order

    print("Just euler angles intended for a different application order, with t2meuler('xyz', T):")
    print(tfm.t2meuler( "xyz", T ))  # get euler angles intended to be applied in xyz order

if __name__ == "__main__":
    __main__()
