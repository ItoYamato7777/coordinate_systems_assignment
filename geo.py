from math import *
import numpy as np

def sign(x) :
  if x > 0 :
    return 1
  elif x < 0 :
    return -1
  else :
    return 0

def arctan2(y, x) :
  return atan2(y, x)

class VECTOR(list) :
  def dot(self, other) :
    return(self[0]*other[0] +
           self[1]*other[1] +
           self[2]*other[2])
  def __neg__(self) :
    tmp = VECTOR();
    for i in [0,1,2] :
      tmp[i] = - self[i]
    return(tmp)
  def __add__(self, other) :
    tmp = VECTOR();
    for i in [0,1,2] :
      tmp[i] = self[i] + other[i]
    return(tmp)
  def __sub__(self, other) :
    tmp = VECTOR();
    for i in [0,1,2] :
      tmp[i] = self[i] - other[i]
    return(tmp)
  def __mul__(self, other) :
    tmp = VECTOR();
    tmp[0] = self[1]*other[2] - self[2]*other[1]
    tmp[1] = self[2]*other[0] - self[0]*other[2]
    tmp[2] = self[0]*other[1] - self[1]*other[0]
    return(tmp)
  def __rmul__(self, other) :
    tmp = VECTOR();
    if isinstance(other, VECTOR) :
      tmp[0] = other[1]*self[2] - other[2]*self[1]
      tmp[1] = other[2]*self[0] - other[0]*self[2]
      tmp[2] = other[0]*self[1] - other[1]*self[0]
    elif isinstance(other, MATRIX) :
      pass
    else :
      for i in [0,1,2] :
        tmp[i] = other * self[i]
    return(tmp)
  def __abs__(self) :
    return(sqrt(self.dot(self)))
  def abs(self) :
     return(sqrt(self.dot(self)))
  def __repr__(self) :
    return("v:" + list.__repr__(self))
  def __init__(self, x=0.0, y=0.0, z=0.0, vec=[]) :
    list.__init__(self)
    self.append(float(x))
    self.append(float(y))
    self.append(float(z))
    if vec != []:
      for i in [0,1,2] :
        self[i] = vec[i]
  def toarray(self):
    return np.array([self[0], self[1], self[2]])

class MATRIX(list) :
  def col(self, idx, arg=[]) :
    tmp = VECTOR()
    if arg :
      for i in [0,1,2] :
        self[i][idx] = arg[i]
    for i in [0,1,2] :
      tmp[i] = self[i][idx]
    return(tmp)
  def row(self, idx, arg=[]) :
    tmp = VECTOR()
    if arg :
      for i in [0,1,2] :
        self[idx][i] = arg[i]
    for i in [0,1,2] :
      tmp[i] = self[idx][i]
    return(tmp)
  def rot_axis(self) :
    axis = VECTOR()
    co=(self[0][0]+self[1][1]+self[2][2]-1.0)/2.0;
    if co <= -1.0 :
      angl = pi;
      tmp=min(1.0,max(0,(self[0][0] + 1.0)/2.0))
      axis[0] = sqrt(tmp)
      tmp=min(1.0,max(0,(self[1][1] + 1.0)/2.0))
      axis[1] = sign(self[0][1])*sqrt(tmp)
      tmp=min(1.0,max(0,(self[2][2] + 1.0)/2.0))
      axis[2] = sign(self[0][2])*sqrt(tmp)
      si=0.0
    elif co < 1.0 :
      axis[0] = self[2][1] - self[1][2]
      axis[1] = self[0][2] - self[2][0]
      axis[2] = self[1][0] - self[0][1]
      an = abs(axis)
      if(an != 0.0) :
        for i in [0,1,2] :
          axis[i] = axis[i]/an
          si = an/2.0
        angl = arctan2(si,co);
      else :
        angl = 0.0
        axis[0] = 1.0
        axis[1] = 0.0
        axis[2] = 0.0
    else :
      angl = 0.0
      axis[0] = 1.0
      axis[1] = 0.0
      axis[2] = 0.0
    return([angl,axis])
  def abc(self) :
    if self[0][2]>= 1.0 :
      b = pi/2;
      a = 0.0;
      c = arctan2(self[2][1],self[1][1])
    elif self[0][2] <= -1.0 :
      b = -pi/2;
      a = 0.0;
      c = arctan2(self[1][0],self[2][0])
    else :
      b = asin(self[0][2]);
      a = arctan2(- self[1][2],self[2][2]);
      c = arctan2(- self[0][1],self[0][0]);
    return([a, b, c])
  def rpy(self):
    return [-x for x in self.trans().abc()]
  def trans(self) :
    tmp = MATRIX()
    for i in [0,1,2] :
      for j in [0,1,2] :
        tmp[i][j] = self[j][i]
    return(tmp)
  def __neg__(self) :
    return(self.trans())
  def __mul__(self, other) :
    tmp = None
    if isinstance(other, MATRIX) :
      tmp = MATRIX()
      for i in [0,1,2] :
        for j in [0,1,2] :
          tmp[i][j] = (self[i][0] * other[0][j] +
                       self[i][1] * other[1][j] +
                       self[i][2] * other[2][j])
    elif isinstance(other, VECTOR) :
      tmp = VECTOR()
      for i in [0,1,2] :
        tmp[i] = (self[i][0] * other[0] +
                  self[i][1] * other[1] +
                  self[i][2] * other[2])
    return(tmp)
  def __repr__(self) :
    return("m:" + list.__repr__(self))
  def __init__(self, mat=[], a=0.0, b=0.0, c=0.0, angle=0.0, axis=[]) :
    list.__init__(self)
    self.append([1.0, 0.0, 0.0])
    self.append([0.0, 1.0, 0.0])
    self.append([0.0, 0.0, 1.0])
    if mat != [] :
      for i in [0,1,2] :
        for j in [0,1,2] :
          self[i][j] = mat[i][j]
    elif a != 0.0 :
      self[0][0] = 1.0;
      self[1][1] = cos(a);
      self[2][2] = cos(a);
      self[1][2] = -sin(a);
      self[2][1] = sin(a);
    elif b != 0.0 :
      self[1][1] = 1.0;
      self[2][2] = cos(b);
      self[0][0] = cos(b);
      self[2][0] = -sin(b);
      self[0][2] = sin(b);
    elif c != 0.0 :
      self[2][2] = 1.0;
      self[0][0] = cos(c);
      self[1][1] = cos(c);
      self[0][1] = -sin(c);
      self[1][0] = sin(c);
    if axis :
      len = abs(axis)
      if len != 0.0 and angle != 0.0 :
        co = cos(angle)
        si = sin(angle)
        atmp = (1.0/len) * axis
        self[0][0] = atmp[0]*atmp[0]*(1.0 - co) + co
        self[1][0] = atmp[1]*atmp[0]*(1.0 - co) + atmp[2]*si
        self[2][0] = atmp[2]*atmp[0]*(1.0 - co) - atmp[1]*si
        self[0][1] = atmp[0]*atmp[1]*(1.0 - co) - atmp[2]*si
        self[1][1] = atmp[1]*atmp[1]*(1.0 - co) + co
        self[2][1] = atmp[2]*atmp[1]*(1.0 - co) + atmp[0]*si
        self[0][2] = atmp[0]*atmp[2]*(1.0 - co) + atmp[1]*si
        self[1][2] = atmp[1]*atmp[2]*(1.0 - co) - atmp[0]*si
        self[2][2] = atmp[2]*atmp[2]*(1.0 - co) + co
  def toarray(self):
    return np.array([self[0],self[1],self[2]])

class FRAME :
  def xyzabc(self) :
    tmp = self.mat.abc()
    return([self.vec[0],self.vec[1],self.vec[2],
            tmp[0],tmp[1],tmp[2]])
  def xyzrpy(self):
    r,p,y = self.mat.rpy()
    return [self.vec[0], self.vec[1], self.vec[2], r, p, y]
  def __neg__(self) :
    return(FRAME(mat=(-self.mat),vec=(-self.mat)*(-self.vec)))
  def __mul__(self, other) :
    tmp = None
    if isinstance(other, FRAME) :
      tmp_mat = self.mat * other.mat
      tmp_vec = (self.mat * other.vec) + self.vec
      tmp = FRAME(mat=tmp_mat, vec=tmp_vec)
    elif isinstance(other, VECTOR) :
      tmp = (self.mat * other) + self.vec
    return(tmp)
  def __repr__(self) :
    return("f:(" + repr(self.mat) + "," + repr(self.vec) + ")")
  def __init__(self, frm=[], mat=[], vec=[], xyzabc=[], xyzrpy=[]) :
    if isinstance(frm, np.ndarray):
      self.mat = MATRIX(mat = frm[0:3,0:3])
      self.vec = VECTOR(vec = frm[0:3,3])
    elif frm :
      self.mat = MATRIX(mat=frm.mat)
      self.vec = VECTOR(vec=frm.vec)
    elif xyzabc :
      self.vec = VECTOR(xyzabc[0],xyzabc[1],xyzabc[2])
      tmp_a = MATRIX(a=xyzabc[3])
      tmp_b = MATRIX(b=xyzabc[4])
      tmp_c = MATRIX(c=xyzabc[5])
      self.mat = tmp_a * tmp_b * tmp_c
    elif xyzrpy :
      self.vec = VECTOR(xyzrpy[0],xyzrpy[1],xyzrpy[2])
      tmp_r = MATRIX(a=xyzrpy[3])
      tmp_p = MATRIX(b=xyzrpy[4])
      tmp_y = MATRIX(c=xyzrpy[5])
      self.mat = tmp_y * tmp_p * tmp_r
    else :
      self.mat = MATRIX(mat=mat)
      self.vec = VECTOR(vec=vec)
  def toarray(self):
    return np.array([self.mat[0] + [self.vec[0]],
                     self.mat[1] + [self.vec[1]],
                     self.mat[2] + [self.vec[2]],
                     [0.,0.,0.,1.,]])