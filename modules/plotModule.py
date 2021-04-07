import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, lines
from mpl_toolkits.mplot3d import Axes3D
from modules.density_extraction import calc_all_dists
from mpl_toolkits.axes_grid1 import make_axes_locatable



def calc_dists(R):
  return np.concatenate([
      np.expand_dims(np.sqrt(np.sum((R[:,0,:] - R[:,1,:])**2, axis=-1)), -1),
      np.expand_dims(np.sqrt(np.sum((R[:,1,:] - R[:,2,:])**2, axis=-1)), -1),
      np.expand_dims(np.sqrt(np.sum((R[:,0,:] - R[:,2,:])**2, axis=-1)), -1)], -1)


def plot3d(geometries, ax=None, ax_lims=None, fileName="density3d.png", alpha=0.99):
  print("INP shape", geometries.shape)
  geometries_centSub = geometries - geometries[:,np.array([1]),:]
  R = calc_dists(geometries)
  print("MEAN", R.shape, np.mean(R, axis=0))
  print(np.mean(geometries, 0))
  angles = np.arccos(
      np.maximum(np.sum(geometries_centSub[:,0,:]*geometries_centSub[:,2,:], axis=-1)\
          /(R[:,0]*R[:,1]), -1))

  print(geometries[0].shape)
  xm, xM = 1.05, 1.35
  ym, yM = 1.05, 1.35
  zm, zM = np.pi*0.7, np.pi*0.8
  if ax_lims is not None:
    xm, xM = ax_lims[0]
    ym, yM = ax_lims[1]
    zm, zM = ax_lims[2]
    if isinstance(xm, list):
      ym, yM = xm
      xm, xM = xM
  #zm, zM = 1.8, 2.4
  Nhbins = 50
  Nlevels = 7

  print("LLLLLLLLLLLLLLLLLLLLLLLL")
  print(ax_lims)
  print(xm,xM)
  print(ym,yM)
  print(zm,zM)

  size=0.005
  if ax is None:
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
  X,Y,Z = R[:,0], R[:,1], R[:,2]
  X,Y,Z = R[:,0], R[:,1], angles #(angles[:] - np.mean(angles[:]))*np.sqrt(36) + angles[:]
  print("STD compare", np.std(R[:,2]), np.std(angles[:]), np.std(R[:,2])/np.std(angles[:]))

  xdt = (xM-xm)/Nhbins
  xEdges = np.linspace(xm, xM+xdt, num=Nhbins+1) - xdt/2.
  x = np.linspace(xm, xM, num=Nhbins)
  ydt = (yM-ym)/Nhbins
  yEdges = np.linspace(ym, yM+ydt, Nhbins+1) - ydt/2.
  y = np.linspace(ym, yM, Nhbins)
  zdt = (zM-zm)/Nhbins
  zEdges = np.linspace(zm, zM+zdt, Nhbins+1) - zdt/2.
  z = np.linspace(zm, zM, Nhbins)

  #ax.scatter(X,Y,Z, color='gray', s=size, alpha=alpha)

  Zhist,_,_ = np.histogram2d(X,Y, [xEdges, yEdges])
  xp,yp = np.meshgrid(x,y)
  cset = ax.contour(xp, yp, Zhist.transpose(), zdir='z',
      offset=zm, levels=Nlevels, cmap=cm.viridis)

  Xhist,_,_ = np.histogram2d(Y,Z, [yEdges, zEdges])
  yp,zp = np.meshgrid(y,z)
  zp,yp = np.meshgrid(z,y)
  cset = ax.contour(Xhist, yp, zp, zdir='x',
      offset=xm, levels=Nlevels, cmap=cm.viridis)

  Yhist,_,_ = np.histogram2d(X,Z, [xEdges, zEdges])
  zp,xp = np.meshgrid(z,x)
  cset = ax.contour(xp, Yhist, zp, zdir='y',
      offset=yM, levels=Nlevels, cmap=cm.viridis)
  #cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

  ax.xaxis.pane.fill = False
  ax.yaxis.pane.fill = False
  ax.zaxis.pane.fill = False
  ax.xaxis.pane.set_edgecolor('w')
  ax.yaxis.pane.set_edgecolor('w')
  ax.zaxis.pane.set_edgecolor('w')

  ax.set_xlabel("$NO^{(1)}$ $[\AA]$")
  ax.set_xlim([xm+xdt, xM-xdt])
  ax.set_ylabel("$NO^{(2)}$ $[\AA]$")
  ax.set_ylim([ym+ydt, yM-ydt])
  ax.set_zlabel("Angle [rad]")
  ax.set_zlim([zm+zdt, zM-zdt])

  if ax is None:
    fig.subplots_adjust(top=0.99)
    fig.subplots_adjust(bottom=0.005)
    fig.subplots_adjust(right=0.99)
    fig.subplots_adjust(left=0.005)
    fig.savefig(fileName)
  return ax


def get_ax_lims(mol, chain, ptype):
  if "symbreak" in mol:
    ang = 2.34 #2.34048
    no1 = 1.35
    no2 = 1.05
    oo = 2.21279 
  else:
    ang = 2.34052 #2.34048
    no = 1.19335
    oo = 2.19769

  if np.amax(chain[:,:2]) > 1.25 and np.amin(chain[:,:2]) < 1.12 and "symbreak" not in mol:
    ax_lims = [[1.05, 1.35], [2.1, 2.34], [0.7*np.pi, 0.8*np.pi]]
  elif "2dof" in mol:
    if ptype == "density":
      ax_lims = [[no-0.03, no+0.03], [oo-0.06, oo+0.06], [ang-0.075, ang+0.075]]
    else:
      ax_lims = [[no-0.0007, no+0.0007], [oo-0.0015, oo+0.0015], [ang-0.0017, ang+0.0017]]
  elif "symbreak" in mol:
    if ptype == "density":
      ax_lims = [[[no1-0.05, no1+0.05], [no2-0.05, no2+0.05]],\
          [oo-0.06, oo+0.06], [ang-0.075, ang+0.075]]
    else:
      ax_lims = [[[no1-0.0018, no1+0.0018], [no2-0.0018, no2+0.0018]],\
          [oo-0.0018, oo+0.0018], [ang-0.0026, ang+0.0026]]
  else:
    ax_lims = None
 
  if mol == "NO2":
    if ptype == "optimal":
      ax_lims = [[no-0.02, no+0.02], [oo-0.02, oo+0.02], [ang-0.015, ang+0.015]]
    else:
      ax_lims = [[1.05, 1.35], [2.1, 2.34], [2.3, 2.8]]


  print("OUTP", ax_lims)
  return ax_lims


def plot_molecule_ensemble(xyzFile):
  atom_types      = []
  atom_positions  = []
  with open(xyzFile) as file:
    for i,ln in enumerate(file):
      if i == 0:
        Natoms = int(ln)
      elif i > 1:
        vals = ln.split()
        print(vals)
        atom_types.append(vals[0])
        pos = [float(x) for x in vals[1:]]
        atom_positions.append(np.array([pos]))


  molecule = np.concatenate(atom_positions, axis=0)
  on1 = np.linalg.norm(molecule[0] - molecule[1])
  on2 = np.linalg.norm(molecule[2] - molecule[1])
  ang = np.pi - 2*np.arccos(molecule[0,2]/on1)
  std_r = 0.01
  std_a = 0.02

  dists = np.random.normal(size=(1000000, 3))
  dists *= np.expand_dims(np.array([std_r, std_r, std_a]), 0)
  dists += np.expand_dims(np.array([on1, on2, ang]), 0)

  NOdists = np.concatenate([dists[:,0], dists[:,1]])
  NOdists[NOdists > on1] = np.nan
  NOlg = np.nanmean(NOdists)
  NOdists = np.concatenate([dists[:,0], dists[:,1]])
  NOdists[NOdists < on1] = np.nan
  NOsm = np.nanmean(NOdists)
  print("NO mean large / small : {} / {}".format(NOlg, NOsm))

  N = dists.shape[0]
  O1 = np.array([np.zeros(N),
      dists[:,0]*np.sin((np.pi-dists[:,2])/2),
      dists[:,0]*np.cos((np.pi-dists[:,2])/2)]).transpose()
  O2 = np.array([np.zeros(N),
      dists[:,1]*np.sin((np.pi-dists[:,2])/2),
      -1*dists[:,1]*np.cos((np.pi-dists[:,2])/2)]).transpose()
  molecules = np.concatenate(
      [np.expand_dims(O1, 1), np.zeros((N, 1, 3)), np.expand_dims(O2, 1)], axis=1)

  return molecules


##########################
#####  Plot Results  #####
##########################

def plot_NO2_results(molecules, N_prj=3, input_fileName=None,
    fig_prj=None, ax_prj=None, ax_lims=None, isdensity=True,
    fit_dist=True, line_xs=None, line_ys=None, sort_by_length=False):


  isdensity = (N_prj==3)
  #plot3d(molecules,
  #    fileName=os.path.join("plots", input_fileName + "_density3d.png"))

  if sort_by_length:
    inds = np.linalg.norm(molecules[:,0,:] - molecules[:,1,:], axis=-1)\
        < np.linalg.norm(molecules[:,2,:] - molecules[:,1,:], axis=-1)
    temp = molecules[inds,0]
    molecules[inds,0] = molecules[inds,2]
    molecules[inds,2] = temp 
  all_dists = calc_all_dists(molecules)
  angles = np.arccos(
      np.sum((molecules[:,0,:] - molecules[:,1,:])\
        *(molecules[:,2,:] - molecules[:,1,:]), -1)\
        /(all_dists[:,1,0,0]*all_dists[:,2,1,0]))

  Nbins = 75
  if fig_prj is not None and ax_prj is not None:
    fig = fig_prj
    ax = ax_prj
  else:
    fig, ax = plt.subplots(1,3,figsize=(12*0.83,5*0.85), #(12,5)
        gridspec_kw={'width_ratios': [2.25, 1.5, 1]}) #[2.25, 1.5, 1]

  no_lims = [1.05, 1.35]
  oo_lims = [2.1, 2.34]
  ang_lims = [0.7*np.pi, 0.8*np.pi]
  if ax_lims is not None:
    no_lims = ax_lims[0]
    oo_lims = ax_lims[1]
    ang_lims = ax_lims[2]
    if isinstance(no_lims[0], list):
      no1_lims = no_lims[0]
      no2_lims = no_lims[1]
    else:
      no1_lims = copy(no_lims)
      no2_lims = copy(no_lims)


  mult = 100
  ax_delta = int(mult*(oo_lims[1] - oo_lims[0])/4)/mult
  while ax_delta == 0:
    mult *= 10
    ax_delta = int(mult*(oo_lims[1] - oo_lims[0])/4)/mult
  a = oo_lims[0]//ax_delta + 1
  ln = oo_lims[1]//ax_delta - a + 1
  oo_ticks = (np.arange(ln) + a)*ax_delta
  ax_delta = int(100*(no1_lims[1] - no1_lims[0])/4)/100.
  mult = 100
  while ax_delta == 0:
    mult *= 10
    ax_delta = int(mult*(no1_lims[1] - no1_lims[0])/4)/mult
  a = no1_lims[0]//ax_delta + 1
  ln = no1_lims[1]//ax_delta - a + 1
  no1_ticks = (np.arange(ln) + a)*ax_delta
  ax_delta = int(100*(no2_lims[1] - no2_lims[0])/4)/100.
  mult = 100
  while ax_delta == 0:
    mult *= 10
    ax_delta = int(mult*(no2_lims[1] - no2_lims[0])/4)/mult
  a = no2_lims[0]//ax_delta + 1
  ln = no2_lims[1]//ax_delta - a + 1
  no2_ticks = (np.arange(ln) + a)*ax_delta




  if N_prj == 3:
    h, xh, yh, _ = ax[0].hist2d(all_dists[:,1,2,0], all_dists[:,0,1,0],
        bins=[np.linspace(no2_lims[0], no2_lims[1], Nbins),
          np.linspace(no1_lims[0], no1_lims[1], Nbins)])
  elif N_prj == 2:
    h, xh, yh, _ = ax[0].hist2d(all_dists[:,0,2,0], all_dists[:,0,1,0],
        bins=[np.linspace(oo_lims[0], oo_lims[1], Nbins),
          np.linspace(no2_lims[0], no2_lims[1], Nbins)])
  h /= np.sum(h)

  x = (xh[:-1] + xh[1:])/2.
  y = (yh[:-1] + yh[1:])/2.
  am = np.argmax(h)
  rMax = am//h.shape[1]
  cMax = am%h.shape[1]
  print("NO", x[rMax], "NO", y[cMax])


  if fit_dist:
    Nd = 7
    h_div = np.expand_dims(np.expand_dims(h, 1), -1)
    h_div = np.tile(h_div, (1, Nd, 1, Nd))
    h_div = np.reshape(h_div, (h.shape[0]*Nd,h.shape[1],Nd))
    h_div = np.reshape(h_div, (h.shape[0]*Nd,h.shape[1]*Nd))/(Nd**2)
    x_delta = (x[1]-x[0])/Nd
    x_div = np.arange(x.shape[0]*Nd)*x_delta + x[0] - x_delta*Nd//2
    y_delta = (y[1]-y[0])/Nd
    y_div = np.arange(y.shape[0]*Nd)*y_delta + y[0] - y_delta*Nd//2

    yMaxs = y[np.argmax(h, axis=1)]
    mask_edges = np.cumsum(np.sum(h, axis=1))
    mask_edge_low = np.argmin(np.abs(mask_edges - 0.001))
    mask_edge_high = np.argmin(np.abs(mask_edges - 0.999))
    mask = (x > x[mask_edge_low])*(x < x[mask_edge_high])
    x_fit = x[mask]
    yMaxs = yMaxs[mask]
    ax[0].plot(x_fit, yMaxs, '-w')


    Ncoeffs = 4
    X_fit = np.array([x_fit**i for i in range(Ncoeffs)])
    X_plot = np.array([x**i for i in range(Ncoeffs)])
    if np.linalg.det(np.matmul(X_fit, np.transpose(X_fit))) == 0:
      print("Cannot plot {} due to singular matrix".format(
          input_fileName, np.sum(h), np.sum(np.isnan(h)), np.sum(h/np.sum(h))))
      if input_fileName is not None:
        plt.tight_layout()
        fileName = os.path.join("plots", input_fileName)
        fig.savefig(fileName + ".png")
      return

    fit_coeffs = np.expand_dims(np.matmul(
        np.linalg.inv(np.matmul(X_fit, np.transpose(X_fit))),
        np.matmul(X_fit, yMaxs)), -1)

    def no_no_fit(x):
        X = np.array([x**i for i in range(fit_coeffs.shape[0])])
        if len(X.shape) == 1:
            X = np.expand_dims(X, -1)
        return np.sum(fit_coeffs*X, 0)

    def der_no_no_fit(x):
        X = np.array([x**i for i in range(fit_coeffs.shape[0]-1)])
        coeffs = np.expand_dims(np.arange(fit_coeffs.shape[0]), -1)*fit_coeffs
        coeffs = coeffs[1:,:]
        if len(X.shape) == 1:
            X = np.expand_dims(X, -1)
        return np.sum(coeffs*X, 0)

    def get_normal(x0):
        m = -1./der_no_no_fit(x0)
        b = no_no_fit(x0) - m*x0
        return m, b

    y_fit_div = no_no_fit(x_div)
    def get_xIntrcpt(y):
        ind = np.argmin(np.abs(y_fit_div - y))
        x_ = np.linspace(x_div[ind]-1.5*x_delta, x_div[ind]+1.5*x_delta, 500)
        return x_[np.argmin(np.abs(no_no_fit(x_) - y))]


    no_no_relation = np.array([x,no_no_fit(x)])
    ax[0].plot(x, no_no_fit(x), '-w', linewidth=0.5)
    #ax[0].plot(x_fit, yMaxs)

    y_divT = np.tile(np.expand_dims(y_div, 0), (len(x_div), 1))

    projectionX = []
    for ix in range(x.shape[0]):
        x0,x1 = xh[ix],xh[ix+1]
        m1,b1 = get_normal(x0)
        m0,b0 = get_normal(x1)

        y1 = m1*x_div + b1
        y0 = m0*x_div + b0

        mask = (np.expand_dims(y1, -1)>=y_divT)*(np.expand_dims(y0, -1)<y_divT)
        projectionX.append(np.sum(h_div[mask]))

    projectionX = np.array(projectionX)
    projectionY = []
    for iy in range(y.shape[0]):
        y0,y1 = yh[iy],yh[iy+1]
        x0,x1 = get_xIntrcpt(y0), get_xIntrcpt(y1)
        m1,b1 = get_normal(x0)
        m0,b0 = get_normal(x1)

        y1 = m1*x_div + b1
        y0 = m0*x_div + b0

        mask = (np.expand_dims(y0, -1)>=y_divT)*(np.expand_dims(y1, -1)<y_divT)
        projectionY.append(np.sum(h_div[mask]))

    projectionY = np.array(projectionY)
  else:
    projectionX = np.sum(h, axis=1)
    projectionY = np.sum(h, axis=0)

  print("MAX PROB", x[np.argmax(projectionX)], y[np.argmax(projectionY)])

  xlim = []
  lo_size = "18%"
  divider = make_axes_locatable(ax[0])
  axt0 = divider.append_axes("top", size=lo_size, pad=0.03)
  axt0.xaxis.tick_top()
  axt0.plot(x, projectionX, '-k')
  axr0 = divider.append_axes("right", size=lo_size, pad=0.03)
  axr0.yaxis.tick_right()
  axr0.xaxis.tick_bottom()
  axr0.plot(projectionY, y, '-k')

  ax[0].yaxis.set_ticks(no1_ticks)
  if N_prj == 3:
    ax[0].xaxis.set_ticks(no2_ticks)
    axt0.set_xlim(no2_lims)
  elif N_prj == 2:
    ax[0].xaxis.set_ticks(oo_ticks)
    axt0.set_xlim(oo_lims)


  axt0.set_ylim([0, np.amax(projectionX)*1.1])
  axt0.xaxis.set_visible(False)
  axt0.yaxis.set_visible(False)
  axr0.yaxis.set_ticks([])
  #axt0.yaxis.get_major_ticks()[0].set_visible(False)
  #axt0.set_ylabel("[Arb]")
  axr0.set_ylim(no1_lims)
  axr0.set_xlim([0, np.amax(projectionY)*1.1])
  axr0.yaxis.set_visible(False)
  axr0.xaxis.set_visible(False)
  axr0.xaxis.set_ticks([])
  #axr0.xaxis.get_major_ticks()[0].set_visible(False)
  #axr0.set_xlabel("[Arb]")

  """
  axt0.xaxis.set_ticks(no_ticks)
  strticks = [str(np.round(x*1000)/1000.) for x in no_no_fit(no_ticks)]
  axt0.set_xticklabels(strticks)
  axt0.xaxis.set_label_position("top")
  axt0.set_xlabel("$\mathrm{NO}^{(2)}$  $[\AA]$")
  axr0.set_ylim(no_lims)
  axr0.yaxis.set_ticks(no_ticks)
  strticks = [str(np.round(get_xIntrcpt(x)*1000)/1000.) for x in no_ticks]
  axr0.set_yticklabels(strticks)
  axr0.set_ylabel("$\mathrm{NO}^{(1)}$ $[\AA]$")
  axr0.yaxis.set_label_position("right")
  """

  ax[0].set_aspect('equal')
  ax[0].set_ylabel("$\mathrm{NO}^{(1)}$ $[\AA]$")
  if N_prj == 3:
    ax[0].set_xlabel("$\mathrm{NO}^{(2)}$ $[\AA]$")
  elif N_prj == 2:
    ax[0].set_xlabel("$\mathrm{OO}$ $[\AA]$")


  if N_prj == 3:
    xlim = []
    divider = make_axes_locatable(ax[1])
    axt1 = divider.append_axes("top", size=lo_size, pad=0.03)
    axt1.set_xticklabels([])
    """
    axr1 = divider.append_axes("right", size=lo_size, pad=0.03)
    axr1.set_yticklabels([])
    axr1.xaxis.tick_bottom()
    """

    #x_delta = no_lims[1] - no_lims[0]
    x_lims = oo_lims
    y_lims = no1_lims
    h, xh, yh, _ = ax[1].hist2d(all_dists[:,0,2,0], all_dists[:,0,1,0],
        bins=[np.linspace(x_lims[0], x_lims[1], Nbins), np.linspace(y_lims[0], y_lims[1    ], Nbins)])
    h /= np.sum(h)
    x = (xh[:-1] + xh[1:])/2.
    y = (yh[:-1] + yh[1:])/2.

    axt1.plot(x, np.sum(h, axis=1), '-k')
    #axr1.plot(projectionX, y)
    xlim = [xh[0], xh[-1]]
    ylim = [yh[0], yh[-1]]
    ax[1].set_xlim(xlim)
    axt1.set_xlim(xlim)
    #axr1.set_ylim(ylim)

    """
    strticks = ["{0:0.4g} / {1:0.4g}".format(x,y) 
        for x,y in zip(no_ticks, no_no_fit(no_ticks))]
    ax[1].set_yticks(no_ticks)
    ax[1].set_yticklabels(strticks)
    axr1.set_yticks(no_ticks)
    """

    ax[1].yaxis.set_visible(False)
    axt1.xaxis.set_visible(False)
    axt1.yaxis.set_visible(False)
    axt1.set_xlim(oo_lims)
    axt1.set_ylim([0, np.amax(np.sum(h, axis=1))*1.1])
    #axt1.yaxis.get_major_ticks()[0].set_visible(False)
    #tck = np.floor(np.amax(np.amax(np.sum(h, axis=1)))*1000)/1000.
    #axt1.yaxis.set_ticks([tck])
    """
    axr1.yaxis.set_visible(False)
    axr1.set_ylim(no_lims)
    axr1.set_xlim([0, np.amax(projectionX)*1.05])
    axr1.xaxis.get_major_ticks()[0].set_visible(False)
    """

    ax[1].set_aspect('equal')
    ax[1].set_ylabel("$\mathrm{NO}^{(2)}$ $[\AA]$")
    ax[1].set_xlabel("$\mathrm{OO}$ $[\AA]$")


  #####  Plot Angle vs NO  #####
  #figA, ax[2] = plt.subplots()
  xlim = []
  divider = make_axes_locatable(ax[-1])
  axtA = divider.append_axes("top", size=lo_size, pad=0.03)
  axtA.xaxis.set_visible(False)
  axtA.yaxis.set_visible(False)
  #axtA.set_xticklabels([])
  #axrA = divider.append_axes("right", size=lo_size, pad=0.1)
  #axrA.set_yticklabels([])
  #axrA.xaxis.tick_top()

  x_lims = ang_lims
  y_lims = no1_lims
  print("LIMS", x_lims, y_lims, np.mean(angles))
  h, xh, yh, _ = ax[-1].hist2d(angles, all_dists[:,0,1,0],
      bins=[np.linspace(x_lims[0], x_lims[1], Nbins), np.linspace(y_lims[0], y_lims[1],     Nbins)])
  #h, xh, yh, _ = ax[2].hist2d(all_dists[:,0,2,0], angles/np.pi,
  #    bins=[np.linspace(x_lims[0], x_lims[1], Nbins), np.linspace(y_lims[0], y_lims[1]    , Nbins)])
  h /= np.sum(h)
  x = (xh[:-1] + xh[1:])/2.
  y = (yh[:-1] + yh[1:])/2.

  print("MAX PROB", x[np.argmax(np.sum(h, axis=1))])
  axtA.plot(x, np.sum(h, axis=1), '-k')
  #axrA.plot(np.sum(h, axis=0), y)
  xlim = [xh[0], xh[-1]]
  ylim = [yh[0], yh[-1]]
  ax[-1].set_xlim(xlim)
  ax[-1].yaxis.set_visible(False)
  axtA.set_xlim(xlim)
  axtA.set_ylim([0, np.amax(np.sum(h, axis=1))*1.1])
  #tck = np.floor(np.amax(np.amax(np.sum(h, axis=1)))*1000)/1000.
  #axtA.yaxis.set_ticks([tck])
  #axrA.set_ylim(ylim)
  ax[-1].set_xlabel("$\mathrm{O}\mathrm{N}\mathrm{O}$ Angle [rad]")
  #ax[2].set_xlabel("$\mathrm{OO}$  $[\AA]$")

  if line_xs is not None:
    xs = line_xs
    ys = line_ys
  else:
    xs = [1.05, 1.4056]
    ys = [no_ticks[0], 1.4056]

  lns = []
  for i,yv in enumerate(no1_ticks):
    if i > 0 and i < len(no2_ticks)-1:
      print("aslkfjasdlfjsd")
      if N_prj == 3:
        lns.append(
            lines.Line2D(xs, [yv, yv], transform=ax[0].transData, figure=fig,
              linestyle=":", color="k", alpha=0.75))
        lns.append(
            lines.Line2D(oo_lims, [yv, yv], transform=ax[1].transData, figure=fig,
              linestyle=":", color="k", alpha=0.75))
        lns.append(
            lines.Line2D(ang_lims, [yv, yv], transform=ax[2].transData, figure=fig,
              linestyle=":", color="k", alpha=0.75))
      elif N_prj == 2:
        lns.append(
            lines.Line2D(oo_lims, [yv, yv], transform=ax[0].transData, figure=fig,
              linestyle=":", color="k", alpha=0.75))
        lns.append(
            lines.Line2D(ang_lims, [yv, yv], transform=ax[1].transData, figure=fig,
              linestyle=":", color="k", alpha=0.75))

  """
  if line_xs is not None:
    xs = line_xs
    ys = line_ys
  else:
    xs = [1.05, 1.4056]
    ys = [no_ticks[0], 1.4056]
  lns = []
  for i,yv in enumerate(no_ticks):
    if i > 0 and i < len(no_ticks)-1:
      lns.append(
          lines.Line2D(xs, [yv, yv], transform=ax[0].transData, figure=fig,
            linestyle=":", color="k", alpha=0.75))
      lns.append(
          lines.Line2D(oo_lims, [yv, yv], transform=ax[1].transData, figure=fig,
            linestyle=":", color="k", alpha=0.75))
      lns.append(
          lines.Line2D(ang_lims, [yv, yv], transform=ax[2].transData, figure=fig,
            linestyle=":", color="k", alpha=0.75))
    if yv > np.amax(no_no_relation[1,:]) or yv < np.amin(no_no_relation[1,:]):
      continue
    x_int = get_xIntrcpt(yv)
    lns.append(
        lines.Line2D([x_int, x_int], ys, linestyle=":",
          transform=ax[0].transData, figure=fig, color="k", alpha=0.75))
  fig.lines.extend(lns)
  """


  if input_fileName is not None:
    plt.tight_layout()
    fileName = os.path.join("plots", input_fileName)
    fig.savefig(fileName + ".png")

  if N_prj == 3:
    return (fig, ax, axt0, axr0, axt1, axtA)
  elif N_prj == 2:
    return (fig, ax, axt0, axr0, axtA)

