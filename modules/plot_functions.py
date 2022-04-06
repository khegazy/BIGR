import corner
import numpy as np
import scipy as sp
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib import cm, lines, patches, ticker
from copy import copy


def plot3d(samples, ranges, ax=None, scat_inds=None, fileName="density3d.png",
    alpha=0.99, labels=None, label_size=10, tick_size=None):
  """
  Plot the 3d distribution of the theta (model) parameters in a scatter plot while
  projecting the distribution onto the walls of the plots using contours

      Parameters
      ----------
      samples : 2D np.array of type float [N,theta]
          The cartesian coordinates of points sampled from the distribution to plot
      ranges : list of tuples [(min,max),...]
          The plotted range for each theta parameter
      ax : pyplot axis
          The 3d pyplot axis instance to plot on, if None then make one
      scat_inds : 1D np.array of type int [N]
          The indices of which samples to use, if None then use all of them
      fleName : string
          The output file name, "densty3d.png" by default
      alpha : float
          The alpha parameter of the scatter plot that determines the opacity
      labels : list of strings
          The axes labels for each theta parameter
      label_size : float
          The text size of the axes labels
      tick_size : float
          The text size of the tick labels
  """
  
  ax.view_init(elev=28., azim=37)
  if labels is None:
      labels = ["d1", "d2", "d3"]

  if ranges is None:
    raise RuntimeError("Must specify ranges for 3d plot, but found None")

  xm, xM = ranges[0]
  ym, yM = ranges[1]
  zm, zM = ranges[2]
  Nhbins = 50
  Nlevels = 7

  size=0.005
  if ax is None:
      fig = plt.figure(figsize=(10,10))
      ax = fig.gca(projection='3d')
  X,Y,Z = samples[:,0], samples[:,1], samples[:,2] #(angles[:] - np.mean(angles[:]))*np.sqrt(36) + angles[:]

  print(np.mean(X), np.mean(Y), np.mean(Z))
  xdt = (xM-xm)/Nhbins
  xEdges = np.linspace(xm, xM+xdt, num=Nhbins+1) - xdt/2.
  x = np.linspace(xm, xM, num=Nhbins)
  ydt = (yM-ym)/Nhbins
  yEdges = np.linspace(ym, yM+ydt, Nhbins+1) - ydt/2.
  y = np.linspace(ym, yM, Nhbins)
  zdt = (zM-zm)/Nhbins
  zEdges = np.linspace(zm, zM+zdt, Nhbins+1) - zdt/2.
  z = np.linspace(zm, zM, Nhbins)

  if scat_inds is not None:
      print("SSS", size)
      ax.scatter(X[scat_inds], Y[scat_inds], Z[scat_inds], color='gray', s=size, alpha=alpha)
  else:
      print("SSS", X.shape, Y.shape, Z.shape)
      ax.scatter(X,Y,Z, color='gray', s=size, alpha=alpha)

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
      offset=ym, levels=Nlevels, cmap=cm.viridis)
  #cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
  

  ax.xaxis.pane.fill = False
  ax.yaxis.pane.fill = False
  ax.zaxis.pane.fill = False
  ax.xaxis.pane.set_edgecolor('w')
  ax.yaxis.pane.set_edgecolor('w')
  ax.zaxis.pane.set_edgecolor('w')

  ax.set_xlabel(labels[0], fontsize=label_size, labelpad=13.5)
  ax.set_xlim([xm+xdt, xM-xdt])
  ax.set_ylabel(labels[1], fontsize=label_size, labelpad=13.5)
  ax.set_ylim([ym+ydt, yM-ydt])
  ax.set_zlabel(labels[2], fontsize=label_size, labelpad=13.5)
  ax.set_zlim([zm+zdt, zM-zdt])
  
  if tick_size is not None:
      ax.tick_params(axis='x', labelsize=tick_size)
      ax.tick_params(axis='y', labelsize=tick_size)
      ax.tick_params(axis='z', labelsize=tick_size)

  if ax is None:
      fig.subplots_adjust(top=0.99)
      fig.subplots_adjust(bottom=0.005)
      fig.subplots_adjust(right=0.99)
      fig.subplots_adjust(left=0.005)
      fig.savefig(fileName)
  
  return ax


def corner_column(fig, dims, samples, modes=None, centers=None, 
    ranges_proj=None, ranges_3d=None, labels_proj=None, labels_3d=None,
    set_ticks=None, label_size=10, tick_size=10, do_3d=None, vdims_3d=None):
  """
  Make a corner plot comparison while addng the 3d theta distribution.

      Parameters
      ----------
      fig : instance of matplotlib.figure
          The figure instance to plot on
      dims : Ntuple of floats
          The dimension parameters that determine the plot dimension
          vdim : vertical dimension of corner plot panels
          hdim : horizontal dimension of corner plot panels
          vsSep : vertical seperation between corner plots of different samples
          vpSep : vertical seperation between corner plot panels
          hpSep : horizontal seperation between corner plot panels
          bp : bottom pad
          tp : top pad
          lp : left pad
      samples : A list of 2D np.arrays of type float [[N,theta], ...]
          A list of the theta parameter chains that will be plotted as
          a corner plot and possibly a 3d distribution
      modes : A list of 1D arrays of type float [[thetas], ...]
          The modes of every theta parameter for each sample, if not None they
          are plotted onto the corner plot
      centers : list of type float [thetas]
          The expected results of each theta parameter
      ranges_proj : list of 2D np.array of type float [[theta,2], ...]
          The corner plot axis ranges for every theta parameter for each sample
      ranges_3d : 2D np.array of type float [[theta,2], ...]
          The 3D plot axis ranges for every theta parameter
      labels_proj : 2D list of strings [samples,theta]
          The corner plot axis labelss for every theta parameter for each sample
      labels_3d : 2D list of strings [samples,theta]
          The 3D plot axis labels for every theta parameter
      set_ticks : list of booleans [samples]
          If True then set ticks to pre-assigned values for that sample
      label_size : float
          The text size of the axis labels
      tick_size : float
          The text size of the axis ticks
      do_3d : list of booleans [samples]
          If not None then when True for a sample it plots the 3d theta
          distribution as an inset
      vdims_3d : list of tuples [(y_start,yd), ...]
          If not None then specifies the lower bound of the inset for the 3d
          plot (y_start) and its heigth (yd)

      Returns
      -------
      all_axes : list of pyplot.axis
          A lst of all the axes created and plotted
  """

    vdim, hdim, vsSep, vpSep, hpSep, bp, tp, lp = dims
    lw, letters, ltr_c = 2, ['a', 'b', 'c', 'd', 'e', 'f'], 0
    if centers is None:
        centers = [None,]*len(samples)
    if do_3d is None:
        do_3d = [True,]*len(samples)
    if set_ticks is None:
        set_ticks = [True,]*len(samples)
    if ranges_proj is None:
        ranges_proj = [None,]*len(samples)
    if labels_proj is None:
        labels_proj= []
        for i in range(len(samples)):
            labels_proj.append(["Theta {}".format(i)\
                for i in range(len(samples[i].shape[-1]))])

    centers_, do_3d_, set_ticks_ = copy(centers), copy(do_3d), copy(set_ticks)
    ranges_proj_, labels_proj_ = copy(ranges_proj), copy(labels_proj)
    
    all_axs = []
    N_thetas = samples[0].shape[-1]
    samples.reverse()
    if modes is not None:
        modes.reverse()
    centers_.reverse()
    do_3d_.reverse()
    ranges_proj_.reverse()
    labels_proj_.reverse()
    set_ticks_.reverse()
    for s, sr in enumerate(samples):
        centers, do_3d, set_ticks = centers_[s], do_3d_[s], set_ticks_[s]
        ranges_proj, labels_proj = ranges_proj_[s], labels_proj_[s]
        
        # 3d
        if samples[0].shape[-1] > 3 and do_3d:
            if modes[s] is None:
                srTD, plt_thetas = [], []
                inds = np.arange(len(sr))
                rnd.shuffle(inds)
                for i in inds[:200]:
                    thetas_ = rnd.normal(0, 1, size=(1000, 3))\
                        *np.array([sr[i,1], sr[i,3], sr[i,5]])
                    thetas_ = thetas_ + np.array([sr[i,0], sr[i,2], sr[i,4]])
                    plt_thetas.append(copy(thetas_))
                srTD = np.concatenate(plt_thetas)
            else:
                srTD = rnd.normal(0, 1, size=(100000, 3))\
                        *np.array([modes[s][1], modes[s][3], modes[s][5]])
                srTD = srTD + np.array([modes[s][0], modes[s][2], modes[s][4]])

            if len(samples) == 3:
                oy = 0.175#0.18
                yd = 0.165
            elif len(samples) == 2:
                oy = 0.283
                yd = 0.24
            elif len(samples) == 1:
                oy = 0.59
                yd = 0.43
            else:
                oy = 0.58
                yd = 0.43
            y_start = (1./len(samples))*s+oy
            
            if vdims_3d is not None:
                y_start, yd = vdims_3d[s]
            ax3d = fig.add_axes((0.54, y_start, 0.43, yd), projection='3d')

            plot3d(np.array(srTD), ranges_3d, ax=ax3d, labels=labels_3d,
                   label_size=label_size*1.1, tick_size=tick_size*1.1, alpha=0.2)

            ax3d.text(0.4, 0.81, 4.1, s="P$^{(gauss)}(\mathbf{r},\Theta^*|C)$",
                transform=ax3d.transAxes, fontsize=22)
            #tmp_planes = ax3d.zaxis._PLANES 
            #ax3d.zaxis._PLANES = ( tmp_planes[3], tmp_planes[2], 
            #             tmp_planes[0], tmp_planes[1], 
            #             tmp_planes[4], tmp_planes[5])

        vert_anch = []
        extra_v = 0
        for v in range(N_thetas):
            #vert_anch.append(((1.-tp)/len(samples)-4*vpSep)*s + s*4*vpSep + bp + v*(vdim + vpSep))
            vert_anch.append((N_thetas*vdim + (N_thetas-1)*vpSep)*s + s*vsSep + bp + v*(vdim + vpSep))

        horz_anch = [lp]
        for h in range(N_thetas):
            horz_anch.append(horz_anch[-1] + (hdim + hpSep))

        axs0 = []
        for v in range(N_thetas):
            axs0.append([])
            for h in range(N_thetas):
                axs0[-1].append(fig.add_axes((horz_anch[h], vert_anch[v], hdim, vdim)))

        if ranges_proj is None:
            bins = [51,]*len(sr)
        else:
            bins = []
            for ir in ranges_proj:
                bins.append(np.linspace(ir[0], ir[1], 51))
        for h in range(N_thetas):
            for v in range(N_thetas):
                if h + v == N_thetas-1:
                    n, x = np.histogram(sr[:,-1*v-1], bins=bins[-1*v-1])
                    n = n/np.amax(n)
                    axs0[v][h].bar((x[1:]+x[:-1])/2., n, width=x[1]-x[0], align='center')
                    #axs0[v][h].plot((x[1:]+x[:-1])/2., n, '--k')

                    if centers is not None:
                        axs0[v][h].plot([centers[-1*v-1], centers[-1*v-1]],
                            [0,1.1], '-k', linewidth=lw)
                    if modes[s] is not None:
                        axs0[v][h].plot([modes[s][-1*v-1], modes[s][-1*v-1]], [0,1.1],
                                ':', color='red', linewidth=lw)
                    axs0[v][h].set_xlim([x[0], x[-1]])
                    axs0[v][h].set_ylim([0, 1.1*np.amax(n)])

                elif h + v > N_thetas-1:
                    axs0[v][h].xaxis.set_visible(False)
                    axs0[v][h].yaxis.set_visible(False)
                    axs0[v][h].set_visible(False)
                else:
                    counts, x, y,image = axs0[v][h].hist2d(sr[:,h], sr[:,-1*v-1],
                        bins=(bins[h], bins[-1*v-1]))
                    if centers is not None:
                        axs0[v][h].scatter(centers[h], centers[-1*v-1],
                            marker='x', color='k', s=50, zorder=100)
                    #axs0[v][h].contour(counts.transpose(), levels=4,
                    #    extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1, colors='lightgray')
                    #    #extent=[bins[-1*v-1].min(),bins[-1*v-1].max(),bins[h].min(),bins[h].max()],linewidths=1)
                    if modes[s] is not None:
                        axs0[v][h].plot([modes[s][h],]*2, [y[0], y[-1]],
                                ":", color="red", linewidth=lw)
                        axs0[v][h].plot([x[0], x[-1]], [modes[s][-1*v-1],]*2,
                                ":", color="red", linewidth=lw)

                if v > 0:
                    axs0[v][h].xaxis.set_visible(False)
                else:
                    if N_thetas > 4:
                        axs0[v][h].set_xlabel(labels_proj[h], fontsize=label_size, rotation=13)
                    else:
                        axs0[v][h].set_xlabel(labels_proj[h], fontsize=label_size)
                    axs0[v][h].tick_params(axis='x', labelsize=tick_size)
                    if set_ticks is not None:
                        if set_ticks:
                            if h == 0:
                                axs0[v][h].set_xticks([1.346, 1.354])
                            elif (N_thetas==6 and h==2) or (N_thetas==3 and h==1):
                                axs0[v][h].set_xticks([1.046, 1.054])
                            elif (N_thetas==6 and h==4) or (N_thetas==3 and h==2):
                                axs0[v][h].set_xticks([2.336, 2.344])
                            else:
                                axs0[v][h].set_xticks([0.05])
                if h > 0 or v == N_thetas:
                    axs0[v][h].yaxis.set_visible(False)
                elif h < N_thetas and v < N_thetas:
                    if N_thetas > 4:
                        axs0[v][h].set_ylabel(labels_proj[-1*v-1], fontsize=label_size, labelpad=20, rotation=75)
                    else:
                        axs0[v][h].set_ylabel(labels_proj[-1*v-1], fontsize=label_size)
                    axs0[v][h].tick_params(axis='y', labelsize=tick_size)
                if v == N_thetas and h == 0:
                    axs0[v][h].tick_params(axis='y', labelsize=tick_size)
        
        if len(samples) > 1:
            if N_thetas > 4:
                if samples[0].shape[-1] > 3 and do_3d:
                    ax3d.text(0.5, 0.41, 3.95, s=letters[len(samples)+np.sum(do_3d_)-1-ltr_c],
                        transform=ax3d.transAxes, fontsize=30)
                    ltr_c += 1
                axs0[-1][0].text(-1.15, 0.7, s=letters[len(samples)+np.sum(do_3d_)-1-ltr_c],
                        transform=axs0[-1][0].transAxes, fontsize=30)
                ltr_c += 1
                print("letter 0", ltr_c, samples[0].shape[-1] > 3 and do_3d, np.sum(do_3d_))
                
            else:
                axs0[-1][0].text(2.8, 0.7, s=letters[len(samples)-1-ltr_c],
                        transform=axs0[-1][0].transAxes, fontsize=30)
                ltr_c += 1

                
        all_axs.append(copy(axs0))
    
    return all_axs


def plot_trends_single(x, precision, modes, centers, x_label, corrs=None,
    labels=None, ylim=None, label_size=10, tick_size=10, colors=['b', 'r', 'k']):
  """
  Plots the precision, modes, and correlations for the retrieved marginalized posterior
  as a function of the input independent variable x

      Parameters
      ----------
      x : 1D array-like of floats [N]
          The independent variable being varied to evaluate the precision, ...
      precision : 2D np.array of type float [N,thetas]
          The marginal likelihood precision values as a function of x
      modes : 2D np.array of type float [N,thetas]
          The modes of the marginal likelihood as a function of x
      centers : list of type float [thetas]
          The expected results of each theta parameter
      x_label : string
          The label of the x axis
      corrs : 2D np.array of type float [N,thetas]
          If not None, then plot the correlations between all theta parameters
      labels : list of strings [thetas]
          The name of each theta parameter
      ylim : array-like of type float [2]
          If not None, then specifies the range of the y axis
      label_size : float
          The text size of the axis labels
      tick_size : float
          The text size of the axis ticks
      colors : list of strings
          The colors associated to each degree of freedom and corresponding
          width

      Returns
      -------
      fig : matplotlb.Figure instance
          The figure this plot was plotted on
  """
    
  if corrs is None:
      fig, ax = plt.subplots(2, 1, 
          gridspec_kw={'height_ratios': [4, 1], 'hspace':0.02},
          figsize=(13,8))
  else:
      fig, ax = plt.subplots(3, 1,
          gridspec_kw={'height_ratios': [4, 1, 1], 'hspace':0.02},
          figsize=(13,8*7./6))

  if labels is None:
    labels = ["Theta {}".format(i) for i in range(precision.shape[1])]
      
  for j in range(precision.shape[1]):
      if precision.shape[-1] > 3:
          clr = colors[j//2]
          if j%2 == 1:
              ltp = '--'
          else:
              ltp = '-'
      else:
          ltp = '-'
          clr = colors[j]

      #lbl = lbl[:lbl.find("$ ")+1]
      
      ax[0].plot(x, precision[:,j], ltp+clr, label=labels[j])
      ax[1].plot(x, 100*np.abs((modes[:,j]-centers[j])/centers[j]), ltp+clr, label=labels[j])

  if corrs is not None:
      ax[2].plot(x, corrs, ':g', label="Correlations")
      ax[2].set_xlim([x[0], x[-1]])
      ax[2].set_ylabel("Correlation", fontsize=label_size*1.2)
      ax[2].set_xlabel(x_label, fontsize=label_size*1.2)
      ax[2].tick_params(axis='x', labelsize=tick_size*1.2)
      ax[2].tick_params(axis='y', labelsize=tick_size*1.2)
      ax[2].set_yscale('log')
      ax[1].xaxis.set_visible(False)
  else:
      ax[1].set_xlabel(x_label, fontsize=label_size*1.2)
      ax[1].tick_params(axis='x', labelsize=tick_size*1.2)
  

  ax[0].xaxis.set_visible(False)
  ax[0].set_xlim([x[0], x[-1]])
  ax[1].set_xlim([x[0], x[-1]])
  
  ax[0].set_ylabel("$\sigma^{(\Theta)}$ $[\AA]$", fontsize=label_size*1.2)
  ax[1].set_ylabel(r"$\Theta^*$ Error [%]", fontsize=label_size*1.2)

  ax[0].tick_params(axis='y', labelsize=tick_size*1.2)
  ax[0].set_yscale('log')
  ax[1].tick_params(axis='y', labelsize=tick_size*1.2)
  ax[1].set_yscale('log')
  if ylim is not None:
      ax[0].set_ylim(ylim)

  #min,ymax = ax[0].get_ylim()
  #ax[0].set_ylim(ymin-0.2*(ymax-ymin), ymax)

  ax[0].legend(loc='lower left', fontsize=15, ncol=3)

  plt.tight_layout()
  return fig




def column_compare_dists(samples, centers, ranges, labels, xticks=None):
  """
  """
  Ns = len(samples)
  Ncrn = samples[0].shape[-1]
  do_3d = 1
  if Ncrn == 2:
      do_3d = 0
  top_pad, bot_pad = 0.04, 0.08
  vert_pad, dns_vert = 0.005, 0.27*2/Ns
  vert_dim = (1 - (bot_pad + top_pad + (8+(Ns-1)+Ncrn)*vert_pad + do_3d*Ns*dns_vert))/3

  left_pad, right_pad, horz_pad = 0.075, 0.04, 0.015
  cbar_pad, cbar_sz = 0.01, 0.27
  dns_horz, dns_horz_pad = 0.27, 0.045
  horz_dim = (1 - (left_pad + right_pad + 3*horz_pad))/3

  dns_labels = ["$P_{opt} ( \mathbf{r}^{(mf)} )$ from a single molecule",
      "Input Ensemble $|\psi(\mathbf{r}^{(mf)})|^2$",
      "$P_{opt} ( \mathbf{r}^{(mf)} )$ from an ensemble"]
  fig_ratio = vert_dim/horz_dim
  sz=6
  fig = plt.figure(figsize=(sz,sz/fig_ratio))

  axs = []
  vert = bot_pad
  for v in range(Ncrn+Ns):
      if v < Ncrn:
          axs.append([])
          horz = left_pad
          for h in range(Ncrn):
              axs[-1].append(fig.add_axes((horz, vert, horz_dim, vert_dim)))
              horz += horz_dim + horz_pad
          vert += vert_dim + vert_pad
      elif do_3d:
          if v == Ncrn and Ns > 1:
              print("WWWWWWTTTTTTTTTFFFFFFFF")
              vert += 8*vert_pad
          axs.append(fig.add_axes(
              (0.75*left_pad, vert, 1-(left_pad+right_pad), dns_vert), projection='3d'))
          vert += dns_vert + vert_pad

  bins = []
  for ir in ranges:
      bins.append(np.linspace(ir[0], ir[1], 51))
  for v in range(Ncrn+Ns):
      if v < Ncrn:
          for h in range(Ncrn):
              if h + v == Ncrn-1:
                  print("BINS", bins[-1*v-1][0], bins[-1*v-1][-1])
                  n, _, _ = axs[v][h].hist(samples[0][:,-1*v-1], bins=bins[-1*v-1], density=True)
                  if Ns > 1:
                      nn, _ = np.histogram(samples[1][:,-1*v-1], bins=bins[-1*v-1], density=True)
                      axs[v][h].plot((bins[-1*v-1][1:]+bins[-1*v-1][:-1])/2, nn, '--k')
                  axs[v][h].plot([centers[-1*v-1], centers[-1*v-1]], [0,1.25*np.amax(n)], '-k')
                  axs[v][h].set_xlim([bins[-1*v-1][0], bins[-1*v-1][-1]])
                  axs[v][h].set_ylim([0, 1.1*np.amax(n)])
              elif h + v > Ncrn-1:
                  axs[v][h].xaxis.set_visible(False)
                  axs[v][h].yaxis.set_visible(False)
                  axs[v][h].set_visible(False)
              else:
                  counts, x, y,image = axs[v][h].hist2d(samples[0][:,h], samples[0][:,-1*v-1],
                      bins=(bins[h], bins[-1*v-1]))
                  if Ns > 1:
                      counts, x, y = np.histogram2d(samples[1][:,h], samples[1][:,-1*v-1],
                          bins=(bins[h], bins[-1*v-1]))
                      axs[v][h].contour(counts.transpose(), levels=5,
                              extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1, colors='w')
                  axs[v][h].scatter(centers[h], centers[-1*v-1], marker='x', color='k')
                          #extent=[bins[-1*v-1].min(),bins[-1*v-1].max(),bins[h].min(),bins[h].max()],linewidths=1)

              if v > 0:
                  axs[v][h].xaxis.set_visible(False)
              else:
                  axs[v][h].set_xlabel(labels[h], fontsize=label_size)
                  axs[v][h].tick_params(axis='x', labelsize=tick_size)
                  if xticks is not None:
                      if xticks[h] is not None:
                          axs[v][h].set_xticks(xticks[h])

              if h > 0:
                  axs[v][h].yaxis.set_visible(False)
              elif h < Ncrn-1 and v < Ncrn-1:
                  axs[v][h].set_ylabel(labels[-1*v-1], fontsize=label_size)
                  axs[v][h].tick_params(axis='y', labelsize=tick_size)
              if v == Ncrn-1 and h == 0:
                  axs[v][h].tick_params(axis='y', labelsize=tick_size)      
      elif do_3d:
          rnd_inds = np.random.permutation(samples[-1-(v-Ncrn)].shape[0])[:100000]
          plot3d(samples[-1-(v-Ncrn)], ranges, scat_inds=rnd_inds, ax=axs[v],
                 labels=labels, label_size=label_size, tick_size=tick_size, alpha=0.2)
  return fig, axs

########################
#####  Deprecated  #####
########################

def plot_2dRows(samples, samples_single, labels,
        label_size=15, tick_size=13, bins=np.ones(3)*50,
        row_labels=None, row_label_size=15, xticks=None):
  Nrows = len(samples)

  top_pad = 0.025
  bot_pad = 0.065
  vert_pad = 0.03*4/Nrows
  hist_size = 0.0#5
  vert_dim = (1 - (bot_pad + top_pad + Nrows*(vert_pad + hist_size)-vert_pad))/Nrows


  horz_pad = 0.11
  left_pad = 0.115
  right_pad = 0.035
  cbar_pad = 0.01
  cbar_sz = 0.035
  horz_dim = (1 - (left_pad + right_pad + 2*horz_pad))/3

  print(vert_dim, horz_dim)
  fig_ratio = vert_dim/horz_dim
  fig = plt.figure(figsize=(10,10/fig_ratio))


  vert_anch = []
  for v in range(Nrows):
      vert_anch.append(bot_pad + v*(vert_dim + hist_size + vert_pad))

  horz_anch = []
  for h in range(3):
      horz_anch.append(left_pad + h*(horz_dim + horz_pad))

  axs, axs_hists = [], []
  for v in range(Nrows):
      axs.append([])
      axs_hists.append([])
      for h in range(3):
          axs[-1].append(fig.add_axes((horz_anch[h], vert_anch[v], horz_dim, vert_dim)))
          #axs_hists[-1].append(fig.add_axes((horz_anch[h], vert_anch[v]+vert_dim, horz_dim, hist_size)))


  inds = [[1,0], [0,2], [2,1]]


  for h in range(3):
      for v in range(Nrows):
          counts, x, y,image = axs[v][h].hist2d(samples_single[v][:,inds[h][0]], samples_single[v][:,inds[h][1]], bins=(bins[inds[h][0]], bins[inds[h][1]]))
          counts, x, y = np.histogram2d(samples[v][:,inds[h][0]], samples[v][:,inds[h][1]], bins=(bins[inds[h][0]], bins[inds[h][1]]))
          axs[v][h].contour(counts.transpose(), levels=6,
                  extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1, colors='lightgray')
          axs[v][h].scatter(centers[inds[h][0]], centers[inds[h][1]], marker='x', color='k')
          #axs_hists[v][h].hist(samples[v][:,inds[h][0]], bins=bins[inds[h][0]])
          #axs_hists[v][h].set_xlim([bins[inds[h][0]][0], bins[inds[h][0]][-1]])
          #if h + v == 2:
          #    axs[v][s*3+h].hist(samples[s][:,-1*v-1], bins=bins[-1*v-1], density=True)
          axs[v][h].set_xlabel(labels[inds[h][0]], fontsize=label_size)
          axs[v][h].set_ylabel(labels[inds[h][1]], fontsize=label_size)
          axs[v][h].tick_params(axis='x', labelsize=tick_size)
          axs[v][h].tick_params(axis='y', labelsize=tick_size)
          
          if v > 0:
              axs[v][h].xaxis.set_visible(False)
          #axs_hists[v][h].xaxis.set_visible(False)
          #axs_hists[v][h].yaxis.set_visible(False)
          else:
              if xticks is not None:
                  if xticks[h] is not None:
                      axs[v][h].set_xticks(xticks[h])
          
          if h == 2 and row_labels is not None:
              axs[v][h].text(1.1, 0.5, row_labels[v], fontsize=row_label_size,
                      transform=axs[v][h].transAxes, rotation='vertical',
                      horizontalalignment="center", verticalalignment="center") 

  return fig

def plot_2dproj_column(fig, dims, theta_gauss=None, theta_delta=None):   

  vdim, vpSep, VSEP, hdim, hpSep, bp, lp = dims
    
  # Bottom Portion
  vert_anch = []
  extra_v = 0
  for v in range(3):
      vert_anch.append(bp + v*(vdim + vpSep))

  horz_anch = [lp]
  for h in range(2):
      horz_anch.append(horz_anch[-1] + (hdim + hpSep))

  axs0 = []
  for v in range(3):
      axs0.append([])
      for h in range(3):
          axs0[-1].append(fig.add_axes((horz_anch[h], vert_anch[v], hdim, vdim)))

  samples = [theta_gauss]
  if theta_delta is not None:
      samples.append(theta_delta)
  if theta_gauss is None:
      samples = [theta_delta]
  if theta_gauss is None and theta_delta is None:
      raise ValueError("Must specify either theta_gauss or theta_delta")
  for s, sr in enumerate(samples):
      if s == 0 and theta_gauss is not None:
          sr = sr[:,np.array([0,2,4])]
      
      if sr is None:
          continue
          
      bins = []
      for ir in ranges_theta:
          bins.append(np.linspace(ir[0], ir[1], 51))
      for h in range(3):
          for v in range(3):
              if h + v == 2:
                  n, x = np.histogram(sr[:,-1*v-1], bins=bins[-1*v-1])
                  n = n/np.amax(n)
                  if s == 0:
                      axs0[v][h].bar((x[1:]+x[:-1])/2., n, width=x[1]-x[0], align='center')
                  else:
                      axs0[v][h].plot((x[1:]+x[:-1])/2., n, '--k')

                  axs0[v][h].plot([cn[-1*v-1], cn[-1*v-1]], [0,1.1], '-k')
                  axs0[v][h].set_xlim([bins[-1*v-1][0], bins[-1*v-1][-1]])
                  axs0[v][h].set_ylim([0, 1.1*np.amax(n)])
                      
              elif h + v > 2:
                  axs0[v][h].xaxis.set_visible(False)
                  axs0[v][h].yaxis.set_visible(False)
                  axs0[v][h].set_visible(False)
              else:
                  if s == 0:
                      counts, x, y,image = axs0[v][h].hist2d(sr[:,h], sr[:,-1*v-1], bins=(bins[h], bins[-1*v-1]))
                  else:
                      counts, x, y = np.histogram2d(sr[:,h], sr[:,-1*v-1], bins=(bins[h], bins[-1*v-1]))
                      axs0[v][h].contour(counts.transpose(), levels=5,
                              extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1, colors='w')
                  axs0[v][h].scatter(cn[h], cn[-1*v-1], marker='x', color='k')
                      #extent=[bins[-1*v-1].min(),bins[-1*v-1].max(),bins[h].min(),bins[h].max()],linewidths=1)

              if v > 0:
                  axs0[v][h].xaxis.set_visible(False)
              else:
                  axs0[v][h].set_xlabel(labels[h], fontsize=label_size)
                  axs0[v][h].tick_params(axis='x', labelsize=tick_size)
                  if s == 0:
                      if h == 0:
                          axs0[v][h].set_xticks([1.349, 1.351])
                      elif h == 1:
                          axs0[v][h].set_xticks([1.049, 1.051])
                      else:
                          axs0[v][h].set_xticks([2.339, 2.341])
              if h > 0 or v == 2:
                  axs0[v][h].yaxis.set_visible(False)
              elif h < 2 and v < 2:
                  axs0[v][h].set_ylabel(labels[-1*v-1], fontsize=label_size)
                  axs0[v][h].tick_params(axis='y', labelsize=tick_size)
              if v == 2 and h == 0:
                  axs0[v][h].tick_params(axis='y', labelsize=tick_size)
  
  # Top Portion
  if theta_gauss is not None:
      vert_anch = []
      extra_v = 0
      for v in range(6):
          vert_anch.append(bp + VSEP + (v+1)*(vdim + vpSep))

      horz_anch = [lp]
      for h in range(2):
          horz_anch.append(horz_anch[-1] + (hdim + hpSep))

      axs1 = []
      for v in range(6):
          axs1.append([])
          for h in range(3):
              axs1[-1].append(fig.add_axes((horz_anch[h], vert_anch[v], hdim, vdim)))

      sr = theta_gauss

      bins = []
      rg_inds = np.array([1,3,5,0,2,4])
      for ir in ranges_gauss:
          bins.append(np.linspace(ir[0], ir[1], 51))
      for h in range(3):
          for v in range(6):
              if h + v == 2:
                  n, _, _ = axs1[v][h].hist(sr[:,rg_inds[h]], bins=bins[rg_inds[h]], density=True)
                  axs1[v][h].plot([cn_gauss[rg_inds[h]], cn_gauss[rg_inds[h]]], [0,1.25*np.amax(n)], '-k')
                  axs1[v][h].set_xlim([bins[rg_inds[h]][0], bins[rg_inds[h]][-1]])
                  axs1[v][h].set_ylim([0, 1.1*np.amax(n)])
              elif h + v < 2:
                  axs1[v][h].xaxis.set_visible(False)
                  axs1[v][h].yaxis.set_visible(False)
                  axs1[v][h].set_visible(False)
              else:
                  if v < 3:
                      vv = 2 - v
                  else: 
                      vv = v
                  counts, x, y,image = axs1[v][h].hist2d(sr[:,rg_inds[h]], sr[:,rg_inds[vv]], bins=(bins[rg_inds[h]], bins[rg_inds[vv]]))
                  axs1[v][h].scatter(cn_gauss[rg_inds[h]], cn_gauss[rg_inds[vv]], marker='x', color='k')
                          #extent=[bins[-1*v-1].min(),bins[-1*v-1].max(),bins[h].min(),bins[h].max()],linewidths=1)

              if v != 5:
                  axs1[v][h].xaxis.set_visible(False)
              else:
                  axs1[v][h].xaxis.tick_top()
                  axs1[v][h].xaxis.set_label_position('top')
                  axs1[v][h].set_xlabel(labels_sig[h], fontsize=label_size)
                  axs1[v][h].tick_params(axis='x', labelsize=tick_size)

              if h < 2 or v == 0:
                  axs1[v][h].yaxis.set_visible(False)
              elif v > 0:
                  if v < 3:
                      vv = 2 - v
                  else: 
                      vv = v
                  axs1[v][h].yaxis.tick_right()
                  axs1[v][h].yaxis.set_label_position('right')
                  if v > 2:
                      lb = labels[v-3]
                  else:
                      lb = labels_sig[vv]
                  axs1[v][h].set_ylabel(lb, fontsize=label_size)
                  axs1[v][h].tick_params(axis='y', labelsize=tick_size)
              if v == 2 and h == 0:
                  axs1[v][h].tick_params(axis='y', labelsize=tick_size)
