import sys, os
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

# Some Function that we take the expectation value of 
def h(x, m, s):
  return np.exp(-0.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi))


norm_option = "both" # normalize, unormalize, both

# pr=prior | m=mean | s=std | post=posterior
pr_m = 5
pr_s =1.2 
h_s = 0.3
N=1000
Ndata = 100

burnIn_time = 1e3
autoCorr_time = 50
end_length = 1e5
rnd.seed(1400)


# Build Prior
prior = rnd.normal(loc=pr_m, scale=pr_s, size=N)

# Create data
data_x = (np.arange(Ndata)-Ndata/2.)*8.0*pr_s/(Ndata-1) + pr_m
data = np.array([h(data_x, m, h_s) for m in prior])
std = np.std(data, axis=0)
data = np.mean(data, axis=0)
fig, ax = plt.subplots()
ax.errorbar(data_x, data, yerr=std, fmt="o")
fig.savefig("mcdata.png")
plt.close()

step = 0.03

# Apply MHS
norms = []
norm_labels = []
if norm_option == "normalize":
  norms.append(Ndata)
  norm_labels.append("Normalized")
elif norm_option == "unnormalize":
  norms.append(1)
  norm_labels.append("Unnormalized")
elif norm_option == "both":
  norms = [Ndata, 1]
  norm_labels = ["Normalized", "Unnormalized"]
else:
  print("Cannot handle norm_option: " + norm_option)
  sys.exit(0)


posts = []
for norm in norms:
  post = []
  p0 = pr_m *0.3
  count = 0
  while len(post) < end_length:

    p  = p0 + rnd.uniform(-1, 1)*step
    r = np.exp(-0.5/norm*np.sum(
      ((h(data_x, p, h_s) - data)**2\
      - (h(data_x, p0, h_s) - data)**2)/std**2))

    """
    p0 = 5
    plt.plot(data_x, h(data_x, p0, h_s))
    plt.savefig("mctt.png")
    sys.exit(0)
    """

    if r > rnd.uniform(0, 1):
      p0 = p

    if count > burnIn_time:
      post.append(p0)
      
      if len(post) % 10000 == 0:
        print("Size: ", len(post))

    count += 1

  posts.append(np.array(post))

posts = np.array(posts)
autoCorr_inds = np.arange((posts.shape[1]-1)//autoCorr_time).astype(int)*autoCorr_time


# Plot Results
gs = {
    "hspace" : 0,
    "height_ratios" : [1,1,1,0.25,1]}
if posts.shape[0] == 1:
  gs["height_ratios"].pop(0)
letters = ['a', 'b', 'c', 'd', 'e']
font_size = 15

fig, ax = plt.subplots(2+len(posts)+1, 1, figsize=(6,15), gridspec_kw=gs)
ax[-2].set_visible(False)


bins = np.linspace(pr_m-5*pr_s, pr_m+5*pr_s, 100) 
hist, _ = np.histogram(prior, bins=bins)
hist = hist/np.sum(hist)
ax[0].bar(bins[:-1], hist, align='edge', color='k')
ax[0].set_xlim([bins[0], bins[-1]])
ax[0].xaxis.set_visible(False)
ax[0].set_ylabel("[arb]", fontsize=15)
ax[0].text(0.82, 0.88, "Prior", fontsize=font_size,
    transform=ax[0].transAxes, horizontalalignment='center')
ax[0].text(0.1, 0.85, "a",
    fontweight='bold', fontsize=25,
    transform=ax[0].transAxes, horizontalalignment='center')
ax[0].tick_params(axis='y', labelsize=14)
for i in range(posts.shape[0]):
  hist, _ = np.histogram(posts[i,autoCorr_inds], bins=bins)
  hist = hist/np.sum(hist)
  ax[1+i].bar(bins[:-1], hist, align='edge', color='k')
  ax[1+i].text(0.82, 0.88, "Posterior", fontsize=font_size,
      transform=ax[1+i].transAxes, horizontalalignment='center')
  ax[1+i].text(0.82, 0.8, norm_labels[i], fontsize=font_size,
      transform=ax[1+i].transAxes, horizontalalignment='center')
  ax[1+i].text(0.1, 0.85, letters[i+1],
      fontweight='bold', fontsize=25,
      transform=ax[1+i].transAxes, horizontalalignment='center')
  ax[1+i].tick_params(axis='y', labelsize=14)
  ax[1+i].set_ylabel("[arb]", fontsize=15)
  #ax[1+i].set_xlim([bins[0], bins[-1]])
ax[posts.shape[0]-1].xaxis.set_visible(False)
ax[posts.shape[0]].set_xlabel("M", fontsize=15)
ax[posts.shape[0]].tick_params(axis='x', labelsize=14)

ax[-1].errorbar(data_x, data, yerr=std, 
    fmt="_k")
ax[-1].set_xlim([data_x[0], data_x[-1]])
ax[-1].set_xlabel("q", fontsize=15)
ax[-1].text(0.1, 0.85, letters[len(ax)-2],
    fontweight='bold', fontsize=25,
    transform=ax[-1].transAxes, horizontalalignment='center')
ax[-1].tick_params(axis='x', labelsize=14)
ax[-1].tick_params(axis='y', labelsize=14)
ax[-1].set_ylabel("[arb]", fontsize=15)

plt.tight_layout()
fig.savefig("mctest.png")




sys.exit(0)





############################################
#####  Use two gaussians in the prior  #####
############################################
# If you change the prior check that the data range (data_x) is wide enough

prior = np.concatenate(
    [rnd.normal(loc=pr_m, scale=pr_s, size=N),
      rnd.normal(loc=2*pr_m, scale=pr_s, size=N)])
data_x = (np.arange(Ndata)-Ndata/2.)*8.0*pr_s/(Ndata-1) + 3.*pr_m/2
data = np.array([h(data_x, m, h_s) for m in prior])
std = np.std(data, axis=0)
data = np.mean(data, axis=0)
fig, ax = plt.subplots()
ax.errorbar(data_x, data, yerr=std, fmt="o", markersize=1)
fig.savefig("mcdata_2dist.png")
plt.close()

step = 0.03

post = [[2,11]]
while len(post) < 1e5:

  p00 = post[-1][0]
  p01 = post[-1][1]
  p0  = p00 + rnd.uniform(-1, 1)*step
  p1  = p01 + rnd.uniform(-1, 1)*step
  r = np.exp(-0.5/norm*np.sum(
    ((h(data_x, p0, h_s) + h(data_x, p1, h_s) - data)**2\
    - (h(data_x, p00, h_s) + h(data_x, p01, h_s) - data)**2)/std**2))

  """
  p0 = 5
  p1 = 10
  plt.plot(data_x, h(data_x, p0, h_s) + h(data_x, p1, h_s))
  plt.savefig("mctt.png")
  sys.exit(0)
  """
  if r > 1:
    post.append([p0, p1])
  elif r > rnd.uniform(0, 1):
    post.append([p0, p1])

  if len(post) % 1000 == 0:
    print("Size: ", len(post))



print(np.mean(prior), np.mean(post[1000:]))
fig, ax = plt.subplots(1, 2)
ax[0].hist(prior, bins=50)
ax[1].hist(np.reshape(np.array(post[1000:]), (-1)), bins=50)

fig.savefig("mctest_2dist.png")



