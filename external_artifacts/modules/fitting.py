import numpy as np
from copy import copy as copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.sparse import csr_matrix
from numpy.polynomial import Legendre


def normal_eqn_vects(X, Y, W, var):
    overlap = np.einsum('bai,bi,bci->bac', X, W, X, optimize='greedy')
    if np.any(np.linalg.det(overlap) == 0.0):
        print("Overlap matrix is singular, should figure out why")
        return np.ones((Y.shape[0], X.shape[0]))*np.nan, np.ones((Y.shape[0], X.shape[0], X.shape[0]))*np.nan
    
    # Fit
    denom = np.linalg.inv(overlap)
    numer = np.einsum('bai,bi,bi->ba', X, W, Y, optimize='greedy')
    
    # Covariance
    cov = np.einsum('bai,bij,bj,bj,bj,bkj,bkc->bac', denom, X, W, var, W, X, denom, optimize='greedy')
    
    return np.einsum('abi,ai->ab', denom, numer, optimize='greedy'), cov



def fit_legendres_images(images, centers, lg_inds, rad_inds, maxPixel, rotate=0,
        image_stds=None, image_counts=None, image_nanMaps=None, image_weights=None,
        chiSq_fit=False, rad_range=None):
    """
    Fits legendre polynomials to an array of single images (3d) or a list/array of 
    an array of scan images, possible dimensionality:
        1) [NtimeSteps, image_rows, image_cols]
        2) [NtimeSteps (list), Nscans, image_rows, image_cols]
    """

    if image_counts is None:
        image_counts = []
        for im in range(len(images)):
            image_counts.append(np.ones_like(images[im]))
            image_counts[im][np.isnan(images[im])] = 0
    
    if chiSq_fit and (image_stds is None):
        print("If using the chiSq fit you must supply image_stds")
        return None
    
    if image_stds is None:
        image_stds = []
        for im in range(len(images)):
            image_stds.append(np.ones_like(images[im]))
            image_stds[im][np.isnan(images[im])] = 0    

    with_scans = len(images[0].shape)+1 >= 4


    rad_zero_var = "" 
    img_fits = [[] for x in range(len(images))]
    img_covs = [[] for x in range(len(images))]
    for ir,rad in enumerate(rad_inds.keys()):
        if rad >= maxPixel:
          break
        if rad_range is not None:
            if rad < rad_range[0] or rad >= rad_range[1]:
                continue
        if rad % 25 == 0:
          print("Fitting radius {}".format(rad))

        pixels, nans, angles = [], [], []
        all_angles = np.arctan2(rad_inds[rad][1].astype(float), rad_inds[rad][0].astype(float))
        all_angles[all_angles<0] += 2*np.pi
        all_angles = np.mod(all_angles + rotate, 2*np.pi)
        all_angles[all_angles > np.pi] -= 2*np.pi
        if np.sum(np.mod(lg_inds, 2)) == 0:
            all_angles[np.abs(all_angles) > np.pi/2.] -= np.pi*np.sign(all_angles[np.abs(all_angles) > np.pi/2.])
        angles = np.unique(np.abs(all_angles))
        ang_sort_inds = np.argsort(angles)
        angles = angles[ang_sort_inds]
        Nangles = angles.shape[0]
        
        if len(angles) == len(all_angles):
            do_merge = False
        else:
            do_merge = True
            mi_rows, mi_cols, mi_data = [], [], []
            pr,pc,pv = [],[],[]
            for ia,ang in enumerate(angles):
                inds = np.where(np.abs(all_angles) == ang)[0]
                mi_rows.append(np.ones_like(inds)*ia)
                mi_cols.append(inds)
            mi_rows, mi_cols = np.concatenate(mi_rows), np.concatenate(mi_cols)

            merge_indices = csr_matrix((np.ones_like(mi_rows), (mi_rows, mi_cols)),
                    shape=(len(angles), len(all_angles)))

        
    
        for im in range(len(images)):
            if with_scans:
                angs_tile = np.tile(angles, (images[im].shape[0], 1))
                scn_inds, row_inds, col_inds = [], [], []
                for isc in range(images[im].shape[0]):
                    scn_inds.append(np.ones(rad_inds[rad][0].shape[0], dtype=int)*isc)
                    row_inds.append(rad_inds[rad][0]+centers[im][isc,0])
                    col_inds.append(rad_inds[rad][1]+centers[im][isc,1])
                    
                scn_inds = np.concatenate(scn_inds)
                row_inds = np.concatenate(row_inds)
                col_inds = np.concatenate(col_inds)
                img_pixels = np.reshape(
                        copy(images[im][scn_inds,row_inds,col_inds]),
                        (images[im].shape[0], -1))
                img_counts = np.reshape(
                        copy(image_counts[im][scn_inds,row_inds,col_inds]),
                        (images[im].shape[0], -1))
                img_stds = np.reshape(
                        copy(image_stds[im][scn_inds,row_inds,col_inds]),
                        (images[im].shape[0], -1))
                if image_nanMaps is not None:
                    ii = np.reshape(image_nanMaps[im][scn_inds,row_inds,col_inds],
                            (images[im].shape[0], -1)).astype(bool)
                    img_pixels[ii] = np.nan
                    img_counts[ii] = 0
                if image_weights is not None:
                    img_weights = np.reshape(
                            copy(image_weights[im][scn_inds,row_inds,col_inds]),
                            (images[im].shape[0], -1))
            else:
                angs_tile = np.expand_dims(angles, 0)
                row_inds = rad_inds[rad][0]+centers[im,0]
                col_inds = rad_inds[rad][1]+centers[im,1]
                img_pixels = np.reshape(copy(images[im][row_inds,col_inds]), (1, -1))
                img_counts = np.reshape(copy(image_counts[im][row_inds,col_inds]), (1, -1))
                img_stds = np.reshape(copy(image_stds[im][row_inds,col_inds]), (1, -1))
                if image_nanMaps is not None:
                    img_pixels[np.reshape(image_nanMaps[im][row_inds,col_inds], (1, -1)).astype(bool)] = np.nan
                    img_counts[np.reshape(image_nanMaps[im][row_inds,col_inds], (1, -1)).astype(bool)] = 0
                if image_weights is not None:
                    img_weights = np.reshape(copy(image_weights[im][row_inds,col_inds]), (1, -1))
                    
            img_pix = img_pixels*img_counts
            img_var = img_counts*(img_stds**2)
            img_pix[np.isnan(img_pixels)] = 0
            img_var[np.isnan(img_pixels)] = 0
            
            if do_merge: 
                
                img_pixels[np.isnan(img_pixels)] = 0
                
                img_pix = np.transpose(merge_indices.dot(np.transpose(img_pix)))
                img_var = np.transpose(merge_indices.dot(np.transpose(img_var)))               
                
                img_counts = np.transpose(merge_indices.dot(np.transpose(img_counts)))
                
                if image_weights is not None:
                    print("Must fill this in, don't forget std option")
                    sys.exit(0)
            else:
                img_pix = img_pix[:,ang_sort_inds]
                img_var = img_var[:,ang_sort_inds]
                img_counts = img_counts[:,ang_sort_inds]
            img_pix /= img_counts
            img_var /= img_counts
            
            
            Nnans = np.sum(np.isnan(img_pix), axis=-1)
            ang_inds = np.where(img_counts > 0)
            arr_inds = np.concatenate([np.arange(Nangles-Nn) for Nn in Nnans])

            shp = [img_pix.shape[0], np.max([1, img_pix.shape[1]-1])]
            img_pixels = np.zeros(shp)
            img_vars = np.zeros(shp)
            img_angs = np.zeros(shp)
            img_dang = np.zeros(shp)

            img_pixels[ang_inds[0][:-1], arr_inds[:-1]] =\
                    (img_pix[ang_inds[0][:-1], ang_inds[1][:-1]] + img_pix[ang_inds[0][1:], ang_inds[1][1:]])/2.
            img_vars[ang_inds[0][:-1], arr_inds[:-1]] =\
                    (img_var[ang_inds[0][:-1], ang_inds[1][:-1]] + img_var[ang_inds[0][1:], ang_inds[1][1:]])/2.
            img_angs[ang_inds[0][:-1], arr_inds[:-1]] =\
                    (angs_tile[ang_inds[0][:-1], ang_inds[1][:-1]] + angs_tile[ang_inds[0][1:], ang_inds[1][1:]])/2.
            img_dang[ang_inds[0][:-1], arr_inds[:-1]] =\
                    (angs_tile[ang_inds[0][1:], ang_inds[1][1:]] - angs_tile[ang_inds[0][:-1], ang_inds[1][:-1]])

            for isc in range(Nnans.shape[0]):
                if Nnans[isc]:
                    # Using angle midpoint => one less angle => Nnans[isc]+1
                    img_pixels[isc,-1*(Nnans[isc]):] = 0
                    img_vars[isc,-1*(Nnans[isc]):] = 0
                    img_angs[isc,-1*(Nnans[isc]):] = 0
                    img_dang[isc,-1*(Nnans[isc]):] = 0
            

            if image_weights is not None:
                print("Must fill this in and check below")
                sys.exit(0)
            elif chiSq_fit:
                img_weights = 1./img_vars
                if np.sum(img_vars==0):
                  rad_zero_var += str(rad)+", "
                  #print("WARNING: Variance is 0 in {} locations!!!".format(
                  #    np.sum(img_vars==0)))
                img_weights[img_vars==0] = 0
            else:
                img_weights = np.ones_like(img_pixels)
            img_weights *= np.sin(img_angs)*img_dang
            lgndrs = []
            for lg in lg_inds:
                lgndrs.append(Legendre.basis(lg)(np.cos(img_angs)))
            lgndrs = np.transpose(np.array(lgndrs), (1,0,2))

            
            empty_scan = np.sum(img_weights.astype(bool), -1) < 2
            overlap = np.einsum('bai,bi,bci->bac',
                    lgndrs[np.invert(empty_scan)],
                    img_weights[np.invert(empty_scan)],
                    lgndrs[np.invert(empty_scan)],
                    optimize='greedy')
            empty_scan[np.invert(empty_scan)] = (np.linalg.det(overlap) == 0.0)
        
            if np.any(empty_scan):
                fit = np.ones((img_pixels.shape[0], len(lg_inds)))*np.nan
                cov = np.ones((img_pixels.shape[0], len(lg_inds), len(lg_inds)))*np.nan
            
                if np.any(np.invert(empty_scan)):
                    img_pixels  = img_pixels[np.invert(empty_scan)]
                    img_weights = img_weights[np.invert(empty_scan)]
                    img_vars    = img_vars[np.invert(empty_scan)]
                    lgndrs      = lgndrs[np.invert(empty_scan)]
                    
                    fit[np.invert(empty_scan)], cov[np.invert(empty_scan)] =\
                        normal_eqn_vects(lgndrs, img_pixels, img_weights, img_vars)
            else:
                fit, cov = normal_eqn_vects(lgndrs, img_pixels, img_weights, img_vars)
            img_fits[im].append(np.expand_dims(fit, 1))
            img_covs[im].append(np.expand_dims(cov, 1))
        

    #print("ZERO VARIANCE IN RAD", rad_zero_var)
    Nscans = None
    for im in range(len(img_fits)):
        img_fits[im] = np.concatenate(img_fits[im], 1)
        img_covs[im] = np.concatenate(img_covs[im], 1)
        if Nscans is None:
            Nscans = img_fits[im].shape[0]
        elif Nscans != img_fits[im].shape[0]:
            Nscans = -1
    if Nscans > 0:
        img_fits = np.array(img_fits)
        img_covs = np.array(img_covs)
       
    #print("OUTSHAPE", img_fits.shape, img_covs.shape)
    #print("NANS", np.sum(np.sum(np.isnan(img_fits), 0), 0), np.sum(np.sum(np.isnan(img_covs), 0), 0))
    if with_scans:
        return img_fits, img_covs
    else:
        return img_fits[:,0,:,:], img_covs[:,0,:,:]

