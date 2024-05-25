import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import logging
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
from urllib.parse import urlparse
from google.colab.patches import cv2_imshow
import tempfile
import cv2
from scipy.stats import mode
import fastremap
from . import transforms, dynamics, utils, plot, metrics, adain_function, adain_net
import torch
#     from GPUtil import showUtilization as gpu_usage #for gpu memory debugging 
from torch import nn
from torch.utils import mkldnn as mkldnn_utils
from . import resnet_torch
TORCH_ENABLED = True

core_logger = logging.getLogger(__name__)
tqdm_out = utils.TqdmToLogger(core_logger, level=logging.INFO)

def parse_model_string(pretrained_model):
    if isinstance(pretrained_model, list):
        model_str = os.path.split(pretrained_model[0])[-1]
    else:
        model_str = os.path.split(pretrained_model)[-1]
    if len(model_str)>3 and model_str[:4]=='unet':
        cp = False
        nclasses = max(2, int(model_str[4]))
    elif len(model_str)>7 and model_str[:8]=='cellpose':
        cp = True
        nclasses = 3
    else:
        return 3, True, True, False
    
    if 'residual' in model_str and 'style' in model_str and 'concatentation' in model_str:
        ostrs = model_str.split('_')[2::2]
        residual_on = ostrs[0]=='on'
        style_on = ostrs[1]=='on'
        concatenation = ostrs[2]=='on'
        return nclasses, residual_on, style_on, concatenation
    else:
        if cp:
            return 3, True, True, False
        else:
            return nclasses, False, False, True

def use_gpu(gpu_number=0, use_torch=True):
    """ check if gpu works """
    if use_torch:
        return _use_gpu_torch(gpu_number)
    else:
        raise ValueError('cellpose only runs with pytorch now')

def _use_gpu_torch(gpu_number=0):
    try:
        device = torch.device('cuda:' + str(gpu_number))
        _ = torch.zeros([1, 2, 3]).to(device)
        core_logger.info('** TORCH CUDA version installed and working. **')
        return True
    except:
        core_logger.info('TORCH CUDA version not installed/working.')
        return False

def assign_device(use_torch=True, gpu=False, device=0):
    mac = False
    cpu = True
    if isinstance(device, str):
        if device=='mps':
            mac = True 
        else:
            device = int(device)
    if gpu and use_gpu(use_torch=True):
        device = torch.device(f'cuda:{device}')
        gpu=True
        cpu=False
        core_logger.info('>>>> using GPU')
    elif mac:
        try:
            device = torch.device('mps')
            gpu=True
            core_logger.info('>>>> using GPU')
        except:
            cpu = True 
            gpu = False

    if cpu:
        device = torch.device('cpu')
        core_logger.info('>>>> using CPU')
        gpu=False
    return device, gpu

def check_mkl(use_torch=True):
    #core_logger.info('Running test snippet to check if MKL-DNN working')
    mkl_enabled = torch.backends.mkldnn.is_available()
    if mkl_enabled:
        mkl_enabled = True
        #core_logger.info('MKL version working - CPU version is sped up.')
    else:
        core_logger.info('WARNING: MKL version on torch not working/installed - CPU version will be slightly slower.')
        core_logger.info('see https://pytorch.org/docs/stable/backends.html?highlight=mkl')
    return mkl_enabled

class UnetModel():
    def __init__(self, gpu=False, pretrained_model=False,pretrained_model_adain='',
                    diam_mean=30., net_avg=False, device=None,
                    residual_on=False, style_on=False, concatenation=True,
                    style_weight = 10,content_weight = 1, seg_weight = 2,
                    nclasses=3, nchan=1, save_image_dir = None):
        print('created Unet Model')                
        self.save_img_dir = save_image_dir
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.seg_weight = seg_weight
        print(self.content_weight,self.style_weight)                
        self.unet = True
        self.torch = True
        self.mkldnn = None
        if device is None:
            sdevice, gpu = assign_device(self.torch, gpu)
        self.device = device if device is not None else sdevice
        if device is not None:
            device_gpu = self.device.type=='cuda'
        self.gpu = gpu if device is None else device_gpu
        if not self.gpu:
            self.mkldnn = check_mkl(True)
        self.pretrained_model = pretrained_model
        self.diam_mean = diam_mean

        ostr = ['off', 'on']
        self.net_type = 'unet{}_residual_{}_style_{}_concatenation_{}'.format(nclasses,
                                                                                ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])                                             
        if pretrained_model:
            core_logger.info(f'u-net net type: {self.net_type}')
        # create network
        self.nclasses = nclasses
        self.nbase = [32,64,128,256]
        self.nchan = nchan
        self.nbase = [nchan, 32, 64, 128, 256]
        self.net = resnet_torch.CPnet(self.nbase, 
                                        self.nclasses, 
                                        sz=3,
                                        residual_on=residual_on, 
                                        style_on=style_on,
                                        concatenation=concatenation,
                                        mkldnn=self.mkldnn,
                                        diam_mean=diam_mean).to(self.device)
        
        if pretrained_model is not None and isinstance(pretrained_model, str):
            self.net.load_model(pretrained_model, device=self.device)
        
        style_decoder = adain_net.decoder
        style_vgg = adain_net.vgg
        style_vgg = nn.Sequential(*list(style_vgg.children())[:31])
        self.style_network = adain_net.Net(style_vgg, style_decoder).to(self.device)
        self.style_network.train()
        
    def eval(self, x, batch_size=8, channels=None, channels_last=False, invert=False, normalize=True,
             rescale=None, do_3D=False, anisotropy=None, net_avg=False, augment=False,
             channel_axis=None, z_axis=None, nolist=False,
             tile=True, cell_threshold=None, boundary_threshold=None, min_size=15, 
             compute_masks=True):
        """ segment list of images x

            Parameters
            ----------
            x: list or array of images
                can be list of 2D/3D images, or array of 2D/3D images, or 4D image array

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            channel_axis: int (optional, default None)
                if None, channels dimension is attempted to be automatically determined

            z_axis: int (optional, default None)
                if None, z dimension is attempted to be automatically determined

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            normalize: bool (optional, default True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

            net_avg: bool (optional, default False)
                runs the 4 built-in networks and averages them if True, runs one network if False

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            cell_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            boundary_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = the cell distance field
                flows[k][3] = the cell boundary

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """        
        x = [transforms.convert_image(xi, channels, channel_axis, z_axis, do_3D, 
                                    normalize, invert, nchan=self.nchan) for xi in x]
        nimg = len(x)
        self.batch_size = batch_size

        styles = []
        flows = []
        masks = []
        if rescale is None:
            rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)
        if nimg > 1:
            iterator = trange(nimg, file=tqdm_out)
        else:
            iterator = range(nimg)

        if isinstance(self.pretrained_model, list):
            model_path = self.pretrained_model[0]
            if not net_avg:
                self.net.load_model(self.pretrained_model[0], device=self.device)
        else:
            model_path = self.pretrained_model

        if cell_threshold is None or boundary_threshold is None:
            try:
                thresholds = np.load(model_path+'_cell_boundary_threshold.npy')
                cell_threshold, boundary_threshold = thresholds
                core_logger.info('>>>> found saved thresholds from validation set')
            except:
                core_logger.warning('WARNING: no thresholds found, using default / user input')

        cell_threshold = 2.0 if cell_threshold is None else cell_threshold
        boundary_threshold = 0.5 if boundary_threshold is None else boundary_threshold

        if not do_3D:
            for i in iterator:
                img = x[i].copy()
                shape = img.shape
                # rescale image for flow computation
                img = transforms.resize_image(img, rsz=rescale[i])
                y, style = self._run_nets(img, None ,net_avg=net_avg, augment=augment, 
                                          tile=tile)
                if compute_masks:
                    maski = utils.get_masks_unet(y, cell_threshold, boundary_threshold)
                    maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                    maski = transforms.resize_image(maski, shape[-3], shape[-2], 
                                                        interpolation=cv2.INTER_NEAREST)
                else:
                    maski = None
                masks.append(maski)
                styles.append(style)
        else:
            for i in iterator:
                tic=time.time()
                yf, style = self._run_3D(x[i], rsz=rescale[i], anisotropy=anisotropy, 
                                         net_avg=net_avg, augment=augment, tile=tile)
                yf = yf.mean(axis=0)
                core_logger.info('probabilities computed %2.2fs'%(time.time()-tic))
                if compute_masks:
                    maski = utils.get_masks_unet(yf.transpose((1,2,3,0)), cell_threshold, boundary_threshold)
                    maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                else:
                    maski = None
                masks.append(maski)
                styles.append(style)
                core_logger.info('masks computed %2.2fs'%(time.time()-tic))
                flows.append(yf)

        if nolist:
            masks, flows, styles = masks[0], flows[0], styles[0]
        
        return masks, flows, styles

    def _to_device(self, x):
        X = torch.from_numpy(x).float().to(self.device)
        return X

    def _from_device(self, X):
        x = X.detach().cpu().numpy()
        return x

    def network(self, x, style_anchor, return_conv=False):
        """ convert imgs to torch and run network model and return numpy """
  
        x=x/255
        X = self._to_device(x)
        self.net.eval()
        if self.mkldnn:
            self.net = mkldnn_utils.to_mkldnn(self.net)
            
        styles = np.zeros((X.shape[0],3,224,224))
        
        for i in range(styles.shape[0]):
            styles[i,:,:,:] = style_anchor
        ########################
        with torch.no_grad():
            full_x = self._to_device(np.stack((x[:,0,:,:],x[:,0,:,:],x[:,0,:,:]),axis=1))
            ST = self._to_device(styles)
            sample_num = x.shape[0]
            self.style_network.eval()
            loss_c, loss_s,t_img_3,t_img = self.style_network(full_x,ST,alpha=1) #t_img: batch,1,224,224
            y,style = self.net(t_img)#y: batch,3,224,224
            masks = y[:,2,:,:].cpu().numpy()
            masks[masks<=0]=0
            masks[masks>0]=1
            #print(np.unique(masks))
            #print(masks.shape)
            transferred_img = t_img_3.detach()
        del X
        y = self._from_device(y)
        t_img = self._from_device(t_img)
        #style = self._from_device(style)
        if return_conv:
            conv = self._from_device(conv)
            y = np.concatenate((y, conv), axis=1)
        
        return y, t_img
                
    def _run_nets(self, img, style_anchor, augment=False, tile_overlap=0.1, bsize=224, 
                  return_conv=False, progress=None):
        """ run network (if more than one, loop over networks and average results

        Parameters
        --------------

        img: float, [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        net_avg: bool (optional, default False)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU memory usage limited (recommended)

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

        Returns
        ------------------

        y: array [3 x Ly x Lx] or [3 x Lz x Ly x Lx]
            y is output (averaged over networks);
            y[0] is Y flow; y[1] is X flow; y[2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles,
            but not averaged over networks.

        """
        y,stylized = self._run_net(img,style_anchor, augment=augment, tile_overlap=tile_overlap,
                                     bsize=bsize, return_conv=return_conv)
            
        return y, stylized

    def _run_net(self, imgs, style_anchor,augment=False, tile_overlap=0.1, bsize=224,
                 return_conv=False):
        """ run network on image or stack of images

        (faster if augment is False)

        Parameters
        --------------

        imgs: array [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient(s) for image

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended);
            cannot be turned off for 3D segmentation

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        Returns
        ------------------

        y: array [Ly x Lx x 3] or [Lz x Ly x Lx x 3]
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles

        """   

        # make image nchan x Ly x Lx for net
        imgs = np.transpose(imgs, (2,0,1))
        detranspose = (1,2,0)

        # pad image for net so Ly and Lx are divisible by 4
        imgs, ysub, xsub = transforms.pad_image_ND(imgs)
        # slices from padding
#         slc = [slice(0, self.nclasses) for n in range(imgs.ndim)] # changed from imgs.shape[n]+1 for first slice size 
        slc = [slice(0, imgs.shape[n]+1) for n in range(imgs.ndim)]
        slc[-3] = slice(0, self.nclasses + 32*return_conv + 1)
        slc[-2] = slice(ysub[0], ysub[-1]+1)
        slc[-1] = slice(xsub[0], xsub[-1]+1)
        slc = tuple(slc)
        dices = None
        # run network
        y, stylized = self._run_tiled(imgs, style_anchor,augment=augment, bsize=bsize, 
                                      tile_overlap=tile_overlap, 
                                      return_conv=return_conv)
        

        # slice out padding
        y = y[slc]
        stylized = stylized[slc]
        # transpose so channels axis is last again
        y = np.transpose(y, detranspose)
        stylized = np.transpose(stylized, detranspose)
        
        return y, stylized
    
    def _run_tiled(self, imgi, style_anchor, augment=False, bsize=224, tile_overlap=0.1, return_conv=False):
        """ run network in tiles of size [bsize x bsize]

        First image is split into overlapping tiles of size [bsize x bsize].
        If augment, tiles have 50% overlap and are flipped at overlaps.
        The average of the network output over tiles is returned.

        Parameters
        --------------

        imgi: array [nchan x Ly x Lx] or [Lz x nchan x Ly x Lx]

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]
         
        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        Returns
        ------------------

        yf: array [3 x Ly x Lx] or [Lz x 3 x Ly x Lx]
            yf is averaged over tiles
            yf[0] is Y flow; yf[1] is X flow; yf[2] is cell probability

        styles: array [64]
            1D array summarizing the style of the image, averaged over tiles

        """

        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize, 
                                                            augment=augment, tile_overlap=tile_overlap)
        ny, nx, nchan, ly, lx = IMG.shape
        IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
        batch_size = self.batch_size
        niter = int(np.ceil(IMG.shape[0] / batch_size))
        nout = self.nclasses + 32*return_conv
        y = np.zeros((IMG.shape[0], nout, ly, lx))
        stylized_all = np.zeros((IMG.shape[0], nout, ly, lx))
        dicess = []
        #print(niter)
        for k in range(niter):
            irange = np.arange(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
            y0, stylized = self.network(IMG[irange],style_anchor, return_conv=return_conv)
            y0 = y0.reshape(len(irange), y0.shape[-3], y0.shape[-2], y0.shape[-1])
            stylized = stylized.reshape(len(irange), stylized.shape[-3], stylized.shape[-2], stylized.shape[-1])
            y[irange] = y0
            stylized_all[irange] = stylized
        yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
        stylizedf = transforms.average_tiles(stylized_all, ysub, xsub, Ly, Lx)
        yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
        stylizedf = stylizedf[:,:imgi.shape[1],:imgi.shape[2]]
        return yf, stylizedf

    def _run_3D(self, imgs, rsz=1.0, anisotropy=None, net_avg=False, 
                augment=False, tile=True, tile_overlap=0.1, 
                bsize=224, progress=None):
        """ run network on stack of images

        (faster if augment is False)

        Parameters
        --------------

        imgs: array [Lz x Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient(s) for image

        anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

        net_avg: bool (optional, default False)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended);
            cannot be turned off for 3D segmentation

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI


        Returns
        ------------------

        yf: array [Lz x Ly x Lx x 3]
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles

        """ 
        sstr = ['YX', 'ZY', 'ZX']
        if anisotropy is not None:
            rescaling = [[rsz, rsz],
                         [rsz*anisotropy, rsz],
                         [rsz*anisotropy, rsz]]
        else:
            rescaling = [rsz] * 3
        pm = [(0,1,2,3), (1,0,2,3), (2,0,1,3)]
        ipm = [(3,0,1,2), (3,1,0,2), (3,1,2,0)]
        yf = np.zeros((3, self.nclasses, imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.float32)
        for p in range(3 - 2*self.unet):
            xsl = imgs.copy().transpose(pm[p])
            # rescale image for flow computation
            shape = xsl.shape
            xsl = transforms.resize_image(xsl, rsz=rescaling[p])  
            # per image
            core_logger.info('running %s: %d planes of size (%d, %d)'%(sstr[p], shape[0], shape[1], shape[2]))
            y = self._run_nets(xsl, net_avg=net_avg, augment=augment, tile=tile, 
                                      bsize=bsize, tile_overlap=tile_overlap)
            y = transforms.resize_image(y, shape[1], shape[2])    
            yf[p] = y.transpose(ipm[p])
            if progress is not None:
                progress.setValue(25+15*p)
        return yf, None#style

    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """
        # if available set boundary pixels to 2
        if lbl.shape[1]>1 and self.nclasses>2:
            boundary = lbl[:,1]<=4
            lbl = lbl[:,0]
            lbl[boundary] *= 2
        else:
            lbl = lbl[:,0]
        lbl = self._to_device(lbl).long()
        loss = 8 * 1./self.nclasses * self.criterion(y, lbl)
        return loss

    def train(self, train_data, train_labels, train_files=None, 
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, save_path=None, save_every=100, save_each=False,
              learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001, batch_size=8, 
              nimg_per_epoch=None, min_train_masks=5, rescale=False, model_name=None):
        """ train function uses 0-1 mask label and boundary pixels for training """

        nimg = len(train_data)

        train_data, train_labels, test_data, test_labels,style, run_test = transforms.reshape_train_test(train_data, train_labels,
                                                                                                   test_data, test_labels,
                                                                                                   channels, normalize) #here, style is still a list of images, but axis swapped and normalized
        train_labels = [fastremap.renumber(label, in_place=True)[0] for label in train_labels]
        # add dist_to_bound to labels
        if self.nclasses==3:
            core_logger.info('computing boundary pixels for training data')
            train_classes = [np.stack((label, label>0, utils.distance_to_boundary(label)), axis=0).astype(np.float32)
                                for label in tqdm(train_labels, file=tqdm_out)]
        else:
            train_classes = [np.stack((label, label>0), axis=0).astype(np.float32)
                                for label in tqdm(train_labels, file=tqdm_out)]
        if run_test:
            test_labels = [fastremap.renumber(label, in_place=True)[0] for label in test_labels]
            if self.nclasses==3:
                core_logger.info('computing boundary pixels for test data')
                test_classes = [np.stack((label, label>0, utils.distance_to_boundary(label)), axis=0).astype(np.float32)
                                    for label in tqdm(test_labels, file=tqdm_out)]
            else:
                test_classes = [np.stack((label, label>0), axis=0).astype(np.float32)
                                    for label in tqdm(test_labels, file=tqdm_out)]
        else:
            test_classes = None
        
        nmasks = np.array([label[0].max()-1 for label in train_classes])
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            core_logger.warning(f'{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set')
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            train_data = [train_data[i] for i in ikeep]
            train_classes = [train_classes[i] for i in ikeep]
            train_labels = [train_labels[i] for i in ikeep]

        # split train data into train and val
        val_data = train_data[::8]
        val_classes = train_classes[::8]
        val_labels = train_labels[::8]
        del train_data[::8], train_classes[::8], train_labels[::8]
        model_path = self._train_net(train_data, train_classes, style,test_data, test_classes,
                                    save_path=save_path, save_every=save_every, save_each=save_each,
                                    learning_rate=learning_rate, n_epochs=n_epochs, momentum=momentum, 
                                    weight_decay=weight_decay, SGD=False, batch_size=batch_size, 
                                    nimg_per_epoch=nimg_per_epoch, rescale=rescale, model_name=model_name)

        # find threshold using validation set
        core_logger.info('>>>> finding best thresholds using validation set')
        cell_threshold, boundary_threshold = self.threshold_validation(val_data, val_labels)
        np.save(model_path+'_cell_boundary_threshold.npy', np.array([cell_threshold, boundary_threshold]))
        return model_path

    def threshold_validation(self, val_data, val_labels):
        cell_thresholds = np.arange(-4.0, 4.25, 0.5)
        if self.nclasses==3:
            boundary_thresholds = np.arange(-2, 2.25, 1.0)
        else:
            boundary_thresholds = np.zeros(1)
        aps = np.zeros((cell_thresholds.size, boundary_thresholds.size, 3))
        for j,cell_threshold in enumerate(cell_thresholds):
            for k,boundary_threshold in enumerate(boundary_thresholds):
                masks = []
                for data in val_data:
                    output,style = self._run_net(data.transpose(1,2,0), augment=False)
                    masks.append(utils.get_masks_unet(output, cell_threshold, boundary_threshold))
                ap = metrics.average_precision(val_labels, masks)[0]
                ap0 = ap.mean(axis=0)
                aps[j,k] = ap0
            if self.nclasses==3:
                kbest = aps[j].mean(axis=-1).argmax()
            else:
                kbest = 0
            if j%4==0:
                core_logger.info('best threshold at cell_threshold = {} => boundary_threshold = {}, ap @ 0.5 = {}'.format(cell_threshold, boundary_thresholds[kbest], 
                                                                        aps[j,kbest,0]))   
        if self.nclasses==3: 
            jbest, kbest = np.unravel_index(aps.mean(axis=-1).argmax(), aps.shape[:2])
        else:
            jbest = aps.squeeze().mean(axis=-1).argmax()
            kbest = 0
        cell_threshold, boundary_threshold = cell_thresholds[jbest], boundary_thresholds[kbest]
        core_logger.info('>>>> best overall thresholds: (cell_threshold = {}, boundary_threshold = {}); ap @ 0.5 = {}'.format(cell_threshold, boundary_threshold, 
                                                          aps[jbest,kbest,0]))
        return cell_threshold, boundary_threshold
        
    def get_random_crop(self,image, crop_height, crop_width):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        crop = image[y: y + crop_height, x: x + crop_width]
        return crop, x,y
    def get_crop(self,image, crop_height, crop_width,xy):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height
        x = xy[0]
        y = xy[1]
        crop = image[y: y + crop_height, x: x + crop_width]
        return crop
   
    
    def _find_scale(self,imgi,styles, scales = [1,0.8,1.5,2]):
        self.style_network.eval()
        self.net.eval()
        bs = imgi.shape[0]
        style_num = styles.shape[0]
        #imgi.shape = (batch,2,224,224)
        #ST = self._to_device(styles)
        imgi = imgi[:,0:1,:,:]
        #full_imgi.shape = (batch,1,224,224)
        imgi = np.transpose(imgi,(0,2,3,1))
        #imgi.shape = (batch,224,224,1)
        h_ori,w_ori = 224,224
        h_, w_ = [int(h_ori*scale) for scale in scales],[int(w_ori*scale) for scale in scales]
        all_dices = np.zeros((len(scales)*styles.shape[0],bs))#(3*3,bs)
        all_imgs = np.zeros((len(scales),bs,224,224,1))
        all_imgs[0,:,:,:,:] = imgi
        random_x_y = []
        #iterate each scale
        for ind,(h,w) in enumerate(zip(h_,w_)):
            img_tmp = np.transpose(np.reshape(imgi,(bs,224,224)),(1,2,0)) #224,224,bs
            img_curr=None
            if ind!=0:
                img_curr = cv2.resize(img_tmp, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
                if ind==1:#shrink
                    #tmp = np.zeros((h_ori,w_ori,3))
                    img_tmp = np.zeros((224,224,bs))
                    img_tmp[22:(22+h),22:(22+w),:] = img_curr 
                    img_curr = img_tmp
                elif ind==2 or ind==3:#enlarge
                    img_curr,x,y = self.get_random_crop(img_curr, 224, 224)
                    random_x_y.append((x,y))
            else:
                img_curr = img_tmp
            
            imgi_o = np.float32(img_curr) #(224,224,bs*1)
            imgi_o = np.transpose(np.reshape(imgi_o,(224,224,bs,1)), (2,0,1,3))
            
            all_imgs[ind,:,:,:,:] = imgi_o
            
            imgi_r = cv2.rotate(np.float32(img_curr), cv2.ROTATE_90_CLOCKWISE)
            imgi_r = np.transpose(np.reshape(imgi_r,(224,224,bs,1)), (2,0,1,3))
            
            imgi_d = cv2.rotate(np.float32(img_curr), cv2.ROTATE_180)
            imgi_d = np.transpose(np.reshape(imgi_d,(224,224,bs,1)), (2,0,1,3))
            
            #iterate each style
            for style_ind in range(style_num):
                curr_style = styles[style_ind,:,:,:]#shape = 3,h,w
                curr_style_b = [curr_style for _ in range(bs)]
                curr_style_b = np.stack(curr_style_b, axis=0)#shape = bs,3,h,w
                ST = self._to_device(curr_style_b)
                y_all_orientation = []
                #at each scale, each orientation, perform forward
                for index, imgs_orientation in enumerate([imgi_o,imgi_r,imgi_d]):
                    
                    imgs_orientation = np.transpose(imgs_orientation,(0,3,1,2))#shape = (bs,1,224,224)
                    #print('1050',imgs_orientation.shape)
                    imgs_orientation = np.stack((imgs_orientation[:,0,:,:],imgs_orientation[:,0,:,:],imgs_orientation[:,0,:,:]),axis=1)
                    #print('1052',imgs_orientation.shape)
                    imgs_orientation = self._to_device(imgs_orientation)
                    img_fix = None
                    with torch.no_grad():
                        loss_c, loss_s,t_img_3,t_img = self.style_network(imgs_orientation,ST,alpha=1.)
                        y = self.net(t_img)[0].cpu().numpy() #(b,3,224,224)
                        #img_fix = np.expand_dims(np.float32((y[:,2,:,:]>0)),axis=-1) #(b,224,224)
                        img_fix = np.float32(y[:,2,:,:]>0)
                        ####################
                        #print('img_fix shape', img_fix.shape)
                        img_fix = np.transpose(img_fix,(1,2,0))#224,224,b
                        if index==1:
                            img_fix = cv2.rotate(img_fix, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        elif index==2:
                            img_fix = cv2.rotate(img_fix, cv2.ROTATE_180)
                    #y_all_orientation.append(np.reshape(img_fix,(bs,-1))>0) #append(b,50000)
                    to_append = np.reshape(np.transpose(img_fix,(2,0,1)) , (bs,-1))
                    y_all_orientation.append( to_append>0)
                    ####################
                one = y_all_orientation[0] #(8,50000)
                #print('one.shape=', one.shape)
                two = y_all_orientation[1] #(8,50000)
                three = y_all_orientation[2] #(8,50000)
                intersection1 = (one * two)
                intersection2 = (one * three)
                intersection4 = (two * three)
                dice1 = (2. * np.sum(intersection1,axis=1) + 1) / (np.sum(one,axis=1) + np.sum(two,axis=1) + 1) #shape = bs
                dice2 = (2. * np.sum(intersection2,axis=1) + 1) / (np.sum(one,axis=1) + np.sum(three,axis=1) + 1)
                dice4 = (2. * np.sum(intersection4,axis=1) + 1) / (np.sum(two,axis=1) + np.sum(three,axis=1) + 1)
                dice_total = (dice1+dice2+dice4)/3 #shape =bs
                all_dices[ind*style_num+style_ind,:] = dice_total

        selection = np.argmax(all_dices, axis=0) #(0,1,2,3,1,2,3,3) (3,6,0,7)/3 = (1,2,0,2)
        scale_selection = np.floor_divide(selection,style_num)
        style_selction = selection%style_num
        final_imgs = np.array([all_imgs[scale_selection[idx],idx,:,:] for idx in range(bs)])
        final_styles = np.array([styles[style_selction[idx],:,:,:] for idx in range(bs)])
        return final_imgs,scale_selection,random_x_y, final_styles
    
    def adjust_lbl(self, selections, lbl,random_xy, scales = [1,0.8,1.5,2]):
        w_ori,h_ori = lbl.shape[3], lbl.shape[2]
        lbl = lbl
        #print(lbl.shape) 8,3,224,224
        with torch.no_grad():
            for i, select in enumerate(selections):
                if select==0:
                    continue
                w_new, h_new = int(w_ori*scales[select]),int(h_ori*scales[select])
                img_curr_1 = cv2.resize(np.transpose(lbl[i,0:1,:,:],(1,2,0)), dsize=(int(w_ori*scales[select]), int(h_ori*scales[select])),interpolation=cv2.INTER_NEAREST)
                img_curr_1 = np.expand_dims(img_curr_1, axis=2)
                #img_curr_1 = 224,224,1
                img_curr_23 = cv2.resize(np.transpose(lbl[i,1:,:,:],(1,2,0)), dsize=(int(w_ori*scales[select]), int(h_ori*scales[select])),interpolation=cv2.INTER_LINEAR)
                #img_curr_1 = 224,224,2
                #print(img_curr_1.shape, img_curr_23.shape)
                img_curr = np.concatenate((img_curr_1, img_curr_23), axis=2)
                if select==1:
                    img_tmp = np.zeros((224,224,3))
                    img_tmp[22:(22+h_new),22:(22+w_new),:] = img_curr 
                    img_curr = img_tmp
                else:
                    if select==2:
                        img_curr = self.get_crop(img_curr, 224, 224,random_xy[0])
                    else:#when select==3
                        img_curr = self.get_crop(img_curr, 224, 224,random_xy[1])
                #print(lbl.dtype)
                #print(img_curr.dtype)
                tmp1 = np.transpose(img_curr,(2,0,1))
                lbl[i,:,:,:] = tmp1
        return lbl
    def _train_step(self, x, lbl,styles):
        full_x = np.stack((x[:,:,:,0],x[:,:,:,0],x[:,:,:,0]),axis=3)#bs,224,224,3
        full_x = np.transpose(full_x,(0,3,1,2))
        full_x = self._to_device(full_x)
        
        ST = self._to_device(styles)
        
        sample_num = x.shape[0]
        self.style_network.train()
        if ST.shape[0]!=full_x.shape[0]:
                ST = ST[:full_x.shape[0],:,:,:]
        loss_c, loss_s,t_img_3,t_img = self.style_network(full_x,ST,alpha=1.)
        c = int(self.content_weight)
        s = int(self.style_weight)
        loss_st = c * (loss_c/5) + s * (loss_s/5)
        self.optimizer.zero_grad()
        self.net.train()
        y_ = self.net(t_img)
        y = y_[0]
        seg = int(self.seg_weight)
        loss_seg = self.loss_fn(lbl,y)
        loss_total = loss_seg*seg + loss_st
        loss_total.backward()
        train_loss = loss_total.item()
        self.optimizer.step()
        train_loss *= len(x)
        return loss_c.item(), loss_s.item(), train_loss

    def _test_eval(self, x, lbl, styles, epoch = 0, batch = 0):
        #X = self._to_device(x)
        self.net.eval()
        self.style_network.eval()
        sample_num = x.shape[0]
        if self.save_img_dir == None:
            save_dir = '/content/drive/MyDrive/Hardvard_BC_research/multiscale_co_training/save_image/'
            new_dir =  'sc_ratio'+str(int(self.style_weight/self.content_weight))+'_seg_ratio'+str(self.seg_weight)
            path = os.path.join(save_dir, new_dir)
            if not os.path.exists(path):
                os.mkdir(path)
            self.save_img_dir = path
        with torch.no_grad():
            full_x = np.stack((x[:,:,:,0],x[:,:,:,0],x[:,:,:,0]),axis=3)#bs,224,224,3
            
            full_x = np.transpose(full_x,(0,3,1,2)) 
            full_x = self._to_device(full_x)
            ST = self._to_device(styles)
            if ST.shape[0]!=full_x.shape[0]:
                ST = ST[:full_x.shape[0],:,:,:]
            loss_c, loss_s,t_img_3,t_img = self.style_network(full_x,ST,alpha=1.)
            y_ = self.net(t_img)
            y = y_[0]
        
            loss_seg = self.loss_fn(lbl,y)
            test_loss = loss_seg.item()
            test_loss *= len(x)
        return loss_c.item(), loss_s.item(), test_loss
        
    def _set_optimizer(self, learning_rate, momentum, weight_decay, SGD=False):
        print(learning_rate)
        param = list(self.net.parameters()) + list(self.style_network.decoder.parameters())+list(self.style_network.final_projection.parameters())
        if SGD:
            self.optimizer = torch.optim.SGD(param, lr=learning_rate,
                                        momentum=momentum, weight_decay=weight_decay)
        else:
            import torch_optimizer as optim # for RADAM optimizer
            self.optimizer = optim.RAdam(param, lr=learning_rate, betas=(0.95, 0.999), #changed to .95
                                        eps=1e-08, weight_decay=weight_decay)
            core_logger.info('>>> Using RAdam optimizer')
            self.optimizer.current_lr = learning_rate
        
    def _set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _set_criterion(self):
        if self.unet:
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        else:
            self.criterion  = nn.MSELoss(reduction='mean')
            self.criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
            
    def _train_net(self, train_data, train_labels, styles,
              test_data=None, test_labels=None,
              save_path=None, save_every=100, save_each=False,
              learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001, 
              SGD=True, batch_size=8, nimg_per_epoch=None, rescale=True, model_name=None): 
        """ train function uses loss function self.loss_fn in models.py"""
        print('trainnet core 1051')
        a1 = train_data[1]
        d = datetime.datetime.now()
        self.n_epochs = n_epochs
        if isinstance(learning_rate, (list, np.ndarray)):
            if isinstance(learning_rate, np.ndarray) and learning_rate.ndim > 1:
                raise ValueError('learning_rate.ndim must equal 1')
            elif len(learning_rate) != n_epochs:
                raise ValueError('if learning_rate given as list or np.ndarray it must have length n_epochs')
            self.learning_rate = learning_rate
            self.learning_rate_const = mode(learning_rate)[0][0]
        else:
            self.learning_rate_const = learning_rate
            # set learning rate schedule    
            if SGD:
                LR = np.linspace(0, self.learning_rate_const, 10)
                if self.n_epochs > 250:
                    LR = np.append(LR, self.learning_rate_const*np.ones(self.n_epochs-100))
                    for i in range(10):
                        LR = np.append(LR, LR[-1]/2 * np.ones(10))
                else:
                    LR = np.append(LR, self.learning_rate_const*np.ones(max(0,self.n_epochs-10)))
            else:
                LR = self.learning_rate_const * np.ones(self.n_epochs)
            self.learning_rate = LR

        self.batch_size = batch_size
        self._set_optimizer(self.learning_rate[0], momentum, weight_decay, SGD)
        self._set_criterion()
        
        nimg = len(train_data)

        # compute average cell diameter
        diam_train = np.array([utils.diameters(train_labels[k][0])[0] for k in range(len(train_labels))])
        diam_train_mean = diam_train[diam_train > 0].mean()
        self.diam_labels = diam_train_mean
        if rescale:
            diam_train[diam_train<5] = 5.
            if test_data is not None:
                diam_test = np.array([utils.diameters(test_labels[k][0])[0] for k in range(len(test_labels))])
                diam_test[diam_test<5] = 5.
            scale_range = 0.5
            core_logger.info('>>>> median diameter set to = %d'%self.diam_mean)
        else:
            scale_range = 1.0
            
        core_logger.info(f'>>>> mean of training label mask diameters (saved to model) {diam_train_mean:.3f}')
        self.net.diam_labels.data = torch.ones(1, device=self.device) * diam_train_mean

        nchan = train_data[0].shape[0]
        core_logger.info('>>>> training network with %d channel input <<<<'%nchan)
        core_logger.info('>>>> LR: %0.5f, batch_size: %d, weight_decay: %0.5f'%(self.learning_rate_const, self.batch_size, weight_decay))
        
        if test_data is not None:
            core_logger.info(f'>>>> ntrain = {nimg}, ntest = {len(test_data)}')
        else:
            core_logger.info(f'>>>> ntrain = {nimg}')
        
        tic = time.time()

        
        lavg, nsum = 0, 0

        if save_path is not None:
            _, file_label = os.path.split(save_path)
            file_path = os.path.join(save_path, 'models/')

            if not os.path.exists(file_path):
                os.makedirs(file_path)
        else:
            core_logger.warning('WARNING: no save_path given, model not saving')

        ksave = 0
        rsc = 1.0

        # cannot train with mkldnn
        self.net.mkldnn = False

        # get indices for each epoch for training
        np.random.seed(0)
        inds_all = np.zeros((0,), 'int32')
        if nimg_per_epoch is None or nimg > nimg_per_epoch:
            nimg_per_epoch = nimg 
        core_logger.info(f'>>>> nimg_per_epoch = {nimg_per_epoch}')
        while len(inds_all) < n_epochs * nimg_per_epoch:
            rperm = np.random.permutation(nimg)
            inds_all = np.hstack((inds_all, rperm))
        
        for iepoch in range(self.n_epochs):    
            if SGD:
                self._set_learning_rate(self.learning_rate[iepoch])
            np.random.seed(iepoch)
            rperm = inds_all[iepoch*nimg_per_epoch:(iepoch+1)*nimg_per_epoch]
            for ibatch in range(0,nimg_per_epoch,batch_size):
                inds = rperm[ibatch:ibatch+batch_size]
                rsc = diam_train[inds] / self.diam_mean if rescale else np.ones(len(inds), np.float32)
                # now passing in the full train array, need the labels for distance field
                imgi, lbl, scale,styles_np = transforms.random_rotate_and_resize(
                                        [train_data[i] for i in inds], styles, Y=[train_labels[i][1:] for i in inds],
                                        rescale=rsc, scale_range=scale_range, unet=self.unet)
                #the returned styles_np is np array (nstyles,3,h,w)
                if self.unet and lbl.shape[1]>1 and rescale:
                    lbl[:,1] *= scale[:,np.newaxis,np.newaxis]**2#diam_batch[:,np.newaxis,np.newaxis]**2

                imgi,selection,random_x_y,final_styles = self._find_scale(imgi, styles_np)
                lbl = self.adjust_lbl(selection, lbl,random_x_y)
                
                loss_c,loss_s,train_loss = self._train_step(imgi, lbl,final_styles )
                lavg += train_loss
                nsum += len(imgi) 
            
            if iepoch%10==0 or iepoch==5:
                lavg = lavg / nsum
                if test_data is not None:
                    lavgt,content_sum, style_sum, nsum = 0.,0.,0., 0
                    np.random.seed(42)
                    rperm = np.arange(0, len(test_data), 1, int)
                    for ibatch in range(0,len(test_data),batch_size):
                        inds = rperm[ibatch:ibatch+batch_size]
                        rsc = diam_test[inds] / self.diam_mean if rescale else np.ones(len(inds), np.float32)
                        imgi, lbl, scale,styles_np = transforms.random_rotate_and_resize(
                                            [test_data[i] for i in inds],styles, Y=[test_labels[i][1:] for i in inds], 
                                            scale_range=0., rescale=rsc, unet=self.unet) 
                        if self.unet and lbl.shape[1]>1 and rescale:
                            lbl[:,1] *= scale[:,np.newaxis,np.newaxis]**2
                        if iepoch>-1:
                            imgi,selection,random_x_y,final_styles = self._find_scale(imgi, styles_np)
                            lbl = self.adjust_lbl(selection, lbl,random_x_y)
                        else:
                            imgi = np.transpose(imgi[:,0:1,:,:],(0,2,3,1))
                        loss_c,loss_s,test_loss = self._test_eval(imgi, lbl, final_styles, batch = ibatch, epoch = iepoch)
                        lavgt += test_loss
                        content_sum += loss_c
                        style_sum += loss_s
                        nsum += len(imgi)

                    core_logger.info('Epoch %d, Time %4.1fs, Loss Test %2.4f,content Loss Test %2.4f,style Loss Test %2.4f, LR %2.4f'%
                            (iepoch, time.time()-tic, lavgt/nsum,content_sum/nsum,style_sum/nsum, self.learning_rate[iepoch]))
                else:
                    core_logger.info('Epoch %d, Time %4.1fs, Loss %2.4f, LR %2.4f'%
                            (iepoch, time.time()-tic, lavg, self.learning_rate[iepoch]))
                
                lavg, nsum = 0, 0
                            
            if save_path is not None:
                if iepoch==self.n_epochs-1 or iepoch%save_every==1:
                    # save model at the end
                    if save_each: #separate files as model progresses 
                        if model_name is None:
                            file_name = '{}_{}_{}_{}'.format(self.net_type, file_label, 
                                                             d.strftime("%Y_%m_%d_%H_%M_%S.%f"),
                                                             'epoch_'+str(iepoch)) 
                            file_name_decoder = '{}_{}_{}_{}'.format('adain_decoder', file_label, 
                                                             d.strftime("%Y_%m_%d_%H_%M_%S.%f"),
                                                             'epoch_'+str(iepoch))
                        else:
                            file_name = '{}_{}'.format(model_name, 'epoch_'+str(iepoch))
                            file_name_decoder = '{}_{}'.format('adain_decoder', 'epoch_'+str(iepoch))
                    else:
                        if model_name is None:
                            file_name = '{}_{}_{}'.format(self.net_type, file_label, d.strftime("%Y_%m_%d_%H_%M_%S.%f"))
                            file_name_decoder = '{}_{}_{}'.format('adain_decoder', file_label, d.strftime("%Y_%m_%d_%H_%M_%S.%f"))
                        else:
                            file_name = model_name
                            file_name_decoder = 'adain_decoder'
                  
                    file_name = os.path.join(file_path, file_name)
                    file_name_decoder = os.path.join(file_path, file_name_decoder)
                    
                    ksave += 1
                    core_logger.info(f'saving network parameters to {file_name}')
                    self.net.save_model(file_name)
                    print('saving net to ',file_name)
                    self.style_network.save_model(file_name_decoder)
                    print('saving adain net to ',file_name_decoder)
                    
                    
                    
                    #torch.save(self.state_dict(), file_name_decoder)
            else:
                file_name = save_path

        # reset to mkldnn if available
        self.net.mkldnn = self.mkldnn
        return file_name
