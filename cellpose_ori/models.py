import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
from pathlib import Path
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import torch
import cv2
import glob
import logging
models_logger = logging.getLogger(__name__)

from . import transforms, dynamics, utils, plot
from .core import UnetModel, assign_device, check_mkl, parse_model_string

_MODEL_URL = 'https://www.cellpose.org/models'
_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = pathlib.Path.home().joinpath('.cellpose', 'models')
MODEL_DIR = pathlib.Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT

MODEL_NAMES = ['cyto','nuclei','tissuenet','livecell', 'cyto2', 'general',
                'CP', 'CPx', 'TN1', 'TN2', 'TN3', 'LC1', 'LC2', 'LC3', 'LC4']

MODEL_LIST_PATH = os.fspath(MODEL_DIR.joinpath('gui_models.txt'))

def model_path(model_type, model_index, use_torch=True):
    torch_str = 'torch'
    if model_type=='cyto' or model_type=='cyto2' or model_type=='nuclei':
        basename = '%s%s_%d' % (model_type, torch_str, model_index)
    else:
        basename = model_type
    return cache_model_path(basename)

def size_model_path(model_type, use_torch=True):
    torch_str = 'torch'
    basename = 'size_%s%s_0.npy' % (model_type, torch_str)
    return cache_model_path(basename)

def cache_model_path(basename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f'{_MODEL_URL}/{basename}'
    cached_file = os.fspath(MODEL_DIR.joinpath(basename)) 
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        utils.download_url_to_file(url, cached_file, progress=True)
    return cached_file

def get_user_models():
    model_strings = []
    if os.path.exists(MODEL_LIST_PATH):
        with open(MODEL_LIST_PATH, 'r') as textfile:
            lines = [line.rstrip() for line in textfile]
            if len(lines) > 0:
                model_strings.extend(lines)
    return model_strings
    

class Cellpose():
    """ main model which combines SizeModel and CellposeModel

    Parameters
    ----------

    gpu: bool (optional, default False)
        whether or not to use GPU, will check if GPU available

    model_type: str (optional, default 'cyto')
        'cyto'=cytoplasm model; 'nuclei'=nucleus model; 'cyto2'=cytoplasm model with additional user images

    net_avg: bool (optional, default False)
        loads the 4 built-in networks and averages them if True, loads one network if False

    device: torch device (optional, default None)n
        device used for model running / training 
        (torch.device('cuda') or torch.device('cpu')), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. torch.device('cuda:1'))

    """
    def __init__(self, gpu=False, model_type='cyto', net_avg=False, device=None):
        super(Cellpose, self).__init__()
        self.torch = True
        
        # assign device (GPU or CPU)
        sdevice, gpu = assign_device(self.torch, gpu)
        self.device = device if device is not None else sdevice
        self.gpu = gpu
        
        model_type = 'cyto' if model_type is None else model_type
        
        self.diam_mean = 30. #default for any cyto model 
        nuclear = 'nuclei' in model_type
        if nuclear:
            self.diam_mean = 17. 
        
        self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                model_type=model_type,
                                diam_mean=self.diam_mean,
                                net_avg=net_avg)
        self.cp.model_type = model_type
        
        # size model not used for bacterial model
        self.pretrained_size = size_model_path(model_type, self.torch)
        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)
        self.sz.model_type = model_type
        
    def eval(self, x, batch_size=8, channels=None, channel_axis=None, z_axis=None,
             invert=False, normalize=True, diameter=30., do_3D=False, anisotropy=None,
             net_avg=False, augment=False, tile=True, tile_overlap=0.1, resample=True, interp=True,
             flow_threshold=0.4, cellprob_threshold=0.0, min_size=15, stitch_threshold=0.0, 
             rescale=None, progress=None, model_loaded=False):
        """ run cellpose and get masks

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
            invert image pixel intensity before running network (if True, image is also normalized)

        normalize: bool (optional, default True)
            normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

        diameter: float (optional, default 30.)
            if set to None, then diameter is automatically estimated if size model is loaded

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

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        resample: bool (optional, default True)
            run dynamics at original image size (will be slower but create more accurate boundaries)

        interp: bool (optional, default True)
                interpolate during 2D dynamics (not available in 3D) 
                (in previous versions it was False)

        flow_threshold: float (optional, default 0.4)
            flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

        cellprob_threshold: float (optional, default 0.0)
            all pixels with value above threshold kept for masks, decrease to find more and larger masks

        min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

        stitch_threshold: float (optional, default 0.0)
            if stitch_threshold>0.0 and not do_3D and equal image sizes, masks are stitched in 3D to return volume segmentation

        rescale: float (optional, default None)
            if diameter is set to None, and rescale is not None, then rescale is used instead of diameter for resizing image

        progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI

        model_loaded: bool (optional, default False)
            internal variable for determining if model has been loaded, used in __main__.py

        Returns
        -------
        masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

        flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
            flows[k][0] = XY flow in HSV 0-255
            flows[k][1] = XY flows at each pixel
            flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics)
            flows[k][3] = final pixel locations after Euler integration 

        styles: list of 1D arrays of length 256, or single 1D array (if do_3D=True)
            style vector summarizing each image, also used to estimate size of objects in image

        diams: list of diameters, or float (if do_3D=True)

        """        

        tic0 = time.time()
        channels = [0,0] if channels is None else channels # why not just make this a default in the function header?

        estimate_size = True if (diameter is None or diameter==0) else False
        
        if estimate_size and self.pretrained_size is not None and not do_3D and x[0].ndim < 4:
            tic = time.time()
            models_logger.info('~~~ ESTIMATING CELL DIAMETER(S) ~~~')
            diams, _ = self.sz.eval(x, channels=channels, channel_axis=channel_axis, invert=invert, batch_size=batch_size, 
                                    augment=augment, tile=tile, normalize=normalize)
            rescale = self.diam_mean / np.array(diams)
            diameter = None
            models_logger.info('estimated cell diameter(s) in %0.2f sec'%(time.time()-tic))
            models_logger.info('>>> diameter(s) = ')
            if isinstance(diams, list) or isinstance(diams, np.ndarray):
                diam_string = '[' + ''.join(['%0.2f, '%d for d in diams]) + ']'
            else:
                diam_string = '[ %0.2f ]'%diams
            models_logger.info(diam_string)
        elif estimate_size:
            if self.pretrained_size is None:
                reason = 'no pretrained size model specified in model Cellpose'
            else:
                reason = 'does not work on non-2D images'
            models_logger.warning(f'could not estimate diameter, {reason}')
            diams = self.diam_mean 
        else:
            diams = diameter

        tic = time.time()
        models_logger.info('~~~ FINDING MASKS ~~~')
        masks, flows, styles = self.cp.eval(x, 
                                            batch_size=batch_size, 
                                            invert=invert, 
                                            normalize=normalize,
                                            diameter=diameter,
                                            rescale=rescale, 
                                            anisotropy=anisotropy, 
                                            channels=channels,
                                            channel_axis=channel_axis, 
                                            z_axis=z_axis,
                                            augment=augment, 
                                            tile=tile, 
                                            do_3D=do_3D, 
                                            net_avg=net_avg, 
                                            progress=progress,
                                            tile_overlap=tile_overlap,
                                            resample=resample,
                                            interp=interp,
                                            flow_threshold=flow_threshold, 
                                            cellprob_threshold=cellprob_threshold,
                                            min_size=min_size, 
                                            stitch_threshold=stitch_threshold,
                                            model_loaded=model_loaded)
        models_logger.info('>>>> TOTAL TIME %0.2f sec'%(time.time()-tic0))
    
        return masks, flows, styles, diams

class CellposeModel(UnetModel):
    """

    Parameters
    -------------------

    gpu: bool (optional, default False)
        whether or not to save model to GPU, will check if GPU available
        
    pretrained_model: str or list of strings (optional, default False)
        full path to pretrained cellpose model(s), if None or False, no model loaded
        
    model_type: str (optional, default None)
        any model that is available in the GUI, use name in GUI e.g. 'livecell' 
        (can be user-trained or model zoo)
        
    net_avg: bool (optional, default False)
        loads the 4 built-in networks and averages them if True, loads one network if False
        
    diam_mean: float (optional, default 30.)
        mean 'diameter', 30. is built in value for 'cyto' model; 17. is built in value for 'nuclei' model; 
        if saved in custom model file (cellpose>=2.0) then it will be loaded automatically and overwrite this value
        
    device: torch device (optional, default None)
        device used for model running / training 
        (torch.device('cuda') or torch.device('cpu')), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. torch.device('cuda:1'))

    residual_on: bool (optional, default True)
        use 4 conv blocks with skip connections per layer instead of 2 conv blocks
        like conventional u-nets

    style_on: bool (optional, default True)
        use skip connections from style vector to all upsampling layers

    concatenation: bool (optional, default False)
        if True, concatentate downsampling block outputs with upsampling block inputs; 
        default is to add 
    
    nchan: int (optional, default 2)
        number of channels to use as input to network, default is 2 
        (cyto + nuclei) or (nuclei + zeros)
    
    """
    
    def __init__(self, gpu=False, pretrained_model=False,
                    pretrained_model_adain='.',
                    model_type=None, net_avg=False,
                    diam_mean=30., device=None,
                    residual_on=True, style_on=True, concatenation=False,
                    nchan=1,style_weight = 10,content_weight = 1, seg_weight = 2, style_anchor_dir = None):
        self.torch = True
        if isinstance(pretrained_model, np.ndarray):
            pretrained_model = list(pretrained_model)
        elif isinstance(pretrained_model, str):
            pretrained_model = [pretrained_model]
        self.diam_mean = diam_mean
        builtin = True
        self.style_anchor_dir = style_anchor_dir
        print('models.py353324g',model_type, pretrained_model)
        
        builtin = False
        if pretrained_model:
            pretrained_model_string = pretrained_model[0]
            params = parse_model_string(pretrained_model_string)
            if params is not None:
                _, residual_on, style_on, concatenation = params 
            models_logger.info(f'>>>> loading model {pretrained_model_string}')
            

                
        # initialize network
        super().__init__(gpu=gpu, pretrained_model=False,
                         diam_mean=self.diam_mean, net_avg=net_avg, device=device,
                         residual_on=residual_on, style_on=style_on, concatenation=concatenation,
                         style_weight = style_weight,content_weight = content_weight, seg_weight = seg_weight,
                        nchan=nchan)

        self.unet = False
        self.pretrained_model = pretrained_model
        self.pretrained_model_adain = pretrained_model_adain
        if self.pretrained_model:
            
            self.net.load_model(self.pretrained_model[0], device=self.device)
            self.style_network.load_state_dict(torch.load(self.pretrained_model_adain, map_location=torch.device('cpu')))
            self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
            self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]
            models_logger.info(f'>>>> model diam_mean = {self.diam_mean: .3f} (ROIs rescaled to this size during training)')
            if not builtin:
                models_logger.info(f'>>>> model diam_labels = {self.diam_labels: .3f} (mean diameter of training ROIs)')
        
        ostr = ['off', 'on']
        self.net_type = 'cellpose_residual_{}_style_{}_concatenation_{}'.format(ostr[residual_on],
                                                                                   ostr[style_on],
                                                                                   ostr[concatenation]
                                                                                 ) 
    
    def eval(self, x, batch_size=8, channels=None, channel_axis=None, 
             z_axis=None, normalize=False, invert=False, 
             rescale=None, diameter=None, do_3D=False, anisotropy=None, net_avg=False, 
             augment=False, tile=True, tile_overlap=0.1,
             resample=True, interp=True,
             flow_threshold=0.4, cellprob_threshold=0.0,
             compute_masks=True, min_size=15, stitch_threshold=0.0, progress=None, model_loaded=False):
        print('channel_axis', channel_axis)
        print('z_axis', z_axis)
        print('normalize', normalize)
        print('invert', invert)
        print('rescale', rescale)
        print('diameter', diameter)
        print('do_3D', do_3D)
        print('anisotropy', anisotropy)
        print('net_avg', net_avg)
        print('augment', augment)
        print('tile', tile)
        print('tile_overlap', tile_overlap)
        print('resample', resample)
        print('interp', interp)
        print('progress', progress)
        
        """
            segment list of images x, or 4D array - Z x nchan x Y x X

            Parameters
            ----------
            x: list or array of images
                can be list of 2D/3D/4D images, or array of 2D/3D/4D images

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

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            diameter: float (optional, default None)
                diameter for each image, 
                if diameter is None, set to diam_mean or diam_train if available

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0;
                (only used if diameter is None)

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

            tile_overlap: float (optional, default 0.1)
                fraction of overlap of tiles when computing flows

            resample: bool (optional, default True)
                run dynamics at original image size (will be slower but create more accurate boundaries)

            interp: bool (optional, default True)
                interpolate during 2D dynamics (not available in 3D) 
                (in previous versions it was False)

            flow_threshold: float (optional, default 0.4)
                flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

            cellprob_threshold: float (optional, default 0.0) 
                all pixels with value above threshold kept for masks, decrease to find more and larger masks

            compute_masks: bool (optional, default True)
                Whether or not to compute dynamics and return masks.
                This is set to False when retrieving the styles for the size model.

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            stitch_threshold: float (optional, default 0.0)
                if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation

            progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI
                            

            model_loaded: bool (optional, default False)
                internal variable for determining if model has been loaded, used in __main__.py

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = XY flows at each pixel
                flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics)
                flows[k][3] = final pixel locations after Euler integration 

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """
        three_dices = None

        


        
        self.net.load_model(self.pretrained_model[0], device=self.device)
        print('498', x.shape)
        # reshape image (normalization happens in _run_cp)
        x = transforms.convert_image(x, channels, channel_axis=channel_axis, z_axis=z_axis,
                                         do_3D=(do_3D or stitch_threshold>0), 
                                         normalize=False, invert=False, nchan=self.nchan)
    
        x = x[np.newaxis,...]
        self.batch_size = batch_size


        diameter = self.diam_labels
        rescale = self.diam_mean / diameter
        print(x.shape,'before run_cp')
        masks, styles, dP, cellprob, p = self._run_cp(x,compute_masks=compute_masks,
                                                          normalize=normalize,
                                                          invert=invert,
                                                          rescale=rescale, 
                                                          net_avg=net_avg, 
                                                          resample=resample,
                                                          augment=augment, 
                                                          tile=tile, 
                                                          tile_overlap=tile_overlap,
                                                          flow_threshold=flow_threshold,
                                                          cellprob_threshold=cellprob_threshold, 
                                                          interp=interp,
                                                          min_size=min_size, 
                                                          do_3D=do_3D, 
                                                          anisotropy=anisotropy,
                                                          stitch_threshold=stitch_threshold,
                                                          )
            
        flows = [plot.dx_to_circ(dP), dP, cellprob, p]
        return masks, flows, styles

    def _run_cp(self, x, compute_masks=True, normalize=True, invert=False,
                rescale=1.0, net_avg=False, resample=True,
                augment=False, tile=True, tile_overlap=0.1,
                cellprob_threshold=0.0, 
                flow_threshold=0.4, min_size=15,
                interp=True, anisotropy=1.0, do_3D=False, stitch_threshold=0.0
                ):
        
        tic = time.time()
        shape = x.shape
        nimg = shape[0]        
        
        bd, tr = None, None

        tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
        iterator = trange(nimg, file=tqdm_out) if nimg>1 else range(nimg)
        styles = np.zeros((nimg, self.nbase[-1]), np.float32)
        

        dP = np.zeros((2, nimg, shape[1], shape[2]), np.float32)
        cellprob = np.zeros((nimg, shape[1], shape[2]), np.float32)
        for i in iterator:
                
            img = np.asarray(x[i])
            imgs_o = [img.copy()]
            all_styles_file = glob.glob(self.style_anchor_dir + '/*')
            #all_styles_file = [all_styles_file[0]]
            style_num = len(all_styles_file)
            styles_list = []
            for file in all_styles_file:
                style_green = (cv2.imread(file)/255)[:,:,1]
                style_green = cv2.resize(style_green, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
                style_green = np.stack((style_green,style_green,style_green))
                #stylei[ind,:,:,:] = style_green #3,224,224
                styles_list.append(style_green)

            h_ori,w_ori = img.shape[0], img.shape[1]
            #add this line
            scales = [1.8,3.0,4.5]
            #scales = [3.0]
            
            sizes_heights = [int(h_ori*scale) for scale in scales]
            sizes_widths = [int(w_ori*scale) for scale in scales]
            

            resized_images = [[cv2.resize(img.copy(), dsize=(w_curr, h_curr), interpolation=cv2.INTER_CUBIC)] for (h_curr, w_curr) in zip(sizes_heights,sizes_widths)]
            resized_images.append(imgs_o)
            scales.append(1.0)
            
            for idx, s in enumerate(resized_images):
                s.append(cv2.rotate(s[0], cv2.ROTATE_90_CLOCKWISE))
                s.append(cv2.rotate(s[0], cv2.ROTATE_180))
            for aa in resized_images:
                for bb in aa:
                    print(bb.shape)
                
            for ii, rot_images_scales in enumerate(resized_images):
                for j,rots in enumerate(rot_images_scales):
                    resized_images[ii][j] = transforms.resize_image(resized_images[ii][j], rsz=rescale)

            all_y = [[[] for j in range(style_num)] for ii in range(len(scales))]      

            
            for ii, rot_images_scales in enumerate(resized_images):
                for j, s in enumerate(styles_list):
                    for k, rots in enumerate(rot_images_scales):
                        y_tmp, style = self._run_nets(rots, s, net_avg=net_avg,augment=augment, tile=tile,tile_overlap=tile_overlap)
                        print(y_tmp.shape)
                        if k==1: y_tmp = cv2.rotate(y_tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        if k==2: y_tmp = cv2.rotate(y_tmp, cv2.ROTATE_180)
                        y_tmp = transforms.resize_image(y_tmp, shape[1], shape[2])
                        all_y[ii][j].append(y_tmp)
                
            #resampling
            dices_total = []
            for ii, style_rot in enumerate(all_y):
                dices_total.append([])
                for j,stylez in enumerate(style_rot):
                    one = (stylez[0]>0).flatten()
                    two = (stylez[1]>0).flatten()
                    three = (stylez[2]>0).flatten()
                    intersection1 = (one * two)
                    intersection2 = (one * three)
                    intersection3 = (two * three)
                    dice1 = (2. * intersection1.sum() + 1) / (one.sum() + two.sum() + 1)
                    dice2 = (2. * intersection2.sum() + 1) / (one.sum() + three.sum() + 1)
                    dice3 = (2. * intersection3.sum() + 1) / (two.sum() + three.sum() + 1)
                    rot_consistency = ((dice1+dice2+dice3)/3)
                    print('scale:',scales[ii], ' style:', j+1,' rot_cons:', rot_consistency)
                    dices_total[ii].append(rot_consistency)
                        
            best_scale, best_style = 0, 0
            max_dice = 0
            for scale_ind, scale in enumerate(dices_total):
                for style_ind, scale_style in enumerate(scale):
                    if scale_style > max_dice:
                        best_scale = scale_ind
                        best_style = style_ind
                        max_dice = scale_style
            
           
            yf = all_y[best_scale][best_style][0]
            cellprob[i] = yf[:,:,2]
            dP[:, i] = yf[:,:,:2].transpose((2,0,1)) 
            if self.nclasses == 4:
                if i==0:
                    bd = np.zeros_like(cellprob)
                bd[i] = yf[:,:,3]
            styles[i] = style
        del yf, style

        styles = styles.squeeze()
        
        
        net_time = time.time() - tic
        
        tic=time.time()
        niter = 200 if (do_3D and not resample) else (1 / rescale * 200)
        masks, p = [], []
        resize = [shape[1], shape[2]] if not resample else None
        for i in iterator:
            outputs = dynamics.compute_masks(dP[:,i], cellprob[i], niter=niter, cellprob_threshold=cellprob_threshold,
                                                         flow_threshold=flow_threshold, interp=interp, resize=resize,
                                                         use_gpu=self.gpu, device=self.device)
            masks.append(outputs[0])
            p.append(outputs[1])
                    
        masks = np.array(masks)
        p = np.array(p)
                
       
        flow_time = time.time() - tic

        masks, dP, cellprob, p = masks.squeeze(), dP.squeeze(), cellprob.squeeze(), p.squeeze()
            
        return masks, styles, dP, cellprob, p

    #def yf_to_mask(self,yf):
        
    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """
        veci = 5. * self._to_device(lbl[:,1:])
        lbl  = self._to_device(lbl[:,0]>.5)
        loss = self.criterion(y[:,:2] , veci) 
        loss /= 2.
        loss2 = self.criterion2(y[:,2] , lbl)
        loss = loss + loss2
        return loss        


    def train(self, train_data, train_labels,styles, train_files=None, 
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, 
              save_path=None, save_every=100, save_each=False,
              learning_rate=0.2, n_epochs=500, momentum=0.9, SGD=True,
              weight_decay=0.00001, batch_size=8, nimg_per_epoch=None,
              rescale=True, min_train_masks=5,random_style=False,
              model_name=None):

        """ train network with images train_data 
        
            Parameters
            ------------------

            train_data: list of arrays (2D or 3D)
                images for training

            train_labels: list of arrays (2D or 3D)
                labels for train_data, where 0=no masks; 1,2,...=mask labels
                can include flows as additional images

            train_files: list of strings
                file names for images in train_data (to save flows for future runs)

            test_data: list of arrays (2D or 3D)
                images for testing

            test_labels: list of arrays (2D or 3D)
                labels for test_data, where 0=no masks; 1,2,...=mask labels; 
                can include flows as additional images
        
            test_files: list of strings
                file names for images in test_data (to save flows for future runs)

            channels: list of ints (default, None)
                channels to use for training

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            save_path: string (default, None)
                where to save trained model, if None it is not saved

            save_every: int (default, 100)
                save network every [save_every] epochs

            learning_rate: float or list/np.ndarray (default, 0.2)
                learning rate for training, if list, must be same length as n_epochs

            n_epochs: int (default, 500)
                how many times to go through whole training set during training

            weight_decay: float (default, 0.00001)

            SGD: bool (default, True) 
                use SGD as optimization instead of RAdam

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            nimg_per_epoch: int (optional, default None)
                minimum number of images to train on per epoch, 
                with a small training set (< 8 images) it may help to set to 8

            rescale: bool (default, True)
                whether or not to rescale images to diam_mean during training, 
                if True it assumes you will fit a size model after training or resize your images accordingly,
                if False it will try to train the model to be scale-invariant (works worse)

            min_train_masks: int (default, 5)
                minimum number of masks an image must have to use in training set

            model_name: str (default, None)
                name of network, otherwise saved with name as params + training start time

        """
        train_data, train_labels, test_data, test_labels, styles, run_test = transforms.reshape_train_test(train_data, train_labels,styles,
                                                                                                   test_data, test_labels,
                                                                                                   channels, normalize)
        # check if train_labels have flows
        # if not, flows computed, returned with labels as train_flows[i][0]
        train_flows = dynamics.labels_to_flows(train_labels, files=train_files, use_gpu=self.gpu, device=self.device)
        
        if run_test:
            test_flows = dynamics.labels_to_flows(test_labels, files=test_files, use_gpu=self.gpu, device=self.device)
            #print(len(test_flows)) = 40
            #print(len(test_flows[0])) = 4
        else:
            test_flows = None
        
        nmasks = np.array([label[0].max() for label in train_flows])
        nremove = (nmasks < min_train_masks).sum()
        print('nremove:',nremove)
        if nremove > 0:
            models_logger.warning(f'{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set')
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            train_data = [train_data[i] for i in ikeep]
            train_flows = [train_flows[i] for i in ikeep]
        if channels is None:
            models_logger.warning('channels is set to None, input must therefore have nchan channels (default is 2)')
        model_path = self._train_net(train_data, train_flows, styles,
                                     test_data=test_data, test_labels=test_flows,
                                     save_path=save_path, save_every=save_every, save_each=save_each,
                                     learning_rate=learning_rate, n_epochs=n_epochs, 
                                     momentum=momentum, weight_decay=weight_decay, 
                                     SGD=SGD, batch_size=batch_size, nimg_per_epoch=nimg_per_epoch, 
                                     rescale=rescale, model_name=model_name)
        self.pretrained_model = model_path
        return model_path


