import sys, os, argparse, glob, pathlib, time
import subprocess

import numpy as np
from natsort import natsorted
from tqdm import tqdm
from cellpose import utils, models, io, core

try:
    from cellpose.gui import gui 
    GUI_ENABLED = True 
except ImportError as err:
    GUI_ERROR = err
    GUI_ENABLED = False
    GUI_IMPORT = True
except Exception as err:
    GUI_ENABLED = False
    GUI_ERROR = err
    GUI_IMPORT = False
    raise
    
import logging

# settings re-grouped a bit
def main():
    parser = argparse.ArgumentParser(description='cellpose parameters')
    
    # settings for CPU vs GPU
    hardware_args = parser.add_argument_group("hardware arguments")
    hardware_args.add_argument('--use_gpu', action='store_true', help='use gpu if torch with cuda installed')
    hardware_args.add_argument('--gpu_device', required=False, default='0', type=str, help='which gpu device to use, use an integer for torch, or mps for M1')
    hardware_args.add_argument('--check_mkl', action='store_true', help='check if mkl working')
        
    # settings for locating and formatting images
    
    
    input_img_args = parser.add_argument_group("input image arguments")
    
    input_img_args.add_argument('--style_anchor',
                        default='/content/drive/MyDrive/Hardvard_BC_research/cellpose-main/style_anchors_3', type=str, help='folder containing the style anchor images.')
    input_img_args.add_argument('--style_selection_random_fix',
                        default=False, type=bool, help='if the style selection is random or fixed.')
    
    input_img_args.add_argument('--dir',
                        default=[], type=str, help='folder containing data to run or train on.')
    input_img_args.add_argument('--image_path',
                        default=[], type=str, help='if given and --dir not given, run on single image instead of folder (cannot train with this option)')
    input_img_args.add_argument('--look_one_level_down', action='store_true', help='run processing on all subdirectories of current folder')
    input_img_args.add_argument('--img_filter',
                        default=[], type=str, help='end string for images to run on')
    input_img_args.add_argument('--channel_axis',
                        default=None, type=int, help='axis of image which corresponds to image channels')
    input_img_args.add_argument('--z_axis',
                        default=None, type=int, help='axis of image which corresponds to Z dimension')
    input_img_args.add_argument('--chan',
                        default=0, type=int, help='channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s')
    input_img_args.add_argument('--chan2',
                        default=0, type=int, help='nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s')
    input_img_args.add_argument('--invert', action='store_true', help='invert grayscale channel')
    input_img_args.add_argument('--all_channels', action='store_true', help='use all channels in image if using own model and images with special channels')
    
    # model settings 
    model_args = parser.add_argument_group("model arguments")
    model_args.add_argument('--pretrained_model', required=False, default='cyto', type=str, help='model to use for running or starting training')
    model_args.add_argument('--pretrained_model_adain', required=False, default='cyto', type=str, help='style transfer model to use for running or starting training')
    model_args.add_argument('--add_model', required=False, default=None, type=str, help='model path to copy model to hidden .cellpose folder for using in GUI/CLI')
    model_args.add_argument('--unet', action='store_true', help='run standard unet instead of cellpose flow output')
    model_args.add_argument('--nclasses',default=3, type=int, help='if running unet, choose 2 or 3; cellpose always uses 3')

    # algorithm settings
    algorithm_args = parser.add_argument_group("algorithm arguments")
    algorithm_args.add_argument('--content_weight', default=1,type=int)
    algorithm_args.add_argument('--style_weight', default=10,type=int)
    algorithm_args.add_argument('--seg_weight', default=2,type=int)
   
    algorithm_args.add_argument('--no_resample', action='store_true', help="disable dynamics on full image (makes algorithm faster for images with large diameters)")
    algorithm_args.add_argument('--net_avg', action='store_true', help='run 4 networks instead of 1 and average results')
    algorithm_args.add_argument('--no_interp', action='store_true', help='do not interpolate when running dynamics (was default)')
    algorithm_args.add_argument('--no_norm', action='store_true', help='do not normalize images (normalize=False)')
    algorithm_args.add_argument('--do_3D', action='store_true', help='process images as 3D stacks of images (nplanes x nchan x Ly x Lx')
    algorithm_args.add_argument('--diameter', required=False, default=30., type=float, 
                        help='cell diameter, if 0 will use the diameter of the training labels used in the model, or with built-in model will estimate diameter for each image')
    algorithm_args.add_argument('--stitch_threshold', required=False, default=0.0, type=float, help='compute masks in 2D then stitch together masks with IoU>0.9 across planes')
    algorithm_args.add_argument('--fast_mode', action='store_true', help='now equivalent to --no_resample; make code run faster by turning off resampling')
    
    algorithm_args.add_argument('--flow_threshold', default=0.4, type=float, help='flow error threshold, 0 turns off this optional QC step. Default: %(default)s')
    algorithm_args.add_argument('--cellprob_threshold', default=0, type=float, help='cellprob threshold, default is 0, decrease to find more and larger masks')
    
    algorithm_args.add_argument('--anisotropy', required=False, default=1.0, type=float,
                        help='anisotropy of volume in 3D')
    algorithm_args.add_argument('--exclude_on_edges', action='store_true', help='discard masks which touch edges of image')
    
    # output settings
    output_args = parser.add_argument_group("output arguments")
    output_args.add_argument('--save_png', action='store_true', help='save masks as png and outlines as text file for ImageJ')
    output_args.add_argument('--save_tif', action='store_true', help='save masks as tif and outlines as text file for ImageJ')
    output_args.add_argument('--no_npy', action='store_true', help='suppress saving of npy')
    output_args.add_argument('--savedir',
                        default=None, type=str, help='folder to which segmentation results will be saved (defaults to input image directory)')
    output_args.add_argument('--dir_above', action='store_true', help='save output folders adjacent to image folder instead of inside it (off by default)')
    output_args.add_argument('--in_folders', action='store_true', help='flag to save output in folders (off by default)')
    output_args.add_argument('--save_flows', action='store_true', help='whether or not to save RGB images of flows when masks are saved (disabled by default)')
    output_args.add_argument('--save_outlines', action='store_true', help='whether or not to save RGB outline images when masks are saved (disabled by default)')
    output_args.add_argument('--save_ncolor', action='store_true', help='whether or not to save minimal "n-color" masks (disabled by default')
    output_args.add_argument('--save_txt', action='store_true', help='flag to enable txt outlines for ImageJ (disabled by default)')

    # training settings
    training_args = parser.add_argument_group("training arguments")
    training_args.add_argument('--train', action='store_true', help='train network using images in dir')
    training_args.add_argument('--train_size', action='store_true', help='train size network at end of training')
    training_args.add_argument('--test_dir',
                        default=[], type=str, help='folder containing test data (optional)')
    training_args.add_argument('--mask_filter',
                        default='_masks', type=str, help='end string for masks to run on. use "_seg.npy" for manual annotations from the GUI. Default: %(default)s')
    training_args.add_argument('--diam_mean',
                        default=30., type=float, help='mean diameter to resize cells to during training -- if starting from pretrained models it cannot be changed from 30.0')
    training_args.add_argument('--learning_rate',
                        default=0.2, type=float, help='learning rate. Default: %(default)s')
    training_args.add_argument('--weight_decay',
                        default=0.00001, type=float, help='weight decay. Default: %(default)s')
    training_args.add_argument('--n_epochs',
                        default=500, type=int, help='number of epochs. Default: %(default)s')
    training_args.add_argument('--batch_size',
                        default=8, type=int, help='batch size. Default: %(default)s')
    training_args.add_argument('--min_train_masks',
                        default=5, type=int, help='minimum number of masks a training image must have to be used. Default: %(default)s')
    training_args.add_argument('--residual_on',
                        default=1, type=int, help='use residual connections')
    training_args.add_argument('--style_on',
                        default=1, type=int, help='use style vector')
    training_args.add_argument('--concatenation',
                        default=0, type=int, help='concatenate downsampled layers with upsampled layers (off by default which means they are added)')
    training_args.add_argument('--save_every',
                        default=100, type=int, help='number of epochs to skip between saves. Default: %(default)s')
    training_args.add_argument('--save_each', action='store_true', help='save the model under a different filename per --save_every epoch for later comparsion')
    
    # misc settings
    parser.add_argument('--verbose', action='store_true', help='show information about running and settings and save to log')
    
    args = parser.parse_args()

    if args.check_mkl:
        mkl_enabled = models.check_mkl()
    else:
        mkl_enabled = True
    
    pretrained_model_adain = ""

    from .io import logger_setup
    logger, log_file = logger_setup()
    use_gpu = False
    channels = [args.chan, args.chan2]
    # find images
    if len(args.img_filter)>0:
        imf = args.img_filter
    else:
        imf = None


    # Check with user if they REALLY mean to run without saving anything 
    if not (args.train or args.train_size):
        saving_something = args.save_png or args.save_tif or args.save_flows or args.save_ncolor or args.save_txt
                    
    device, gpu = models.assign_device(use_torch=True, gpu=args.use_gpu, device=args.gpu_device)

    if args.pretrained_model is None or args.pretrained_model == 'None' or args.pretrained_model == 'False' or args.pretrained_model == '0':
        pretrained_model = False
    else:
        print('set model path from args here!!!!!!!!!!!!!!!!!!!!!!')
        pretrained_model = args.pretrained_model
        pretrained_model_adain = args.pretrained_model_adain
            
    model_type = None
    print('main189',pretrained_model,os.path.exists(pretrained_model),'aaaabbc3exss')
    if pretrained_model and not os.path.exists(pretrained_model):
        model_type = pretrained_model if pretrained_model is not None else 'cyto'
        model_strings = models.get_user_models()
        all_models = models.MODEL_NAMES.copy() 
        all_models.extend(model_strings)
        if ~np.any([model_type == s for s in all_models]):
            model_type = 'cyto'
            logger.warning('pretrained model has incorrect path')

        if model_type=='nuclei':
            szmean = 17. 
        else:
            szmean = 30.
    print('203model type:',model_type)
    builtin_size = model_type == 'cyto' or model_type == 'cyto2' or model_type == 'nuclei'
    print('203: ',pretrained_model)
    print('204: ',pretrained_model_adain)
    if len(args.image_path) > 0 and (args.train or args.train_size):
        raise ValueError('ERROR: cannot train model with single image input')
    print('209model type:',model_type)

    print('2model type:',model_type)
    test_dir = None if len(args.test_dir)==0 else args.test_dir
    output = io.load_train_test_data(args.dir,args.style_anchor, test_dir, imf, args.mask_filter, args.unet, args.look_one_level_down)
    images, labels, image_names, test_images, test_labels, image_names_test, style_train, style_test = output# here style is a list of cv2 read images
    # training with all channels
    if args.all_channels:
        img = images[0]
        if img.ndim==3:
            nchan = min(img.shape)
        elif img.ndim==2:
            nchan = 1
        channels = None 
    else:
        nchan = 2 
            
    # model path
    szmean = args.diam_mean
    if not os.path.exists(pretrained_model) and model_type is None:
        if not args.train:
            error_message = 'ERROR: model path missing or incorrect - cannot train size model'
            logger.critical(error_message)
            raise ValueError(error_message)
        pretrained_model = False
        logger.info('>>>> training from scratch')
    print('334: ',pretrained_model)
    print('335: ',pretrained_model_adain)
    if args.train:
        logger.info('>>>> during training rescaling images to fixed diameter of %0.1f pixels'%args.diam_mean)
                
            # initialize model
    model = models.CellposeModel(device=device,
                                        pretrained_model=pretrained_model,
                                        pretrained_model_adain=pretrained_model_adain,
                                        model_type=model_type, 
                                        diam_mean=szmean,
                                        residual_on=args.residual_on,
                                        style_on=args.style_on,
                                        concatenation=args.concatenation,
                                        nchan=1,style_weight = args.style_weight,content_weight = args.content_weight, seg_weight = args.seg_weight)
                                            #nchan=nchan)
            
            # train segmentation model
    if args.train:
        cpmodel_path = model.train(images, labels,style_train, train_files=image_names,
                                          test_data=test_images, test_labels=test_labels, test_files=image_names_test,
                                          learning_rate=args.learning_rate, 
                                          weight_decay=args.weight_decay,
                                          channels=channels,
                                          save_path=os.path.realpath(args.dir), save_every=args.save_every,
                                          save_each=args.save_each,
                                          n_epochs=args.n_epochs,
                                          batch_size=args.batch_size, 
                                          min_train_masks=args.min_train_masks)
        model.pretrained_model = cpmodel_path
        logger.info('>>>> model trained and saved to %s'%cpmodel_path)

            # train size model
    if args.train_size:
        sz_model = models.SizeModel(cp_model=model, device=device)
        masks = [lbl[0] for lbl in labels]
        test_masks = [lbl[0] for lbl in test_labels] if test_labels is not None else test_labels
        # data has already been normalized and reshaped
        sz_model.train(images, masks, test_images, test_masks, 
                            channels=None, normalize=False,
                                batch_size=args.batch_size)
        if test_images is not None:
            predicted_diams, diams_style = sz_model.eval(test_images, 
                                                            channels=None,
                                                            normalize=False)
            ccs = np.corrcoef(diams_style, np.array([utils.diameters(lbl)[0] for lbl in test_masks]))[0,1]
            cc = np.corrcoef(predicted_diams, np.array([utils.diameters(lbl)[0] for lbl in test_masks]))[0,1]
            logger.info('style test correlation: %0.4f; final test correlation: %0.4f'%(ccs,cc))
            np.save(os.path.join(args.test_dir, '%s_predicted_diams.npy'%os.path.split(cpmodel_path)[1]), 
                            {'predicted_diams': predicted_diams, 'diams_style': diams_style})

if __name__ == '__main__':
    main()
    
