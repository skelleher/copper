from collections import namedtuple
import numpy as np
import math
import requests
import io
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plot
from mpl_toolkits.axes_grid1 import ImageGrid 
import sklearn
from sklearn.metrics import confusion_matrix
import itertools
import visdom
import logging
from enum import Enum
import tornado
import tornado.log # just to silence Visdom

Rectangle = namedtuple("Rectangle", "x, y, width, height")

Normalization = namedtuple( "Normalization", "mean, std" )
NORMALIZE_IMAGENET = Normalization( mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ] )


def image_from_url(url):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    return img

def tensor_from_bytes(data):
    return torch.tensor( [ b for b in data ] )

def tensor_from_image(img):
    return transforms.ToTensor()(img)


# Assumes tensor is in Torch's canonical CxHxW format
def image_from_tensor( tensor, bpp=None, width=None, height=None ):
    tensor_np = tensor.numpy()

#    tensor_np = np.uint8( tensor_np * 255 )
    
    # PIL requires images in HxWxC order
#    tensor_np = np.transpose( tensor_np, (1,2,0) )
    
    if ( 3 == tensor.dim() ):
        bpp    = tensor.size()[0]
        height = tensor.size()[1]
        width  = tensor.size()[2]

        # PIL requires images in HxWxC order
        tensor_np = np.transpose( tensor_np, (1,2,0) )
        #print("3D: %d x %d x %d" % (width, height, bpp))
        
    elif ( 2 == tensor.dim() ):
        bpp    = 1
        height = tensor.size()[0]
        width  = tensor.size()[1]

        #print("2D: %d x %d" % (width, height))
        
    elif ( 1 == tensor.dim() ):
        # If flat and CxWxH not specified, assume a square image and hope for the best
        length = tensor.size()[0]
        print( "WARN: image_from_tensor() without dimensions, assuming square" )
        
        # If divisible by 3, assume RGB
        if ( (not bpp) and length % 3 == 0 ):
            bpp = 3
            width = width or (int) (math.sqrt( length / bpp ))
            height = height or width
            tensor_np = tensor_np.reshape(bpp, height, width)
            tensor_np = np.transpose( tensor_np, (1,2,0) )
        else:
            # Assume grayscale
            bpp = 1
            tensor_np = tensor_np.reshape(height, width)
        
        print("1D: %d x %d x %d" % (width, height, bpp))


    if bpp == 1:
        # Normalize the values so Image doesn't clip them
#        tensor_np = tensor_np / tensor_np.max()

        # PIL expects HxW image to be of type 'L' (luminance / 8-bit) 
        if False:
            # if range [0, 255] ?
            tensor_np = tensor_np.reshape(height, width).astype( 'byte' ) # BUG: clipping values?
            img = Image.fromarray( tensor_np, 'L' )
        else:
            # if range [-1, 1] ?
            tensor_np = tensor_np.reshape(height, width)
            img = Image.fromarray( tensor_np, 'F' ).convert( 'L' )

    elif bpp == 3:
        tensor_np = tensor.numpy()
        tensor_np = tensor_np.reshape(height, width, bpp)
        img = Image.fromarray( tensor_np, 'RGB' )
        
    return img


def images_from_tensors(tensors, bpp, width, height):
    images = []

    for tensor in tensors:
        #tensor_np = tensor.numpy()
        ###img = Image.fromarray( tensor_np, 'F' )
        img = image_from_tensor( tensor, bpp, width, height )
        images.append( img )

    return images


def images_from_tensor(tensor, count, bpp, width, height):

    # NO: there is transforms.ToPILImage()( tensor ) for this
    # But it does not support grayscale in the stable release

    if tensor.is_cuda:
        tensor = tensor.cpu()

    tensor_np = tensor.numpy()
    tensor_bitmaps = tensor_np.reshape( (count, bpp, width, height) )

    images = []
    for bitmap in tensor_bitmaps:
        bitmap = np.uint8( bitmap * 255 )
        bitmap = np.transpose( bitmap, (1,2,0) )
        
        if bpp == 3:
            img = Image.fromarray(bitmap);
        if bpp == 1:
            img = Image.fromarray(bitmap, 'L')

        images.append( img )

    return images


# TODO: new up a MatPlotLib or Visdom instance, which exposes these methods, for
# showing a) inline in a notebook or b) remotely to a visdom server
def image_grid_show(images, cols=None, rows=None, width=512, height=512, labels=None):
    if not cols or not rows:
        batch_size = len( tensors )
        rows, cols = grid_size( batch_size )
    
    dpi = 192
    width_inches  = ( width  * cols ) / dpi 
    height_inches = ( height * rows ) / dpi
    print( "%d x %d : %d x %d pix = %f x %f in" % (cols, rows, width, height, width_inches, height_inches) )

    fig  = plot.figure(1, (width_inches, height_inches))
    grid = ImageGrid(fig, 
                    111,
                    nrows_ncols = (rows, cols),
                    axes_pad = 0.01,
                    label_mode="1")

    for i in range(len(images)):
        cell = grid[i]
        cell.get_xaxis().set_visible(False)
        cell.get_yaxis().set_visible(False)
        cell.imshow(images[i], cmap='gray')

        if labels:
            cell.annotate( labels[ i ], fontsize=12, xy=(12, 25), backgroundcolor="white", color="black"  )

    plot.show()


def tensor_grid_show(tensors, cols=None, rows=None, bpp=3, width=512, height=512, labels=None):
    #if len(tensors) < rows * cols:
    #    print( "Error: tensor_grid_show: %d tensors < %d x %d" % (len(tensors), cols, rows) )
    #    return;

    if type(tensors) is list:
        tensors = torch.stack(tensors)

    #if not type(tensors) is list:
    if tensors.dim() <= 3:
        return tensor_show( tensors, bpp, width, height, labels )

    if not cols or not rows:
        batch_size = len( tensors )
        rows, cols = grid_size( batch_size )
   
    #print( "tensor_grid_show: %s" % str(tensors.size()) ) 

    dpi = 192
    width_inches  = ( width  * cols ) / dpi 
    height_inches = ( height * rows ) / dpi
    print( "%d x %d : %d x %d pix = %f x %f in" % (cols, rows, width, height, width_inches, height_inches) )
    
    fig  = plot.figure(1, (width_inches, height_inches))                         
    grid = ImageGrid(fig,                                          
                    111,                                           
                    nrows_ncols = (rows, cols),                    
                    axes_pad = 0.001,                              
                    label_mode="1")                                

    for i in range( len( tensors ) ):                                  
        cell = grid[i]                                             
        cell.get_xaxis().set_visible(False)                        
        cell.get_yaxis().set_visible(False)
        
        # Convert from CUDA to CPU tensor, if needed
        if tensors[i].is_cuda:
            tensors[i] = tensors[i].cpu()
        
        if bpp == 1:
            img = image_from_tensor( tensors[i], bpp, width, height )
            cell.imshow( img, cmap='gray' )
        else:
            img = transforms.ToPILImage()( tensors[i] )
            cell.imshow( img )

        if labels:
            cell.annotate( labels[ i ], fontsize=12, xy=(12, 25), backgroundcolor="white", color="black" )
        
    plot.show()


def tensor_show( tensor, bpp = 3, width = 512, height = 512, label = None ):
    dpi = 192
    width_inches  = width  / dpi 
    height_inches = height / dpi
    print( "%d x %d pix = %f x %f in" % (width, height, width_inches, height_inches) )
    
    fig  = plot.figure(1, (width_inches, height_inches))                         
    grid = ImageGrid(fig,                                          
                    111,                                           
                    nrows_ncols = (1, 1),                    
                    axes_pad = 0.001,                              
                    label_mode="1")                                

    cell = grid[0]                                             
    cell.get_xaxis().set_visible(False)                        
    cell.get_yaxis().set_visible(False)
        
    # Convert from CUDA to CPU tensor, if needed
    if tensor.is_cuda:
        tensor = tensor.cpu()
        
    if bpp == 1:
        img = image_from_tensor( tensor, bpp, width, height )
        cell.imshow( img, cmap='gray' )
    else:
        img = transforms.ToPILImage()( tensor )
        cell.imshow( img )

    if label:
        cell.annotate( label, fontsize=12, xy=(12, 25), backgroundcolor="white", color="black" )
        
    plot.show()



#
# To change a numpy HxWxC array to CxHxW, and get the same behavior as if you called ToPILImage() and then ToTensor(), do
# npimg = np.transpose(npimg,(2,0,1))
# 

def image_show_visdom( vis, image, label = None ):
    array = np.array( image ).transpose( (2, 0, 1) )
    options = dict( title = label, caption = label )
    vis.image( array, opts = options )


def images_show_visdom( vis, images, label = None ):
    grid = []
    for img in images:
        array = np.array( img ).transpose( (2, 0, 1) )
        grid.append( array )
    grid = np.asarray( grid )
    options = dict( caption = label )
    vis.images( grid, opts = options )
    

def plot_visdom( vis, values, title = None, window = None ):
    _values = np.asarray( values )
    _indices = np.arange( len(_values) )
    
    # Create new plot, or append to existing
    if not window:
        plot = vis.line(
            Y = _values,
            X = _indices,
            opts=dict(
                title=title,
                markersymbol='cross-thin-open',
            ),
        )        
    else:
        plot = vis.line(
            win = window,
            update="replace",
            Y = _values,
            X = _indices,
            opts=dict(
                title=title,
                markersymbol='cross-thin-open',
            ),
        )
        
    return plot    


def plot_confusion_matrix( cm, class_names,
                           normalize = False,
                           title = "Confusion matrix",
                           cmap = plot.cm.Blues ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plot.figure()
    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = np.arange(len(class_names))
    plot.xticks(tick_marks, class_names, rotation=45)
    plot.yticks(tick_marks, class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    np.set_printoptions( precision = 2 )
    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plot.tight_layout()
    plot.ylabel( "True label" )
    plot.xlabel( "Predicted label" )
    plot.show()
    

def confusion_matrix( class_ids, predicted_class_ids, class_names ):
    cm = sklearn.metrics.confusion_matrix( class_ids, predicted_class_ids )
 
    # Plot non-normalized confusion matrix
    plot_confusion_matrix( cm, class_names )

    return cm


def factor_pairs( integer ):
    return [((int)(x), (int)(integer/x)) for x in range(1, int(math.sqrt(integer))+1) if integer % x == 0]


# return optimal grid size for an integer
# e.g. a minibatch of N is best shown as a grid of X x Y images,
def grid_size( integer ):
    pairs = factor_pairs( integer )
    pair = pairs[ len(pairs) - 1 ]
    
    # Optimize for wider images, and more rows
    return max(pair), min(pair)


#def image_scale( img, scale ):

class IMAGE_REGION( Enum ):
    CENTER = 1
    NW     = 2
    NE     = 3
    SW     = 4
    SE     = 5
    N      = 6
    S      = 7
    E      = 8
    W      = 9

def tensor_crop( tensor, x, y, width, height ):
    bpp = tensor.size()[0]
    
    crop_x_end = x + width
    crop_y_end = y + height
    
    crop = tensor[ 0:bpp, x:crop_x_end, y:crop_y_end ]
    return crop


def tensor_crop( tensor, crop_width, crop_height, region = IMAGE_REGION.CENTER ):
    #print( "tensor_crop %s" % str(region) )

    bpp    = tensor.size()[0]
    width  = tensor.size()[1]
    height = tensor.size()[2]
    
    if region == IMAGE_REGION.N:
        left   = (int)((width - crop_width) / 2)
        top    = 0

    elif region == IMAGE_REGION.NE:
        left   = (int)(width - crop_width)
        top    = 0

    elif region == IMAGE_REGION.E:
        left   = (int)(width - crop_width)
        top    = (int)((height - crop_height) / 2)

    elif region == IMAGE_REGION.SE:
        left   = (int)(width - crop_width)
        top    = (int)(height - crop_height)

    elif region == IMAGE_REGION.S:
        left   = (int)((width - crop_width) / 2)
        top    = (int)(height - crop_height)
                                                     
    elif region == IMAGE_REGION.SW:
        left   = 0
        top    = (int)(height - crop_height)

    elif region == IMAGE_REGION.W:
        left   = 0
        top    = (int)((height - crop_height) / 2)

    elif region == IMAGE_REGION.NW:
        left   = 0
        top    = 0

    elif region == IMAGE_REGION.CENTER:
        left   = (int)((width - crop_width) / 2)
        top    = (int)((height - crop_height) / 2)

    right  = left + crop_width
    bottom = top + crop_height

    #print( "%d %d -> %d %d" % (left, top, right, bottom) )

    crop = tensor[ 0:bpp, left:right, top:bottom ]

    return crop


def image_crop( image, width, height, region = IMAGE_REGION.CENTER ):
    #print( "image_crop %s" % str(region) )

    if region == IMAGE_REGION.N:
        left   = (int)((image.width - width) / 2) 
        top    = 0

    elif region == IMAGE_REGION.NE:
        left   = (int)(image.width - width)
        top    = 0

    elif region == IMAGE_REGION.E:
        left   = (int)(image.width - width)
        top    = (int)((image.height - height) / 2)

    elif region == IMAGE_REGION.SE:
        left   = (int)(image.width - width)
        top    = (int)(image.height - height)

    elif region == IMAGE_REGION.S:
        left   = (int)((image.width - width) / 2) 
        top    = (int)(image.height - height)

    elif region == IMAGE_REGION.SW:
        left   = 0 
        top    = (int)(image.height - height)

    elif region == IMAGE_REGION.W:
        left   = 0
        top    = (int)((image.height - height) / 2)

    elif region == IMAGE_REGION.NW:
        left   = 0
        top    = 0

    elif region == IMAGE_REGION.CENTER:
        left   = (int)((image.width - width) / 2) 
        top    = (int)((image.height - height) / 2)

    right  = left + width
    bottom = top + height

    #print( "%d %d -> %d %d" % (left, top, right, bottom) )

    crop = image.crop( (left, top, right, bottom) ) 
    crop = transforms.ToTensor()( crop )

    return crop


def convolution_output_shape(input_size, kernel_size, stride = (1,1), padding = (0,0)):
#    print("input_size = %d, %d" % (input_size[0], input_size[1]))
#    print("kernel_size = %d, %d" % (kernel_size[0], kernel_size[1]))
#    print("stride = %d, %d" % (stride[0], stride[1]))
#    print("padding = %d, %d" % (padding[0], padding[1]))
    
    width  = np.floor( (input_size[0] - kernel_size[0] + (2 * padding[0])) / stride[0] ) + 1
    height = np.floor( (input_size[1] - kernel_size[1] + (2 * padding[1])) / stride[1] ) + 1    
    
    #print("conv %d x %d (%d x %d) stride %dx%d pad %dx%d -> %f x %f" % (input_shape[0], input_shape[1], kernel_shape[0], kernel_shape[1], stride[0], stride[1], padding[0], padding[1], width, height))
    
    return (width, height)

    
def same_padding(input_size, kernel_size, stride = (1,1)):
    conv_width, conv_height = convolution_output_shape(input_size, kernel_size, stride, (0,0))

    padding_width  = max(0,  (input_size[0]/stride[0] - conv_width)  / 2) + stride[0] - 1
    padding_height = max(0,  (input_size[1]/stride[1] - conv_height) / 2) + stride[1] - 1
    
    padding_width  = int( np.floor(padding_width) )
    padding_height = int( np.floor(padding_height) )
    
    width, height = convolution_output_shape(input_size, kernel_size, stride, (padding_width, padding_height))

#    print("input %d x %d filter %d x %d stride %d,%d -> %d x %d (same padding: %d,%d -> %d x %d)" % 
#          (input_size[0], input_size[1],
#           kernel_size[0], kernel_size[1],
#           stride[0], stride[1],
#           conv_width, conv_height,
#           padding_width, padding_height,
#           width, height))

    return (padding_width, padding_height)

    
