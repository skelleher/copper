import os
import torch
import random
import signal
import numpy as np
from collections import namedtuple
from copper import index
from copper import utils

# for loading and transforming images; move this outside of DataProvider?
# Could map across many procs via zeromq?
from PIL import Image
from torchvision import transforms
from functools import partial
from multiprocessing import Pool 

# Tell workers to ignore Ctrl-C; let the parent process handle it
def _init_pool_worker():
    signal.signal( signal.SIGINT, signal.SIG_IGN )

pool = Pool( processes = 4, initializer = _init_pool_worker ) 

#Sample_v1 = namedtuple( "Sample", "classid, classname, path" )  
#Sample    = namedtuple( "Sample", "path, num_objects, objects" )  
#Object    = namedtuple( "Object", "classid, classname, rect" )


# Class methods because we want to dispatch them to multiple processes.
# Pickling/de-pickling an entire DataProvider per proc/call is very slow.
#_load_function = None
#_augment_function = None

# Extending Dataset allows us to work in concert with torch.utils.data.DataLoader,
# which is convenient
class DataProvider( torch.utils.data.Dataset ):
    INVALID_CLASS_ID    = -1

    def __init__( self, args = None ):
        super().__init__()
        
        self._args              = args
        self._index             = None
#        self._class_table       = None
#        self._path_table        = None
        self._num_items         = 0
        self._load_function     = self._default_load
        self._augment_function  = self._default_augment
        self._sample_size       = (3, 256, 256)
        self._crop_size         = (3, 224, 224)
        self._batch_dim         = (128, 3, 224, 224) # Assume ImageNet-type usage
        self._normalization     = utils.NORMALIZE_IMAGENET

        print( "DataProvider: sample_size:%s crop_size:%s" % 
              (str(self._sample_size), str(self._crop_size)) )
        

    def classname_from_id( class_id ):
        return self._class_table[ class_id ]
    
    
    @staticmethod
    def from_index( path ):
        #print( "Loading index %s" % path )
        provider = DataProvider()
        provider.index = index.Index.from_path( path )
        
        return provider

    # Assume samples are images; use PIL to load them
    # TODO: return object descriptions, if present
    def _default_load( self, sample, sample_size ):
        if not sample:
            raise IOError( "_default_load: sample is null" )
        
        try:
            class_path = self.index.path_table.classpath_from_id( sample.path_id )
            path = class_path + os.path.sep + sample.filename

            image = Image.open( path )

            if sample_size:
                bpp    = sample_size[0]
                width  = sample_size[1]
                height = sample_size[2]

                # If image is smaller than (width, height) resize before cropping
                image_width  = image.width
                image_height = image.height

                scale_width  = width  / image_width
                scale_height = height / image_height
                scale = max(scale_width, scale_height)

                old_size = image_width, image_height
                new_size = (int)(scale * image_width), (int)(scale * image_height)

                #print( "_default_load %d x %d -> %d x %d" % ( image.width, image.height, new_size[0], new_size[1] ) )

                image = image.resize( new_size ) # don't use thumbnail; it won't size up, only down
    
            if image.mode != 'RGB':
                image = image.convert( 'RGB' )
        except (IOError, TypeError) as error:
            print( "_default_load %s" % (error) );
            print( sample )
            raise IOError from error
            return None

        return image;

    #
    # TODO: move augmentation functions into class methods or even a separate module
    #
    def _no_augment_center_crop( self, image, crop_size ):
        bpp    = crop_size[ 0 ]
        width  = crop_size[ 1 ]
        height = crop_size[ 2 ] 

        if not image:
            raise IOError( "_no_augment_center_crop: image is null" )

        left   = (int)((image.width - width) / 2) 
        right  = left + width
        top    = (int)((image.height - height) / 2)
        bottom = top + height

        crop = image.crop( (left, top, right, bottom) ) 

        print( "_no_augment_center_crop %d x %d -> %d x %d" % (image.width, image.height, crop.width, crop.height) )

        crop = transforms.ToTensor()( crop )

        if self._normalization:
            crop = transforms.Normalize(
                mean = self._normalization.mean, 
                std  = self._normalization.std )( crop )

        return crop

    # return the full image (for validation / inference; the 5-crops or 10-crops will be performed 
    # in the main loop because DataLoader doesn't really support in-place expansion of the number of samples)
    def _no_augment_no_crop( self, image, crop_size ):
        bpp    = crop_size[ 0 ]
        width  = crop_size[ 1 ]
        height = crop_size[ 2 ] 
        #top    = 0
        #left   = 0
        #bottom = height
        #right  = width

        if not image:
            raise IOError( "_no_augment_no_crop: image is null" )

        img = image.resize( (width, height), resample = Image.BILINEAR )
        #img = img.crop( (left, top, right, bottom) )
        #print( "_no_augment_no_crop %d x %d -> %d x %d" % (image.width, image.height, img.width, img.height) )
        img = transforms.ToTensor()( img )

        if self._normalization:
            img = transforms.Normalize(
                mean = self._normalization.mean,
                std  = self._normalization.std )( img )

        return img

    # Assume samples are images, and apply standard ImageNet-ish transformations
    # Should return a float tensor, not a CUDA tensor, becuase this will be run in another process
    # that can't use CUDA
    def _default_augment( self, image, crop_size ):
        bpp    = crop_size[ 0 ]
        width  = crop_size[ 1 ]
        height = crop_size[ 2 ]

        if not image:
            raise IOError( "_default_augment: image is null" )

        crop = transforms.RandomResizedCrop( size = width )( image ) # equivalent to scale(0.08 - 1.0), then crop
        crop = transforms.RandomHorizontalFlip()( crop )
        crop = transforms.ToTensor()( crop )

        if self._normalization:
            crop = transforms.Normalize(
                mean = self._normalization.mean,
                std  = self._normalization.std )( crop )


#        if torch.cuda.is_available():
#            crop = crop.cuda( async = True )
            
        return crop

    # If a sample fails to load (e.g. malformed PNG or JPEG) we need to return
    # a dummy tensor of the proper dimensions.
    # Otherwise iterating over the DataProvider via a torch.utils.data.DataLoader will explode.
    def _null_sample( self, sample_size ):
        bpp    = sample_size[0]
        width  = sample_size[1]
        height = sample_size[2]
        tensor = torch.FloatTensor( bpp, width, height ).zero_()
        class_id = random.randint( 0, len(self.index.class_table) - 1 )

        #print( "_null_sample( %d )" % class_id )
        
        return tensor, class_id
        

    def _get_sample( self, sample ):
        # Load the training targets (one or more class_ids and optional bounding boxes)
        class_ids = []
        bounding_boxes = []
        for obj in sample.objects:
            class_ids.append( obj.class_id )
            bounding_boxes.append( obj.bounding_box )

        class_ids = torch.from_numpy(np.array(class_ids))
        bounding_boxes = torch.from_numpy(np.array(bounding_boxes))
            
        if len(class_ids) == 1:
            class_ids = class_ids.squeeze()
            bounding_boxes = bounding_boxes.squeeze()

        # Load and transform the image
        try:
            img    = self._load_function( sample, self._sample_size )
            tensor = self._augment_function( img, self._crop_size )
        except IOError as error:
            # DataLoader pukes if we return None, so return an empty Sample
            print( "__getitem__ %s : %s - returning zero tensor" % (sample.path, error) )
            tensor, class_ids, bounding_boxes = self._null_sample( self._sample_size )

#        print( "class_ids = ", class_ids )
#        print( "bounding_boxes = ", bounding_boxes )
            
        return tensor, class_ids, bounding_boxes
    
    
    # DataProvider supports [] operator
    # It returns images and their training targets after loading, augmenting (crop, flip, etc), and normalizing
    # TODO: should it also convert to tensor?  Can't call CUDA from multiple threads, but
    # might well map iteration of DataProvider to multiple threads, so convert to CUDA tensors in the reducer?
    def __getitem__( self, key ):
        if isinstance( key, slice ):
            samples = [self._index[ i ] for i in range( *key.indices(self._num_items) ) ]  # assumes Python 3 range()
            
            batch_tensors = []
            batch_class_ids = []
            batch_bounding_boxes = []

            for sample in samples:
                tensor, class_ids, bounding_boxes = self._get_sample( sample )
    
                batch_tensors.append( tensor )
                batch_class_ids.append( class_ids )
                batch_bounding_boxes.append( bounding_boxes )
            
            return batch_tensors, batch_class_ids, batch_bounding_boxes
                
        if key < 0:
            key += self._num_items
            # wrap-around and fall-through
        
        if key < 0 or key >= self._num_items:
            raise IndexError

        sample = self._index[ key ]
        tensor, class_ids, bounding_boxes = self._get_sample( sample )

        return tensor, class_ids, bounding_boxes

    
    # DataProvider is iterable
    # Iteration may apply run-time augmentation filters: crop, flip, scale, warp, color, noise
    # Iteration may load (and cache) samples on the fly
    def __len__(self):
        return len( self._index )


    def __iter__( self ):
        self._idx = 0
        return self

    
    def __next__( self ):
        if self._idx >= self._num_items:
            raise StopIteration
            
        sample, label = self.__getitem__(  self._idx )
        self._idx += 1
        
        return sample, label
        
    

    #
    # Properties
    #
    
    def _get_index( self ):
        return self._index

    def _set_index( self, index ):
        self._index = index
        self._class_table = index.class_table
        self._path_table = index.path_table
        self._num_items = index.num_items
        
    def _get_name( self ):
        if self._index:
            return self._index.name
        else:
            return "Unknown"

    def _get_path( self ):
        if self._index:
            return self._index.index_path
        else:
            return "Unknown"

    def _get_class_table( self ):
        return self._class_table

    def _get_path_table( self ):
        return self._path_table
       
    def _get_num_items( self ):
        return self._num_items

    def _get_num_classes( self ):
        return len( self._class_table )

    def _set_transforms( self, transforms ):
        self._transforms = transforms
        
    def _get_transforms( self ):
        return self._transforms
    
    def _set_sample_size( self, sample_size ):
        self._sample_size = sample_size
        
    def _get_sample_size( self ):
        return self._sample_size

    def _set_crop_size( self, crop_size ):
        self._crop_size = crop_size
        
    def _get_crop_size( self ):
        return self._crop_size

    def _get_augment_function( self ):
        return self._augment_function
        #return _augment_function

    def _set_augment_function( self, augment_function ):
        if augment_function:
            self._augment_function = augment_function
        else:
            #self._augment_function = self._no_augment_center_crop
            self._augment_function = self._no_augment_no_crop
    
    def _get_load_function( self ):
        return self._load_function
    
    def _set_load_function( self, load_function ):
        self._load_function = load_function

    def _get_invalid_class_id( self ):
        return -1
    
    #def _get_labels( self ):
    #    labels = []
    #    for sample in self._index:
    #        labels.append( sample.class_id )

    def _get_normalization( self ):
        return self._normalization

    def _set_normalization( self, normalization ):
        if normalization and type(normalization) != Normalization:
            print( "DataProvider.normalization must be of type Normalization( mean, std )" );
            return
        self._normalization = normalization
            

    index            = property( _get_index, _set_index )
    name             = property( _get_name, None )
    path             = property( _get_path, None )
    class_table      = property( _get_class_table, None )
    path_table       = property( _get_path_table, None )
    num_items        = property( _get_num_items, None )
    num_classes      = property( _get_num_classes, None )
    transforms       = property( _get_transforms, _set_transforms )
    sample_size      = property( _get_sample_size, _set_sample_size )
    crop_size        = property( _get_crop_size, _set_crop_size )
    augment_function = property( _get_augment_function, _set_augment_function )
    load_function    = property( _get_load_function, _set_load_function )
    #labels           = property( _get_labels, None )
    normalize        = property( _get_normalization, _set_normalization )
