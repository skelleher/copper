import os
import numpy as np
import math
import string
import sys

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms

from copper.class_table import ClassTable
from copper.timer import Timer
from copper import utils

from pydoc import locate # load classes dynamically, e.g. the particular model (AlexNet.py, ResNet, etc)
from collections import OrderedDict
from tqdm import tqdm


# define a view (reshape without copy) layer that
# we can insert into a Sequential model, because PyTorch
# doesn't include one??
class View( nn.Module ):
    def __init__( self, shape ):
        super( View, self ).__init__()
        self.shape = shape

    def forward( self, input ):
        # first dimension is -1 because input is a minibatch, and we don't know the length of
        # the minibatch 
        return input.view( -1, self.shape )


class Model( object ):
    MODEL_SIGNATURE     = 0xFACE0FFF
    CHECKPOINT_VERSION  = 1

    # Build a lookup table for the pre-defined / pre-trained models provided
    # by Torch
    _built_in_models = {}
    for name in models.__dict__:
        if name.islower() and not name.startswith( "__" ) and callable( models.__dict__[ name ] ):
            _built_in_models[ name ] = models.__dict__[ name ]

    #print( _built_in_models.keys() )


    def __init__( self ):
        super( Model, self ).__init__()

        self._args                  = None
        self._model_name            = None
        self._model                 = None # Model has an nn.Module as a component, rather than inheriting from it - SO GROSS MUST FIX
        self._is_classifier         = True 
        self._num_classes           = 0
        self._class_table           = None
        self._optimizer             = None
        self._input_width           = None
        self._input_height          = None
        self._normalization         = utils.NORMALIZE_IMAGENET
        self._epoch                 = 0
        self._gpus                  = None
        self._vis                   = None


    @staticmethod
    # TODO: models often want values passed to their constructor; just return a factory method to caller?
    def create( model_name ):
        # First check for a match against the built-in models provided by Torch:
        try:
            pretrained = False
            if "_pretrained" in model_name:
                pretrained = True
                model_name = model_name.replace( "_pretrained", "" )

            if Model._built_in_models[ model_name ]:
                _class = Model._built_in_models[ model_name ]
                #print( "built-in model %s %s" % ( model_name, "(pretrained)" if pretrained else "(NOT pretrained)" ) )
                _model = _class( pretrained = pretrained )
        except KeyError as error:
            # Next check for a fully-qualified Python class that implements the model
            _class = locate( model_name )
            _model = _class()

        model = Model()
        model._model_name = model_name
        model._model      = _model

        try:
            model._is_classifier = _model._is_classifier # SO GROSS MUST FIX
        except:
            pass

        #print( model._model )

        return model


    # Serialize the model to disk, optionally saving it with a new classname
    # (this is useful if you have deleted or added any layers)
    def save( self, path, classname = None ):
        dirname = os.path.dirname( path )
        if dirname and not os.path.exists( dirname ):
            os.makedirs( dirname )

        # Flatten DataParallel models so we can re-load them on a single GPU (or CPU)
        # Remove "module." prefix from each item
        model_state_dict = self._model.state_dict()
        new_state_dict = OrderedDict()

        for key, value in model_state_dict.items():
            key = key.replace( "module.", "" )
            new_state_dict[ key ] = value

        if classname:
            print("Change model from class %s to %s" % (self._model_name, classname))

        model_state_dict = new_state_dict

        checkpoint = {
            "signature"             : Model.MODEL_SIGNATURE,
            "version"               : Model.CHECKPOINT_VERSION,
            "model"                 : classname if classname else self._model_name,
            "model_state_dict"      : model_state_dict,
            "num_classes"           : self._num_classes,
            "class_table"           : self._class_table,
            "optimizer"             : self._optimizer.__class__.__name__,
            "optimizer_state_dict"  : self._optimizer.state_dict() if self._optimizer else None,
            "args"                  : self._args,
            "epoch"                 : self._epoch,
            "cmdline"               : sys.argv,
            "normalization"         : self._normalization,
        }

        #for key in enumerate( checkpoint[ "model_state_dict" ] ):
        #    print( key )

        #print("*** saved %d layers" % (len(model_state_dict)))

        torch.save( checkpoint, path )

    @staticmethod
    # TODO: should not return criterion or args.
    # These are only useful for fine-tuning or resuming training of a model; most
    # callers of load() won't care.  And they can be optional members of teh returned model instead.
    def load( path_or_model_name ):
        # try to load a matching built-in Torch model,
        # else load a previously-trained model checkpoint
        model, criterion, args = Model.load_model( path_or_model_name )
        if not model:
            model, criterion, args = Model.load_checkpoint( path_or_model_name )

        # Save the input width/height that this model expects
#        if args:
#            print( "args = ", args )
#            self._input_width  = args.w
#            self._input_height = args.h

        return model, criterion, args


    @staticmethod
    def load_model( model_name ):
        # Check for a match against the built-in models provided by Torch:
        try:
            pretrained = False
            if "_pretrained" in model_name:
                pretrained = True
                model_name = model_name.replace( "_pretrained", "" )

            if Model._built_in_models[ model_name ]:
                _class = Model._built_in_models[ model_name ]
                print( "built-in model %s %s" % ( model_name, "(pretrained)" if pretrained else "(NOT pretrained)" ) )
                _model = _class( pretrained = pretrained )
        except KeyError as error:
            return None, None, None

        return _model, None, None


    @staticmethod
    def load_checkpoint( path ):
        if not os.path.isfile( path ):
            print( "load_checkpoint: %s not found\n" % path )
            return None, None, None

        print( "Loading checkpoint %s " % path )

        checkpoint = torch.load( path )

        if checkpoint[ "signature" ] != Model.MODEL_SIGNATURE:
            print( "Error: %s is not a valid model checkpoint" % path )
            return None, None, None

        if checkpoint[ "version" ] != Model.CHECKPOINT_VERSION:
            print( "Error: checkpoint version %d not supported" % checkpoint[ "version" ] )
            return None, None, None
        
        # Create an empty model; we load the checkpoint into it
        print( "Model: ", checkpoint[ "model" ] )
        model = Model.create( checkpoint[ "model" ] )

        args = checkpoint[ "args" ]
        saved_state_dict = checkpoint[ "model_state_dict" ]

        try:
            print( "Model was created with: python " + " ".join( checkpoint[ "cmdline" ] ) + "\n")
        except KeyError as error:
            pass

        # Load the class_table, if model is a classifier
        try:
            # Set class_table using the property, which properly resizes the output layer
            model.class_table = checkpoint[ "class_table" ]
            #model._num_classes = len(model._class_table)
        except:
            pass

        # Load the state_dict.
        saved_layers = len(saved_state_dict)
        model_layers = len(model._model.state_dict())
        if saved_layers < model_layers:
            print("WARNING: checkpoint has fewer layers than model; this is OK if you are restoring a truncated model.")
            print("Loading %d of %d layers\n" % (saved_layers, model_layers))
            model._model.load_state_dict( saved_state_dict, strict = False )
        elif saved_layers > model_layers:
            print("\n*** ERROR: checkpoint has %d layers but model only expects %d layers; code/data mis-match?\n" % (saved_layers, model_layers))
            return model, None, None
        else:
            model._model.load_state_dict( saved_state_dict )


        # Load the data normalization
        try:
            self._normalization = checkpoint[ "normalization" ]
            print( "Data normalization = ", self._normalization )
        except:
            pass

        # Load the optimizer state, if included in the checkpoint
        try:
            optimizer_name = checkpoint[ "optimizer" ]
        except KeyError as error:
            optimizer_name = None

        optimizer = None
        if optimizer_name == "SGD":
            optimizer = optim.SGD( model._model.parameters(), lr = args.lr, weight_decay = args.wd, momentum = 0.9 )
#            #optimizer = optim.SGD( model._model.parameters(), lr = 0.01 )
        elif optimizer_name == "Adam":
            optimizer = optim.Adam( model._model.parameters(), lr = args.lr, weight_decay = args.wd )
#            #optimizer = optim.Adam( model._model.parameters(), lr = 0.01 )
   
        if optimizer:
            print( "Loaded optimizer %s" % optimizer )
            optimizer.load_state_dict( checkpoint[ "optimizer_state_dict" ] )

        try:
            model._epoch = checkpoint[ "epoch" ]
        except KeyError as error:
            model._epoch = 0

        #print("loaded checkpoint %s" % path )
    
        return model, optimizer, args


    def forward( self, minibatch ):
        return self._model.forward( minibatch )


    @staticmethod
    def _weights_init( module ):
        if isinstance( module, nn.Linear ):
            #print( "Initialize Linear" )
            size     = module.weight.size()
            fan_out  = size[0] # number of rows
            fan_in   = size[1] # number of columns
            variance = math.sqrt( 2.0 / (fan_in + fan_out) )
            module.weight.data.normal_( 0.0, variance )
    
        if isinstance( module, nn.Conv2d ):
            #print( "Initialize Conv2d" )
            init.xavier_uniform_( module.weight, gain=math.sqrt(2.0) )
            init.constant_( module.bias, 0.1 )


    @staticmethod
    def _conv_layer( input_filters, output_filters, kernel_size, stride, pad, width, height, verbose = False ):
        layer = nn.Sequential(
            nn.Conv2d( input_filters, output_filters, kernel_size, stride, pad ),
            nn.BatchNorm2d( output_filters ),
            nn.ReLU( inplace = True ),
        )
 
        width  = math.floor( ((width  - kernel_size + 2*pad) / stride) + 1)
        height = math.floor( ((height - kernel_size + 2*pad) / stride) + 1)

        if verbose:
            print( "Size after C layer (stride %d) = %d x %d x %d" % (stride, output_filters, width, height) )

        layer.apply( Model._weights_init )

        return layer, width, height


    @staticmethod
    def _pool_layer( kernel_size, stride, pad, width, height, verbose = False ):
        layer  = nn.MaxPool2d(kernel_size, stride, pad)

        width  = math.floor( ((width  - kernel_size + 2*pad) / stride) + 1)
        height = math.floor( ((height - kernel_size + 2*pad) / stride) + 1)
    
        if verbose:
            print( "Size after Pooling layer = %d x %d" % (width, height) )

        return layer, width, height

    
    # Define a residual block that a) does NOT downsample (stride = 1) and b) outputs SAME number of channels as the input
    # this allows use of an IDENTITY skip layer which does NOT need to reshape the input before adding it to the output
    # Contrast with the Residual Convolutional Block
    @staticmethod
    def _residual_identity_block( input_size, input_channels, kernel_size, layer_filters, verbose = False ):
        
        assert(isinstance(input_size, tuple))
        assert(isinstance(input_channels, int))
        assert(isinstance(kernel_size, tuple))
        assert(isinstance(layer_filters, list))
        assert(input_channels == layer_filters[-1])
        
        # assumes input_shape is a minibatch of form NxCxWxH
        #input_filters = input_shape[1]
        F1, F2, F3 = layer_filters

        pad_width, pad_height = utils.same_padding(input_size, kernel_size, stride = (1,1))
        assert(pad_width == pad_height)
        pad = pad_width
        
        block = nn.Sequential(
            nn.Conv2d( input_channels, F1, kernel_size = (1, 1), stride = (1, 1), padding = 0 ),
            nn.BatchNorm2d( F1 ),
            nn.ReLU(),
            
            nn.Conv2d( F1, F2, kernel_size = kernel_size, stride = (1, 1), padding = pad),
            nn.BatchNorm2d( F2 ),
            nn.ReLU(),

            nn.Conv2d( F2, F3, kernel_size = (1, 1), stride = (1, 1), padding = 0),
            nn.BatchNorm2d( F3 )
        )

        block.apply( Model._weights_init )
        
        if verbose:
            w = input_size[0]
            h = input_size[1]
            print( "Size after residual identity %d x %d" % (w, h) )
        
        # Caller must add X and block output, and apply final ReLU(), in the forward() function
        # see Model.residual_block_forward()

        return block


    # Define a residual block that a) downsamples using stride = 2 and b) outputs different number of channels than the input
    # this requires a CONVOLUTIONAL skip layer which RESHAPES the input before adding it to the output
    # Contrast with the Residual Identity Block
    @staticmethod
    def _residual_convolutional_block( input_size, input_channels, kernel_size, layer_filters, stride = (1,1), verbose = False ):
        
        assert(isinstance(input_size, tuple))
        assert(isinstance(input_channels, int))
        assert(isinstance(kernel_size, tuple))
        assert(isinstance(layer_filters, list))
        assert(isinstance(stride, tuple))

        F1, F2, F3 = layer_filters

        pad_width, pad_height = utils.same_padding(input_size, kernel_size, stride = stride)
        assert(pad_width == pad_height)
        pad = pad_width
        #print("res_conv: pad: ", pad)
        
        block = nn.Sequential(
                nn.Conv2d( input_channels, F1, kernel_size = (1, 1), stride = stride, padding = 0),
                nn.BatchNorm2d( F1 ),
                nn.ReLU(),

                nn.Conv2d( F1, F2, kernel_size = kernel_size, stride = (1, 1), padding = pad),
                nn.BatchNorm2d( F2 ),
                nn.ReLU(),

                nn.Conv2d( F2, F3, kernel_size = (1, 1), stride = (1, 1), padding = 0 ),
                nn.BatchNorm2d( F3 )
        )
        
        # input channels != output channels, so pass input through a 1x1 convolutional skip connection
        # to reshape it before adding to the output
        shortcut = nn.Sequential(
            nn.Conv2d( input_channels, F3, kernel_size = (1, 1), stride = stride, padding = 0 ),
            nn.BatchNorm2d( F3 )
        )
        
        block.apply( Model._weights_init )
        shortcut.apply( Model._weights_init )

        # Caller must add X and block output, and apply final ReLU(), in the forward() function

        #w = input_size[0] / stride[0]
        #h = input_size[1] / stride[1]
        w = math.floor( ((input_size[0] - kernel_size[0] + 2*pad) / stride[0]) + 1)
        h = math.floor( ((input_size[1] - kernel_size[1] + 2*pad) / stride[1]) + 1)
        
        if verbose:
            print( "Size after residual convolution (stride %d) %d x %d" % (stride[0], w, h) )
        
        return block, shortcut, w, h

    
    # Perform forward pass on a residual block, of type identity or convolutional.
    @staticmethod
    def residual_block_forward( X, block, shortcut = None ):
        output = block(X)
        
        if shortcut:
            X_shortcut = shortcut(X)
        else:
            X_shortcut = X
            
        output = output + X_shortcut
        output = torch.nn.functional.relu(output)
                
        return output
    
    
    def train_epoch( self, dataloader, criterion, optimizer ):
        self._optimizer = optimizer

        self._model.train()
        torch.set_grad_enabled( True )

        num_batches = len( dataloader )
        batch_idx = 0
        total_loss = 0
        total_acc = 0
        
        # Iterate over batches (automatically handled by the dataloader)
        for batch, labels, bounding_boxes in tqdm( dataloader ):
            with Timer() as perf:
                
                # TEST TEST
                #if self._vis and self._epoch % 10 == 0:
                #    num    = batch.size()[0]
                #    bpp    = batch.size()[1]
                #    width  = batch.size()[2]
                #    height = batch.size()[3]
                #    utils.images_show_visdom( self._vis, utils.images_from_tensor( batch, num, bpp, width, height ) )
                
                if torch.cuda.is_available() and self._gpus:
                    batch  = batch.cuda() 
                    labels = labels.cuda()

                # Forward Pass
                optimizer.zero_grad()

                output      = self._model.forward( batch )
                loss        = criterion( output, labels ) # The NLL criterion returns average loss for the minibatch
                
                # Backprop
                loss.backward()
                optimizer.step()
    
                predictions = Model._get_predictions( output )
                accuracy    = Model._get_accuracy( predictions, labels.cpu().numpy() )
        
                total_loss += float(loss.item())
                total_acc  += accuracy
        
                # Sanity check: print initial classificaiton loss
                if self._epoch == 0 and batch_idx == 0 and self._num_classes > 0:  
                    # print first loss, should be -ln(1/num_classes)
                    print('Epoch: {}, Batch: {}, Avg. Loss: {} (should be {} for {} classes)'.format(self._epoch, batch_idx, total_loss, -math.log(1/self._num_classes), self._num_classes))
                batch_idx += 1;

#        print( "batch %s ms" % perf.msecs ) # 1500 ms per batch on ImageNet; why so long?
    
        mean_loss = total_loss / num_batches
        mean_acc  = total_acc  / num_batches

        self._epoch += 1
 
        return mean_loss, mean_acc


    def test( self, test_loader, criterion, crop_size = (224, 224) ):
        self._model.eval()
        torch.set_grad_enabled( False )

        num_batches = len( test_loader )    
        total_loss  = 0
        total_acc   = 0

        width  = crop_size[ 0 ]
        height = crop_size[ 1 ]

        # iterate over minibatchs, testing up to 9 crops per image and averaging results
        for batch, labels, bounding_boxes in tqdm( test_loader ):
            if torch.cuda.is_available() and self._gpus:
                labels = labels.cuda() 

            num_crops = 0

            #for region in [utils.IMAGE_REGION.CENTER, utils.IMAGE_REGION.NW, utils.IMAGE_REGION.NE, utils.IMAGE_REGION.SW, utils.IMAGE_REGION.SE]:
            for region in [utils.IMAGE_REGION.CENTER]:
                crops = []
                num_crops += 1

                for sample in batch:
                    crop = utils.tensor_crop( sample, width, height, region )
                    crops.append( crop )

                # Convert back to minibatch
                cropped_batch = torch.stack( crops )
                if torch.cuda.is_available() and self._gpus:
                    cropped_batch = cropped_batch.cuda()

                # TEST TEST
                #if self._vis and self._epoch % 10 == 0:
                #    num    = cropped_batch.size()[0]
                #    bpp    = cropped_batch.size()[1]
                #    width  = cropped_batch.size()[2]
                #    height = cropped_batch.size()[3]
                #    utils.images_show_visdom( self._vis, utils.images_from_tensor( cropped_batch, num, bpp, width, height ) )

                output      = self._model.forward( cropped_batch )
                loss        = criterion( output, labels )   # returns average loss for the minibatch

                predictions = Model._get_predictions( output )
                accuracy    = Model._get_accuracy( predictions, labels.cpu().numpy() )

                #print( "batch = %s labels = %s" % (str(cropped_batch.size()), str(labels.size())))
                #print( "test loss: %f test acc: %f" % (loss, accuracy) )             
                #print( "labels = %s" % str(labels) )
                #print( "predictions = %s" % str(predictions) )

                total_loss += float(loss.item())
                total_acc  += accuracy

        mean_loss = total_loss / num_batches / num_crops
        mean_acc  = total_acc  / num_batches / num_crops

        return mean_loss, mean_acc

    # TODO: better to take an image, and perform the cropping / normalization / to-tensor
    # here rather than pushing it onto the caller?
    # details of crops size and normalization are properties of the model
    def classify( self, image_tensor ):
        self._model.eval()
        torch.set_grad_enabled( False )
        
        # Normalize the data to match model's training parameters
        if self._normalization:
            image_tensor = transforms.Normalize(
                mean = self._normalization.mean,
                std  = self._normalization.std )( image_tensor )

        # Add dimension to create a minibatch of 1 (Torch requires 3D tensors always)
        image_tensor = image_tensor.unsqueeze_( 0 ) 
        
        if torch.cuda.is_available() and self._gpus:
            image_tensor = image_tensor.cuda()

        output = self._model.forward( image_tensor )
        predictions, probabilities = Model._get_predictions( output )
        #print( "p = %s" % str(predictions) )
    
        labels = []
        for p in predictions:
            labels.append( self._class_table[ p ] )

        return predictions, probabilities, labels


    @staticmethod
    def _get_predictions( model_output ):
        output = model_output.data

#        print( "_get_predictions: ", output.size() ) 
#        print( output )
        
#        for i in range( len( output ) ):
#            print( "min/mean/max = ", output[ i ].min(), output[ i ].mean(), output[ i ].max() )

        # if output was not from logsoft max, we must apply it first:
        model_output = nn.functional.log_softmax( output, dim = 1 ) 
#        print( "softmax = ", model_output )

        val, idx = torch.max( model_output, dim = 1 )

#        print( val.item() )
#        print( idx.item() )

        #class_ids     = idx.data.cpu().view( -1 ).numpy()
        #probabilities = val.data.cpu().view( -1 ).numpy()
        class_ids     = idx.cpu().view( -1 ).numpy()
        probabilities = val.cpu().view( -1 ).numpy()
        
        # convert from negative log probability to probability
        probabilities = [ math.exp( p ) for p in probabilities ]
        #print( probabilities )
        
        return class_ids, probabilities


    @staticmethod
    def _get_accuracy( predictions, targets ):
        correct = np.sum( predictions[0] == targets )

        #print( "predictions = \n%s" % str(predictions[0]) )
        #print( "labels      = \n%s" % str(targets) )
        #print( "correct     = %d of %d" % (correct, len(targets)) )

        return correct / len( targets )


    # Deprecated; use torch.optim.lr_scheduler instead
    def adjust_learning_rate( self, epoch, lr_decay = 0.1, lr_decay_epoch = 10 ):
        if epoch == 0 or epoch % lr_decay_epoch:
            return
      
        print( "Epoch %d: update learning rate:" % epoch )
        for param_group in self._optimizer.param_groups:
            print( "%f -> %f" % (param_group["lr"], param_group["lr"] * lr_decay) )
            param_group['lr'] *= lr_decay

        return


    #
    # Properties
    #
       
    def _get_class_path( self ):
        return self._model_name

    def _get_class_table( self ):
        return self._class_table

    def _set_class_table( self, class_table ):
        if not self._is_classifier:
            print( "ERROR: set_class_table not allowed; model is not a classifier" )
            return

        self._class_table = class_table
        self._num_classes = len(class_table)

        # Resize the output layer if output != num_classes
        # Assumes model's classifier layer is an attribute named "fc"! # TODO: fix this gross hack
        try:
            output = self._model.fc
            if type(output) is nn.Linear and output.out_features != self._num_classes:
                del self._model.fc
                self._model.fc = nn.Linear( output.in_features, self._num_classes )
                print( "Set output layer: %s" % str( self._model.fc ) )
        except Exception as e:
            print( "ERROR: set_class_table: ", e ) 
            return 

        print( "Set class_table: ", len(class_table) )
        #print( self._model )
        
 
    def _get_class_names( self ):
        return self._class_table.names

    # For saving command-line arguments to checkpoints; informational use only
    def _set_model_args( self, args ):
        self._args = args

    def _get_training_epoch( self ):
        return self._epoch

    def _get_model_args( self ):
        return self._args

    def _get_model_gpus( self ):
        return self._gpus

    def _set_model_gpus( self, gpus ):
        self._gpus = gpus
        if gpus:
            self._model = self._model.cuda( device = gpus[0] ) # PyTorch issue #1150
            #self._model = self._model.cuda()
            if len( gpus ) == 1:
                print( "model using gpu %s" % gpus )
            elif len( gpus ) > 1:
                self._model = nn.DataParallel( self._model, device_ids = gpus )
                print( "model spans gpus %s" % gpus )

    def _set_normalization( self, normalization ):
        self._normalization = normalization
   
    def _get_normalization( self ):
        return self._normalization

    def _get_visdom( self ):
        return self._vis

    def _set_visdom( self, vis ):
        self._vis = vis
   
    model_name  = property( _get_class_path,        None )
    class_table = property( _get_class_table,       _set_class_table )
    class_names = property( _get_class_names,       None )
    epoch       = property( _get_training_epoch,    None )
    args        = property( _get_model_args,        _set_model_args )
    gpus        = property( _get_model_gpus,        _set_model_gpus )
    vis         = property( _get_visdom,            _set_visdom )
    normalization = property( _get_normalization,   _set_normalization )
   
 
