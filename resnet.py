import torch
import torch.nn as nn
import torch.nn.functional as funcs
from copper.model import Model
from copper.model import View
from copper.class_table import ClassTable
from copper import utils

# Inherit from nn.Module, not copper.Model, because latter was PITA with 
# resolving methods like forward() and train()

# Don't change capitalization of classname; old serialized models expect it
class ResNet( nn.Module ):
    def __init__( self, num_classes = 2, input_width = 224, input_height = 224 ):
        super( ResNet, self ).__init__()

        verbose = True 
        
        self._logger = None

        self._is_classifier = True
        self._class_table  = ClassTable( num_classes = num_classes )
        self._num_classes  = num_classes
        self._input_width  = input_width
        self._input_height = input_height

        self.features = nn.Sequential()
#        self.features.zeroPad = torch.nn.ZeroPad2d(3)

        # Stage 1
#        pad, _ = utils.same_padding((self._input_width, self._input_height), (7, 7), (2,2))

        self.features.conv1, w, h = Model._conv_layer(3, 64, 7, 2, 3, input_width, input_height, verbose)
        self.features.pool1, w, h = Model._pool_layer(3, 2, 1, w, h, verbose)

        # Stage 2
        block, shortcut, w, h = Model._residual_convolutional_block((w, h), 64, (3,3), [64, 64, 256], (1,1), verbose)
        self.features.residual2a = block
        self.features.residual2a_shortcut = shortcut
        self.features.residual2b = Model._residual_identity_block((w, h), 256, (3,3), [64, 64, 256], verbose)
        self.features.residual2c = Model._residual_identity_block((w, h), 256, (3,3), [64, 64, 256], verbose)
        
        # Stage 3: downsample
        block, shortcut, w, h = Model._residual_convolutional_block((w, h), 256, (3,3), [128, 128, 512], (2,2), verbose)
        self.features.residual3a = block
        self.features.residual3a_shortcut = shortcut
        self.features.residual3b = Model._residual_identity_block((w, h), 512, (3,3), [128, 128, 512], verbose)
        self.features.residual3c = Model._residual_identity_block((w, h), 512, (3,3), [128, 128, 512], verbose)
        self.features.residual3d = Model._residual_identity_block((w, h), 512, (3,3), [128, 128, 512], verbose)
        
        # Stage 4: downsample
        block, shortcut, w, h = Model._residual_convolutional_block((w, h), 512, (3,3), [256, 256, 1024], (2,2), verbose)
        self.features.residual4a = block
        self.features.residual4a_shortcut = shortcut
        self.features.residual4b = Model._residual_identity_block((w, h), 1024, (3,3), [256, 256, 1024], verbose)
        self.features.residual4c = Model._residual_identity_block((w, h), 1024, (3,3), [256, 256, 1024], verbose)
        self.features.residual4d = Model._residual_identity_block((w, h), 1024, (3,3), [256, 256, 1024], verbose)
        self.features.residual4e = Model._residual_identity_block((w, h), 1024, (3,3), [256, 256, 1024], verbose)
        self.features.residual4f = Model._residual_identity_block((w, h), 1024, (3,3), [256, 256, 1024], verbose)
                
        # Stage 5: downsample
        block, shortcut, w, h = Model._residual_convolutional_block((w, h), 1024, (3,3), [512, 512, 2048], (2,2), verbose)
        self.features.residual5a = block
        self.features.residual5a_shortcut = shortcut
        self.features.residual5b = Model._residual_identity_block((w, h), 2048, (3,3), [512, 512, 2048], verbose)
        self.features.residual5c = Model._residual_identity_block((w, h), 2048, (3,3), [512, 512, 2048], verbose)

        self.avgpool = torch.nn.AvgPool2d( (7,7) ) # Bx7x7 features -> Bx7x7 avgpool == Bx1x1 output
        self.reshape = View( 2048 )
        
#        self.classifier = nn.Sequential(
#            nn.Linear(2048, self._num_classes)
#        )

        # final layer must be named .fc, so that model.class_table = <foo> can find it and resize it
        self.fc = nn.Linear(2048, self._num_classes)

        # Save handle to final layer, since we may replace it at runtime

        print( "Size after classifier layer: ", self._num_classes )

        # Initialize weights
        self.apply( Model._weights_init )

        
    def forward(self, x):
        x = self.features.conv1(x)
        x = self.features.pool1(x)

        x = Model.residual_block_forward(x, self.features.residual2a, self.features.residual2a_shortcut)
        x = Model.residual_block_forward(x, self.features.residual2b)
        x = Model.residual_block_forward(x, self.features.residual2c)
        
        x = Model.residual_block_forward(x, self.features.residual3a, self.features.residual3a_shortcut)
        x = Model.residual_block_forward(x, self.features.residual3b)
        x = Model.residual_block_forward(x, self.features.residual3c)
        x = Model.residual_block_forward(x, self.features.residual3d)

        x = Model.residual_block_forward(x, self.features.residual4a, self.features.residual4a_shortcut)
        x = Model.residual_block_forward(x, self.features.residual4b)
        x = Model.residual_block_forward(x, self.features.residual4c)
        x = Model.residual_block_forward(x, self.features.residual4d)
        x = Model.residual_block_forward(x, self.features.residual4e)
        x = Model.residual_block_forward(x, self.features.residual4f)

        x = Model.residual_block_forward(x, self.features.residual5a, self.features.residual5a_shortcut)
        x = Model.residual_block_forward(x, self.features.residual5b)
        x = Model.residual_block_forward(x, self.features.residual5c)

        x = self.avgpool(x)

        x = self.reshape(x)

#        x = self.classifier(x)
        x = self.fc(x)

        return x

    
    # backward() function is auto-defined by the autograd package



# This is a Resnet50 model that has the classifier layer removed, and replaced
# with an embedding layer.  The output vectors are suitable for image search or captioning.

class ResnetCBIR( nn.Module ):
    def __init__( self, embedding_length = 512, input_width = 224, input_height = 224 ):
        super( ResnetCBIR, self ).__init__()

        verbose = True 
        
        self._logger = None
       
        self._is_classifier = False
        self._embedding_length = embedding_length
        self._input_width  = input_width
        self._input_height = input_height

        self.features = nn.Sequential()

        # Stage 1
        self.features.conv1, w, h = Model._conv_layer(3, 64, 7, 2, 3, input_width, input_height, verbose)
        self.features.pool1, w, h = Model._pool_layer(3, 2, 1, w, h, verbose)

        # Stage 2
        block, shortcut, w, h = Model._residual_convolutional_block((w, h), 64, (3,3), [64, 64, 256], (1,1), verbose)
        self.features.residual2a = block
        self.features.residual2a_shortcut = shortcut
        self.features.residual2b = Model._residual_identity_block((w, h), 256, (3,3), [64, 64, 256], verbose)
        self.features.residual2c = Model._residual_identity_block((w, h), 256, (3,3), [64, 64, 256], verbose)
        
        # Stage 3: downsample
        block, shortcut, w, h = Model._residual_convolutional_block((w, h), 256, (3,3), [128, 128, 512], (2,2), verbose)
        self.features.residual3a = block
        self.features.residual3a_shortcut = shortcut
        self.features.residual3b = Model._residual_identity_block((w, h), 512, (3,3), [128, 128, 512], verbose)
        self.features.residual3c = Model._residual_identity_block((w, h), 512, (3,3), [128, 128, 512], verbose)
        self.features.residual3d = Model._residual_identity_block((w, h), 512, (3,3), [128, 128, 512], verbose)
        
        # Stage 4: downsample
        block, shortcut, w, h = Model._residual_convolutional_block((w, h), 512, (3,3), [256, 256, 1024], (2,2), verbose)
        self.features.residual4a = block
        self.features.residual4a_shortcut = shortcut
        self.features.residual4b = Model._residual_identity_block((w, h), 1024, (3,3), [256, 256, 1024], verbose)
        self.features.residual4c = Model._residual_identity_block((w, h), 1024, (3,3), [256, 256, 1024], verbose)
        self.features.residual4d = Model._residual_identity_block((w, h), 1024, (3,3), [256, 256, 1024], verbose)
        self.features.residual4e = Model._residual_identity_block((w, h), 1024, (3,3), [256, 256, 1024], verbose)
        self.features.residual4f = Model._residual_identity_block((w, h), 1024, (3,3), [256, 256, 1024], verbose)
                
        # Stage 5: downsample
        block, shortcut, w, h = Model._residual_convolutional_block((w, h), 1024, (3,3), [512, 512, 2048], (2,2), verbose)
        self.features.residual5a = block
        self.features.residual5a_shortcut = shortcut
        self.features.residual5b = Model._residual_identity_block((w, h), 2048, (3,3), [512, 512, 2048], verbose)
        self.features.residual5c = Model._residual_identity_block((w, h), 2048, (3,3), [512, 512, 2048], verbose)

        self.avgpool = torch.nn.AvgPool2d( (7,7) ) # Bx7x7 features -> Bx7x7 avgpool == Bx1x1 output
        self.reshape = View( 2048 )
        
        # final layer must be named .fc, so that model.class_table = <foo> can find it and resize it
        self.fc = nn.Linear(2048, self._embedding_length)

        print( "Size after embedding layer: ", self._embedding_length )

        # Initialize weights
        self.apply( Model._weights_init )

        
    def forward(self, x):
        x = self.features.conv1(x)
        x = self.features.pool1(x)

        x = Model.residual_block_forward(x, self.features.residual2a, self.features.residual2a_shortcut)
        x = Model.residual_block_forward(x, self.features.residual2b)
        x = Model.residual_block_forward(x, self.features.residual2c)
        
        x = Model.residual_block_forward(x, self.features.residual3a, self.features.residual3a_shortcut)
        x = Model.residual_block_forward(x, self.features.residual3b)
        x = Model.residual_block_forward(x, self.features.residual3c)
        x = Model.residual_block_forward(x, self.features.residual3d)

        x = Model.residual_block_forward(x, self.features.residual4a, self.features.residual4a_shortcut)
        x = Model.residual_block_forward(x, self.features.residual4b)
        x = Model.residual_block_forward(x, self.features.residual4c)
        x = Model.residual_block_forward(x, self.features.residual4d)
        x = Model.residual_block_forward(x, self.features.residual4e)
        x = Model.residual_block_forward(x, self.features.residual4f)

        x = Model.residual_block_forward(x, self.features.residual5a, self.features.residual5a_shortcut)
        x = Model.residual_block_forward(x, self.features.residual5b)
        x = Model.residual_block_forward(x, self.features.residual5c)

        x = self.avgpool(x)

        x = self.reshape(x)

        x = self.embedding(x)

        return x

    
    # backward() function is auto-defined by the autograd package


#    def log(self, msg):
#        if self.logger:
#            self.logger.debug(msg)
            

    # Properties
    
#    def _get_logger( self ):
#        return self._logger
    
#    def _set_logger( self, logger ):
#        self._logger = logger

#    def _get_num_classes( self ):
#        return self._num_classes

#    logger          = property( _get_logger,        _set_logger )
#    num_classes     = property( _get_num_classes,   None )


