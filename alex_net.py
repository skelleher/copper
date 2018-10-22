import torch
import torch.nn as nn
import torch.nn.functional as funcs
from copper.model import Model
from copper.model import View

# Inherit from nn.Module, not copper.Model, because latter was PITA with 
# resolving methods like forward() and train()

class AlexNet( nn.Module ):
    def __init__( self, num_classes = 2, input_width = 224, input_height = 224 ):
        super( AlexNet, self ).__init__()

        self._logger = None
        
        self._num_classes  = num_classes
        self._input_width  = input_width
        self._input_height = input_height

        self.features = nn.Sequential()
        self.features.conv1, width, height = Model._conv_layer(3, 96,  11, 4, 0, self._input_width, self._input_height)
        self.features.pool1, width, height = Model._pool_layer(3, 2, 1, width, height)
        self.features.conv2, width, height = Model._conv_layer(96, 256, 5, 1, 1, width, height)
        self.features.pool2, width, height = Model._pool_layer(3, 2, 1, width, height)
        self.features.conv3, width, height = Model._conv_layer(256, 384, 3, 1, 1, width, height)
        self.features.conv4, width, height = Model._conv_layer(384, 384, 3, 1, 1, width, height)
        self.features.conv5, width, height = Model._conv_layer(384, 256, 3, 1, 1, width, height)
        self.features.pool5, width, height = Model._pool_layer(3, 2, 1, width, height)

        self.reshape = View( 256 * width * height )

#        self.features.conv1 = nn.Conv2d( 3, 64, kernel_size = 11, stride = 4, padding = 2 )
#        self.features.pool1 = nn.MaxPool2d( kernel_size = 3, stride = 2 )
#        self.features.relu1 = nn.ReLU( inplace = True )
#        self.features.bn1   = nn.BatchNorm2d( 64, eps = 1e-3, momentum = 0.1, affine = True )
 
#        self.features.conv2 = nn.Conv2d( 64, 192, kernel_size = 5, stride = 1, padding = 2 )
#        self.features.pool2 = nn.MaxPool2d( kernel_size = 3, stride = 2 )
#        self.features.relu2 = nn.ReLU( inplace = True )
#        self.features.bn2   = nn.BatchNorm2d( 192, eps = 1e-3, momentum = 0.1, affine = True )
 
#        self.features.conv3 = nn.Conv2d( 192, 384, kernel_size = 3, stride = 1, padding = 1 )
#        self.features.relu3 = nn.ReLU( inplace = True )
#        self.features.bn3   = nn.BatchNorm2d( 384, eps = 1e-3, momentum = 0.1, affine = True )
 
#        self.features.conv4 = nn.Conv2d( 384, 256, kernel_size = 3, stride = 1, padding = 1 )
#        self.features.relu4 = nn.ReLU( inplace = True )
#        self.features.bn4   = nn.BatchNorm2d( 256, eps = 1e-3, momentum = 0.1, affine = True )
 
#        self.features.conv5 = nn.Conv2d( 256, 256, kernel_size = 3, stride = 1, padding = 1 )
#        self.features.pool5 = nn.MaxPool2d( kernel_size = 3, stride = 2 )
#        self.features.relu5 = nn.ReLU( inplace = True )
#        self.features.bn5   = nn.BatchNorm2d( 256, eps = 1e-3, momentum = 0.1, affine = True )
 
#        width = 6
#        height = 6
#        self.reshape        = View( 256 * width*height )

        self.classifier = nn.Sequential(
            # FC1
            nn.Dropout( ),
            nn.Linear(256 * width*height, 4096),
            # TODO: batch norm
            nn.ReLU( inplace = True ),

            # FC2
            nn.Dropout( ),
            nn.Linear(4096, 4096),
            # TODO: batch norm
            nn.ReLU( inplace = True ),

            # Classifier
            # dropout?  PyTorch example AlexNet does not
            nn.Linear(4096, self._num_classes),

            # how to append a softmax in the PyTorch way?  Do it functionally in the forward() pass?
            # Linear + LogSoftMax + ClassNLLCriterion
###            nn.LogSoftmax()
        )
    
        self.fc = self.classifier[6]
        print("self.classifier = ", self.classifier)
            
        # Initialize weights
        self.apply( Model._weights_init )

    def forward(self, x):
        #x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2))
        #x = func.max_pool2d(func.relu(self.conv2(x)), 2) # only specify a single number for a square size WTF?
        #x = x.view(-1, self.num_flat_features(x))

        x = self.features.conv1(x)
        x = self.features.pool1(x)
        x = self.features.conv2(x)
        x = self.features.pool2(x)
        x = self.features.conv3(x)
        x = self.features.conv4(x)
        x = self.features.conv5(x)
        x = self.features.pool5(x)

        # View should not copy the data, just change the dimensions.
        # In this case we flatten the features into a 1D vector for the classifier
#        x = x.view(-1, self.num_flat_features(x))
        x = self.reshape(x)

        x = self.classifier(x)

##        return nn.functional.log_softmax(x)
        return x

    
    # backward() function is auto-defined by the autograd package
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dims except the batch size
        num_features = 1
        for s in size:
            num_features *= s
        #print("num_flat_features = %d" % num_features)
        return num_features 

    def log(self, msg):
        if self.logger:
            self.logger.debug(msg)
            

    # Properties
    
    def _get_logger( self ):
        return self._logger
    
    def _set_logger( self, logger ):
        self._logger = logger

#    def _get_num_classes( self ):
#        return self._num_classes

    logger          = property( _get_logger,        _set_logger )
#    num_classes     = property( _get_num_classes,   None )


