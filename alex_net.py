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
        
        self._is_classifier = True
        self._class_table  = ClassTable( num_classes )
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
        )
    
        # Save handle to final layer, since we may replace it at runtime
        self.fc = self.classifier[-1]
            
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
        x = self.reshape(x)

        x = self.classifier(x)

        return x

    
    # backward() function is auto-defined by the autograd package

