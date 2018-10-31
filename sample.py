# Represent a training Sample (an image), containing 1 or more Object instances.
#
# The relationship is: DataProvider -> Index -> Sample -> Image -> Objects
#
# In the case of MS-COCO, each Sample has a list of Objects, with a class_id and bounding_box.
#
# In the case where bounding boxes aren't provided (ImageNet, CIFAR, MNIST, etc)
# the Sample has a single Object with class_id = <foo> and bounding_box == <image size>
# 
# Samples are immutable, held by an Index, and only describe the image.
# To keep Samples small, they do not contain the image itself (or even the full pathname; they add up quickly).
#
# This means an entire DataProvider (Index of Samples) can be loaded into RAM quickly.
# The Index of Samples is then iterated, loading / cropping / augmenting images on the fly.

# namedtuple to keep it small and immutable; we are creating millions of them.
# TODO: just use __slots__ to reap the memory savings, but still use Sample and Object as classes:
#     __slots__ = ('foo', 'bar', 'baz')
# This effectively overides the class's dictionary so that it is hard-coded and can't change at runtime
from collections import namedtuple
import simplejson as json
from copper.utils import Rectangle

# deprecated:
Sample_v1 = namedtuple( "Sample", "class_id, classname, filename" )

# TODO: import python.typing and give all fields type hints

# Sample (an image) contains one or more objects and their bounding boxes
Sample    = namedtuple( "Sample", "path_id, filename, num_objects, objects" )
Object    = namedtuple( "Object", "class_id, bounding_box" )


def _object_decode(dct):
    if "x" in dct and "y" in dct and "width" in dct and "height" in dct:
        return Rectangle( *dct.values() )

    if "class_id" in dct and "bounding_box" in dct:
        return Object( *dct.values() )

    if "path_id" in dct and "filename" in dct and "num_objects" in dct and "objects" in dct:
        return Sample( *dct.values() )

    return dct


def from_string(json_string):
    try:
        sample = json.loads( json_string, object_hook=_object_decode )
    except:
        print("Error: sample.from_string() input is not JSON \"%s\"" % json_string)
        sample = None
        
    return sample


def to_string( sample ):
    # Serializing None to JSON doesn't work; fails to deserialize.
    # So replace missing box with an empty one
    objects = []
    for obj in sample.objects:
        if obj.bounding_box is None:
            obj = Object( obj.class_id, Rectangle( 0,0,0,0 ) )
            objects.append( obj )
            
    sample = Sample( sample.path_id, sample.filename, sample.num_objects, objects )
    
    #return json.dumps( sample, default=object_encode )
    return json.dumps( sample )
