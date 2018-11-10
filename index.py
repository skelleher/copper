import os
import sys
import csv
from stat import *
from base64 import *

from copper.class_table import ClassTable
from copper.path_table import PathTable
from copper.sample import Sample, Object
import copper.sample

from IPython.core.debugger import set_trace

# TODO: refactor so Index is a list of Samples, which contain one or more Objects with one or more bounding boxes
# Essentially make Index the container and DataProvider the iterator/augmentor.


# An index of class_id / filename pairs
# Also has an associated ClassTable, which describes each class (the class_id, classname, and the folder)
class Index(object):
    INDEX_SIGNATURE = 0xDEADBEEF

    _files_to_ignore = [
        "@eaDir",
        ".DS_Store",
        "._.DS_Store",
        ".txt"
    ]

    def __init__( self, path = None, args = None ):
        self._args                = args
        self._name                = None
        self._index_path          = None
        self._index               = None
        self._class_table_path    = None
        self._class_table         = None
        self._path_table_path     = None
        self._path_table          = None
        self._num_items           = 0
        self._has_bounding_boxes  = False

        self.index_path           = path

    @staticmethod
    def _ignore_file( name ):
        for ignore in Index._files_to_ignore:
            if ignore in name:
                return True
            if name[0] == '.':
                return True

        return False

    @staticmethod
    def from_path( path ):
        index = Index()
        index.index_path = path
        return index

    @staticmethod
    def _class_table_path_from_index( path ):
        if ( not path ):
            return None
        
        if ( not os.path.exists( path ) or not os.path.isfile( path ) ):
            return None
        
        with open( path, mode = 'r' ) as infile:                                   
            reader = csv.reader( infile )
                                           
            # Confirm the header is an index and not some other file
            signature = next( reader )
            try:
                signature = int(signature[0])
            except:
                print( "Error: %s has invalid signature" % path )
                return None
            
            if signature != Index.INDEX_SIGNATURE:
                print( "Error: %s is not a valid index" % path )
                return None

            # What ClassTable does this index point to?
            line = next( reader )
            try:
                prefix, class_table_path = line[0].split( "=" )
            except:
                prefix, class_table_path = ( None, None )

            if ( prefix != "class_table" ):
                print( "index %s doesn't point to a class_table" % path )
                class_table_path = None
            
            # append the filename to the index's working directory
            directory = os.path.dirname( path )
            class_table_path = directory + os.path.sep + class_table_path

            #print( "*** class_table_path = %s" % class_table_path )
            infile.close()
            return class_table_path
       
        # If we got here, couldn't open index file
        directory = os.path.dirname( path )
        filename = os.path.basename( path )
        class_table_path = directory + os.path.sep + filename + ".classes"
        #print( "*** class_table_path = %s" % class_table_path )
        return class_table_path
    
        
    @staticmethod
    def _path_table_path_from_index( path ):
        assert path

        if ( not os.path.exists( path ) or not os.path.isfile( path ) ):
            print("Error: %s: file not found %s" %(__file__, path) )
            return None
        
        with open( path, mode = 'r' ) as infile:                                   
            reader = csv.reader( infile )
                                           
            # Confirm the header is an index and not some other file
            signature = next( reader )
            try:
                signature = int(signature[0])
            except:
                print( "Error: %s has invalid signature" % path )
                return None
            
            if signature != Index.INDEX_SIGNATURE:
                print( "Error: %s is not a valid index" % path )
                return None


            # Skip over the ClassTable
            next( reader )

            # What PathTable does this index point to?
            line = next( reader )
            try:
                prefix, path_table_path = line[0].split( "=" )
            except:
                prefix, path_table_path = ( None, None )

            if ( prefix != "path_table" ):
                print( "index %s doesn't point to a path_table" % path )
                path_table_path = None
            
            # append the filename to the index's working directory
            directory = os.path.dirname( path )
            path_table_path = directory + os.path.sep + path_table_path

            #print( "*** path_table_path = %s" % path_table_path )
            infile.close()
            return path_table_path
       
        # If we got here, couldn't open index file
        directory = os.path.dirname( path )
        filename = os.path.basename( path )
        path_table_path = directory + os.path.sep + filename + ".classes"
        return path_table_path


    def _load_class_table( self ):
        if ( not self._class_table_path ):
            return
        
        self._class_table = ClassTable.load( self._class_table_path )


    def _load_path_table( self ):
        if ( not self._path_table_path ):
            print( "Error: %s self._path_table_path nil" % os.path.basename(__file__) )
            return

        self._path_table = PathTable.load( self._path_table_path )


    def _load_index( self ):
        if ( not self._index_path ):
            return
        
        if ( not os.path.exists( self._index_path ) or not os.path.isfile( self._index_path ) ):
            return
        
        #print( "Loading index [%s]" % self._index_path )
        with open (self._index_path, mode = 'r' ) as infile:
            reader = csv.reader( infile )
            
            # Confirm the header is an index and not some other file
            signature = next( reader )
            try:
                signature = int(signature[0])
            except:
                print( "Error: %s has invalid signature" % self._index_path )
                return None
            
            if signature != Index.INDEX_SIGNATURE:
                print( "Error: %s is not a valid index" % self._index_path )
                return None
            
            # Skip ClassTable line
            next( reader )

            # Skip PathTable line
            next( reader )
            
            # Skip CSV header
            next( reader )

            # Load the Samples, which were serialized as JSON strings
            self._index = []
            for row in list(reader):
                json_str = ", ".join( row )
                sample = copper.sample.from_string( json_str )
                self._index.append( sample )
            self._num_items = len( self._index )


    def _index_file( self, path, index, class_id ):
        elements  = path.split( os.sep )
        pathname  = os.path.dirname( path )
        filename  = elements[ -1 ]

        sys.stdout.write( "." )
        sys.stdout.flush()
    
        # Some images have multiple objects / bounding boxes (MS-COCO)
        # Otherwise (e.g. ImageNet), set bounding_box = None
        
        obj = Object(class_id, None)
        sample = Sample( self._path_table[pathname], filename, 1, [obj] )
        
        index.write( copper.sample.to_string( sample ) + "\n" )

        return 1


    # Recursively calls itself for all subfolders
    def _index_folder( self, path, index, class_id ):
        numfiles = 0

        print( "_index_folder %s" % path )

        classname = os.path.basename( path )
        classpath = path.rstrip( os.path.sep )

        #print( "classname = [%s] classpath = [%s]" % ( classname, classpath ) )
        #print( self._class_table )

        files = sorted( os.listdir( path ) )

        if len( files ) <= 0:
            print( "Empty folder: %s" % path )
###            return # No; reserve the slot in the classtable (test set may have matching files for this class)

        # Is class already in our ClassTable?  If not, add it
        if None == self._class_table[ classname ]:
            print( "Add class: %d = %s" % (class_id, classname ) )
            self._class_table[ class_id ] = classname
        else:
            class_id = self.class_id_from_name( classname, class_id )

        # Is path already in our Pathtable?  If not, add it
        if None == self._path_table[ classpath ]:
            print( "Add path: %d = %s" % ( class_id, classpath ) )
            self._path_table[ class_id ] = classpath


        print( "index_folder: %s (%d)" % ( path, class_id ) )

        next_class_id = class_id + 1

        for name in files:
            filename = path + os.path.sep + name

            if self._ignore_file( name ):
                continue

            if os.path.isfile( filename ):
                numfiles += self._index_file( filename, index, class_id )
            if os.path.isdir( filename ):
                next_class_id, numfiles = self._index_folder( filename, index, next_class_id )

        print("\n")
        return next_class_id, numfiles

    
    def index_all( self, path, index_path, existing_class_table_path = None, force = False ):
        if ( not path ):
            return
        
        print( "index_all: %s -> %s %s" % ( path, index_path, "(force)" if force else "" ) )
        
        path = path.rstrip( os.path.sep )

        # Check if input path exists.
        if not os.path.exists( path ):
            print( "Error: %s not found" % path )
            return -1

        mode = os.stat( path ).st_mode
        if not S_ISDIR( mode ):
            print( "Error: %s is not a directory" % path )
            return -1

        self._index_path = index_path
        
        # Check if index file exists.
        if os.path.exists( self._index_path ) and not force:
            print( "Error: %s exists; use force=True to overwrite" % self._index_path )
            return -1

        # Check if ClassTable exists.
        if not existing_class_table_path:
            root = os.path.dirname( self._index_path )
            filename = os.path.basename( self._index_path )
            filename = filename.replace( ".index", "" )
            self._class_table_path = root + os.path.sep + filename  + ".classes"
            create_classtable = True 
            self._class_table = None
        else:
            create_classtable = False
            self._class_table_path = existing_class_table_path
            print( "classtable = %s" % self._class_table )
            self._load_class_table()

        if os.path.exists( self._class_table_path ) and create_classtable and not force:
            print("Error: %s exists; use --force to overwrite" % self._class_table_path )
            return -1

        if create_classtable:
            self._class_table = ClassTable.create( self._class_table_path )

        # Check if PathTable exists.
        root = os.path.dirname( self._index_path )
        filename = os.path.basename( self._index_path )
        filename = filename.replace( ".index", "" )
        self._path_table_path = root + os.path.sep + filename  + ".paths"
        self._path_table = None

        if os.path.exists( self._path_table_path ) and not force:
            print( "Error: %s exists; use --force to overwrite" % self._path_table_path )
            return -1

        self._path_table = PathTable.create( self._path_table_path )

        print( ">>> index = [%s]" % path )
        print( ">>> classtable = [%s] (%s)" % (self._class_table_path, "Create" if create_classtable else "Re-use") )
        print( ">>> pathtable = [%s]" % self._path_table_path )

        print("Indexing [%s] -> %s, %s, %s" % ( path, self._index_path, self._class_table_path, self._path_table_path ) ) 


        with open( self._index_path, "wt" ) as index:
            class_id = 0 
            numfiles = 0

            # Don't print spaces between column names; confuses Pandas
            index.write( "%d\n" % Index.INDEX_SIGNATURE )
            index.write( "class_table=%s\n" % os.path.basename( self._class_table_path ) )
            index.write( "path_table=%s\n" % os.path.basename( self._path_table_path ) )
            index.write( "sample_json\n" )

            files = sorted( os.listdir( path ) )     
            if len( files ) <= 0:                        
                print( "Empty folder: %s" % path ) 
                ##return

            for name in files:
                filename = path + os.path.sep + name

                if self._ignore_file( name ):
                    continue

                if os.path.isfile( filename ):
                    numfiles += self._index_file( filename, index, class_id )

                elif os.path.isdir( filename ):
                    class_id, files = self._index_folder( filename, index, class_id )
                    numfiles += files

            print( "Index %d files, %d classes\n" % ( numfiles, class_id ) )

            index.flush()
            index.close()

            if create_classtable:
                self._class_table.save()

            self._path_table.save()


    def class_id_from_name( self, classname, class_id ):

        if self._class_table[ classname ]:
            return self._class_table[ classname ]
        else:
            return class_id


    def class_id_from_path( self, path, class_id ):
        classname = os.path.basename( path )   
        classpath = path.rstrip( os.path.sep ) 

        if self._path_table[ classpath ]:
            return self._path_table[ classpath ]
        else:
            return class_id


    def classname_from_id( self, class_id ):
        return self._class_table[ class_id ]

        
    def load( self ):
        # Load the class_table and index (index may be very large)
        self._load_class_table()
        self._load_path_table()
        self._load_index()
        

    # Index supports [] operator
    def __getitem__( self, key ):
        if isinstance( key, slice ):
            return [self._index[ i ] for i in range( *key.indices(self._num_items) ) ] # assumes Python 3 range()

        if key < 0:
            key += self._num_items
        
        if key < 0 or key >= self._num_items:
            raise IndexError
            
        return self._index[ key ]
    
    
    # Index is iterable
    def __len__(self):
        return len( self._index )
    
    def __iter__( self ):
        self._idx = 0
        return self
    
    def __next__( self ):
        if self._idx > self._num_items:
            raise StopIteration
            
        item = self._index[ self._idx ]
        self._idx += 1
        
        return item
        

    #
    # Properties
    #
        
    def _get_index_name( self ):
        return self._name
    

    def _get_index_path( self ):
        return self._index_path
    
    
    def _set_index_path( self, path ):
        if ( not path ):
            return
        
        path = os.path.normpath( path )
        directory = os.path.dirname( path )
        filename = os.path.basename( path )
        
        if ( not directory ):
            directory = os.getcwd()
            path = directory + os.path.sep + path

        #print( "Index: %s" % path )
        self._index_path = path
        self._name, _ = os.path.splitext( filename )

        if ( not self._class_table_path ):
            class_table_path = Index._class_table_path_from_index( path )

        if ( not self._path_table_path ):
            path_table_path = Index._path_table_path_from_index( path )

        self._class_table_path = class_table_path
        self._path_table_path = path_table_path
        
        self.load()

    def _get_num_items( self ):
        return self._num_items
        
    def _get_class_table( self ):
        return self._class_table

    def _get_path_table( self ):
        return self._path_table

    def _get_class_names( self ):
        return self._class_table.names
    
    # Wrap has_bounding_boxes in a property to make it read-only
    def _get_has_bounding_boxes( self ):
        return self._has_bounding_boxes
    
    name        = property( _get_index_name, None )
    index_path  = property( _get_index_path, _set_index_path )
    num_items   = property( _get_num_items, None )
    class_table = property( _get_class_table, None )
    path_table  = property( _get_path_table, None )
    class_names = property( _get_class_names, None )
    has_bounding_boxes = property( _get_has_bounding_boxes, None )
   
    
