import os
import sys
import csv
from stat import *
from base64 import *


class PathTable( object ):
    PATH_TABLE_SIGNATURE = 0xC0FFEEEE

    def __init__( self, path = None, args = None ):
        self._args                  = args
        self._name                  = None
        self._path_table            = None
        self._path_table_by_name    = None
        self._path_table_path       = None
        self._num_items             = 0
      

    @staticmethod 
    def create( path ):
        #print( "Creating PathTable [%s]" % path )

        with open( path, "wt" ) as new_path_table:
            new_path_table.write( "%d\n" % PathTable.PATH_TABLE_SIGNATURE )
            new_path_table.write( "classid,classpath\n" )

            new_path_table.flush()
            new_path_table.close()
        
        ct = PathTable()
        ct._load( path )
        return ct


    @staticmethod
    def load( path ):
        ct = PathTable()
        ct._load( path )
        return ct


    def _load( self, path ):
        self._path_table_path = path
        #print( "PathTable %s" % self._path_table_path )

        if not os.path.exists( path ) or not os.path.isfile( path ):
            print( "Error: %s not found" % path )
            return None

        with open( self._path_table_path, mode = 'r' ) as infile:                                   
            reader = csv.reader( infile )
                                           
            # Confirm the header is a path_table and not some other file
            signature = next( reader )
            try:
                signature = int(signature[0])
            except:
                print( "Error: %s has invalid signature" % self._path_table_path )
                return None
            
            if signature != PathTable.PATH_TABLE_SIGNATURE:
                print( "Error: %s is not a valid PathTable" % self._path_table_path )
                return None

            # Skip the CSV header
            next( reader )

            self._path_table = { int(rows[0]) : rows[1] for rows in reader }
            self._num_items = len( self._path_table )

            self._path_table_by_name = { row[1] : row[0] for row in self._path_table.items() }
            #print( "PathTable: %s %d classes" % ( self._path_table_path, len(self._path_table) ) )

    
    def save( self ):
        print( "PathTable: save to %s" % self._path_table_path )

        with open( self._path_table_path, "wt" ) as new_path_table:
            new_path_table.write( "%d\n" % PathTable.PATH_TABLE_SIGNATURE )
            new_path_table.write( "classid,classpath\n" )

            for classid in self._path_table:
                classpath = self._path_table[ classid ]
                #print( "%d, %s" % (classid, classpath) )
                new_path_table.write( "%d,%s\n" % (classid, classpath) )

            new_path_table.flush()
            new_path_table.close()

    def classpath_from_id( self, classid ):
        try:
            return self._path_table[ classid ]
        except:
            print("Error: classpath_from_id(): ", classid)
            return None
        
    # PathTable supports [] operator
    # the key may be a numeric class_id, or string class_name for reverse lookup
    def __getitem__( self, key ):
        if isinstance( key, slice ):
            return [self._path_table[ i ] for i in range( *key.indices(self._num_items) ) ] # assumes Python 3 range()

        if isinstance( key, str ):
            try:
                return self._path_table_by_name[ key ]
            except KeyError as error:
                return None

        if key < 0:
            key += self._num_items
        
         # ClassTable can have sparse indices, e.g. 0,1,2,9,27,862 - so don't clamp keys to _num_items
#        if key < 0 or key >= self._num_items:
#            raise IndexError
        
        try:
            return self._path_table[ key ]
        except KeyError as error:
            return None
  
    # Allow insertion of new class descriptions at runtime
    def __setitem__( self, key, value ):
        if isinstance( key, str ):
            classpath = key
            classid = value
        elif isinstance( key, int ):
            classid = key
            classpath = value
        else:
            print( "Error" )
            raise KeyError( "key must be str or int" )

        #print( "PathTable %d = [%s]" % (classid, classpath) )

        self._path_table[ classid ] = classpath
        self._path_table_by_name[ classpath ] = classid
        self._num_items += 1

        #print( self._path_table )
        #print( self._path_table_by_name )

    
    # Index is iterable
    def __len__(self):
        return len( self._path_table )

    
    def __iter__( self ):
        self._idx = 0
        return self

    
    def __next__( self ):
        if self._idx > self._num_items:
            raise StopIteration
        
        try:
            item = self._path_table[ self._idx ]
        except KeyError as error:
            item = None

        self._idx += 1
        
        return item
        

    #
    # Properties
    #
        
    def _get_name( self ):
        return self._name
    

    def _get_path( self ):
        return self._path_table_path
    
    
    def _set_path( self, path ):
        if ( not path ):
            return
        
        path = os.path.normpath( path )
        directory = os.path.dirname( path )
        filename = os.path.basename( path )
        
        if ( not directory ):
            directory = os.getcwd()
            path = directory + os.path.sep + path

        print( "PathTable: %s" % path )
        self._path_table_path = path
        self._name, _ = os.path.splitext( filename )

        self.load()

    def _get_num_items( self ):
        return self._num_items
        
    def _get_class_paths( self ):
        class_paths = []
        for k, v in sorted( self._path_table.items() ):
            class_paths.append( v )
        return class_paths

    
    name        = property( _get_name, None )
    path        = property( _get_path, _set_path )
    num_items   = property( _get_num_items, None )
    class_paths = property( _get_class_paths, None )
    
 
