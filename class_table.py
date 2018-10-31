import os
import sys
import csv
from stat import *
from base64 import *


class ClassTable( object ):
    CLASS_TABLE_SIGNATURE   = 0xC001C0DE

    def __init__( self, classes = None, path = None, args = None, num_classes = 0 ):
        self._args                  = args
        self._name                  = None
        self._class_table           = {}
        self._class_table_path      = None
        self._num_items             = 0
        self._class_table_by_name   = {}
        
        if classes:
            self._class_table = classes
            self._num_items = len( self._class_table )

            self._class_table_by_name = { row[1] : row[0] for row in self._class_table.items() }
            #print( "self._class_table_by_name = ", self._class_table_by_name )
        else:
            self._class_table = {}
            self._num_items = num_classes
            for class_id in range( self._num_items ):
                class_name = "class_" + str(class_id)
                self[ class_id ] = class_name

        #print( "Created ClassTable with %d classes" % len(self._class_table) )
        #print( self._class_table )
      


    @staticmethod
    def create( path_or_num_classes ):
        if type( path_or_num_classes) is str:
            return ClassTable.create_from_file( path_or_num_classes )
        else:
            return ClassTable.create_with_num_classes( path_or_num_classes )

    @staticmethod
    def create_with_num_classes( num_classes ):
        return ClassTable( num_classes = num_classes )


    @staticmethod 
    def create_from_file( path ):
        #print( "Creating ClassTable [%s]" % path )

        with open( path, "wt" ) as new_classtable:
            new_classtable.write( "%d\n" % ClassTable.CLASS_TABLE_SIGNATURE )
            new_classtable.write( "classid,classname\n" )

            new_classtable.flush()
            new_classtable.close()
        
        ct = ClassTable()
        ct._load( path )
        return ct


    @staticmethod
    def load( path ):
        ct = ClassTable()
        ct._load( path )
        return ct


    def _load( self, path ):
        self._class_table_path = path

        if not os.path.exists( path ) or not os.path.isfile( path ):
            print( "Error: %s not found" % path )
            return None

        with open( self._class_table_path, mode = 'r' ) as infile:                                   
            reader = csv.reader( infile )
                                           
            # Confirm the header is a classtable and not some other file
            signature = next( reader )
            try:
                signature = int(signature[0])
            except:
                print( "Error: %s has invalid signature" % self._class_table_path )
                return None
            
            if signature != ClassTable.CLASS_TABLE_SIGNATURE:
                print( "Error: %s is not a valid class table" % self._class_table_path )
                return None

            # Skip the CSV header
            next( reader )

            self._class_table = { int(rows[0]) : rows[1] for rows in reader }
            self._num_items = len( self._class_table )

            self._class_table_by_name = { row[1] : row[0] for row in self._class_table.items() }
            #print( "ClassTable: %s %d classes" % ( self._class_table_path, len(self._class_table) ) )
    
    def save( self ):
        print( "ClassTable: save to %s" % self._class_table_path )

        with open( self._class_table_path, "wt" ) as new_classtable:
            new_classtable.write( "%d\n" % ClassTable.CLASS_TABLE_SIGNATURE )
            new_classtable.write( "classid,classname\n" )

            for classid in self._class_table:
                classname = self._class_table[ classid ]
                #print( "%d, %s" % (classid, classname) )
                new_classtable.write( "%d,%s\n" % (classid, classname) )

            new_classtable.flush()
            new_classtable.close()


    def classid_from_name( self, classname, classid ):
        if not self._class_table_by_name:
            return classid

        if classname in self._class_table_by_name:
            classid   = self._class_table_by_name[ classname ]
            #print("%s = %d" % (classname, classid))
            return classid


    def classname_from_id( self, classid ):
        return self._class_table[ classid ].class_name
        

    # ClassTable supports [] operator
    # the key may be a numeric class_id, or string class_name for reverse lookup
    def __getitem__( self, key ):
        if isinstance( key, slice ):
            return [self._class_table[ i ] for i in range( *key.indices(self._num_items) ) ] # assumes Python 3 range()

        if isinstance( key, str ):
            try:
                return self._class_table_by_name[ key ]
            except KeyError as error:
                return None

        if key < 0:
            key += self._num_items
 
        # ClassTable can have sparse indices, e.g. 0,1,2,9,27,862 - so don't clamp keys to _num_items
#        if key < 0 or key >= self._num_items:
#            raise IndexError
        
        try:
            return self._class_table[ key ]
        except KeyError as error:
            return None
  
    # Allow insertion of new class descriptions at runtime
    def __setitem__( self, key, value ):
        if isinstance( key, str ):
            classname = key
            classid   = value
        elif isinstance( key, int ):
            classid = key
            classname = value
        else:
            print( "Error" )
            raise KeyError( "key must be str or int" )

        #print( "ClassTable %d = [%s]" % (classid, classname) )

        self._class_table[ classid ] = classname
        self._class_table_by_name[ classname ] = classid
        self._num_items += 1

        #print( self._class_table )
        #print( self._class_table_by_name )

    
    # Index is iterable
    def __len__(self):
        return len( self._class_table )

    
    def __iter__( self ):
        self._idx = 0
        return self

    
    def __next__( self ):
        if self._idx > self._num_items:
            raise StopIteration
        
        try:
            item = self._class_table[ self._idx ]
        except KeyError as error:
            item = None

        self._idx += 1
        
        return item


    def __str__( self ):
        return str( self._class_table )


    #
    # Properties
    #
        
    def _get_name( self ):
        return self._name
    

    def _get_path( self ):
        return self._class_table_path
    
    
    def _set_path( self, path ):
        if ( not path ):
            return
        
        path = os.path.normpath( path )
        directory = os.path.dirname( path )
        filename = os.path.basename( path )
        
        if ( not directory ):
            directory = os.getcwd()
            path = directory + os.path.sep + path

        print( "ClassTable: %s" % path )
        self._class_table_path = path
        self._name, _ = os.path.splitext( filename )

        self.load()

    def _get_num_items( self ):
        return self._num_items
        
    def _get_class_names( self ):
        class_names = []
        for key, value in sorted( self._class_table.items() ):
            class_names.append( value )
        return class_names

    
    name        = property( _get_name, None )
    path        = property( _get_path, _set_path )
    num_items   = property( _get_num_items, None )
    class_names = property( _get_class_names, None )
    
 
