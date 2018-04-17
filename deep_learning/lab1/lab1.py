import cairo
%matplotlib inline
 
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
 
# A simple function to display an image in an ipython notebook
def nbimage( data ):
    from IPython.display import display, Image
    from PIL.Image import fromarray
    from StringIO import StringIO
 
    s = StringIO()
    fromarray( data ).save( s, 'png' )
    display( Image( s.getvalue() ) )
 
WIDTH = 512
HEIGHT = 288
 
# this is a numpy buffer to hold the image data
data = np.zeros( (HEIGHT,WIDTH,4), dtype=np.uint8 )
 
# this creates a cairo context based on the numpy buffer
ims = cairo.ImageSurface.create_for_data( data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT )
cr = cairo.Context( ims )
 
# Get a random color
alpha = np.random.rand()*.5
r = np.random.rand()*.75 +.125
g = np.random.rand()*.75 +.125
b = np.random.rand()*.75 +.125
cr.set_source_rgba( r,g,b,alpha )

# Make a background color
cr.rectangle(0,0,WIDTH,HEIGHT)
cr.fill()


for i in xrange(0,50):
    
    # Get a random color
    alpha = np.random.rand()*.5 +.5
    r = np.random.rand()*.75 +.125
    g = np.random.rand()*.75 +.125
    b = np.random.rand()*.75 +.125
    
    cr.set_source_rgba(r,g,b,alpha)
    
    # Get a random line width between 5 and 10
    cr.set_line_width(np.random.rand()*5.0+5.0)
    
    # Get a random place to begin a curve
    cr.move_to(np.random.rand()*WIDTH,np.random.rand()*HEIGHT)
    
    # Get vertices for a random curve
    x = np.random.rand(3)*WIDTH
    y = np.random.rand(3)*HEIGHT
    cr.curve_to(x[0],y[0],x[1],y[1],x[2],y[2])
    
    cr.stroke()
 
# display the image
nbimage( data )
