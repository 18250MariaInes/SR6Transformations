"""
Maria Ines Vasquez Figueroa
18250
Gráficas
SR6 Transformations
Main
"""

from gl import Render
from obj import Obj, Texture
from shaders import *

#valores con los que se inicializan la ventana y viewport

width=1000
height=500

#creacion de Window

r = Render(width,height)
#r.glViewPort(200,100,600,300)
#se carga textura
#t = Texture('./models/model.bmp')
#se carga modelo obj con textura, la textura no debe ir obligatoriamente
#r.loadModel('./models/face.obj', (960,300,0), (15,15,15), t)

#r.loadModel('./models/objBarrel.obj', (500,500,0), (300,300,300), t)

r.active_texture = Texture('./models/model.bmp')
#r.active_texture = Texture('./models/earth.bmp')
r.active_shader = gouraud

#r.lightx, r.lighty, r.lightz=1,0,0
#( 3, 0, *profundidad y direccion con -*5)
posModel = ( 0, 0, -3)

#high angle
#r.lookAt(posModel, (0,2,0))

#low angle
#r.lookAt(posModel, (0,-2,0))

#medium shot
#r.lookAt(posModel, (0,0,0))

#Dutch
r.lookAt(posModel, (-2,-2,-0.25))


r.loadModel('./models/model.obj', posModel, (1,1,1),(0,0,0))
#r.loadModel('./models/earth.obj', (500,500,0), (1,1,1))

r.glFinish('output.bmp')
#r.glZBuffer('zbuffer.bmp')





