"""
Maria Ines Vasquez Figueroa
18250
Gráficas
SR5 Textures
Main
"""

from gl import Render, V2, V3
from obj import Obj, Texture
from shaders import *

#valores con los que se inicializan la ventana y viewport

width=768
height=432

#creacion de Window

r = Render(width,height)
#se carga textura
#t = Texture('./models/model.bmp')
#se carga modelo obj con textura, la textura no debe ir obligatoriamente
#r.loadModel('./models/face.obj', (960,300,0), (15,15,15), t)

#r.loadModel('./models/objBarrel.obj', (500,500,0), (300,300,300), t)

r.active_texture = Texture('./models/model.bmp')
#r.active_texture = Texture('./models/earth.bmp')
r.active_shader = gouraud

#r.lightx, r.lighty, r.lightz=1,0,0
posModel = ( 0, 0, -5)

r.lookAt(posModel, (2,2,0))
r.loadModel('./models/model.obj', posModel, (1,1,1),(0,0,0))
#r.loadModel('./models/earth.obj', (500,500,0), (1,1,1))

r.glFinish('output.bmp')
#r.glZBuffer('zbuffer.bmp')





