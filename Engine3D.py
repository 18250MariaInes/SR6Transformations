"""
Maria Ines Vasquez Figueroa
18250
Gráficas
SR5 Textures
Main
"""

from gl import Render
from obj import Obj, Texture
from shaders import *

#valores con los que se inicializan la ventana y viewport

width=1200
height=1200

#creacion de Window

r = Render(width,height)
#se carga textura
#t = Texture('./models/model.bmp')
#se carga modelo obj con textura, la textura no debe ir obligatoriamente
#r.loadModel('./models/face.obj', (960,300,0), (15,15,15), t)

#r.loadModel('./models/objBarrel.obj', (500,500,0), (300,300,300), t)

r.active_texture = Texture('./models/model.bmp')
#r.active_texture = Texture('./models/earth.bmp')
r.active_shader = unlit

#r.lightx, r.lighty, r.lightz=1,0,0

r.loadModel('./models/model.obj', (250,250,0), (150,150,150))
#r.loadModel('./models/earth.obj', (500,500,0), (1,1,1))

r.glFinish('output.bmp')
#r.glZBuffer('zbuffer.bmp')





