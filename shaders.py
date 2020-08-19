from gl import *

#ejemplo de Carlos, base para los hecho por mi
def gouraud(render, **kwargs):
    u, v, w = kwargs['baryCoords']
    tax, tbx, tcx, tay, tby, tcy = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = tax * u + tbx * v + tcx * w
        ty = tay * u + tby * v + tcy * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = (nx, ny, nz)
#producto punto de las funciones que sustituyen a numpy
    intensity = render.dot(normal, render.lightx,render.lighty,render.lightz )

    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0:
        return r, g, b
    else:
        return 0,0,0

def toon(render, **kwargs):
    u, v, w = kwargs['baryCoords']
    tax, tbx, tcx, tay, tby, tcy = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = tax * u + tbx * v + tcx * w
        ty = tay * u + tby * v + tcy * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = (nx, ny, nz)

    intensity = render.dot(normal, render.lightx,render.lighty,render.lightz )
    #para gradiente de toon shader, luego todo es igual al gourad
    if (intensity>=0 and intensity<0.3):
        intensity=0
    elif (intensity>=0.3 and intensity<0.5):
        intensity=0.3
    elif (intensity>=0.5 and intensity<0.8):
        intensity=0.5
    elif (intensity>=0.8 and intensity<=1):
        intensity=0.8

    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0:
        return r, g, b
    else:
        return 0,0,0

def phong(render, **kwargs):
    Ax, Bx, Cx, Ay, By, Cy, Az, Bz, Cz= kwargs['verts']
    u, v, w = kwargs['baryCoords']
    tax, tbx, tcx, tay, tby, tcy = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = tax * u + tbx * v + tcx * w
        ty = tay * u + tby * v + tcy * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = (nx, ny, nz)
    #producto punto de las funciones que sustituyen a numpy
    intensity = render.dot(normal, render.lightx,render.lighty,render.lightz )
    print(intensity)

    #print(intensity)
    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0:
        return r, g, b
    else:
        return 0,0,0

def static(render, **kwargs):

    u, v, w = kwargs['baryCoords']
    tax, tbx, tcx, tay, tby, tcy = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = tax * u + tbx * v + tcx * w
        ty = tay * u + tby * v + tcy * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = (nx, ny, nz)
    #producto punto de las funciones que sustituyen a numpy
    intensity = render.dot(normal, render.lightx,render.lighty,render.lightz )

    #print(intensity)
    b *= intensity
    g *= intensity
    r *= intensity

    prob=random.randint(1,2)
    #print(prob)
    if intensity > 0 and prob==2:
        return r, g, b
    else:
        return 0,0,0

def thermal(render, **kwargs):

    u, v, w = kwargs['baryCoords']
    tax, tbx, tcx, tay, tby, tcy = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = tax * u + tbx * v + tcx * w
        ty = tay * u + tby * v + tcy * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w

    normal = (nx, ny, nz)
    #producto punto de las funciones que sustituyen a numpy
    intensity = render.dot(normal, render.lightx,render.lighty,render.lightz )

    #print(intensity)
    b *= intensity
    g *= intensity
    r *= intensity

    #print(prob)
    if intensity > 0 and intensity <= 0.33 :
        return 0, 0, b
    elif intensity > 0.33 and intensity <= 0.66:
        return r, g, 0
    elif intensity>0.66 and intensity<=1:
        return r,0,b

    else:
        return 0,0,0

def scifi(render, **kwargs):
    Ax, Bx, Cx, Ay, By, Cy, Az, Bz, Cz=kwargs['verts']
    u, v, w = kwargs['baryCoords']
    tax, tbx, tcx, tay, tby, tcy = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = tax * u + tbx * v + tcx * w
        ty = tay * u + tby * v + tcy * w
        
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w
    


    normal = (nx, ny, nz)
#producto punto de las funciones que sustituyen a numpy
    intensity = render.dot(normal, render.lightx,render.lighty,render.lightz )

    b *= intensity
    g *= intensity
    r *= intensity
    #print(render.frobenius(normal))
    if intensity > 0:
        if intensity<0.90:
            return 0,1,0
        return 0, 0, 0
    else:
        return 0,0,0

def stripes(render, **kwargs):
    Ax, Bx, Cx, Ay, By, Cy, Az, Bz, Cz=kwargs['verts']
    u, v, w = kwargs['baryCoords']
    tax, tbx, tcx, tay, tby, tcy = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']
    y,x=kwargs['coordy']

    print(y)

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = tax * u + tbx * v + tcx * w
        ty = tay * u + tby * v + tcy * w
        
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w
    


    normal = (nx, ny, nz)
#producto punto de las funciones que sustituyen a numpy
    intensity = render.dot(normal, render.lightx,render.lighty,render.lightz )

    b *= intensity
    g *= intensity
    r *= intensity
    #print(render.frobenius(normal))
    if intensity > 0:
        if y%50<25:
            return 1,round((y%50)/50),1
        return r,g,b
    else:
        return 0,0,0

def checkered(render, **kwargs):
    Ax, Bx, Cx, Ay, By, Cy, Az, Bz, Cz=kwargs['verts']
    u, v, w = kwargs['baryCoords']
    tax, tbx, tcx, tay, tby, tcy = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']
    y,x=kwargs['coordy']

    print(y)

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = tax * u + tbx * v + tcx * w
        ty = tay * u + tby * v + tcy * w
        
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w
    


    normal = (nx, ny, nz)
#producto punto de las funciones que sustituyen a numpy
    intensity = render.dot(normal, render.lightx,render.lighty,render.lightz )

    b *= intensity
    g *= intensity
    r *= intensity
    #print(render.frobenius(normal))
    if intensity > 0:
        if y%50<25:
            r,g,b=1,round((y%50)/50),1
            
        if x%50<25:
            r,g,b=round((y%50)/50),1,1
        return r,g,b
    else:
        return 0,0,0

