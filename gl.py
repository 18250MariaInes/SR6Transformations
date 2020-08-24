"""
Maria Ines Vasquez Figueroa
18250
Gráficas
SR6 Transformations
Funciones
"""
import struct
from obj import Obj
import random
from numpy import  cos, sin, tan, matrix
import numpy as np
"""
Carlos nos permitio usar numpy para las operaciones de cos, sin, tan y transformar de grados a radianes, todas las otras operaciones matematica
fueron sustituidas y creadas por mi.
"""


def char(c):
    # 1 byte
    return struct.pack('=c', c.encode('ascii'))

def word(w):
    # 2 bytes
    return struct.pack('=h',w)

def dword(d):
    # 4 bytes
    return struct.pack('=l',d)

def color(r, g, b):
    #return bytes([b, g, r])
    return bytes([int(b * 255), int(g * 255), int(r * 255)])

def baryCoords(Ax, Bx, Cx, Ay, By, Cy, Px, Py):
    # u es para la A, v es para B, w para C
    try:
        u = ( ((By - Cy)*(Px - Cx) + (Cx - Bx)*(Py - Cy) ) /
              ((By - Cy)*(Ax - Cx) + (Cx - Bx)*(Ay - Cy)) )

        v = ( ((Cy - Ay)*(Px - Cx) + (Ax - Cx)*(Py - Cy) ) /
              ((By - Cy)*(Ax - Cx) + (Cx - Bx)*(Ay - Cy)) )

        w = 1 - u - v
    except:
        return -1, -1, -1

    return u, v, w


BLACK = color(0,0,0)
WHITE = color(1,1,1)

class Render(object):
    def __init__(self, width, height): #funncion que actua como el glInit
        #self.glInit(width, height)
        self.curr_color = WHITE
        self.curr_color_bg=BLACK
        self.glCreateWindow(width, height)
        self.lightx=0
        self.lighty=0
        self.lightz=1
        self.active_texture = None
        self.active_texture2 = None
        self.active_shader = None
        """self.camPosition=(0,0,0)
        self.camRotation=(0,0,0)"""
        self.createViewMatrix()
        self.createProjectionMatrix()
    
    def createViewMatrix(self, camPosition = (0,0,0), camRotation = (0,0,0)):
        camMatrix = self.createObjectMatrix( translate = camPosition, rotate = camRotation)
        self.viewMatrix=self.getMatrixInverse(camMatrix)
        """print(self.viewMatrix)
        print(np.linalg.inv(matrix(camMatrix)))
        print("-----------")"""
       

    #funciones para sacar inversa de la matriz extraido de: https://stackoverflow.com/questions/32114054/matrix-inversion-without-numpy, editado por mi para acomodarlo al proyecto
    def transposeMatrix(self, m):
        a=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        for i in range(0, 4):
            for j in range(0, 4):
                a[j][i]=m[i][j]
        return a

    def getMatrixMinor(self,m,i,j):
        return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

    def getMatrixDeternminant(self, m):
        #base case for 2x2 matrix
        if len(m) == 2:
            return m[0][0]*m[1][1]-m[0][1]*m[1][0]

        determinant = 0
        for c in range(len(m)):
            determinant += ((-1)**c)*m[0][c]*self.getMatrixDeternminant(self.getMatrixMinor(m,0,c))
        return determinant

    def getMatrixInverse(self, m):
        determinant = self.getMatrixDeternminant(m)
        #special case for 2x2 matrix:
        if len(m) == 2:
            return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                    [-1*m[1][0]/determinant, m[0][0]/determinant]]

        #find matrix of cofactors
        cofactors = []
        for r in range(len(m)):
            cofactorRow = []
            for c in range(len(m)):
                minor = self.getMatrixMinor(m,r,c)
                cofactorRow.append(((-1)**(r+c)) * self.getMatrixDeternminant(minor))
            cofactors.append(cofactorRow)
        cofactors = self.transposeMatrix(cofactors)
        for r in range(len(cofactors)):
            for c in range(len(cofactors)):
                cofactors[r][c] = cofactors[r][c]/determinant
        return cofactors

    def lookAt(self, eye, camPosition = (0,0,0)):

        """print(camPosition)
        print(eye)
        print(np.subtract(camPosition, eye))
        print("+++++++++++++++++++++")"""
        """print("resta mia")
        print(self.subtract(camPosition[0],eye[0],camPosition[1],eye[1],camPosition[2],eye[2]))#funciona"""
        #forward = np.subtract(camPosition, eye)
        pforward=self.subtract(camPosition[0],eye[0],camPosition[1],eye[1],camPosition[2],eye[2])
        pforward=self.division(pforward, self.frobenius(pforward))#si funciona ya
        """print("resta numpy")
        print(np.subtract(camPosition, eye))
        print("normal")
        print(np.linalg.norm(forward))"""
        
        #forward = forward / np.linalg.norm(forward)
        """print("resta numpy/normal")
        print(forward)
        print("mi forward")
        print(pforward)"""
        
        """print("cross con numpy")
        print(np.cross(V3(0,1,0), forward))
        print("mi cross")
        print(self.cross((0,1,0),forward))
        right = np.cross(V3(0,1,0), forward)
        print("right/normal numpy")
        right = right / np.linalg.norm(right)
        print(right)
        print("mi version")"""
        pright=self.division(self.cross((0,1,0),pforward), self.frobenius(pforward))
        #print(pright)

        """up = np.cross(forward, right)
        up = up / np.linalg.norm(up)"""
        pup=self.cross(pforward, pright)
        pup=self.division(pup, self.frobenius(pup))
        #print(pup)

        camMatrix = [[pright[0], pup[0], pforward[0], camPosition[0]],
                            [pright[1], pup[1], pforward[1], camPosition[1]],
                            [pright[2], pup[2], pforward[2], camPosition[2]],
                            [0,0,0,1]]

        
        self.viewMatrix=self.getMatrixInverse(camMatrix)
        #self.viewMatrix=camMatrix
        
        
    #Inicializa objetos internos
    def glInit(self, width, height):
        #esto se establece ahora en la funcion glCreatWindow
        """self.width = width
        self.height = height"""
        self.curr_color = WHITE
        self.curr_color_bg=BLACK
        self.glCreateWindow(width, height)
        """self.glClearColor(red, green, blue)
        self.glClear()"""

    #inicializa framebuffer
    def glCreateWindow(self, width, height):
        self.width = width
        self.height = height
        self.glClear()
        self.glViewPort(0, 0, width, height)

    #define area de dibujo
    def glViewPort(self, x, y, width, height):
        self.vportwidth = width
        self.vportheight = height
        self.vportx = x
        self.vporty = y
        #matriz del viewport para crear camara
        self.viewportMatrix = [[width/2, 0, 0, x + width/2],
                                      [0, height/2, 0, y + height/2],
                                      [0, 0, 0.5, 0.5],
                                      [0, 0, 0, 1]]
    
    def createProjectionMatrix(self, n = 0.1, f = 1000, fov = 60):

        t = tan((fov * np.pi / 180) / 2) * n
        r = t * self.vportwidth / self.vportheight

        self.projectionMatrix = [[n / r, 0, 0, 0],
                                        [0, n / t, 0, 0],
                                        [0, 0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                                        [0, 0, -1, 0]] 
    #cambia el color con el que se llena el mapa de bits (fondo)
    def glClearColor(self, red, green, blue):
        nred=int(255*red)
        ngreen=int(255*green)
        nblue=int(255*blue)
        self.curr_color_bg = color(nred, ngreen, nblue)

    #llena el mapa de bits de un solo color predeterminado antes
    def glClear(self):
        self.pixels = [ [ self.curr_color_bg for x in range(self.width)] for y in range(self.height) ]
        #Z - buffer, depthbuffer, buffer de profudidad
        self.zbuffer = [ [ float('inf') for x in range(self.width)] for y in range(self.height) ]

    
    #dibuja el punto en relación al viewport
    def glVertex(self, x, y):
        nx=int((x+1)*(self.vportwidth/2)+self.vportx)
        ny=int((y+1)*(self.vportheight/2)+self.vporty)

        if nx >= self.width or nx < 0 or ny >= self.height or ny < 0:
            return
        try:
            self.pixels[ny][nx] = self.curr_color
        except:
            pass
    
    #cambia de color con el que se hará el punto con parametros de 0-1
    def glColor(self, red, green, blue):
        nred=int(255*red)
        ngreen=int(255*green)
        nblue=int(255*blue)
        self.curr_color = color(nred, ngreen, nblue)
    
    def glVertex_coord(self, x,y, color = None):#helper para dibujar puntas en la funcion de glLine, 
    #ahora mejorado para solo dibujar cuando no hay nada abajo ya dibujado, más eficiente
        if x < self.vportx or x >= self.vportx + self.vportwidth or y < self.vporty or y >= self.vporty + self.vportheight:
            return
        if x >= self.width or x < 0 or y >= self.height or y < 0:
            return
        try:
            self.pixels[y][x] = color or self.curr_color
        except:
            pass


    #escribe el archivo de dibujo
    def glFinish(self, filename):
        archivo = open(filename, 'wb')

        # File header 14 bytes
        #f.write(char('B'))
        #f.write(char('M'))

        archivo.write(bytes('B'.encode('ascii')))
        archivo.write(bytes('M'.encode('ascii')))

        archivo.write(dword(14 + 40 + self.width * self.height * 3))
        archivo.write(dword(0))
        archivo.write(dword(14 + 40))

        # Image Header 40 bytes
        archivo.write(dword(40))
        archivo.write(dword(self.width))
        archivo.write(dword(self.height))
        archivo.write(word(1))
        archivo.write(word(24))
        archivo.write(dword(0))
        archivo.write(dword(self.width * self.height * 3))
        archivo.write(dword(0))
        archivo.write(dword(0))
        archivo.write(dword(0))
        archivo.write(dword(0))

        # Pixeles, 3 bytes cada uno

        for x in range(self.height):
            for y in range(self.width):
                archivo.write(self.pixels[x][y])


        archivo.close()
    
    def glZBuffer(self, filename):
        archivo = open(filename, 'wb')
        #misma configuracion de espacio que glFinish
        # File header 14 bytes
        archivo.write(bytes('B'.encode('ascii')))
        archivo.write(bytes('M'.encode('ascii')))
        archivo.write(dword(14 + 40 + self.width * self.height * 3))
        archivo.write(dword(0))
        archivo.write(dword(14 + 40))

        # Image Header 40 bytes
        archivo.write(dword(40))
        archivo.write(dword(self.width))
        archivo.write(dword(self.height))
        archivo.write(word(1))
        archivo.write(word(24))
        archivo.write(dword(0))
        archivo.write(dword(self.width * self.height * 3))
        archivo.write(dword(0))
        archivo.write(dword(0))
        archivo.write(dword(0))
        archivo.write(dword(0))

        #Minimo y el maximo del Zbuffer
        minZ = float('inf')
        maxZ = -float('inf')
        for x in range(self.height):
            for y in range(self.width):
                if self.zbuffer[x][y] != -float('inf'):
                    if self.zbuffer[x][y] < minZ:
                        minZ = self.zbuffer[x][y]

                    if self.zbuffer[x][y] > maxZ:
                        maxZ = self.zbuffer[x][y]

        for x in range(self.height):
            for y in range(self.width):
                depth = self.zbuffer[x][y]
                if depth == -float('inf'):
                    depth = minZ
                depth = (depth - minZ) / (maxZ - minZ)
                archivo.write(color(depth,depth,depth))

        archivo.close()

    def glLine(self, x0, y0, x1, y1): #algoritmo de clase modificado por mi en base al algoritmo de Bersenham extraido de : https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/
        x0 = int(( x0 + 1) * (self.vportwidth / 2 ) + self.vportx)
        x1 = int(( x1 + 1) * (self.vportwidth / 2 ) + self.vportx)
        y0 = int(( y0 + 1) * (self.vportheight / 2 ) + self.vporty)
        y1 = int(( y1 + 1) * (self.vportheight / 2 ) + self.vporty)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        inc = dy > dx

        if inc:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        limit = 0.5
    
        #a diferencia del visto en clase, el algoritmo consultado inicializa m como 2 veces el diferencial en y 
        #y offset como la resta entre la pendiente m y 2 veces el diferencial en x
        m=2*(dy)
        offset=m-2*dx
        y = y0
        for x in range(x0, x1 + 1):
            if inc:
                self.glVertex_coord(y, x)
            else:
                self.glVertex_coord(x, y)
            offset += m
            if offset >= limit:
                if y0 < y1:
                    y += 1
                else:
                    y-=1
                limit += 1
                #igualmente cuando offset es mayor o igual que el limite 0.5, se le resta 2 veces el diferencial en x
                offset-=2*dx
    
    def glLine_c(self, x0, y0, x1, y1):#algoritmo realizado con Carlos en clase, lo mantengo como comparacion y el resultado es muy similar al desarrollado por mi
        x0 = int(( x0 + 1) * (self.vportwidth / 2 ) + self.vportx)
        x1 = int(( x1 + 1) * (self.vportwidth / 2 ) + self.vportx)
        y0 = int(( y0 + 1) * (self.vportheight / 2 ) + self.vporty)
        y1 = int(( y1 + 1) * (self.vportheight / 2 ) + self.vporty)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        inc = dy > dx

        if inc:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        offset = 0
        limit = 0.5
        m = dy/dx
        y = y0
        for x in range(x0, x1 + 1):
            if inc:
                self.glVertex_coord(y, x)
            else:
                self.glVertex_coord(x, y)
            offset += m
            if offset >= limit:
                y += 1 if y0 < y1 else -1
                limit += 1

    def glLine_coord(self, x0, y0, x1, y1): #window coordinates en base a mi algoritmo realizado, no da problema con division con cero
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        inc = dy > dx

        if inc:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        limit = 0.5
    
        #a diferencia del visto en clase, el algoritmo consultado inicializa m como 2 veces el diferencial en y 
        #y offset como la resta entre la pendiente m y 2 veces el diferencial en x
        
        m=2*dy
    
        y = y0
        
        offset=m-2*dx
        for x in range(x0, x1 + 1):
            if inc:
                self.glVertex_coord(y, x)
            else:
                self.glVertex_coord(x, y)
            offset += m
            if offset >= limit:
                if y0 < y1:
                    y += 1
                else:
                    y-=1
                limit += 1
                #igualmente cuando offset es mayor o igual que el limite 0.5, se le resta 2 veces el diferencial en x
                offset-=2*dx

    #Barycentric Coordinates
    def triangle_bc(self, Ax, Bx, Cx, Ay, By, Cy, Az, Bz, Cz, tax, tbx, tcx, tay, tby, tcy, normals=(), colorest = WHITE):
        #bounding box
        minX = int(min(Ax, Bx, Cx))
        minY = int( min(Ay, By, Cy))
        maxX = int(max(Ax, Bx, Cx))
        maxY = int(max(Ay, By, Cy))

        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):
                if x >= self.width or x < 0 or y >= self.height or y < 0: #para no dar error al intentar dibujar fuera del zbuffer
                    continue
                u, v, w = baryCoords(Ax, Bx, Cx, Ay, By, Cy, x,y)

                if u >= 0 and v >= 0 and w >= 0:

                    z = Az * u + Bz * v + Cz * w
                    
                    if z < self.zbuffer[y][x] and z <= 1 and z >= -1:
                        
                        if self.active_shader:
                        
                            r, g, b = self.active_shader(
                                self,
                                verts=(Ax, Bx, Cx, Ay, By, Cy, Az, Bz, Cz),
                                baryCoords=(u,v,w),
                                texCoords=(tax, tbx, tcx, tay, tby, tcy),
                                normals=normals,
                                color = colorest or self.curr_color,
                                coordy=(y,x))
                        else:
                            b, g, r = colorest or self.curr_color
                        
                       

                        self.glVertex_coord(x, y, color(r,g,b))
                        self.zbuffer[y][x] = z
                    
    #funciones para reemplazar numpy del ejemplo de Carlos
    #Realiza la resta entre 2 listas
    def subtract(self, x0, x1, y0, y1, z0, z1):
        res=[]
        res.append(x0-x1)
        res.append(y0-y1)
        res.append(z0-z1)
        return res
    #realiza producto cruz entre dos listas
    def cross(self, v0, v1):
        res=[]
        res.append(v0[1]*v1[2]-v1[1]*v0[2])
        res.append(-(v0[0]*v1[2]-v1[0]*v0[2]))
        res.append(v0[0]*v1[1]-v1[0]*v0[1])
        return res

    #Calcula normal de Frobenius
    def frobenius(self, norm):
        return((norm[0]**2+norm[1]**2+norm[2]**2)**(1/2))

    #calcula la division entre elementos de una lista y la normal de frobenius
    def division(self, norm, frobenius):
        #si la division es entre cero regresa un not a number
        if (frobenius==0):
            res=[]
            res.append(float('NaN'))
            res.append(float('NaN'))
            res.append(float('NaN'))
            return res
            #return float('NaN')
        else:
            res=[]
            res.append(norm[0]/ frobenius)
            res.append(norm[1]/ frobenius)
            res.append(norm[2]/ frobenius)
            return res
    
    #realiza producto punto entre la matriz y la luz
    def dot(self, normal, lightx, lighty, lightz):
        return (normal[0]*lightx+normal[1]*lighty+normal[2]*lightz)
    
    def dot4(self, matrix1, matrix2):
        return (matrix1[0]*matrix2[0]+matrix1[1]*matrix2[1]+matrix1[2]*matrix2[2]+matrix1[3]*matrix2[3])

    def multiplicacion(self, matriz1, matriz2, c1, f1, c2, f2): #función para multiplicar matrices
        matriz3 = []
        for i in range(f1):
            matriz3.append( [0] * c2 )

        for i in range(f1):
            for j in range(c2):
                for k in range(f2):
                        numf=matriz1[i][k] * matriz2[k][j]
                        matriz3[i][j] += numf
                    
                    
        return matriz3

    def multiplicacionV(self, G, v, f1, c2): #función para multiplicar matrices, esta fue un fracaso pero la dejo porque tengo fe que algun día funcionará
        result = []
        for i in range(0,f1): #this loops through columns of the matrix
            total = 0
            for j in range(0,c2): #this loops through vector coordinates & rows of matrix
                total +=  v[i] *G[j][i]
            result.append(total)
        return result
        
    def multMaster(self, v, M): #función para multiplicar desde matrices hasta vectores
        c = []
        for i in range(0,len(v)):
            temp=[]
            for j in range(0,len(M[0])):
                s = 0
                for k in range(0,len(v[0])):
                    s += v[i][k]*M[k][j]
                temp.append(s)
            c.append(temp)
        return c
        
                    

    def transform(self, vertex, vMatrix):#sustitucion del transform antiguo

        
        pVertex=[ [vertex[0]], [vertex[1]], [vertex[2]], [1]]
        a=self.multMaster(self.viewportMatrix, self.projectionMatrix)
        b=self.multMaster(a, self.viewMatrix)
        c=self.multMaster(b, vMatrix)
        pVertex=self.multMaster(c, pVertex)
        
        
        pVertex=(pVertex[0][0] / pVertex[3][0] ,
                pVertex[1][0] / pVertex[3][0] ,
                pVertex[2][0]  / pVertex[3][0] )
        
        return pVertex

    def dirTransform(self, vertex, vMatrix):#transform para las normales

        pVertex=[ [vertex[0]], [vertex[1]], [vertex[2]], [0]]
       
        a=self.multMaster(pVertex, vMatrix)
 
        pVertex=(a[0][0],
                a[1][0],
                a[2][0])

        return pVertex

    def createObjectMatrix(self, translate = (0,0,0), scale = (1,1,1), rotate=(0,0,0)):
        #matriz de traslacion
        translateMatrix = [[1, 0, 0, translate[0]],
                                  [0, 1, 0, translate[1]],
                                  [0, 0, 1, translate[2]],
                                  [0, 0, 0, 1]]

        #matriz de la escala
        scaleMatrix = [[scale[0], 0, 0, 0],
                              [0, scale[1], 0, 0],
                              [0, 0, scale[2], 0],
                              [0, 0, 0, 1]]

        #matriz de rotacion
        rotationMatrix = self.createRotationMatrix(rotate)
        #multiplicacion de matrices sin numpy
        a=self.multiplicacion(translateMatrix, rotationMatrix, 4,4,4,4)
        b=self.multiplicacion(a, scaleMatrix, 4,4,4,4)

        return b
    
    def createRotationMatrix(self, rotate=(0,0,0)):

        pitch = np.deg2rad(rotate[0])
        yaw = np.deg2rad(rotate[1])
        roll = np.deg2rad(rotate[2])

        #matriz de rotacion en x
        rotationX = [[1, 0, 0, 0],
                            [0, cos(pitch),-sin(pitch), 0],
                            [0, sin(pitch), cos(pitch), 0],
                            [0, 0, 0, 1]]
        #matriz de rotacion en y
        rotationY = [[cos(yaw), 0, sin(yaw), 0],
                            [0, 1, 0, 0],
                            [-sin(yaw), 0, cos(yaw), 0],
                            [0, 0, 0, 1]]
        #matriz de rotacion en z
        rotationZ = [[cos(roll),-sin(roll), 0, 0],
                            [sin(roll), cos(roll), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]
        #multiplicacion de matrices sin numpy
        a=self.multiplicacion(rotationX, rotationY, 4,4,4,4)
        b=self.multiplicacion(a, rotationZ, 4,4,4,4)
        return (b)

    def loadModel(self, filename, translate= (0,0,0), scale= (1,1,1), rotate=(0,0,0), isWireframe = False): #funcion para crear modelo Obj
        model = Obj(filename)
        modelMatrix = self.createObjectMatrix(translate, scale, rotate)
        rotationMatrix = self.createRotationMatrix(rotate)

        for face in model.faces:
            vertCount = len(face) #conexion entre vertices para crear Wireframe
            if isWireframe:
                for vert in range(vertCount):
                    v0 = model.vertices[ face[vert][0] - 1 ]
                    v1 = model.vertices[ face[(vert + 1) % vertCount][0] - 1]
                    #coordenadas para dibujar linea con escala y traslacion setteado
                    x0 = int(v0[0] * scale[0]  + translate[0])
                    y0 = int(v0[1] * scale[1]  + translate[1])
                    x1 = int(v1[0] * scale[0]  + translate[0])
                    y1 = int(v1[1] * scale[1]  + translate[1])

                    #self.glVertex_coord(x0, y0)
                    
                    self.glLine_coord(x0, y0, x1, y1)
            else:
                v0 = model.vertices[ face[0][0] - 1 ]
                v1 = model.vertices[ face[1][0] - 1 ]
                v2 = model.vertices[ face[2][0] - 1 ]
                v0 = self.transform(v0, modelMatrix)
                v1 = self.transform(v1, modelMatrix)
                v2 = self.transform(v2, modelMatrix)

                
                x0 = v0[0]
                y0 = v0[1]
                z0 = v0[2]
                x1 = v1[0]
                y1 = v1[1]
                z1 = v1[2]
                x2 = v2[0]
                y2 = v2[1]
                z2 = v2[2]

                if vertCount > 3: #asumamos que 4, un cuadrado
                    v3 = model.vertices[ face[3][0] - 1 ]
                    v3 = self.transform(v3, modelMatrix)
                    x3 = v3[0]
                    y3 = v3[1]
                    z3 = v3[2]

                #----------FORMULA CON FUNCIONES POR MI---------------
               #normal=productoCruz(V1-V0, v2-V0)/Frobenius

                if self.active_texture:
                    vt0 = model.texcoords[face[0][1] - 1]
                    vt1 = model.texcoords[face[1][1] - 1]
                    vt2 = model.texcoords[face[2][1] - 1]
                    vt0x=vt0[0]
                    vt0y=vt0[1]
                    vt1x=vt1[0]
                    vt1y=vt1[1]
                    vt2x=vt2[0]
                    vt2y=vt2[1]
                    if vertCount > 3:
                        vt3 = model.texcoords[face[3][1] - 1]
                        vt3x=vt3[0]
                        vt3y=vt3[1]

                else:
                    vt0x=0
                    vt0y=0
                    vt1x=0
                    vt1y=0
                    vt2x=0
                    vt2y=0
                    vt3x=0
                    vt3y=0
                
                vn0 = model.normals[face[0][2] - 1]
                vn1 = model.normals[face[1][2] - 1]
                vn2 = model.normals[face[2][2] - 1]
                #para rotar normales y que la luz no se mueva con el modelo OBJ
                vn0 = self.dirTransform(vn0, rotationMatrix)#sustituir con transform de normales
                vn1 = self.dirTransform(vn1, rotationMatrix)
                vn2 = self.dirTransform(vn2, rotationMatrix)
                if vertCount > 3:
                    vn3 = model.normals[face[3][2] - 1]
                    vn3 = self.dirTransform(vn3, rotationMatrix)


                #normalMI=self.division(self.cross(self.subtract(x1, x0, y1, y0, z1, z0), self.subtract(x2, x0, y2, y0, z2, z0)),self.frobenius(self.cross(self.subtract(x1, x0, y1, y0, z1, z0), self.subtract(x2, x0, y2, y0, z2, z0))) )
                #ProductoCruz(normal,light)

                #intensity = self.dot(normalMI, lightx, lighty, lightz)
                """print("--------------intensity----------------------------")
                print(intensity)
                print(self.dot(normalMI, lightx, lighty, lightz))"""

                #if intensity >=0:
                    
                if vertCount > 3:
                    self.triangle_bc(x0,x2,x3, y0, y2,y3, z0, z2,z3, vt0x, vt2x,vt3x, vt0y,  vt2y, vt3y , normals=(vn0,vn2,vn3))
                self.triangle_bc(x0,x1,x2, y0, y1, y2, z0, z1, z2, vt0x ,  vt1x,vt2x,vt0y,vt1y, vt2y, normals = (vn0,vn1,vn2))


           
                
                
                

                











