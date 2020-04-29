import cv2 
import numpy as np
from matplotlib import pyplot as plt
"""
def threshold():
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
def prueba():
    plt.hist(img.ravel(),256,[0,256])
    plt.show()
    cv2.imshow("res",img)
    cv2.imshow("res2",img)

ret,thresh1 = cv2.threshold(img,MT,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,MT,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,MT,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,MT,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,MT,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

diff=cv2.subtract(th2,img)
"""
"""
img = cv2.imread("chacra.png",0)
MT=180
ret,th = cv2.threshold(img,MT,255,cv2.THRESH_BINARY)

blur = cv2.GaussianBlur(img,(5,5),0)
ret2,th2 = cv2.threshold(blur,MT,255,cv2.THRESH_BINARY_INV)

fig, (ax1, ax2) = plt.subplots(2)
ax1.hist(blur.ravel(),256,[0,256])
ax2.hist(img.ravel(),256,[0,256])

plt.show()
normal=cv2.hconcat([img,th])
gauss=cv2.hconcat([blur,th2])
cv2.imshow('Imagen',cv2.vconcat([normal,gauss]))
"""
"""
def threshold(image,MinT,MaxT):
    h=image.shape[0]
    w=image.shape[1]
    img=np.full((h,w),255)
    for y in range(0,h):
        for x in range(0,w):
           if(MaxT>=image[y,x]>=MinT):
               img[y,x]=0
    return img


img =cv2.imread("chacra.png")
#binary=threshold(img,150,176)

rgb_weights = [0.2989, 0.5870, 0.1140]
grayscale_image = np.dot(img[...,:3], rgb_weights)


fig = plt.figure(figsize=(2,2))
grid = plt.GridSpec(2,2)
ax1 = fig.add_subplot(grid[0,0])
ax2 = fig.add_subplot(grid[0,1])
ax3 = fig.add_subplot(grid[1,0])
ax4=fig.add_subplot(grid[1,1])
ax1.imshow(img,interpolation='nearest')
#ax2.imshow(binary)

ax2.imshow(grayscale_image, cmap=plt.get_cmap("gray"))
ax3.hist(img.ravel(),256,[0,256])
ax4.hist(grayscale_image.ravel(),256,[0,256])
plt.show()
"""

def thresholding(name,MinT,MaxT,Gauss):
    #Cargar Imagen
    img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    #Aplicar filtro Gaussiano para eliminar ruido
    if(Gauss):
        img = cv2.GaussianBlur(img,(5,5),0)
    #Creacion de subplots
    fig,axs=plt.subplots(1,3)
    #Imagen Original
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #Histograma
    axs[1].hist(img.ravel(),256,[0,256])
    #Thresholding
    for y in range(0,img.shape[0]):
        for x in range(0,img.shape[1]):
           img[y,x]=255 if(MaxT>=img[y,x]>=MinT) else 0
    #Imagen resultante
    axs[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
# ELIMINANDO CELULAS VIVAS
#thresholding("celulas_raton.png",145,180,False)
#thresholding("celulas_raton.png",145,180,True)
#thresholding("celulas_raton2.png",100,180,False)
#thresholding("celulas_raton2.png",100,180,True)

# ELIMINANDO CELULAS MUERTAS
#thresholding("celulas_raton.png",193,195,False)
#thresholding("celulas_raton.png",193,195,True)
thresholding("celulas_raton2.png",190,191,False)
thresholding("celulas_raton2.png",190,191,True)
plt.show()




