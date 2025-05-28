import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from functions import fitness,crossover,cross,mutation,mating

img=cv.imread('10081.jpg',cv.IMREAD_GRAYSCALE)
img=torch.tensor(img,dtype=torch.float32)
img=img.unsqueeze(0).unsqueeze(0)


#------------Convolving with sobel----------------------------------------------------------
sobel_x=torch.tensor([[2,3,0,-3,-2],
                     [3,4,0,-4,-3],
                     [6,6,0,-6,-6],
                     [3,4,0,-4,-3],
                     [2,3,0,-3,-2]],dtype=torch.float32)

sobel_45=torch.tensor([[0,-2,-3,-2,-6],
                      [2,0,-4,-6,-2],
                      [3,4,0,-4,-3],
                      [-2,6,4,0,-2],
                      [6,2,3,2,0]],dtype=torch.float32)

sobel_y=sobel_x.t()
sobel_135=sobel_45.t()

kernel=torch.stack([sobel_x,sobel_y,sobel_45,sobel_135])
kernel=kernel.unsqueeze(1)


grad=F.conv2d(img,kernel,padding='same')
grad=grad.squeeze(0)

grad_x=grad[0,:,:]
grad_y=grad[1,:,:]
grad_45=grad[2,:,:]
grad_135=grad[3,:,:]

edge_map=torch.sqrt(torch.square(grad_x)+torch.square(grad_y)+torch.square(grad_45)+torch.square(grad_135))/2

edge_map=edge_map.numpy()


#----------------Initialising Population------------------------------------------------------

population=np.random.randint(math.ceil(np.min(edge_map)),math.floor((np.max(edge_map))),40)
bin_pop=[]
for n in population:
  n=bin(n)
  bin_pop.append(n[2:].zfill(8))

bin_pop=np.array(bin_pop)




#----------------------------------------Genetic algorithm--------------------------------------
generations=75

for i in range(generations):

  mating_pool,fit=mating(bin_pop,edge_map)
  bin_pop=crossover(mating_pool,bin_pop)
  mating_pool,fit=mating(bin_pop,edge_map)
  bin_pop=mutation(bin_pop,fit)



mating_pool,fit=mating(bin_pop,edge_map)

index=np.argmax(fit)

bin_threshold=bin_pop[index]
threshold=int(bin_threshold,2)

mask=edge_map<=threshold

edge_map[mask==1]=1
edge_map[mask!=1]=0


plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(edge_map,cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(img.squeeze(0).squeeze(0).numpy(),cmap='gray')
plt.axis('off')

plt.show()
