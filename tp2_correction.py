#Author: Hélène Urien, 28/04/2019
#Try to be Pep8 compliant !  
#https://www.python.org/dev/peps/pep-0008/

#System imports
from __future__ import print_function
from __future__ import division
import os

#Third part import
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

#Get the current script directory
script_dir = os.path.dirname(__file__)
im_dir = os.path.join(script_dir, "images")

###############################################################################
#I Gamma correction
###############################################################################
#Load the got image
im_arr = Image.open(os.path.join(im_dir, "got.jpg"))
im_arr = np.array(im_arr)
im_arr = im_arr / np.max(im_arr)

#Save the image
fig, ax = plt.subplots(1, 1) 
ax.imshow(im_arr)
ax.axis("off")
fig.savefig("Ex1_im.png")

#Try different gamma values
nb_lines, nb_cols = 2, 3
fig, ax = plt.subplots(nb_lines, nb_cols) 
fig.subplots_adjust(hspace=0, wspace=0)
cols, lines = np.meshgrid(range(nb_cols), range(nb_lines))
gammas = [0.1, 0.5, 0.9, 2, 3, 4]
for gamma, line_id, col_id in zip(gammas,
                                       lines.flatten(),
                                       cols.flatten()):
    filter_arr = im_arr ** gamma 
    ax[line_id, col_id].imshow(filter_arr)
    ax[line_id, col_id].axis("off")
    ax[line_id, col_id].set_title("Gamma = {0}".format(gamma))   
fig.savefig("Ex1_im_gamma_correction.png")

###############################################################################
#II Image filtering
###############################################################################

#Load the boat512 image
im_arr = Image.open(os.path.join(im_dir, "boat512.gif"))
im_arr = np.array(im_arr)
im_arr = im_arr / np.max(im_arr)

#Image dimension
sx, sy = im_arr.shape

#Define useful kernel/filters

#Mean kernel
def mean_kernel(kernel_dim):
    mean_kernel = np.ones((kernel_dim, kernel_dim)) /(kernel_dim ** 2)
    return mean_kernel

#Gaussian kernel
def gaussian_kernel(sigma, kernel_dim):
    c = 1 / (2 * np.pi * sigma ** 2)
    [xk, yk] = np.meshgrid(range(0, kernel_dim), range(0, kernel_dim))
    gaussian_kernel = c * np.exp(- (c * np.pi) *
                                 ((xk - np.floor(kernel_dim / 2)) ** 2
                                + (yk - np.floor(kernel_dim / 2)) ** 2 ))
    gaussian_kernel  = gaussian_kernel / gaussian_kernel.sum()
    return gaussian_kernel

#Median filter
def median_filter(arr, kernel_dim):
    #Initialize filtered image
    M, N = arr.shape
    filter_arr = np.zeros((M + kernel_dim - 1, N + kernel_dim - 1))
    kd2 = int(np.floor(kernel_dim / 2))

    #Pad the image array
    pad_arr = np.zeros((M + kernel_dim - 1, N + kernel_dim - 1))
    pad_arr[kd2: M + kd2, kd2: N + kd2] = arr

    #Go through each pixel
    for x in range(kd2, kd2 + M):
        for y in range(kd2, kd2 + N):
            #Get the neighbourhood of the current pixel
            neighb = pad_arr[x - kd2: x + kd2 + 1, y - kd2: y + kd2 + 1]
            neighb = sorted(neighb.flatten())
            
            #Get the intensity value of the pixel in the middle of
            #the neighboorhood sorted values list
            filtered_val = neighb[int(np.ceil(len(neighb) / 2))]

            #Change the current pixel intensity
            filter_arr[x, y] = filtered_val
    return filter_arr[kd2: M + kd2, kd2: N + kd2]

#Define a 2D convolution function
#For everyday use, choose package function (ex: scipy)

#2D Convolution (with zeros padding)
def convolve(arr, kernel):
    #Initialize filtered image
    M, N = arr.shape
    kernel_dim = kernel.shape[0]  
    filter_arr = np.zeros((M + kernel_dim - 1, N + kernel_dim - 1))
    kd2 = int(np.floor(kernel_dim / 2))

    #Pi rotation of the kernel
    rot_kernel = np.zeros((kernel_dim, kernel_dim))
    for line in range(0, kernel_dim):
        kernel_line = kernel[kernel_dim - 1 - line, :]
        rot_kernel[line, :] = kernel_line[::-1]
    
    #Pad the image array
    pad_arr = np.zeros((M + kernel_dim - 1, N + kernel_dim - 1))
    pad_arr[kd2: M + kd2, kd2: N + kd2] = arr

    #Go through each pixel
    for x in range(kd2, kd2 + M):
        for y in range(kd2, kd2 + N):
            #Get the neighbourhood of the current pixel
            neighb = pad_arr[x - kd2: x + kd2 + 1, y - kd2: y + kd2 + 1]

            #Pixel-wise multiplication
            convolved_val = (neighb * rot_kernel).sum()

            #Change the current pixel intensity
            filter_arr[x, y] = convolved_val
    return filter_arr[kd2: M + kd2, kd2: N + kd2]

#Try different parameter values

#a) Mean filter
nb_lines, nb_cols = 2, 2
fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15)) 
fig.subplots_adjust(hspace=0.1, wspace=0)
cols, lines = np.meshgrid(range(nb_cols), range(nb_lines))
kernel_dims = [3, 5, 7, 9]
for kernel_dim, line_id, col_id in zip(kernel_dims,
                                       lines.flatten(),
                                       cols.flatten()):
    filter_arr = convolve(im_arr, mean_kernel(kernel_dim))
    ax[line_id, col_id].imshow(filter_arr, cmap="gray")
    ax[line_id, col_id].axis("off")
    ax[line_id, col_id].set_title("Kernel dimension = {0}".format(kernel_dim))   
fig.savefig("Ex2_Mean_filter.png")

#b) Gaussian filter
nb_lines, nb_cols = 3, 3
fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15)) 
fig.subplots_adjust(hspace=0.1, wspace=0)
cols, lines = np.meshgrid(range(nb_cols), range(nb_lines))
sigmas, kernel_dims = np.meshgrid([0.5, 1, 2], [3, 5, 9])
for sigma, kernel_dim, line_id, col_id in zip(sigmas.flatten(),
                                              kernel_dims.flatten(),
                                              lines.flatten(),
                                              cols.flatten()):
    filter_arr = convolve(im_arr, gaussian_kernel(sigma, kernel_dim))
    ax[line_id, col_id].imshow(filter_arr, cmap="gray")
    ax[line_id, col_id].axis("off")
    ax[line_id, col_id].set_title("Kernel dimension = {0}, sigma = {1}".
                                  format(kernel_dim, sigma))   
fig.savefig("Ex2_Gaussian_filter.png")
 
#c) Median filter
nb_lines, nb_cols = 2, 2
fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15)) 
fig.subplots_adjust(hspace=0.1, wspace=0)
cols, lines = np.meshgrid(range(nb_cols), range(nb_lines))
kernel_dims = [3, 5, 7, 9]
for kernel_dim, line_id, col_id in zip(kernel_dims,
                                       lines.flatten(),
                                       cols.flatten()):
    filter_arr =  median_filter(im_arr, kernel_dim)
    ax[line_id, col_id].imshow(filter_arr, cmap="gray")
    ax[line_id, col_id].axis("off")
    ax[line_id, col_id].set_title("Kernel dimension = {0}".format(kernel_dim))   
fig.savefig("Ex2_Median_filter.png")

#Add gaussian white noise
sigma = 20
noise_arr = np.random.normal(0, sigma, (sx, sy))
noise_arr = noise_arr.reshape(sx, sy)
noisy_im_arr = im_arr + noise_arr

#Save the noisy image
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.imshow(noisy_im_arr, cmap="gray")
ax.axis("off")
fig.savefig("Ex2_gaussian_white_noise_sigma{0}.png".format(sigma))
    
#Compare the three filters
nb_lines, nb_cols = 3, 4
fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15)) 
fig.subplots_adjust(hspace=0.1, wspace=0)
cols, lines = np.meshgrid(range(nb_cols), range(nb_lines))
kernel_dims = [3, 5, 7]
for line_id, col_id in zip(lines.flatten(),
                           cols.flatten()):
    kernel_dim = kernel_dims[line_id]
    if col_id == 0:
        filter_arr =  median_filter(noisy_im_arr, kernel_dim)
        title = "Median, {0}x{1} kernel".format(kernel_dim, kernel_dim)
    elif col_id == 1:
        filter_arr = convolve(noisy_im_arr, mean_kernel(kernel_dim))
        title = "Mean, {0}x{1} kernel".format(kernel_dim, kernel_dim)
    elif col_id == 2:
        filter_arr = convolve(noisy_im_arr, gaussian_kernel(1, kernel_dim))
        title = "Gaussian, {0}x{1} kernel, sigma = 1".format(kernel_dim, kernel_dim)
    elif col_id == 3:
        filter_arr = convolve(noisy_im_arr, gaussian_kernel(2, kernel_dim))
        title = "Gaussian, {0}x{1} kernel, sigma = 2".format(kernel_dim, kernel_dim)
    ax[line_id, col_id].imshow(filter_arr, cmap="gray")
    ax[line_id, col_id].axis("off")
    ax[line_id, col_id].set_title(title)   
fig.savefig("Ex2_Gaussian_white_noise_sigma{0}_filter.png".format(sigma))



#Add spurious noise
for percent_change in [4, 20]:
    #Get the n number of pixels with intensity to change
    nb_change = int((percent_change / 100) * sx * sy)
    noisy_im_arr = np.copy(im_arr.flatten())

    #Choose pixels randomly  
    pixel_ids = random.sample(range(0, sx * sy), nb_change)

    #Replace n/2 intensities with the intensity minimal value
    noisy_im_arr[pixel_ids[0: int(nb_change / 2)]] = im_arr.min()

    #Replace n/2 intensities with the intensity maximal value
    noisy_im_arr[pixel_ids[int(nb_change / 2):]] = im_arr.max()

    #Transform the array from a sx*sy vector to a sx x sy image
    noisy_im_arr = np.reshape(noisy_im_arr, (sx, sy))

    #Save the noisy image
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(noisy_im_arr, cmap="gray")
    ax.axis("off")
    fig.savefig("Ex2_impulsive_noise_{0}.png".format(percent_change))  
    
    #Compare the three filters
    nb_lines, nb_cols = 3, 4
    fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15)) 
    fig.subplots_adjust(hspace=0.1, wspace=0)
    cols, lines = np.meshgrid(range(nb_cols), range(nb_lines))
    kernel_dims = [3, 5, 7]
    for line_id, col_id in zip(lines.flatten(),
                               cols.flatten()):
        kernel_dim = kernel_dims[line_id]
        if col_id == 0:
            filter_arr =  median_filter(noisy_im_arr, kernel_dim)
            title = "Median, {0}x{1} kernel".format(kernel_dim, kernel_dim)
        elif col_id == 1:
            filter_arr = convolve(noisy_im_arr, mean_kernel(kernel_dim))
            title = "Mean, {0}x{1} kernel".format(kernel_dim, kernel_dim)
        elif col_id == 2:
            filter_arr = convolve(noisy_im_arr, gaussian_kernel(1, kernel_dim))
            title = "Gaussian, {0}x{1} kernel, sigma = 1".format(kernel_dim, kernel_dim)
        elif col_id == 3:
            filter_arr = convolve(noisy_im_arr, gaussian_kernel(2, kernel_dim))
            title = "Gaussian, {0}x{1} kernel, sigma = 2".format(kernel_dim, kernel_dim)
        ax[line_id, col_id].imshow(filter_arr, cmap="gray")
        ax[line_id, col_id].axis("off")
        ax[line_id, col_id].set_title(title)   
    fig.savefig("Ex2_impulsive_noise_{0}_filter.png".format(percent_change))                      


#Gaussian filtering in the Fourier domain
sigma = 5
v_vect = np.array(np.fft.ifftshift(range(int(-sy/2), int(sy/2))))
u_vect = np.array(np.fft.ifftshift(range(int(-sx/2), int(sx/2)))) 
[v, u] = np.meshgrid(u_vect, v_vect)
dtf_gaussian_kernel = np.exp(-0.5 * (sigma **2) * (np.pi ** 2) * ((u / sx) ** 2 + (v / sy) ** 2))
dtf_im = np.fft.fft2(im_arr)
dtf_filter = dtf_im * dtf_gaussian_kernel
filter_arr = np.abs(np.fft.ifft2(dtf_filter))
dtfs_filter = np.fft.fftshift(dtf_filter)
dtfs_im = np.fft.fftshift(dtf_im)
nb_lines, nb_cols = 2, 2
fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15)) 
fig.subplots_adjust(hspace=0.1, wspace=0)
cols, lines = np.meshgrid(range(nb_cols), range(nb_lines))
arrs = [im_arr, np.log(np.abs(dtfs_im) + 1), filter_arr, np.log(np.abs(dtfs_filter) + 1)] 
title = ["Image", "DTF", "Filter image", "DTF filter image"]
for title, line_id, col_id, arr in zip(title, lines.flatten(),
                           cols.flatten(), arrs):
    ax[line_id, col_id].imshow(arr / np.max(arr), cmap='gray')
    ax[line_id, col_id].axis("off")
    ax[line_id, col_id].set_title(title)   
fig.savefig("Ex2_fourier_gaussian_filter.png")   


#Deconvolution
def fourier_gaussian_filter(arr, sigma):
    sx, sy = arr.shape
    v_vect = np.array(np.fft.ifftshift(range(int(-sy/2), int(sy/2))))
    u_vect = np.array(np.fft.ifftshift(range(int(-sx/2), int(sx/2)))) 
    [v, u] = np.meshgrid(u_vect, v_vect)
    dtf_gaussian_kernel = np.exp(-0.5 * (sigma **2) * (np.pi ** 2) * ((u / sx) ** 2 + (v / sy) ** 2))
    dtf_im = np.fft.fft2(arr)
    dtf_filter = dtf_im * dtf_gaussian_kernel
    filter_arr = np.abs(np.fft.ifft2(dtf_filter))
    return filter_arr


titles = ["Filtered image", "Deconvolved image"]
sigmas = [3.5, 10]
nb_lines, nb_cols = len(sigmas), 2
fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15)) 
fig.subplots_adjust(hspace=0.1, wspace=0)
for line_id, sigma in enumerate(sigmas):
    filter_arr = fourier_gaussian_filter(im_arr, sigma) 
    dtf_filter = np.fft.fft2(filter_arr)
    v_vect = np.array(np.fft.ifftshift(range(int(-sy/2), int(sy/2))))
    u_vect = np.array(np.fft.ifftshift(range(int(-sx/2), int(sx/2)))) 
    [v, u] = np.meshgrid(u_vect, v_vect)
    dtf_gaussian_kernel = np.exp(-0.5 * (sigma **2) * (np.pi ** 2) * ((u / sx) ** 2 + (v / sy) ** 2))
    dtf_deconvolved = dtf_filter / dtf_gaussian_kernel
    deconvolved_arr = np.abs(np.fft.ifft2(dtf_deconvolved))
    arrs = [filter_arr, deconvolved_arr]
    for col_id, (arr, title) in enumerate(zip(arrs, titles)):
        ax[line_id, col_id].imshow(arr / np.max(arr), cmap='gray')
        ax[line_id, col_id].axis("off")
        ax[line_id, col_id].set_title("{0} sigma = {1}".format(title, sigma))
fig.savefig("Ex2_gaussian_deconvolution.png")

#Deconvolution adding some noise
nb_lines, nb_cols = len(sigmas), 2
fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15)) 
fig.subplots_adjust(hspace=0.1, wspace=0)
for line_id, sigma in enumerate(sigmas):
    filter_arr = fourier_gaussian_filter(im_arr, sigma)
    + 0.01 * np.random.random((sx, sy))
    dtf_filter = np.fft.fft2(filter_arr)
    v_vect = np.array(np.fft.ifftshift(range(int(-sy/2), int(sy/2))))
    u_vect = np.array(np.fft.ifftshift(range(int(-sx/2), int(sx/2)))) 
    [v, u] = np.meshgrid(u_vect, v_vect)
    dtf_gaussian_kernel = np.exp(-0.5 * (sigma **2) * (np.pi ** 2) * ((u / sx) ** 2 + (v / sy) ** 2))
    dtf_deconvolved = dtf_filter / dtf_gaussian_kernel
    deconvolved_arr = np.abs(np.fft.ifft2(dtf_deconvolved))
    arrs = [filter_arr, deconvolved_arr]
    for col_id, (arr, title) in enumerate(zip(arrs, titles)):
        ax[line_id, col_id].imshow(arr / np.max(arr), cmap='gray')
        ax[line_id, col_id].axis("off")
        ax[line_id, col_id].set_title("{0} sigma = {1}".format(title, sigma))
fig.savefig("Ex2_noisy_gaussian_deconvolution.png")

#Wiener filter
lambda_ = 0.01
nb_lines, nb_cols = len(sigmas), 2
fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15)) 
fig.subplots_adjust(hspace=0.1, wspace=0)
for line_id, sigma in enumerate(sigmas):
    filter_arr = fourier_gaussian_filter(im_arr, sigma)
    + 0.01 * np.random.random((sx, sy))
    dtf_filter = np.fft.fft2(filter_arr)
    v_vect = np.array(np.fft.ifftshift(range(int(-sy/2), int(sy/2))))
    u_vect = np.array(np.fft.ifftshift(range(int(-sx/2), int(sx/2)))) 
    [v, u] = np.meshgrid(u_vect, v_vect)
    dtf_gaussian_kernel = np.exp(-0.5 * (sigma **2) * (np.pi ** 2) * ((u / sx) ** 2 + (v / sy) ** 2))
    dtf_wiener = np.conj(dtf_gaussian_kernel) / (lambda_ + (np.abs(dtf_gaussian_kernel) ** 2))
    dtf_deconvolved = dtf_filter * dtf_gaussian_kernel
    deconvolved_arr = np.abs(np.fft.ifft2(dtf_deconvolved))
    arrs = [filter_arr, deconvolved_arr]
    for col_id, (arr, title) in enumerate(zip(arrs, titles)):
        ax[line_id, col_id].imshow(arr / np.max(arr), cmap='gray')
        ax[line_id, col_id].axis("off")
        ax[line_id, col_id].set_title("{0} sigma = {1}".format(title, sigma))
fig.savefig("Ex2_wiener_noisy_gaussian_deconvolution.png")

###############################################################################
#III Edge detection
###############################################################################

#Load the boat512 image
glim_arr = Image.open(os.path.join(im_dir, "boat512.gif"))
glim_arr = np.array(glim_arr)

#Convert it into grayscale
glim_arr = glim_arr / np.max(glim_arr)
sx, sy = glim_arr.shape

#Gradient using image lines and columns
Gx = glim_arr[0:sx- 1,:] -  glim_arr[1:, :]
Gy = glim_arr[:,0:sy-1] -  glim_arr[:, 1:]
grad_arr = np.sqrt(Gx[:, 0:sy-1] ** 2 + Gy[0:sx-1,:] ** 2)
fig, ax = plt.subplots(1, 4, figsize=(15, 15)) 
fig.subplots_adjust(hspace=0.1, wspace=0)
titles = ["Image", "Gx", "Gy", "Module gradient"]
for cnt, (title, arr) in enumerate(zip(titles, [glim_arr, Gx, Gy, grad_arr])):
    ax[cnt].imshow(arr, cmap='gray')
    ax[cnt].axis("off")
    ax[cnt].set_title(title)
fig.savefig("Ex3_image_gradient.png")

#Gradient using masks
G_x = np.zeros((3, 3, 2))
G_x[:,:, 0]= [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
G_x[:,:, 1]= [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
G_y = np.zeros((3, 3, 2))
G_y[:,:, 0]= [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
G_y[:,:, 1]= [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

titles = ["Image", "Gx", "Gy", "Module gradient", "Binary edge map"]
maps = ["gray", "gray", "gray", "gray", "Greys"]
ops = ["sobel", "prewitt"]
for filter_id in range(0, 2):
    fig, ax = plt.subplots(1, 5, figsize=(15, 15)) 
    fig.subplots_adjust(hspace=0.1, wspace=0)
    gx_arr = convolve(glim_arr , G_x[:,:, filter_id])
    gy_arr = convolve(glim_arr , G_y[:,:, filter_id])
    g_arr = np.sqrt(gx_arr ** 2 + gy_arr ** 2)
    for cnt, (map_, title, arr) in enumerate(zip(maps, titles,
    [glim_arr, gx_arr, gy_arr, g_arr, g_arr > 0.1])):
        ax[cnt].imshow(arr, cmap=map_)
        ax[cnt].axis("off")
        ax[cnt].set_title(title)
 
    fig.savefig("Ex3_image_gradient_{0}.png".format(ops[filter_id]))

#Define the Laplacian filters
G_i = np.zeros((3, 3, 2))
G_i[:,:, 0]= [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
G_i[:,:, 1]= [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
fig, ax = plt.subplots(1, 2, figsize=(15, 15)) 
fig.subplots_adjust(hspace=0.1, wspace=0)
for filter_id in range(0, 2):
    filter_arr = convolve(glim_arr , G_i[:,:, filter_id])
    ax[filter_id].imshow(filter_arr> 0.1, cmap='Greys')
    ax[filter_id].axis("off")
    ax[filter_id].set_title("Binary edge image using G_{0}".format(filter_id + 1))
fig.savefig("Ex3_laplacian_filter.png")
    
#Image sharpening using the Laplacian
param = 0.05
filter_arr = np.copy(glim_arr)
cnt_fig = 0
fig, ax = plt.subplots(1, 2, figsize=(15, 15)) 
fig.subplots_adjust(hspace=0.1, wspace=0)
for iter in range(1, 13):
    dx = filter_arr[0:filter_arr.shape[0]- 1,:] -  filter_arr[1:, :]
    dx2 = dx[0:dx.shape[0] - 1,:] -  dx[1:, :]
    dy = filter_arr[:,0:filter_arr.shape[1]-1] -  filter_arr[:, 1:]
    dy2 = dy[:, 0:dy.shape[1]-1] -  dy[:, 1:]
    dx2 = dx2[:, 0:dx2.shape[1]-2]
    dy2=dy2[0:dy2.shape[0]-2, :]
    lap = dx2+dy2
    filter_arr=filter_arr[0:filter_arr.shape[0]-2,
                          0:filter_arr.shape[1]-2]-param*lap
    if iter in [1, 12]:
        ax[cnt_fig].imshow(filter_arr, cmap="gray")
        ax[cnt_fig].axis("off")
        ax[cnt_fig].set_title("After {0} iteration(s)".format(iter))
        cnt_fig += 1

fig.savefig("Ex3_laplacian_sharpening.png")

