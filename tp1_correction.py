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
from PIL import Image

###############################################################################
#I Discrete Fourier Transform of a synthetic image
###############################################################################

#Create a synthetic image with vertical lines, varying the space between lines
spaces = [5, 10, 20]
fig, ax = plt.subplots(len(spaces), 2, figsize=(15, 15)) 
fig.subplots_adjust(hspace=0.1, wspace=0)
for cnt_space, space in enumerate(spaces):
    im_arr = np.zeros((50, 50))
    for i in range(0, 50, space):
        im_arr[:, i] = 255

    #Show the synthetic image
    ax[cnt_space, 0].imshow(im_arr, cmap="gray")
    ax[cnt_space, 0].axis("off")
        
    #Compute the DFT
    DFT_im = np.fft.fft2(im_arr)

    #Shift the zero-frequency DTF component to the center of the spectrum    
    DFTS_im = np.fft.fftshift(DFT_im)

    #Show the modulus/spectrum of the synthetic image
    ax[cnt_space, 1].imshow(np.abs(DFTS_im))
    ax[cnt_space, 1].axis("off")
    
ax[0, 0].set_title("Synthetic image")
ax[0, 1].set_title("DFT spectrum")
fig.savefig("DFT_synthetic.png")

###############################################################################
#II Discrete Fourier Transform of a grayscale image
###############################################################################

#Get the current script directory
script_dir = os.path.dirname(__file__)
im_dir = os.path.join(script_dir, "images")

#Load the Lena image
im_arr = Image.open(os.path.join(im_dir, "boat512.gif"))
im_arr = np.array(im_arr)
im_arr = im_arr / np.max(im_arr)

#Show the Lena image
fig, ax = plt.subplots(1, 4, figsize=(10, 10)) 
fig.subplots_adjust(hspace=0.1, wspace=0.1)
ax[0].imshow(im_arr, cmap="gray")
ax[0].set_title("Natural image")
ax[0].axis("off")

#Compute the DFT
dtf_im = np.fft.fft2(im_arr)

#reorder the coefficient for "human" visualization.
dtfs_im = np.fft.fftshift(dtf_im)
print(np.abs(dtfs_im.min()))
print(np.abs(dtfs_im.max()))

#Show the modulus of the DTF using logarithm scale	
ax[1].imshow(np.log(np.abs(dtfs_im) + 1))
ax[1].set_title("DFT spectrum")
ax[1].axis("off")

#Show the argument of the DTF 
ax[2].imshow(np.angle(dtfs_im))
ax[2].set_title("DFT phase")
ax[2].axis("off")

#Get back to the original image using the inverse transform
dtf_im = np.fft.ifftshift(dtfs_im)
fim_arr = np.abs(np.fft.ifft2(dtf_im))

ax[3].imshow(fim_arr, cmap='gray')
ax[3].set_title("Image using IDFT")
ax[3].axis("off")
fig.savefig("DFT_real.png")

###############################################################################
#III Phase and modulus exchange
###############################################################################

#a) Change the modulus of the DTF of im1 to the modulus of the DTF of im2

#Transform into cartesian coordinate
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

#Load the first image, then convert it into grayscale image
im1_arr = Image.open(os.path.join(im_dir, "lena.bmp"))
im1_arr = np.array(im1_arr)

#Load the second image, then convert it into grayscale image
im2_arr = Image.open(os.path.join(im_dir, "barbara.png"))
im2_arr = np.array(im2_arr)

#Get the phase of the DFT of image 1
im1_phase = np.angle(np.fft.fft2(im1_arr))

#Get the spectrum of the DFT of image 2
im2_modulus = np.abs(np.fft.fft2(im2_arr))

#Create a new real image, combining the modulus of the DFT of image 2 and
#the phase of the DFT of image 1
real_arr, imaginary_arr = pol2cart(im2_modulus, im1_phase)
im1_mod2_arr = real_arr + 1j * imaginary_arr
im1_mod2_arr = np.fft.ifft2(im1_mod2_arr)
im1_mod2_arr = np.abs(im1_mod2_arr)

#a) Change the phase of the DTF of im1 to the phase of the DTF of im2

#Get the phase of the DTF of image 2
im2_phase = np.angle(np.fft.fft2(im2_arr))

#Get the spectrum of the DTF of image 1
im1_modulus = np.abs(np.fft.fft2(im1_arr))

#Create a new real image, combining the phase of the DFT of image 2 and
#the modulus of the DFT of image 1
real_arr, imaginary_arr = pol2cart(im1_modulus, im2_phase)
im1_phase2_arr = real_arr + 1j * imaginary_arr
im1_phase2_arr = np.fft.ifft2(im1_phase2_arr)
im1_phase2_arr = np.abs(im1_phase2_arr)

#Show the created images
fig, ax = plt.subplots(1, 4, figsize=(15, 15))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
title_list = ["Im1", "Im2", "Im1 with modulus of im2 DFT",
              "Im1 with phase of im2 DFT"]
arr_list = [im1_arr, im2_arr, im1_mod2_arr, im1_phase2_arr]
for cnt, (arr, title) in enumerate(zip(arr_list, title_list)):
    ax[cnt].imshow(arr, cmap="gray")
    ax[cnt].set_title(title)
    ax[cnt].axis("off")
fig.savefig("DFT_exchanges.png")

###############################################################################
#IV Basics transform
###############################################################################

#Image directory
script_dir = os.path.dirname(__file__)
im_dir = os.path.join(script_dir, "images")

#a) Grayscale image translation

#Parameters
Tx = 20.5
Ty = 40.5

#Load a color image
im_arr = Image.open(os.path.join(im_dir, "hibiscus.bmp"))

#Normalize it for better visualization
im_arr = np.array(im_arr).astype(int)
im_arr = im_arr / np.max(im_arr)

#Convert it into grayscale
glim_arr = np.mean(im_arr, 2)

#Get the height and width of the image
sx, sy = glim_arr.shape

#Get the u and v 1D coordinates (Fourier domain)
v_vect = np.array(np.fft.ifftshift(range(int(-sy/2), int(sy/2))))
u_vect = np.array(np.fft.ifftshift(range(int(-sx/2), int(sx/2))))

#Compute the 2D coordinate arrays 
[v, u] = np.meshgrid(u_vect, v_vect)

#Compute the phase 
phase = np.exp(-2 * 1j * np.pi * ((Tx * u) / sx + (Ty * v) / sy))

#Compute the DFT of the grayscale image
dtf_glim = np.fft.fft2(glim_arr)

#Compute the DTF of the translated grayscale image 
dtf_glim_t = dtf_glim * phase

#Compute the inverse DTF to retrieve a real grayscale image
glim_t = np.fft.ifft2(dtf_glim_t)
glim_abs_t = np.abs(glim_t)

"""
#Version 2: the code above is equivalent to:
#1/Compute the shifted DTF   
dtfs_glim = np.fft.fftshift(np.fft.fft2(glim_arr))

#2/Compute the coordinates in the shifted DTF space:
# u \in [-sx/2,sx/2], v \in [-sy/2, sy/2]
u_vect = np.array(range(int(-sx / 2), int(sx / 2)))
v_vect = np.array(range(int(-sy / 2), int(sy / 2)))
[v, u] = np.meshgrid(u_vect, v_vect)

#3/Compute the shifted DTF of the translated image
phase = np.exp(-2 * 1j * np.pi * ((Tx * u) / sx + (Ty * v) / sy))
dtfs_glim_t = dtfs_glim * phase

#4/Retrieve the real image
glim_t = np.fft.ifft2(np.fft.ifftshift(dtfs_glim_t))
glim_abs_t = np.abs(glim_t)
"""
#b) RGB image translation

#Compute the DTF of the RGB image
dft_im = np.fft.fftn(im_arr)

#Compute the array phase (same 2D array in the 3 channels)
phase3D = np.dstack((phase, phase, phase))

#Compute the DTF of the translated RGB image
dft_t = dft_im * phase3D

#Compute the inverse DTF to retrieve a real RGB image
im_t = np.fft.ifftn(dft_t)
im_abs_t = np.abs(im_t)

#Save results
nb_lines = 2
nb_cols = 2
fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
title_list = ["Image", "Translated image"]
arr_list = [glim_arr, glim_abs_t, im_arr, im_abs_t]
cmaps = ["gray", "gray", "hsv", "hsv"]
cols, lines = np.meshgrid(range(nb_cols), range(nb_lines))
for line_id, col_id, arr, cmap in zip(lines.flatten(), cols.flatten(),
                                       arr_list, cmaps):
    ax[line_id, col_id].imshow(arr, cmap=cmap)
    ax[line_id, col_id].axis("off")   
    if line_id == 0:
        ax[line_id, col_id].set_title(title_list[col_id])        
fig.savefig("DFT_translated.png")

#b) Graylevel image rotation

#References:
#M. Unser, P. Thevenaz, and L. Yaroslavsky. Convolution-based 
#interpolation for fast, high-quality rotation ofimages. 
#Image Processing, IEEE Transactions on, 4(10) :1371–1381, 1995.

#angle (degree)
angle = 45.5

#angle (radiant)
angle *= np.pi / 180

#Pad the image
pim_arr = np.mean(glim_arr) * np.zeros((2 * sx, 2 * sy))
pim_arr[int(np.floor(sx / 2)): int(np.floor(sx / 2)) + sx,
        int(np.floor(sy / 2)): int(np.floor(sy / 2)) + sy] = glim_arr

#Get the 1D coordinates in the u and v axis
u_vect = np.array(np.fft.ifftshift(range(-sx, sx)))
v_vect = np.array(np.fft.ifftshift(range(-sy, sy)))

#Initialize the rotated image
rot_arr = np.mean(pim_arr) * np.zeros((2 * sx, 2 * sy))

#Translate the grayscale image of tan(angle/2) among the line 
for i in range(0, 2 * sx):
    rot_arr[i, :] = np.fft.ifft(np.fft.fft(pim_arr[i, :])
                                * np.exp(-2 * 1j * np.pi * (i - sx) *
                                         v_vect * (np.tan(angle / 2)
                                                  / (2 * sy))))
    
#Translate the previous image of sin(angle/2) among the column 
for i in range(0, 2 * sy):
    rot_arr[:,i] = np.conjugate(np.fft.ifft(
        np.fft.fft(np.conjugate(rot_arr[:,i]).T)*
        np.exp(2 * 1j * np.pi * (i - sy) * u_vect *
               (np.sin(angle) / (2 * sx))))).T

#Translate the previous image of tan(angle/2) among the line  
for i in range(0, 2 * sx):
    rot_arr[i, :] = np.real(np.fft.ifft(np.fft.fft(rot_arr[i, :])
                                        *np.exp(-2 * 1j * np.pi *
                                                (i - sx) * v_vect *
                                                (np.tan(angle / 2)
                                                 / (2 * sy)))))

#Save results
nb_lines, nb_cols = 1, 2
fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
title_list = ["Image", "Rotated image"]
arr_list = [pim_arr, rot_arr]
for col_id, arr in enumerate(arr_list):
    ax[col_id].imshow(arr, cmap="gray")
    ax[col_id].axis("off")   
    ax[col_id].set_title(title_list[col_id])        
fig.savefig("DFT_rotated.png")
    
###############################################################################
#V Phase correlation
###############################################################################

#a) Phase correlation between a grayscale image and its translation

#Computation of the cross-power spectrum
cps = dtf_glim * np.ma.conjugate(dtf_glim_t)
cps /= np.abs(cps)

#Computation of the phase correlation
glim_pc =  np.abs(np.fft.ifft2(cps))

#Compute the location of the voxel with maximum correlation
[xm, ym] = np.where(glim_pc == glim_pc.max())

#b) Phase correlation between a noisy grayscale image and its translation

#Add noise to the grayscale image
nglim_arr = glim_arr +  np.random.normal(0, 0.01, ((sx, sy)))

#Get the DTF of the noisy image
dtf_nglim = np.fft.fft2(nglim_arr)

#Get the DTF of the translated image of the noisy image
dtf_nglim_t = dtf_nglim * phase
nglim_t = np.fft.ifft2(dtf_nglim_t)
nglim_abs_t = np.abs(nglim_t)

#Computation of the cross-power spectrum
cps = dtf_nglim * np.ma.conjugate(dtf_nglim_t)
cps /= np.abs(cps)

#Computation of the phase correlation
nglim_pc =  np.abs(np.fft.ifft2(cps))

#Compute the location of the voxel with maximum correlation
[nxm, nym] = np.where(nglim_pc == nglim_pc.max())

#Save results
nb_lines = 2
nb_cols = 3
fig, ax = plt.subplots(nb_lines, nb_cols, figsize=(15, 15))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
title_list = ["Image", "Transformed image", "Phase correlation"]
arr_list = [glim_arr, glim_abs_t, glim_pc,
            nglim_arr, nglim_abs_t, nglim_pc]
max_list = [[xm, ym], [nxm, nym]]
cols, lines = np.meshgrid(range(nb_cols), range(nb_lines))
for line_id, col_id, arr in zip(lines.flatten(), cols.flatten(),
                                       arr_list):
    ax[line_id, col_id].imshow(arr, cmap="gray")
    ax[line_id, col_id].axis("off")   
    if line_id == 0:
        ax[line_id, col_id].set_title(title_list[col_id])
    if col_id == 2:
        ax[line_id, col_id].plot(max_list[line_id][0], max_list[line_id][1],
                                 "or")     
fig.savefig("phase_correlation.png")
