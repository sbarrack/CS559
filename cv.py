#!/bin/python
from scipy.misc import imread, imsave
from scipy.signal import medfilt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from numpy import array, sum as summ, copy, uint8, ceil, log2, empty, log10, hypot
from numpy.random import rand
from numpy.fft import fftn, fftshift, ifftn
#from time import time

pic = imread('picture.jpg')

def gray(img):
	bland = rgb_to_hsv(img)
	bland[:, :, 1] = 0
	return hsv_to_rgb(bland)[:, :, 0]

def isGray(img):
	return img.shape == (img.shape[0], img.shape[1])

# Homework 1

def mirrorH(img):
	return img[:, ::-1]

def mirrorV(img):
	return img[::-1]

def flip(img):
	return mirrorV(mirrorH(img))

def interpolate(img, y, x):
	"""Bilinearly interpolates a pixel from the image"""
	pixel = []
	y1 = int(y)
	x1 = int(x)
	yRatio = y - y1
	xRatio = x - x1
	y2 = min(y1 + 1, img.shape[0] - 1)
	x2 = min(x1 + 1, img.shape[1] - 1)
	for z in range(img.shape[2]):
		bl = img[y1, x1, z]
		br = img[y1, x2, z]
		tl = img[y2, x1, z]
		tr = img[y2, x2, z]
		b = xRatio * br + (1 - xRatio) * bl
		t = xRatio * tr + (1 - xRatio) * tl
		mid = yRatio * t + (1 - yRatio) * b
		pixel.append(int(mid + 0.5))
	return pixel

def smallRBig(img, k=.5):
	"""k - scaling factor; any float, large sizes take longer but do work"""
	if isGray(img):
		bigimg = empty((int(img.shape[0] * k), int(img.shape[1] * k)), dtype=uint8)
	else:
		bigimg = empty((int(img.shape[0] * k), int(img.shape[1] * k), img.shape[2]), dtype=uint8)
	for y in range(bigimg.shape[0]):
		for x in range(bigimg.shape[1]):
			bigimg[y, x] = interpolate(img, y / k, x / k)
	return bigimg

imsave('picture1.jpg', smallRBig(flip(pic)))

# Homework 2
pink = (300, 60)

def keepColor(img, hues):
	singled = rgb_to_hsv(img)
	h0, h1 = hues
	h0 /= 360
	h1 /= 360
	hue = singled[:, :, 0]
	bland = copy(singled)
	bland[:, :, 1] = 0
	if h0 < h1: # Normal
		singled[hue < h0] = bland[hue < h0]
		singled[hue > h1] = bland[hue > h1]
	elif h0 == h1: # Single hue
		singled[hue != h0] = bland[hue != h0]
	else: # Wrap around
		bland[hue < h1] = singled[hue < h1]
		bland[hue > h0] = singled[hue > h0]
		return hsv_to_rgb(bland)
	return hsv_to_rgb(singled)

imsave('picture2.jpg', keepColor(pic, pink))

# Homework 3
kernel = array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

def saltNPepper(img, salt=.025, pepper=.025):
	noisy = copy(img)
	mask = rand(noisy.shape[0], noisy.shape[1])
	noisy[mask < pepper] = 0
	noisy[mask > 1 - salt] = 255
	return noisy

def median(img):
	return medfilt(img).astype(uint8)

def mkEdge(img):
	"""Adds a matching, single-pixel border"""
	if isGray(img):
		more = empty((img.shape[0] + 2, img.shape[1] + 2), dtype=uint8)
	else:
		more = empty((img.shape[0] + 2, img.shape[1] + 2, img.shape[2]), dtype=uint8)
	more[1:-1, 1:-1] = img # Middle
	more[:, 0:1] = more[:, 1:2] # Left
	more[:, -1:] = more[:, -2:-1] # Right
	more[0:1] = more[1:2] # Top
	more[-1:] = more[-2:-1] # Bottom
	return more

def rmEdge(img):
	"""Removes a single-pixel border"""
	return img[1:-1, 1:-1]

def sobel(img):
	if img.shape != (img.shape[0], img.shape[1]):
		img = gray(img)
	edgy = empty((img.shape[0] - 1, img.shape[1] - 1))
	for y in range(1, edgy.shape[0] - 2):
		for x in range(1, edgy.shape[1] - 2):
			chunk = img[y - 1:y + 2, x - 1:x + 2]
			Gx = summ(chunk * kernel)
			Gy = summ(chunk * kernel.T)
			edgy[y - 1, x - 1] = hypot(Gx, Gy)
	return edgy * .05891866913123844731977818853974

def threshold(img, th=127):
	thresh = copy(img)
	thresh[img < th] = 0
	thresh[img >= th] = 255
	return thresh

imsave('picture3a.jpg', sobel(threshold(pic)))
imsave('picture3b.jpg', median(saltNPepper(gray(pic))))

# Homework 4

def ceilPow2(x):
	"""Rounds up to the nearest power of two"""
	return 1 << int(ceil(log2(x)))

def makeSqr(img):
	"""Make an image square for an FFT"""
	square = []
	taller = img.shape[0] > img.shape[1]
	# Side of the square image
	s = ceilPow2(img.shape[0]) if taller else ceilPow2(img.shape[1])
	if isGray(img):
		square = empty((s, s), dtype=uint8)
	else:
		square = empty((s, s, img.shape[2]), dtype=uint8)
	# Copy the OG once
	square[:img.shape[0], :img.shape[1]] = img
	if taller:
		i = img.shape[1]
		# Copy it vertically, once
		square[img.shape[0]:, :img.shape[1]] = img[:s - img.shape[0]]
		j = s - img.shape[1]
		# Copy it horizontally until you reach the end
		while i < j:
			square[:, i:i + img.shape[1]] = square[:, :img.shape[1]]
			i += img.shape[1]
		square[:, i:] = square[:, :s - i]
	else:
		i = img.shape[0]
		# Ditto horizontally, once
		square[:i, img.shape[1]:] = img[:, :s - img.shape[1]]
		j = s - img.shape[0]
		# Ditto vertically until the end
		while i < j:
			square[i:i + img.shape[0]] = square[:img.shape[0]]
			i += img.shape[0]
		square[i:] = square[:s - i]
	return square

def pic2fft(img):
	return fftn(makeSqr(img).astype(float))

def viewFft(imgfft):
	imgfft[imgfft == 0] = .0001
	return (20 * log10(abs(fftshift(imgfft)))).astype(uint8)

def lowFat(imgfft, ro, p=2):
	"""
	Apply Butterworth low-pass filter
	ro - cutoff frequency
	p - order
	"""
	smooth = empty(imgfft.shape)
	for v in range(imgfft.shape[0]):
		for u in range(imgfft.shape[1]):
			smooth[v][u] = 1 / (1 + (hypot(v, u) / ro) ** (p << 1))
	half = smooth.shape[1] >> 1
	smooth[:, half:] = mirrorH(smooth)[:, half:]
	half = smooth.shape[0] >> 1
	smooth[half:] = mirrorV(smooth)[half:]
	# fftshift() ain't just for FFTs :)
	#imsave('picture3.jpg', fftshift(smooth * 255).astype(uint8))
	return smooth * imgfft

def fft2pic(imgfft, ogshape):
	return ifftn(imgfft).real.astype(uint8)[:ogshape[0], :ogshape[1]]

def applyLowFat(img, ro, p=2):
	return fft2pic(lowFat(pic2fft(img), ro, p), img.shape)

imsave('picture4.jpg', applyLowFat(pic, 100, 1))
