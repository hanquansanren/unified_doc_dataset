# import argparse
import sys, os
from os.path import join as pjoin
import pickle
import random
import collections
import json
import numpy as np
import scipy.io as io
import scipy.misc as m
import matplotlib.pyplot as plt
import glob
import math
import time

from PIL import Image
import threading
import multiprocessing as mp
from multiprocessing import Pool
import re
import cv2
import synthesis_code.utils as sutils
import lmdb

class perturbed(sutils.BasePerturbed):
	def __init__(self, path, bg_img_list, save_path, save_suffix):
		self.path = path
		self.bg_img_list = bg_img_list
		self.save_path = save_path
		self.save_suffix = save_suffix

	def save_img(self, fold_curve='fold', repeat_time=4, fiducial_points = 61, relativeShift_position='relativeShift_v2', idx= 'd1'):
		begin_train = time.time()
		error_flag=0
		origin_img = cv2.imread(self.path, flags=cv2.IMREAD_COLOR)
		base_img_bound = [1024, 768]
		input_image_height = origin_img.shape[0] # H
		input_image_width = origin_img.shape[1] # W

		'''随机设定扫描图像的长边要缩小多少'''
		reduce_value = np.random.randint(15,370)
		if input_image_height > input_image_width: # H>W 常见情况
			flag=1 #用于决定是否将背景图像翻转90°
			random_shrink_long_edge = base_img_bound[0] - reduce_value
			scaled_image_height = random_shrink_long_edge
			scaled_image_width = int(input_image_width / input_image_height * random_shrink_long_edge)
		else:      # W>H 较少见情况
			flag=0
			base_img_bound = [768, 1024]
			random_shrink_long_edge = base_img_bound[1] - reduce_value
			scaled_image_width = random_shrink_long_edge
			scaled_image_height = int(input_image_height / input_image_width * random_shrink_long_edge)
			
		
		# if round(scaled_image_height / scaled_image_width, 2) < 0.5 or round(scaled_image_width / scaled_image_height, 2) < 0.5:
		# 	repeat_time = min(repeat_time, 8) #如果原图的横纵比比较极端（比如长票据），则控制最大的弯折次数为8
		edge_padding = 2 #用于调整控制点对不齐合成扭曲图像边界的问题
		scaled_image_height -= scaled_image_height % (fiducial_points-1) - (2*edge_padding)		
		scaled_image_width -= scaled_image_width % (fiducial_points-1) - (2*edge_padding)
		self.origin_img = cv2.resize(origin_img, (scaled_image_width, scaled_image_height), interpolation=cv2.INTER_CUBIC)		
	
		'''参考点网格'''
		im_hight = np.linspace(0, scaled_image_height-edge_padding, fiducial_points, dtype=np.int64)
		im_wide = np.linspace(0, scaled_image_width-edge_padding, fiducial_points, dtype=np.int64)
		im_x, im_y = np.meshgrid(im_hight, im_wide)
		# plt.plot(im_x, im_y,
		# 		 color='limegreen',
		# 		 marker='.',
		# 		 linestyle='')
		# plt.grid(True)
		# plt.savefig('./predict.png')

		# enlarge_img_shrink = [512*4, 512*4]
		enlarge_img_shrink = [2048, 2048]
		bg_img = './dataset/background/' + random.choice(self.bg_img_list)
		perturbed_bg_img = cv2.imread(bg_img, flags=cv2.IMREAD_COLOR)
		if flag==1:
			perturbed_bg_img = cv2.rotate(perturbed_bg_img, cv2.ROTATE_90_CLOCKWISE)
		else:
			None
		# lp2 = cv2.rotate(lp, cv2.ROTATE_90_COUNTERCLOCKWISE)
		mesh_shape = self.origin_img.shape[0:2] # (829, 621, 3)

		self.synthesis_perturbed_img = np.full((enlarge_img_shrink[0], enlarge_img_shrink[1], 3), 256, dtype=np.float32)#np.zeros_like(perturbed_bg_img)
		self.new_shape = self.synthesis_perturbed_img.shape[0:2] #背景图的初始尺寸2048*2048
		perturbed_bg_img = cv2.resize(perturbed_bg_img, (base_img_bound[1], base_img_bound[0]), cv2.INPAINT_TELEA)

		origin_pixel_position = np.argwhere(np.zeros(mesh_shape, dtype=np.uint32) == 0).reshape(mesh_shape[0], mesh_shape[1], 2) #先x后Y
		pixel_position = np.argwhere(np.zeros(self.new_shape, dtype=np.uint32) == 0).reshape(self.new_shape[0], self.new_shape[1], 2)
		self.perturbed_xy_ = np.zeros((self.new_shape[0], self.new_shape[1], 2))
		# self.perturbed_xy_ = pixel_position.copy().astype(np.float32)
		# fiducial_points_grid = origin_pixel_position[im_x, im_y]

		self.synthesis_perturbed_label = np.zeros((self.new_shape[0], self.new_shape[1], 2))
		x_min, y_min, x_max, y_max = self.adjust_position_v2(0, 0, mesh_shape[0], mesh_shape[1], base_img_bound)
		origin_pixel_position += [x_min, y_min] # 从坐标原点移动到(115,86)

		x_min, y_min, x_max, y_max = self.adjust_position(0, 0, mesh_shape[0], mesh_shape[1])
		x_shift = random.randint(-enlarge_img_shrink[0]//16, enlarge_img_shrink[0]//16)
		y_shift = random.randint(-enlarge_img_shrink[1]//16, enlarge_img_shrink[1]//16)
		x_min += x_shift
		x_max += x_shift
		y_min += y_shift
		y_max += y_shift

		'''im_x,y'''
		im_x += x_min
		im_y += y_min

		plt.plot(im_x, im_y,
				 color='limegreen',
				 marker='.',
				 linestyle='')
		plt.grid(True)
		plt.savefig('./predicta.png')

		self.synthesis_perturbed_img[x_min:x_max, y_min:y_max] = self.origin_img
		self.synthesis_perturbed_label[x_min:x_max, y_min:y_max] = origin_pixel_position

		synthesis_perturbed_img_map = self.synthesis_perturbed_img.copy()
		synthesis_perturbed_label_map = self.synthesis_perturbed_label.copy()

		foreORbackground_label = np.full((mesh_shape), 1, dtype=np.int16)
		foreORbackground_label_map = np.full((self.new_shape), 0, dtype=np.int16)
		foreORbackground_label_map[x_min:x_max, y_min:y_max] = foreORbackground_label







		'''*****************************************************************'''
		is_normalizationFun_mixture = self.is_perform(0.2, 0.8)
		# if not is_normalizationFun_mixture:
		normalizationFun_0_1 = False
		# normalizationFun_0_1 = self.is_perform(0.5, 0.5)

		if fold_curve == 'fold':
			fold_curve_random = True
			# is_normalizationFun_mixture = False
			normalizationFun_0_1 = self.is_perform(0.2, 0.8)
			if is_normalizationFun_mixture:
				alpha_perturbed = random.randint(80, 120) / 100
			else:
				if normalizationFun_0_1 and repeat_time < 8:
					alpha_perturbed = random.randint(50, 70) / 100
				else:
					alpha_perturbed = random.randint(70, 130) / 100
		else:
			fold_curve_random = self.is_perform(0.1, 0.9)  # False		# self.is_perform(0.01, 0.99)
			alpha_perturbed = random.randint(80, 160) / 100
			# is_normalizationFun_mixture = False  # self.is_perform(0.01, 0.99)
		synthesis_perturbed_img = np.full_like(self.synthesis_perturbed_img, 256)
		# synthesis_perturbed_img = np.full_like(self.synthesis_perturbed_img, 0, dtype=np.int16)
		synthesis_perturbed_label = np.zeros_like(self.synthesis_perturbed_label)

		alpha_perturbed_change = self.is_perform(0.5, 0.5)
		p_pp_choice = self.is_perform(0.8, 0.2) if fold_curve == 'fold' else self.is_perform(0.1, 0.9)
		for repeat_i in range(repeat_time):

			if alpha_perturbed_change:
				if fold_curve == 'fold':
					if is_normalizationFun_mixture:
						alpha_perturbed = random.randint(80, 120) / 100
					else:
						if normalizationFun_0_1 and repeat_time < 8:
							alpha_perturbed = random.randint(50, 70) / 100
						else:
							alpha_perturbed = random.randint(70, 130) / 100
				else:
					alpha_perturbed = random.randint(80, 160) / 100
			''''''
			linspace_x = [0, (self.new_shape[0] - scaled_image_height) // 2 - 1,
						  self.new_shape[0] - (self.new_shape[0] - scaled_image_height) // 2 - 1, self.new_shape[0] - 1]
			linspace_y = [0, (self.new_shape[1] - scaled_image_width) // 2 - 1,
						  self.new_shape[1] - (self.new_shape[1] - scaled_image_width) // 2 - 1, self.new_shape[1] - 1]
			linspace_x_seq = [1, 2, 3]
			linspace_y_seq = [1, 2, 3]
			r_x = random.choice(linspace_x_seq)
			r_y = random.choice(linspace_y_seq)
			perturbed_p = np.array(
				[random.randint(linspace_x[r_x-1] * 10, linspace_x[r_x] * 10),
				 random.randint(linspace_y[r_y-1] * 10, linspace_y[r_y] * 10)])/10
			if ((r_x == 1 or r_x == 3) and (r_y == 1 or r_y == 3)) and p_pp_choice:
				linspace_x_seq.remove(r_x)
				linspace_y_seq.remove(r_y)
			r_x = random.choice(linspace_x_seq)
			r_y = random.choice(linspace_y_seq)
			perturbed_pp = np.array(
				[random.randint(linspace_x[r_x-1] * 10, linspace_x[r_x] * 10),
				 random.randint(linspace_y[r_y-1] * 10, linspace_y[r_y] * 10)])/10
			''''''

			perturbed_vp = perturbed_pp - perturbed_p
			perturbed_vp_norm = np.linalg.norm(perturbed_vp)

			perturbed_distance_vertex_and_line = np.dot((perturbed_p - pixel_position), perturbed_vp) / perturbed_vp_norm
			''''''
			# perturbed_v = np.array([random.randint(-3000, 3000) / 100, random.randint(-3000, 3000) / 100])
			# perturbed_v = np.array([random.randint(-4000, 4000) / 100, random.randint(-4000, 4000) / 100])
			if fold_curve == 'fold' and self.is_perform(0.6, 0.4):	# self.is_perform(0.3, 0.7):
				# perturbed_v = np.array([random.randint(-9000, 9000) / 100, random.randint(-9000, 9000) / 100])
				perturbed_v = np.array([random.randint(-10000, 10000) / 100, random.randint(-10000, 10000) / 100])
				# perturbed_v = np.array([random.randint(-11000, 11000) / 100, random.randint(-11000, 11000) / 100])
			else:
				# perturbed_v = np.array([random.randint(-9000, 9000) / 100, random.randint(-9000, 9000) / 100])
				# perturbed_v = np.array([random.randint(-16000, 16000) / 100, random.randint(-16000, 16000) / 100])
				perturbed_v = np.array([random.randint(-8000, 8000) / 100, random.randint(-8000, 8000) / 100])
				# perturbed_v = np.array([random.randint(-3500, 3500) / 100, random.randint(-3500, 3500) / 100])
				# perturbed_v = np.array([random.randint(-600, 600) / 10, random.randint(-600, 600) / 10])
			''''''
			if fold_curve == 'fold':
				if is_normalizationFun_mixture:
					if self.is_perform(0.5, 0.5):
						perturbed_d = np.abs(self.get_normalize(perturbed_distance_vertex_and_line))
					else:
						perturbed_d = self.get_0_1_d(np.abs(perturbed_distance_vertex_and_line), random.randint(1, 2))
				else:
					if normalizationFun_0_1:
						perturbed_d = self.get_0_1_d(np.abs(perturbed_distance_vertex_and_line), 2)
					else:
						perturbed_d = np.abs(self.get_normalize(perturbed_distance_vertex_and_line))

			else:
				if is_normalizationFun_mixture:
					if self.is_perform(0.5, 0.5):
						perturbed_d = np.abs(self.get_normalize(perturbed_distance_vertex_and_line))
					else:
						perturbed_d = self.get_0_1_d(np.abs(perturbed_distance_vertex_and_line), random.randint(1, 2))
				else:
					if normalizationFun_0_1:
						perturbed_d = self.get_0_1_d(np.abs(perturbed_distance_vertex_and_line), 2)
					else:
						perturbed_d = np.abs(self.get_normalize(perturbed_distance_vertex_and_line))
			''''''
			if fold_curve_random:
				# omega_perturbed = (alpha_perturbed+0.2) / (perturbed_d + alpha_perturbed)
				# omega_perturbed = alpha_perturbed**perturbed_d
				omega_perturbed = alpha_perturbed / (perturbed_d + alpha_perturbed)
			else:
				omega_perturbed = 1 - perturbed_d ** alpha_perturbed

			'''shadow'''
			if self.is_perform(0.6, 0.4):
				synthesis_perturbed_img_map[x_min:x_max, y_min:y_max] = np.minimum(np.maximum(synthesis_perturbed_img_map[x_min:x_max, y_min:y_max] - np.int16(np.round(omega_perturbed[x_min:x_max, y_min:y_max].repeat(3).reshape(x_max-x_min, y_max-y_min, 3) * abs(np.linalg.norm(perturbed_v//2))*np.array([0.4-random.random()*0.1, 0.4-random.random()*0.1, 0.4-random.random()*0.1]))), 0), 255)
			''''''

			if relativeShift_position in ['position', 'relativeShift_v2']:
				self.perturbed_xy_ += np.array([omega_perturbed * perturbed_v[0], omega_perturbed * perturbed_v[1]]).transpose(1, 2, 0)
			else:
				print('relativeShift_position error')
				exit()


		'''perspective'''
		perspective_shreshold = random.randint(26, 36)*10 # 280
		x_min_per, y_min_per, x_max_per, y_max_per = self.adjust_position(perspective_shreshold, perspective_shreshold, self.new_shape[0]-perspective_shreshold, self.new_shape[1]-perspective_shreshold)
		pts1 = np.float32([[x_min_per, y_min_per], [x_max_per, y_min_per], [x_min_per, y_max_per], [x_max_per, y_max_per]])
		e_1_ = x_max_per - x_min_per
		e_2_ = y_max_per - y_min_per
		e_3_ = e_2_
		e_4_ = e_1_
		perspective_shreshold_h = e_1_*0.02
		perspective_shreshold_w = e_2_*0.02
		a_min_, a_max_ = 70, 110
		# if self.is_perform(1, 0):
		if fold_curve == 'curve' and self.is_perform(0.5, 0.5):
			if self.is_perform(0.5, 0.5):
				while True:
					pts2 = np.around(
						np.float32([[x_min_per - (random.random()) * perspective_shreshold, y_min_per + (random.random()) * perspective_shreshold],
									[x_max_per - (random.random()) * perspective_shreshold, y_min_per - (random.random()) * perspective_shreshold],
									[x_min_per + (random.random()) * perspective_shreshold, y_max_per + (random.random()) * perspective_shreshold],
									[x_max_per + (random.random()) * perspective_shreshold, y_max_per - (random.random()) * perspective_shreshold]]))	# right
					e_1 = np.linalg.norm(pts2[0]-pts2[1])
					e_2 = np.linalg.norm(pts2[0]-pts2[2])
					e_3 = np.linalg.norm(pts2[1]-pts2[3])
					e_4 = np.linalg.norm(pts2[2]-pts2[3])
					if e_1_+perspective_shreshold_h > e_1 and e_2_+perspective_shreshold_w > e_2 and e_3_+perspective_shreshold_w > e_3 and e_4_+perspective_shreshold_h > e_4 and \
						e_1_ - perspective_shreshold_h < e_1 and e_2_ - perspective_shreshold_w < e_2 and e_3_ - perspective_shreshold_w < e_3 and e_4_ - perspective_shreshold_h < e_4 and \
						abs(e_1-e_4) < perspective_shreshold_h and abs(e_2-e_3) < perspective_shreshold_w:
						a0_, a1_, a2_, a3_ = self.get_angle_4(pts2)
						if (a0_ > a_min_ and a0_ < a_max_) or (a1_ > a_min_ and a1_ < a_max_) or (a2_ > a_min_ and a2_ < a_max_) or (a3_ > a_min_ and a3_ < a_max_):
							break
			else:
				while True:
					pts2 = np.around(
						np.float32([[x_min_per + (random.random()) * perspective_shreshold, y_min_per - (random.random()) * perspective_shreshold],
									[x_max_per + (random.random()) * perspective_shreshold, y_min_per + (random.random()) * perspective_shreshold],
									[x_min_per - (random.random()) * perspective_shreshold, y_max_per - (random.random()) * perspective_shreshold],
									[x_max_per - (random.random()) * perspective_shreshold, y_max_per + (random.random()) * perspective_shreshold]]))
					e_1 = np.linalg.norm(pts2[0]-pts2[1])
					e_2 = np.linalg.norm(pts2[0]-pts2[2])
					e_3 = np.linalg.norm(pts2[1]-pts2[3])
					e_4 = np.linalg.norm(pts2[2]-pts2[3])
					if e_1_+perspective_shreshold_h > e_1 and e_2_+perspective_shreshold_w > e_2 and e_3_+perspective_shreshold_w > e_3 and e_4_+perspective_shreshold_h > e_4 and \
						e_1_ - perspective_shreshold_h < e_1 and e_2_ - perspective_shreshold_w < e_2 and e_3_ - perspective_shreshold_w < e_3 and e_4_ - perspective_shreshold_h < e_4 and \
						abs(e_1-e_4) < perspective_shreshold_h and abs(e_2-e_3) < perspective_shreshold_w:
						a0_, a1_, a2_, a3_ = self.get_angle_4(pts2)
						if (a0_ > a_min_ and a0_ < a_max_) or (a1_ > a_min_ and a1_ < a_max_) or (a2_ > a_min_ and a2_ < a_max_) or (a3_ > a_min_ and a3_ < a_max_):
							break

		else:
			while True:
				pts2 = np.around(np.float32([[x_min_per+(random.random()-0.5)*perspective_shreshold, y_min_per+(random.random()-0.5)*perspective_shreshold],
								 [x_max_per+(random.random()-0.5)*perspective_shreshold, y_min_per+(random.random()-0.5)*perspective_shreshold],
								 [x_min_per+(random.random()-0.5)*perspective_shreshold, y_max_per+(random.random()-0.5)*perspective_shreshold],
								 [x_max_per+(random.random()-0.5)*perspective_shreshold, y_max_per+(random.random()-0.5)*perspective_shreshold]]))
				e_1 = np.linalg.norm(pts2[0]-pts2[1])
				e_2 = np.linalg.norm(pts2[0]-pts2[2])
				e_3 = np.linalg.norm(pts2[1]-pts2[3])
				e_4 = np.linalg.norm(pts2[2]-pts2[3])
				if e_1_+perspective_shreshold_h > e_1 and e_2_+perspective_shreshold_w > e_2 and e_3_+perspective_shreshold_w > e_3 and e_4_+perspective_shreshold_h > e_4 and \
					e_1_ - perspective_shreshold_h < e_1 and e_2_ - perspective_shreshold_w < e_2 and e_3_ - perspective_shreshold_w < e_3 and e_4_ - perspective_shreshold_h < e_4 and \
					abs(e_1-e_4) < perspective_shreshold_h and abs(e_2-e_3) < perspective_shreshold_w:
					a0_, a1_, a2_, a3_ = self.get_angle_4(pts2)
					if (a0_ > a_min_ and a0_ < a_max_) or (a1_ > a_min_ and a1_ < a_max_) or (a2_ > a_min_ and a2_ < a_max_) or (a3_ > a_min_ and a3_ < a_max_):
						break

		M = cv2.getPerspectiveTransform(pts1, pts2)
		one = np.ones((self.new_shape[0], self.new_shape[1], 1), dtype=np.int16)
		matr = np.dstack((pixel_position, one))
		new = np.dot(M, matr.reshape(-1, 3).T).T.reshape(self.new_shape[0], self.new_shape[1], 3)
		x = new[:, :, 0]/new[:, :, 2]
		y = new[:, :, 1]/new[:, :, 2]
		perturbed_xy_ = np.dstack((x, y))
		# perturbed_xy_round_int = np.around(cv2.bilateralFilter(perturbed_xy_round_int, 9, 75, 75))
		# perturbed_xy_round_int = np.around(cv2.blur(perturbed_xy_, (17, 17)))
		# perturbed_xy_round_int = cv2.blur(perturbed_xy_round_int, (17, 17))
		# perturbed_xy_round_int = cv2.GaussianBlur(perturbed_xy_round_int, (7, 7), 0)
		perturbed_xy_ = perturbed_xy_-np.min(perturbed_xy_.T.reshape(2, -1), 1)
		# perturbed_xy_round_int = np.around(perturbed_xy_round_int-np.min(perturbed_xy_round_int.T.reshape(2, -1), 1)).astype(np.int16)

		self.perturbed_xy_ += perturbed_xy_

		'''perspective end'''

		'''to img'''
		flat_position = np.argwhere(np.zeros(self.new_shape, dtype=np.uint32) == 0).reshape(
			self.new_shape[0] * self.new_shape[1], 2)
		# self.perturbed_xy_ = cv2.blur(self.perturbed_xy_, (7, 7))
		self.perturbed_xy_ = cv2.GaussianBlur(self.perturbed_xy_, (7, 7), 0)

		'''''''''ggggggggggggggggggggggggggggggggg'''''''''''
		#########################
		'''get fiducial points'''
		fiducial_points_coordinate = self.perturbed_xy_[im_x, im_y]


		vtx, wts = self.interp_weights(self.perturbed_xy_.reshape(self.new_shape[0] * self.new_shape[1], 2), flat_position)
		wts_sum = np.abs(wts).sum(-1)

		# flat_img.reshape(flat_shape[0] * flat_shape[1], 3)[:] = interpolate(pixel, vtx, wts)
		wts = wts[wts_sum <= 1, :]
		vtx = vtx[wts_sum <= 1, :]
		synthesis_perturbed_img.reshape(self.new_shape[0] * self.new_shape[1], 3)[wts_sum <= 1,
		:] = self.interpolate(synthesis_perturbed_img_map.reshape(self.new_shape[0] * self.new_shape[1], 3), vtx, wts)

		synthesis_perturbed_label.reshape(self.new_shape[0] * self.new_shape[1], 2)[wts_sum <= 1,
		:] = self.interpolate(synthesis_perturbed_label_map.reshape(self.new_shape[0] * self.new_shape[1], 2), vtx, wts)

		foreORbackground_label = np.zeros(self.new_shape)
		foreORbackground_label.reshape(self.new_shape[0] * self.new_shape[1], 1)[wts_sum <= 1, :] = self.interpolate(foreORbackground_label_map.reshape(self.new_shape[0] * self.new_shape[1], 1), vtx, wts)
		foreORbackground_label[foreORbackground_label < 0.99] = 0
		foreORbackground_label[foreORbackground_label >= 0.99] = 1

		self.synthesis_perturbed_img = synthesis_perturbed_img
		self.synthesis_perturbed_label = synthesis_perturbed_label
		self.foreORbackground_label = foreORbackground_label


		'''clip'''
		perturbed_x_min, perturbed_y_min, perturbed_x_max, perturbed_y_max = -1, -1, self.new_shape[0], self.new_shape[1]
		for x in range(self.new_shape[0] // 2, perturbed_x_max):
			if np.sum(self.synthesis_perturbed_img[x, :]) == 768 * self.new_shape[1] and perturbed_x_max - 1 > x:
				perturbed_x_max = x
				break
		for x in range(self.new_shape[0] // 2, perturbed_x_min, -1):
			if np.sum(self.synthesis_perturbed_img[x, :]) == 768 * self.new_shape[1] and x > 0:
				perturbed_x_min = x
				break
		for y in range(self.new_shape[1] // 2, perturbed_y_max):
			if np.sum(self.synthesis_perturbed_img[:, y]) == 768 * self.new_shape[0] and perturbed_y_max - 1 > y:
				perturbed_y_max = y
				break
		for y in range(self.new_shape[1] // 2, perturbed_y_min, -1):
			if np.sum(self.synthesis_perturbed_img[:, y]) == 768 * self.new_shape[0] and y > 0:
				perturbed_y_min = y
				break

		if perturbed_x_min == 0 or perturbed_x_max == self.new_shape[0] or perturbed_y_min == self.new_shape[1] or perturbed_y_max == self.new_shape[1]:
			raise Exception('clip error1')

		if perturbed_x_max - perturbed_x_min < scaled_image_height//2 or perturbed_y_max - perturbed_y_min < scaled_image_width//2:
			print("clip error2 here")
			raise Exception('clip error2')
			error_flag=1
			


		perfix_ = self.save_suffix
		is_shrink = False
		if perturbed_x_max - perturbed_x_min > base_img_bound[0] or perturbed_y_max - perturbed_y_min > base_img_bound[1]:
			is_shrink = True
			synthesis_perturbed_img = cv2.resize(self.synthesis_perturbed_img[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max, :].copy(), (scaled_image_width, scaled_image_height), interpolation=cv2.INTER_LINEAR)
			synthesis_perturbed_label = cv2.resize(self.synthesis_perturbed_label[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max, :].copy(), (scaled_image_width, scaled_image_height), interpolation=cv2.INTER_LINEAR)
			foreORbackground_label = cv2.resize(self.foreORbackground_label[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max].copy(), (scaled_image_width, scaled_image_height), interpolation=cv2.INTER_LINEAR)
			foreORbackground_label[foreORbackground_label < 0.99] = 0
			foreORbackground_label[foreORbackground_label >= 0.99] = 1
			'''shrink fiducial points'''
			center_x_l, center_y_l = perturbed_x_min + (perturbed_x_max - perturbed_x_min) // 2, perturbed_y_min + (perturbed_y_max - perturbed_y_min) // 2
			fiducial_points_coordinate_copy = fiducial_points_coordinate.copy()
			shrink_x = scaled_image_height/(perturbed_x_max - perturbed_x_min)
			shrink_y = scaled_image_width/(perturbed_y_max - perturbed_y_min)
			fiducial_points_coordinate *= [shrink_x, shrink_y]
			center_x_l *= shrink_x
			center_y_l *= shrink_y
			# fiducial_points_coordinate[1:, 1:] *= [shrink_x, shrink_y]
			# fiducial_points_coordinate[1:, :1, 0] *= shrink_x
			# fiducial_points_coordinate[:1, 1:, 1] *= shrink_y
			# perturbed_x_min_copy, perturbed_y_min_copy, perturbed_x_max_copy, perturbed_y_max_copy = perturbed_x_min, perturbed_y_min, perturbed_x_max, perturbed_y_max

			perturbed_x_min, perturbed_y_min, perturbed_x_max, perturbed_y_max = self.adjust_position_v2(0, 0, scaled_image_height, scaled_image_width, self.new_shape)

			self.synthesis_perturbed_img = np.full_like(self.synthesis_perturbed_img, 256)
			self.synthesis_perturbed_label = np.zeros_like(self.synthesis_perturbed_label)
			self.foreORbackground_label = np.zeros_like(self.foreORbackground_label)
			self.synthesis_perturbed_img[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max, :] = synthesis_perturbed_img
			self.synthesis_perturbed_label[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max, :] = synthesis_perturbed_label
			self.foreORbackground_label[perturbed_x_min:perturbed_x_max, perturbed_y_min:perturbed_y_max] = foreORbackground_label


		center_x, center_y = perturbed_x_min + (perturbed_x_max - perturbed_x_min) // 2, perturbed_y_min + (perturbed_y_max - perturbed_y_min) // 2
		if is_shrink:
			fiducial_points_coordinate += [center_x-center_x_l, center_y-center_y_l]

			'''draw fiducial points
			stepSize = 0
			fiducial_points_synthesis_perturbed_img = self.synthesis_perturbed_img.copy()
			for l in fiducial_points_coordinate.astype(np.int64).reshape(-1, 2):
				cv2.circle(fiducial_points_synthesis_perturbed_img,
						   (l[1] + math.ceil(stepSize / 2), l[0] + math.ceil(stepSize / 2)), 5, (0, 0, 255), -1)
			cv2.imwrite('/lustre/home/gwxie/program/project/unwarp/unwarp_perturbed/TPS/img/cv_TPS_small.jpg',fiducial_points_synthesis_perturbed_img)
			'''
		self.new_shape = base_img_bound
		self.synthesis_perturbed_img = self.synthesis_perturbed_img[
									   center_x - self.new_shape[0] // 2:center_x + self.new_shape[0] // 2,
									   center_y - self.new_shape[1] // 2:center_y + self.new_shape[1] // 2,
									   :].copy()
		self.synthesis_perturbed_label = self.synthesis_perturbed_label[
										 center_x - self.new_shape[0] // 2:center_x + self.new_shape[0] // 2,
										 center_y - self.new_shape[1] // 2:center_y + self.new_shape[1] // 2,
										 :].copy()
		self.foreORbackground_label = self.foreORbackground_label[
									  center_x - self.new_shape[0] // 2:center_x + self.new_shape[0] // 2,
									  center_y - self.new_shape[1] // 2:center_y + self.new_shape[1] // 2].copy()


		perturbed_x_ = max(self.new_shape[0] - (perturbed_x_max - perturbed_x_min), 0)
		perturbed_x_min = perturbed_x_ // 2
		perturbed_x_max = self.new_shape[0] - perturbed_x_ // 2 if perturbed_x_%2 == 0 else self.new_shape[0] - (perturbed_x_ // 2 + 1)

		perturbed_y_ = max(self.new_shape[1] - (perturbed_y_max - perturbed_y_min), 0)
		perturbed_y_min = perturbed_y_ // 2
		perturbed_y_max = self.new_shape[1] - perturbed_y_ // 2 if perturbed_y_%2 == 0 else self.new_shape[1] - (perturbed_y_ // 2 + 1)



		'''save'''
		pixel_position = np.argwhere(np.zeros(self.new_shape, dtype=np.uint32) == 0).reshape(self.new_shape[0], self.new_shape[1], 2)

		if relativeShift_position == 'relativeShift_v2':
			self.synthesis_perturbed_label -= pixel_position
			fiducial_points_coordinate -= [center_x - self.new_shape[0] // 2, center_y - self.new_shape[1] // 2]
		fiducial_points_coordinate = fiducial_points_coordinate[:, :, ::-1]

		self.synthesis_perturbed_label[:, :, 0] *= self.foreORbackground_label
		self.synthesis_perturbed_label[:, :, 1] *= self.foreORbackground_label
		self.synthesis_perturbed_img[:, :, 0] *= self.foreORbackground_label
		self.synthesis_perturbed_img[:, :, 1] *= self.foreORbackground_label
		self.synthesis_perturbed_img[:, :, 2] *= self.foreORbackground_label


		'''HSV_v2'''
		perturbed_bg_img = perturbed_bg_img.astype(np.float32)
		# if self.is_perform(1, 0):
		# 	if self.is_perform(1, 0):
		if self.is_perform(0.1, 0.9):
			if self.is_perform(0.2, 0.8):
				synthesis_perturbed_img_clip_HSV = self.synthesis_perturbed_img.copy()

				synthesis_perturbed_img_clip_HSV = self.HSV_v1(synthesis_perturbed_img_clip_HSV)

				perturbed_bg_img[:, :, 0] *= 1-self.foreORbackground_label
				perturbed_bg_img[:, :, 1] *= 1-self.foreORbackground_label
				perturbed_bg_img[:, :, 2] *= 1-self.foreORbackground_label

				synthesis_perturbed_img_clip_HSV += perturbed_bg_img
				self.synthesis_perturbed_img = synthesis_perturbed_img_clip_HSV
			else:
				perturbed_bg_img_HSV = perturbed_bg_img
				perturbed_bg_img_HSV = self.HSV_v1(perturbed_bg_img_HSV)

				perturbed_bg_img_HSV[:, :, 0] *= 1-self.foreORbackground_label
				perturbed_bg_img_HSV[:, :, 1] *= 1-self.foreORbackground_label
				perturbed_bg_img_HSV[:, :, 2] *= 1-self.foreORbackground_label

				self.synthesis_perturbed_img += perturbed_bg_img_HSV
				# self.synthesis_perturbed_img[np.sum(self.synthesis_perturbed_img, 2) == 771] = perturbed_bg_img_HSV[np.sum(self.synthesis_perturbed_img, 2) == 771]

		else:
			synthesis_perturbed_img_clip_HSV = self.synthesis_perturbed_img.copy()
			perturbed_bg_img[:, :, 0] *= 1 - self.foreORbackground_label
			perturbed_bg_img[:, :, 1] *= 1 - self.foreORbackground_label
			perturbed_bg_img[:, :, 2] *= 1 - self.foreORbackground_label

			synthesis_perturbed_img_clip_HSV += perturbed_bg_img

			synthesis_perturbed_img_clip_HSV = self.HSV_v1(synthesis_perturbed_img_clip_HSV)

			self.synthesis_perturbed_img = synthesis_perturbed_img_clip_HSV

		''''''
		# cv2.imwrite(self.save_path+'clip/'+perfix_+'_'+fold_curve+str(perturbed_time)+'-'+str(repeat_time)+'.png', synthesis_perturbed_img_clip)

		self.synthesis_perturbed_img[self.synthesis_perturbed_img < 0] = 0
		self.synthesis_perturbed_img[self.synthesis_perturbed_img > 255] = 255
		self.synthesis_perturbed_img = np.around(self.synthesis_perturbed_img).astype(np.uint8)
		label = np.zeros_like(self.synthesis_perturbed_img, dtype=np.float32)
		label[:, :, :2] = self.synthesis_perturbed_label
		label[:, :, 2] = self.foreORbackground_label

		# grey = np.around(self.synthesis_perturbed_img[:, :, 0] * 0.2989 + self.synthesis_perturbed_img[:, :, 1] * 0.5870 + self.synthesis_perturbed_img[:, :, 0] * 0.1140).astype(np.int16)
		# synthesis_perturbed_grey = np.concatenate((grey.reshape(self.new_shape[0], self.new_shape[1], 1), label), axis=2)
		# d1 = self.synthesis_perturbed_img
		self.synthesis_perturbed_color = np.concatenate((self.synthesis_perturbed_img, label), axis=2)

		# '''裁剪背景图'''
		# self.synthesis_perturbed_color = np.zeros_like(synthesis_perturbed_color, dtype=np.float32)
		# # self.synthesis_perturbed_grey = np.zeros_like(synthesis_perturbed_grey, dtype=np.float32)
		# reduce_value_x = int(round(min((random.random() / 2) * (self.new_shape[0] - (perturbed_x_max - perturbed_x_min)), min(reduce_value, reduce_value_v2))))
		# reduce_value_y = int(round(min((random.random() / 2) * (self.new_shape[1] - (perturbed_y_max - perturbed_y_min)), min(reduce_value, reduce_value_v2))))
		# perturbed_x_min = max(perturbed_x_min - reduce_value_x, 0)
		# perturbed_x_max = min(perturbed_x_max + reduce_value_x, self.new_shape[0])
		# perturbed_y_min = max(perturbed_y_min - reduce_value_y, 0)
		# perturbed_y_max = min(perturbed_y_max + reduce_value_y, self.new_shape[1])

		# if scaled_image_height >= scaled_image_width:
		# 	self.synthesis_perturbed_color[:, perturbed_y_min:perturbed_y_max, :] = synthesis_perturbed_color[:, perturbed_y_min:perturbed_y_max, :]
		# 	# self.synthesis_perturbed_grey[:, perturbed_y_min:perturbed_y_max, :] = synthesis_perturbed_grey[:, perturbed_y_min:perturbed_y_max, :]
		# else:
		# 	self.synthesis_perturbed_color[perturbed_x_min:perturbed_x_max, :, :] = synthesis_perturbed_color[perturbed_x_min:perturbed_x_max, :, :]
		# 	# self.synthesis_perturbed_grey[perturbed_x_min:perturbed_x_max, :, :] = synthesis_perturbed_grey[perturbed_x_min:perturbed_x_max, :, :]

		'''blur'''
		if self.is_perform(0.1, 0.9):
			synthesis_perturbed_img_filter = self.synthesis_perturbed_color[:, :, :3].copy()
			if self.is_perform(0.1, 0.9):
				synthesis_perturbed_img_filter = cv2.GaussianBlur(synthesis_perturbed_img_filter, (5, 5), 0)
			else:
				synthesis_perturbed_img_filter = cv2.GaussianBlur(synthesis_perturbed_img_filter, (3, 3), 0)
			if self.is_perform(0.5, 0.5):
				self.synthesis_perturbed_color[:, :, :3][self.synthesis_perturbed_color[:, :, 5] == 1] = synthesis_perturbed_img_filter[self.synthesis_perturbed_color[:, :, 5] == 1]
			else:
				self.synthesis_perturbed_color[:, :, :3] = synthesis_perturbed_img_filter
		else:
			None
		
		'''
		###################################################
		test visualization here
		###################################################
		'''
		if error_flag==1:
			self.check_vis(0, self.synthesis_perturbed_color[:, :, :3], fiducial_points_coordinate)
			error_flag=0
			print("error image has been visualized")
		
		'''
		###################################################
		save data
		###################################################
		'''
		cv2.imwrite(self.save_path + 'd1/' + perfix_ + '_' + fold_curve + '.png', self.synthesis_perturbed_color[:, :, :3])
		# with open(self.save_path+'color/'+perfix_+'_'+fold_curve+'.gw', 'wb') as f:
		# 	pickle_perturbed_data = pickle.dumps(synthesis_perturbed_data)
		# 	f.write(pickle_perturbed_data)

		# with open(self.save_path+'grey/'+perfix_+'_'+fold_curve+'.gw', 'wb') as f:
		# 	pickle_perturbed_data = pickle.dumps(self.synthesis_perturbed_grey)
		# 	f.write(pickle_perturbed_data)
		# cv2.imwrite(self.save_path+'grey_im/'+perfix_+'_'+fold_curve+'.png', self.synthesis_perturbed_color[:, :, :1])

		'''保存为lmdb'''
		'''forward-end'''
		synthesis_perturbed_data = {
			'image': np.uint8(self.synthesis_perturbed_color[:, :, :3]), # np.float32
			'label': fiducial_points_coordinate, # np.float64
		}
		pickle_perturbed_data = pickle.dumps(synthesis_perturbed_data) # byte类型



		'''时间统计'''
		trian_t = time.time() - begin_train
		mm, ss = divmod(trian_t, 60)
		hh, mm = divmod(mm, 60)
		print(fold_curve+'_'+str(repeat_time)+" Time : %02d:%02d:%02d\n" % (hh, mm, ss))

		return pickle_perturbed_data
		# return np.uint8(self.synthesis_perturbed_color[:, :, :3]),fiducial_points_coordinate,np.array((segment_x, segment_y))
		# return np.uint8(self.synthesis_perturbed_color[:, :, :3]),fiducial_points_coordinate
	
	def check_vis(self, idx, im, lbl, interval=None):
		'''
		im : distorted image   # HWC 
		lbl : fiducial_points  # 61*61*2 
		'''
		im=np.uint8(im)
		im=im[:,:,::-1]
		h=im.shape[0]*0.01
		w=im.shape[1]*0.01
		im = Image.fromarray(im)
		im.convert('RGB').save("./data_vis/img_{}.png".format(idx))
		
		# fig= plt.figure(j,figsize = (6,6))
		# fig, ax = plt.subplots(figsize = (10.24,7.68),facecolor='white')
		fig, ax = plt.subplots(figsize = (w,h),facecolor='white')
		ax.imshow(im)
		ax.scatter(lbl[:,:,0].flatten(),lbl[:,:,1].flatten(),s=1.2,c='red',alpha=1)
		ax.axis('off')
		plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
		# plt.tight_layout()
		plt.savefig('./synthesis_code/test/kk_{}.png'.format(idx))
		plt.close()

def image2byte(path):
    with open(path, 'rb') as f:
        bin_image = f.read()
    return bin_image

def check_vis(idx, im, lbl):
	'''
	im : distorted image   # HWC 
	lbl : fiducial_points  # 61*61*2 
	'''
	im=np.uint8(im)
	im=im[:,:,::-1]
	h=im.shape[0]*0.01
	w=im.shape[1]*0.01
	im = Image.fromarray(im)
	im.convert('RGB').save("./data_vis/img_{}.png".format(idx))
	
	# fig= plt.figure(j,figsize = (6,6))
	# fig, ax = plt.subplots(figsize = (10.24,7.68),facecolor='white')
	fig, ax = plt.subplots(figsize = (w,h),facecolor='white')
	ax.imshow(im)
	ax.scatter(lbl[:,:,0].flatten(),lbl[:,:,1].flatten(),s=1.2,c='red',alpha=1)
	ax.axis('off')
	plt.subplots_adjust(left=0,bottom=0,right=1,top=1, hspace=0,wspace=0)
	# plt.tight_layout()
	plt.savefig('./data_vis/point_{}.png'.format(idx))
	plt.close()

def get_syn_image(path, bg_path, deform_type, idx, save_path='./output/'):
	print("begin")
	save_suffix = str.split(path, '/')[-2] # 'new'

	all_bgImg_idx = os.listdir(bg_path)
	# global begin_train
	# begin_train = time.time()
	fiducial_points = 61	# or 31
	# process_pool = Pool(16) # max=33

	save_suffix = str.split(path, '/')[-2]+str.split(path, '/')[-1][0:4] # 'new'
	for m_n in range(10):
		try:
			save = perturbed(path, all_bgImg_idx, save_path, save_suffix)
			if deform_type=='fold':
				repeat_time = min(max(round(np.random.normal(12, 4)), 1), 18) # 随机折叠次数
				pickle_dict = save.save_img('fold', repeat_time, fiducial_points, 'relativeShift_v2', idx)
			elif deform_type=='curve':
				repeat_time = min(max(round(np.random.normal(8, 4)), 1), 13) # 随机弯曲次数
				pickle_dict = save.save_img('curve', repeat_time, fiducial_points, 'relativeShift_v2', idx)
		except BaseException as err:
			print('sssssssssssss')
			print(err)
			continue
		break
	# print('end')

	# process_pool.close()
	# process_pool.join()
	return pickle_dict

def get_wild_img(head_dir, type_dir, img_name):
	head_dir = head_dir[:18]+'image'
	image = cv2.imread(pjoin(head_dir,type_dir,img_name), flags=cv2.IMREAD_COLOR)
	w_dict = { 'image': image }
	w_dict = pickle.dumps(w_dict) # byte类型
	return w_dict

def get_digital_img(head_dir, type_dir, img_name):
	image = cv2.imread(pjoin(head_dir,type_dir,img_name), flags=cv2.IMREAD_COLOR)
	d_dict = { 'image': image }
	d_dict = pickle.dumps(d_dict) # byte类型
	return d_dict

if __name__ == '__main__':
	print("the num of cpu core is {:4d}".format(mp.cpu_count()))
	# data_path='./dataset/smallda' #'./dataset/WarpDoc'
	data_path='./dataset/WarpDoc'
	dataset_name="digital"
	time_begin = time.time()
	data_dir = os.path.expanduser(pjoin(data_path, dataset_name)) # './dataset/WarpDoc/digital'
	print("Loading dataset from {}".format(data_dir))
    
	type_list = os.listdir(pjoin(data_dir))
<<<<<<< HEAD
	# lmdb_path = pjoin(data_path, "{}.lmdb".format(dataset_name)) # './unit_test/train.lmdb'
	lmdb_path = './warp4.lmdb'
	print(lmdb_path, os.getcwd())
	env = lmdb.open(lmdb_path, subdir=True,
=======
	lmdb_path = pjoin(data_path, "{}.lmdb".format(dataset_name)) # './unit_test/train.lmdb'


	env = lmdb.open('./warp1.lmdb', subdir=True,
>>>>>>> edcca2b859471916501946829c1829d9ef64e102
                   map_size=1099511627776, readonly=False,
                   meminit=False, map_async=True)
	'''写入数据'''
	txn = env.begin(write=True)
	id_sum=-1
	bg_path = './dataset/background/'
	deform_type_list=['fold','curve']
	

	for idx1, type_path in enumerate(type_list):
		image_list = os.listdir(pjoin(data_dir, type_path))
<<<<<<< HEAD
		process_pool = Pool(2) # max=33
=======
		process_pool = Pool(8) # max=33
>>>>>>> edcca2b859471916501946829c1829d9ef64e102
		for idx2, image_path in enumerate(image_list):
			id_sum+=1
			res_l = []
			deform_type1=np.random.choice(deform_type_list,p=[0.5,0.5])
			deform_type2=np.random.choice(deform_type_list,p=[0.5,0.5])
			print("deform_type1 of {0} is {1}".format(pjoin(type_path, image_path), deform_type1))
			print("deform_type2 of {0} is {1}".format(pjoin(type_path, image_path), deform_type2))
			w_dict = get_wild_img(data_dir, type_path, image_path)
			d_dict = get_digital_img(data_dir, type_path, image_path)
			# # 原始形式，.get()操作会引入阻塞
			# pickle_dict1 = process_pool.apply_async(func=get_syn_image, args=(pjoin(data_dir, type_path, image_path), bg_path, deform_type1, 'd1')).get()
			# pickle_dict2 = process_pool.apply_async(func=get_syn_image, args=(pjoin(data_dir, type_path, image_path), bg_path, deform_type2, 'd2')).get()
			pickle_dict1 = process_pool.apply_async(func=get_syn_image, args=(pjoin(data_dir, type_path, image_path), bg_path, deform_type1, 'd1'))
			pickle_dict2 = process_pool.apply_async(func=get_syn_image, args=(pjoin(data_dir, type_path, image_path), bg_path, deform_type2, 'd2'))
			# # 无多线程版写法
			# # pickle_dict1 = get_syn_image(path=pjoin(data_dir, type_path, image_path), bg_path=bg_path, deform_type=deform_type1, idx='d1')
			# # pickle_dict2 = get_syn_image(path=pjoin(data_dir, type_path, image_path), bg_path=bg_path, deform_type=deform_type2, idx='d2')
			res_l.append(pickle_dict1)
			res_l.append(pickle_dict2)
			txn.put(key = '{0}_{1}_{2}_{3}'.format(idx1,image_path[0:4],id_sum,'d1').encode(), value = res_l[0].get())
			txn.put(key = '{0}_{1}_{2}_{3}'.format(idx1,image_path[0:4],id_sum,'d2').encode(), value = res_l[1].get())
			txn.put(key = '{0}_{1}_{2}_{3}'.format(idx1,image_path[0:4],id_sum,'w1').encode(), value = w_dict)
			txn.put(key = '{0}_{1}_{2}_{3}'.format(idx1,image_path[0:4],id_sum,'di').encode(), value = d_dict)
			txn.commit()
			txn = env.begin(write=True)

		process_pool.close()
		process_pool.join()
    
	txn.commit()
	keys = [u'{}'.format(k).encode('ascii') for k in range(4*(idx1+1)*(idx2+1))]
	with env.begin(write=True) as txn:
		txn.put(b'__keys__', pickle.dumps(keys))
		txn.put(b'__len__', pickle.dumps(len(keys)))

	print("Flushing database ...")
	env.sync()
	time_end = time.time() - time_begin
	mm, ss = divmod(time_end, 60)
	hh, mm = divmod(mm, 60)
	print("Total time : %02d:%02d:%02d\n" % (hh, mm, ss))


    # 查看数据
	print(env.stat())
<<<<<<< HEAD
	# txn = env.begin(write=False)
	# print(pickle.loads(txn.get(b'__len__')))
	# for num, (key, value) in enumerate(txn.cursor()):
	# 	if num<(pickle.loads(txn.get(b'__len__'))): # if num<32:
	# 		print("{} is over".format(key.decode()))
	# 		value = pickle.loads(value)
	# 		if key.decode()[-2:]=='w1' or key.decode()[-2:]=='di':
	# 			im=np.uint8(value['image'])
	# 			im=im[:,:,::-1]
	# 			im = Image.fromarray(im)
	# 			im.convert('RGB').save("./data_vis/{0}{1}.png".format(num, key.decode()[-2:]))
	# 		else:
	# 			check_vis(num, value['image'], value['label'])
=======
	txn = env.begin(write=False)
	print(pickle.loads(txn.get(b'__len__')))
	for num, (key, value) in enumerate(txn.cursor()):
		if num<(pickle.loads(txn.get(b'__len__'))): # if num<32:
			print("{} is over".format(key.decode()))
			value = pickle.loads(value)
			if key.decode()[-2:]=='w1' or key.decode()[-2:]=='di':
				im=np.uint8(value['image'])
				im=im[:,:,::-1]
				im = Image.fromarray(im)
				im.convert('RGB').save("./data_vis/{0}{1}.png".format(num, key.decode()[-2:]))
			else:
				check_vis(num, value['image'], value['label'])
>>>>>>> edcca2b859471916501946829c1829d9ef64e102
	print("over")
	env.close()






