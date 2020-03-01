'''
ECE276A WI20 HW1
Stop Sign Detector
'''

import os, cv2
import numpy as np



def normalise_image(x):
	return x/255

class StopSignDetector():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		self.w = np.load('./weights_29.npy')


		#raise NotImplementedError

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		# YOUR CODE HERE
		obj = StopSignDetector()
		w = obj.w

		img = img.astype(np.float32)
		img = normalise_image(img)

		dim = img.shape[0] * img.shape[1]
		x = np.reshape(img, (dim, 3))
		o = np.ones((x.shape[0], 1))
		x = np.concatenate((o, x), 1)

		array1 = np.matmul(x, w) >= 0
		array1 = array1.flatten()
		array1 = array1.astype(np.float32)

		array1 = np.reshape(array1, (img.shape[0],img.shape[1]))
		array1 = array1.astype(np.uint8)
		mask_img = array1

		#raise NotImplementedError
		return mask_img

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the stop sign
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''




		mask_img = self.segment_image(img)

		kernel = np.ones((3, 3), np.uint8)
		dilated = cv2.dilate(mask_img, kernel, iterations=2)

		contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		hierarchy = hierarchy[0]  # get the actual inner list of hierarchy descriptions
		boxes = []

		max_h = mask_img.shape[0]
		max_w = mask_img.shape[1]
		img_area = max_h * max_w

		for component in zip(contours, hierarchy):

			currentContour = component[0]
			#currentHierarchy = component[1]




				#boxes.append([a, b, a + c, b + d])

			x1, y1, w, h = cv2.boundingRect(currentContour)
			ecc = w / h
			are = w * h
			if ecc < 1.2 and ecc > 0.8:
				if are / img_area > 0.008:
					#print("Contour for this image has ", len(poly_aprox), "sides")
					boxes.append([x1, max_h - (y1 + h), x1 + w, max_h - y1])
			# sort left to right
			# boxes.sort(key=lambda x: x[0])
		#print(len(boxes), boxes)
		return boxes


if __name__ == '__main__':
	folder = "trainset"
	my_detector = StopSignDetector()

	for filename in os.listdir(folder):
		# read one test image
		img = cv2.imread(os.path.join(folder,filename))
		cv2.imshow('image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		#Display results:
		#(1) Segmented images
		mask_img = my_detector.segment_image(img)


		#(2) Stop sign bounding box
		boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope

