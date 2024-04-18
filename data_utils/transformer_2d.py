import numpy as np
from PIL import Image,ImageOps
import random
from skimage import exposure
from skimage.util import random_noise

import cv2

from skimage.exposure.exposure import rescale_intensity
from skimage.draw import polygon


class RandomErase2D(object):
    '''
    Data augmentation method.
    Args:

    '''
    def __init__(self, window_size=(64,64), scale_flag=True, prob=0.5):
        self.window_size = window_size
        self.scale_flag = scale_flag
        self.prob = prob
    
    def __call__(self, sample):
        if self.scale_flag:
            h_factor = np.random.uniform(0.5, 1)
            w_factor = np.random.uniform(0.5, 1)
            max_h, max_w = np.uint8(self.window_size[0]*h_factor),np.uint8(self.window_size[1]*w_factor)
        else:
            max_h, max_w = self.window_size
        image = sample['image']
        label = sample['label']

        mm = 1 if len(image.shape) > 2 else 0

        h,w = label.shape
        roi_window = []

        if np.sum(label) !=0:
            roi_nz = np.nonzero(label)
            roi_window.append((
                np.maximum((np.amin(roi_nz[0]) - max_h//2), 0),
                np.minimum((np.amax(roi_nz[0]) + max_h//2), h)
            ))

            roi_window.append((
                np.maximum((np.amin(roi_nz[1]) - max_w//2), 0),
                np.minimum((np.amax(roi_nz[1]) + max_w//2), w)
            ))

        else:
            roi_window.append((random.randint(0,64),random.randint(-64,0)))
            roi_window.append((random.randint(0,64),random.randint(-64,0)))

        if np.random.uniform(0, 1) > self.prob:
            direction = random.choice(['t','d','l','r'])
            # print(direction)
            if direction == 't':
                if mm:
                    image[:,:roi_window[0][0],:] = 0
                else:
                    image[:roi_window[0][0],:] = 0
            elif direction == 'd':
                if mm:
                    image[:,roi_window[0][1]:,:] = 0
                else:
                    image[roi_window[0][1]:,:] = 0
            elif direction == 'l':
                if mm:
                    image[:,:,:roi_window[1][0]] = 0
                else:
                    image[:,:roi_window[1][0]] = 0
            elif direction == 'r':
                if mm:
                    image[:,:,roi_window[1][1]:] = 0
                else:
                    image[:,roi_window[1][1]:] = 0
        

        sample['image'] = image
        sample['label'] = label
        return sample


class RandomFlip2D(object):
    '''
    Data augmentation method.
    Flipping the image, including horizontal and vertical flipping.
    Args:
    - mode: string, consisting of 'h' and 'v'. Optional methods and 'hv' is default.
            'h'-> horizontal flipping,
            'v'-> vertical flipping,
            'hv'-> random flipping.

    '''
    def __init__(self, mode='hv'):
        self.mode = mode

    def __call__(self, sample):
        # image: numpy array
        # label: numpy array
        image = sample['image']
        label = sample['label']

        mm = 1 if len(image.shape) > 2 else 0

        if 'h' in self.mode and 'v' in self.mode:
            random_factor = np.random.uniform(0, 1)
            if random_factor < 0.3:
                if mm:
                    image = image[:,:,::-1]
                else:
                    image = image[:,::-1]
                label = label[:,::-1]
            elif random_factor < 0.6:
                if mm:
                    image = image[:,::-1,:]
                else:
                    image = image[::-1,:]
                label = label[::-1,:]

        elif 'h' in self.mode:
            if np.random.uniform(0, 1) > 0.5:
                if mm:
                    image = image[:,:,::-1]
                else:
                    image = image[:,::-1]
                label = label[:,::-1]

        elif 'v' in self.mode:
            if np.random.uniform(0, 1) > 0.5:
                if mm:
                    image = image[:,::-1,:]
                else:
                    image = image[::-1,:]
                label = label[::-1,:]
        
        sample['image'] = image
        sample['label'] = label
        return sample


class RandomRotate2D(object):
    """
    Data augmentation method.
    Rotating the image with random degree.
    Args:
    - degree: the rotate degree from (-degree , degree)
    Returns:
    - rotated image and label
    """

    def __init__(self, degree=[-15,-10,-5,0,5,10,15]):
        self.degree = degree

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        mm = 1 if len(image.shape) > 2 else 0

        cts = []
        if mm:
            for i in range(image.shape[0]):
                cts.append(Image.fromarray(image[i]))
        else:
            cts=[Image.fromarray(image)]
        label = Image.fromarray(np.uint8(label))

        rotate_degree = random.choice(self.degree)

        cts_out = []
        for ct in cts:
            ct = ct.rotate(rotate_degree, Image.BILINEAR)
            ct = np.array(ct).astype(np.float32)
            cts_out.append(ct)

        label = label.rotate(rotate_degree, Image.NEAREST)

        image = np.asarray(cts_out).squeeze()
        label = np.array(label).astype(np.float32)


        # image = Image.fromarray(image)
        # label = Image.fromarray(np.uint8(label))

        # rotate_degree = random.choice(self.degree)
        # image = image.rotate(rotate_degree, Image.BILINEAR)
        # label = label.rotate(rotate_degree, Image.NEAREST)

        # image = np.array(image).astype(np.float32)
        # label = np.array(label).astype(np.float32)


        sample['image'] = image
        sample['label'] = label
        return sample


class RandomZoom2D(object):
    """
    Data augmentation method.
    Zooming the image with random scale.
    Args:
    - scale: the scale factor from the scale
    Returns:
    - zoomed image and label, keep original size
    """

    def __init__(self, scale=(0.8,1.2)):
        assert isinstance(scale,tuple)
        self.scale = scale

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        mm = 1 if len(image.shape) > 2 else 0

        cts = []
        if mm:
            for i in range(image.shape[0]):
                cts.append(Image.fromarray(image[i]))
            # image = Image.fromarray(image.transpose((1,2,0)))
        else:
            cts = [Image.fromarray(image)]
            # image = Image.fromarray(image)
        label = Image.fromarray(np.uint8(label))

        scale_factor = random.uniform(self.scale[0],self.scale[1])
        # print(scale_factor)
        h, w = label.size[0], label.size[1]  # source image width and height
        tw, th = int(h * scale_factor), int(w * scale_factor)  #croped width and height

        if scale_factor < 1.:
            left_shift = []
            label_np = sample['label']
            select_index = np.concatenate([np.where(label_np != 0)], axis=1)
            if select_index.shape[1] == 0:
                left_shift.append([0, (w - tw)])
                left_shift.append([0, (h - th)])
            else:
                x_left = max(0, min(select_index[0]))
                x_right = min(w, max(select_index[0]))
                y_left = max(0, min(select_index[1]))
                y_right = min(h, max(select_index[1]))
                left_shift.append(
                    [max(0, min(x_left, x_right - tw)),
                     min(x_left, w - tw)])
                left_shift.append(
                    [max(0, min(y_left, y_right - th)),
                     min(y_left, h - th)])
            x1 = random.randint(left_shift[1][0], left_shift[1][1])
            y1 = random.randint(left_shift[0][0], left_shift[0][1])

            for i in range(len(cts)):
                cts[i] = cts[i].crop((x1, y1, x1 + tw, y1 + th))

            # image = image.crop((x1, y1, x1 + tw, y1 + th))
            label = label.crop((x1, y1, x1 + tw, y1 + th))
        else:
            pw, ph = tw - w, th - h
            pad_value = [
                int(random.uniform(0, pw / 2)),
                int(random.uniform(0, ph / 2))
            ]
            for i in range(len(cts)):
                cts[i] = ImageOps.expand(cts[i],
                                        border=(pad_value[0], pad_value[1],
                                                tw - w,
                                                th - h),
                                        fill=0)
            # image = ImageOps.expand(image,
            #                         border=(pad_value[0], pad_value[1],
            #                                 tw - w,
            #                                 th - h),
            #                         fill=0)
            label = ImageOps.expand(label,
                                   border=(pad_value[0], pad_value[1],
                                           tw - w,
                                           th - h),
                                   fill=0)
        tw, th = h, w
        for i in range(len(cts)):
            cts[i] = cts[i].resize((tw, th), Image.BILINEAR)
        # image = image.resize((tw, th), Image.BILINEAR)
        label = label.resize((tw, th), Image.NEAREST)
        if mm:
            cts_out = [np.array(ct).astype(np.float32) for ct in cts]
            image = np.asarray(cts_out).squeeze()
            # image = np.array(image).transpose((2,0,1)).astype(np.float32)
        else:
            image = np.array(image).astype(np.float32)
        label = np.array(label).astype(np.float32)

        sample['image'] = image
        sample['label'] = label
        return sample


class RandomAdjust2D(object):
    """
    Data augmentation method.
    Adjust the brightness of the image with random gamma.
    Args:
    - scale: the gamma from the scale
    Returns:
    - adjusted image
    """

    def __init__(self, scale=(0.2,1.8)):
        assert isinstance(scale,tuple)
        self.scale = scale

    def __call__(self, sample):
        image = sample['image']
        mm = 1 if len(image.shape) > 2 else 0
        gamma = random.uniform(self.scale[0],self.scale[1])
        if mm:
            for i in range(image.shape[0]):
                image[i] = exposure.adjust_gamma(image[i], gamma) 
        else:
            image = exposure.adjust_gamma(image, gamma)
        sample['image'] = image
        return sample


class RandomNoise2D(object):
    """
    Data augmentation method.
    Add random salt-and-pepper noise to the image with a probability.
    Returns:
    - adjusted image
    """
    def __call__(self, sample):
        image = sample['image']
        prob = random.uniform(0,1)
        if prob > 0.9:
            image = random_noise(image,mode='s&p') 
        sample['image'] = image
        return sample


class RandomDistort2D(object):
    """
    Data augmentation method.
    Add random salt-and-pepper noise to the image with a probability.
    Returns:
    - adjusted image
    """
    def __init__(self,random_state=None,alpha=200,sigma=20,grid_scale=4,prob=0.5):
        self.random_state = random_state
        self.alpha = alpha
        self.sigma = sigma
        self.grid_scale = grid_scale
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform(0, 1) > self.prob:
            image = sample['image']
            label = sample['label']

            mm = 1 if len(image.shape) > 2 else 0

            if self.random_state is None:
                random_state = np.random.RandomState(None)

            if mm:
                im_merge = np.concatenate([image[i][...,None] for i in range(image.shape[0])] + [label[...,None]], axis=2)
            else:
                im_merge = np.concatenate([image[...,None], label[...,None]], axis=2)
            shape = im_merge.shape
            shape_size = shape[:2]

            self.alpha //= self.grid_scale  # Does scaling these make sense? seems to provide
            self.sigma //= self.grid_scale  # more similar end result when scaling grid used.
            grid_shape = (shape_size[0]//self.grid_scale, shape_size[1]//self.grid_scale)

            blur_size = int(4 * self.sigma) | 1
            rand_x = cv2.GaussianBlur(
                (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
                ksize=(blur_size, blur_size), sigmaX=self.sigma) * self.alpha
            rand_y = cv2.GaussianBlur(
                (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
                ksize=(blur_size, blur_size), sigmaX=self.sigma) * self.alpha
            if self.grid_scale > 1:
                rand_x = cv2.resize(rand_x, shape_size[::-1])
                rand_y = cv2.resize(rand_y, shape_size[::-1])

            grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
            grid_x = (grid_x + rand_x).astype(np.float32)
            grid_y = (grid_y + rand_y).astype(np.float32)

            distorted_img = cv2.remap(im_merge, grid_x, grid_y, borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)
            '''
            alpha, sigma, alpha_affine = im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08
            # Random affine
            center_square = np.float32(shape_size) // 2
            square_size = min(shape_size) // 3
            pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
            pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
            M = cv2.getAffineTransform(pts1, pts2)
            im_merge = cv2.warpAffine(im_merge, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dz = np.zeros_like(dx)
            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
            distorted_img = map_coordinates(im_merge, indices, order=1, mode='reflect').reshape(shape)
            '''
            # print(distorted_img.shape)
            if mm: 
                image = np.asarray([distorted_img[...,i] for i in range(image.shape[0])])
            else:
                image = distorted_img[...,0]
            # print(ct.shape)
            sample['image'] = image
            sample['label']  = distorted_img[...,-1]

        return sample


class Get_ROI(object):
    def __init__(self, keep_size=24, pad_flag=False):
        self.keep_size = keep_size
        self.pad_flag = pad_flag
    
    def __call__(self, sample):
        '''
        sample['image'] must be scaled to (0~1)
        '''
        image = sample['image']
        label = sample['label']

        h,w = image.shape
        roi = self.get_body(image)
        mm = 1 if len(image.shape) > 2 else 0
        if np.sum(roi) != 0:
            roi_nz = np.nonzero(roi)
            roi_bbox = [
                np.maximum((np.amin(roi_nz[0]) - self.keep_size), 0), # left_top x
                np.maximum((np.amin(roi_nz[1]) - self.keep_size), 0), # left_top y
                np.minimum((np.amax(roi_nz[0]) + self.keep_size), h), # right_bottom x
                np.minimum((np.amax(roi_nz[1]) + self.keep_size), w)  # right_bottom y
            ]
        else:
            roi_bbox = [0,0,h,w]

        if mm:
            image = image[:,roi_bbox[0]:roi_bbox[2],roi_bbox[1]:roi_bbox[3]]
            label = label[:,roi_bbox[0]:roi_bbox[2],roi_bbox[1]:roi_bbox[3]]
        else:
            image = image[roi_bbox[0]:roi_bbox[2],roi_bbox[1]:roi_bbox[3]]
            label = label[roi_bbox[0]:roi_bbox[2],roi_bbox[1]:roi_bbox[3]]
        # pad
        if self.pad_flag:
            nh, nw = roi_bbox[2] - roi_bbox[0], roi_bbox[3] - roi_bbox[1]
            if abs(nh - nw) > 1:
                if nh > nw:
                    pad = ((0,0),(int(nh-nw)//2,int(nh-nw)//2))
                else:
                    pad = ((int(nw-nh)//2,int(nw-nh)//2),(0,0))
                if mm:
                    image = np.pad(image,((0,0)) + pad,'constant')
                else:
                    image = np.pad(image,pad,'constant')
                label = np.pad(label,pad,'constant')

        sample['image'] = image
        sample['label'] = label
        sample['bbox'] = roi_bbox
        return sample
    
    def get_body(self,image):
        body_array = np.zeros_like(image, dtype=np.uint8)
        img = rescale_intensity(image, out_range=(0, 255))
        img = img.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        body = cv2.erode(img, kernel, iterations=1)
        blur = cv2.GaussianBlur(body, (5, 5), 0)
        _, body = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        area = [[c, cv2.contourArea(contours[c])] for c in range(len(contours))]
        area.sort(key=lambda x: x[1], reverse=True)
        body = np.zeros_like(body, dtype=np.uint8)
        for j in range(min(len(area),3)):
            if area[j][1] > area[0][1] / 20:
                contour = contours[area[j][0]]
                r = contour[:, 0, 1]
                c = contour[:, 0, 0]
                rr, cc = polygon(r, c)
                body[rr, cc] = 1
        body_array = cv2.medianBlur(body, 5)

        return body_array
