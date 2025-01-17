"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m generate_3DMarket --market_path ../Market/pytorch/
python -m generate_3DMarket --market_path ../Duke/pytorch/

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf
import os
import cv2
import time
from PIL import Image
from src.util import renderer as vis_util
from src.util import image as img_util
from src.tf_smpl import projection as proj_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('dancing_path', '/storage_fast/jyzhu/video/dancing/', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])
    #visualize(img, proc_param, joints[0], verts[0], cams[0])
    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        #vert_shifted, cam=None, img=img, do_alpha=True)
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    # plt.ion()
    fig = plt.figure()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    result = Image.fromarray(rend_img)
    result.save('mesh.jpg')
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    fig.savefig('demo.jpg')
    # import ipdb
    # ipdb.set_trace()


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(dir_path, json_path=None):
    if not os.path.exists('./3DDancing_hmr_mask'):
        os.mkdir('./3DDancing_hmr_mask')
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    face_path = './src/tf_smpl/smpl_faces.npy'
    faces = np.load(face_path)
    count = 0 
    for split in ['dancing1', 'dancing2']:
        print('processing split', dir_path+split)
        if not os.path.exists('./3DDancing_hmr_mask/' + split):
            os.mkdir('./3DDancing_hmr_mask/' + split)
        for root, dirs, files in os.walk(dir_path+split, topdown=True):
            print('got images', len(files), 'in', root)
            for img_path in files:
                count +=1
                if not count % 2 ==1:
                    continue
                if not img_path[-3:]=='jpg':
                    continue 
                img_path = root +'/' + img_path
                print(img_path)
                try:
                    input_img, proc_param, img = preprocess_image(img_path, json_path)
                except Exception as e:
                    with open('exceptions.log', 'a+') as f:
                        f.write(img_path + '\n' + e + '\n')
                    print(e)
                    continue

                input_img = np.expand_dims(input_img, 0)

                # Theta is the 85D vector holding [camera, pose, shape]
                # where camera is 3D [s, tx, ty]
                # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
                # shape is 10D shape coefficients of SMPL
                since = time.time()
                joints, verts, cams, joints3d, theta = model.predict(
                    input_img, get_theta=True)
                # scaling and translation
                print(time.time()-since)
                w, h, _ = img.shape
                img = cv2.resize(img, dsize=(h//8, w//8), interpolation=cv2.INTER_NEAREST) # for filling the hole...
                save_mesh(img, img_path,'', proc_param, joints[0], verts[0], cams[0], faces)
                # if count == 500:
                #     break
        print('somehow it is done.')
def save_mesh(img, img_path, split, proc_param, joints, verts, cam, faces):
    cam_for_render, vert_3d, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])
    cam_for_render, vert_shifted = cam, verts
    #print(proc_param)
    #print(vert_shifted)
    camera  = np.reshape(cam_for_render, [1,3])
    w, h, _ = img.shape
    imgsize = max(w,h)
    # project to 2D
    vert_2d = verts[:, :2] + camera[:, 1:]
    vert_2d = vert_2d * camera[0,0]
    img_copy = img.copy()
    mask_name = './3DDancing_hmr_mask/{}/masks/{}/{}.png'.format(img_path.split('/')[-4], img_path.split('/')[-2], os.path.basename(img_path) )
    store_dir = os.path.dirname(mask_name)
    if not os.path.exists(os.path.dirname(store_dir)):
        os.mkdir(os.path.dirname(store_dir))
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    foreground_index_2d = np.zeros((w,h))+99999
    foreground_value_2d = np.zeros((w,h))+99999
    background = np.zeros((w,h))
    mask = np.zeros((w,h))
    index = 6891
    w, h, _ = img.shape
    imgsize = max(w,h)
    # Decide Forground
    for i in range(vert_2d.shape[0]):
        v2 = vert_2d[i,:]
        v3 = vert_3d[i,:]
        z = v3[2]
        x = int(round( (v2[1]+1)*0.5*imgsize ))
        y = int(round( (v2[0]+1)*0.5*imgsize ))
        if w<h:
            x = int(round(x -h/2 + w/2))
        else:
            y = int(round(y - w/2 + h/2))
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        mask[x, y] = 1
        if z < foreground_value_2d[x,y]:
            foreground_index_2d[x,y] = i
            foreground_value_2d[x,y] = z

    #mask = cv2.resize(mask, dsize=(h//8, w//8), interpolation=cv2.INTER_NEAREST)
    # check the hole
    for t in range(3):
        for i in range(1,w-1):
            for j in range(1,h-1):
                if mask[i,j] == 1:
                    continue
                sum = mask[i-1,j-1] + mask[i,j-1] + mask[i-1,j] + mask[i-1,j+1] \
                    +mask[i+1,j+1] + mask[i,j+1] + mask[i+1,j] + mask[i+1,j-1]
                if sum >= 5:
                    mask[i, j] = 1
    #img_masked = img*mask.reshape(w,h,1)
    #img_masked = Image.fromarray(np.uint8(img_masked))
    #img_masked.save(mask_name+'.jpg')
    #mask = cv2.resize(mask, dsize=(h, w), interpolation=cv2.INTER_NEAREST)
    mask = Image.fromarray(np.uint8(mask*255))
    mask.save(mask_name)



if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.dancing_path, config.json_path)
