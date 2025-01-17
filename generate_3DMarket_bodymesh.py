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
import time
from PIL import Image
from src.util import renderer as vis_util
from src.util import image as img_util
from src.tf_smpl import projection as proj_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('market_path', '../Market/pytorch/', 'Image to run')
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
    if not os.path.exists('./3DMarket_bodymesh'):
        os.mkdir('./3DMarket_bodymesh')
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    face_path = './src/tf_smpl/smpl_faces.npy'
    faces = np.load(face_path)
    count = 0 
    #for split in ['train', 'train_all', 'val', 'gallery', 'query']:
    for split in ['train', 'train_all']:
        for root, dirs, files in os.walk(dir_path+split, topdown=True):
            for img_path in files:
                count +=1
                if not img_path[-3:]=='jpg':
                    continue 
                img_path = root +'/' + img_path
                print(img_path)
                input_img, proc_param, img = preprocess_image(img_path, json_path)
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
                save_mesh(img, img_path, split, proc_param, joints[0], verts[0], cams[0], faces)


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
    obj_mesh_name = './3DMarket_bodymesh/%s/%s/%s.obj'%( split, os.path.basename(os.path.dirname(img_path)), os.path.basename(img_path) )
    store_dir = os.path.dirname(obj_mesh_name)
    if not os.path.exists(os.path.dirname(store_dir)):
        os.mkdir(os.path.dirname(store_dir))
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    foreground_index_2d = np.zeros((w,h))+99999
    foreground_value_2d = np.zeros((w,h))+99999
    background = np.zeros((w,h))
    index = 6891
    with open(obj_mesh_name, 'w') as fp:
        w, h, _ = img.shape
        imgsize = max(w,h)
        # Draw Color
        for i in range(vert_2d.shape[0]):
            v2 = vert_2d[i,:]
            v3 = verts[i,:]
            fp.write( 'v %f %f %f\n' % ( v3[0], v3[1], v3[2]) )

        for f in faces: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )
            break  # skip for saving space . small file

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.market_path, config.json_path)
