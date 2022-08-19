"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo_dancing --video_path /storage_fast/jyzhu/video/dancing/dancing1/images/00043
python -m demo_dancing --video_path data/00000

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf
import cv2
from PIL import Image
from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('video_path', '/storage_fast/jyzhu/video/dancing/dancing1/images/00043', 'Video clip to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')

def visualize(img, img_path, proc_param, joints, verts, cam):
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

    obj_mesh_name = './3DDancing_demo/{}/{}/{}.jpg'.format(img_path.split('/')[-4], img_path.split('/')[-2], os.path.basename(img_path) )
    store_dir = os.path.dirname(obj_mesh_name)
    if not os.path.exists(os.path.dirname(store_dir)):
        os.mkdir(os.path.dirname(store_dir))
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    # plt.ion()
    fig = plt.figure()
    plt.figure(1)

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
    # result = Image.fromarray(rend_img)
    # result.save('mesh.jpg')
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
    fig.savefig(obj_mesh_name)
    plt.close() # clear figure
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


def main(video_path, json_path=None):
    if not os.path.exists('./3DDancing_demo'):
        os.mkdir('./3DDancing_demo')
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    print(video_path)

    for root, dirs, files in os.walk(video_path, topdown=True):
        print('got images', len(files), 'in', root)
        files.sort()
        for img_path in files:
            # count +=1
            # if not count % 2 ==1:
            #     continue
            if not img_path[-3:]=='jpg':
                continue 
            img_path = root +'/' + img_path
            print(img_path)
            input_img, proc_param, img = preprocess_image(img_path, json_path)

            input_img = np.expand_dims(input_img, 0)
            joints, verts, cams, joints3d, theta = model.predict(
                input_img, get_theta=True)

            visualize(img, img_path, proc_param, joints[0], verts[0], cams[0])


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.video_path, config.json_path)
