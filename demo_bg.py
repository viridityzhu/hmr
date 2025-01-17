"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

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
from skimage.transform import resize
import tensorflow as tf
import os

from PIL import Image
from src.util import renderer as vis_util
from src.util import image as img_util
from src.tf_smpl import projection as proj_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', '../3D-Person-reID/2DMarket/query/0001/0001_c1s1_001051_00.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')

percent = 1.0

def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

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


def preprocess_image(img_path, json_path=None, fliplr=False):
    #img = io.imread(img_path)
    #img = Image.fromarray(img)
    img = Image.open(img_path) 
    img = img.resize((64,128))
    img = np.array(img)
    #img = resize(img, (128 , 64))
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if fliplr:
        img = np.fliplr(img)

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


def main(img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    for fliplr in [False]:
        input_img, proc_param, img = preprocess_image(img_path, json_path, fliplr=fliplr)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        # Theta is the 85D vector holding [camera, pose, shape]
        # where camera is 3D [s, tx, ty]
       # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
        # shape is 10D shape coefficients of SMPL
        joints, verts, cams, joints3d, theta = model.predict(
            input_img, get_theta=True)


    # scaling and translation
    save_mesh(img, img_path, proc_param, joints[0], verts[0], cams[0])
    visualize(img, proc_param, joints[0], verts[0], cams[0])

def save_mesh(img, img_path, proc_param, joints, verts, cam):
    cam_for_render, vert_3d, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])
    cam_for_render, vert_shifted = cam, verts
    #print(proc_param)
    #print(vert_shifted)
    camera  = np.reshape(cam_for_render, [1,3])
    w, h, _ = img.shape
    print(w,h)
    imgsize = max(w,h)
    # project to 2D
    # project to 2D
    vert_2d = verts[:, :2] + camera[:, 1:]
    vert_2d = vert_2d * camera[0,0]
    img_copy = img.copy()
    face_path = './src/tf_smpl/smpl_faces.npy'
    faces = np.load(face_path)
    obj_mesh_name = '%s.obj'%os.path.basename(img_path)
    foreground_index_2d = np.zeros((w,h))+99999
    foreground_value_2d = np.zeros((w,h))+99999
    background = np.zeros((w,h))
    mask = np.zeros((w,h))
    index = 6891
    with open(obj_mesh_name, 'w') as fp:
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
            # for every pixel, we only save the closet vertex (smallest depth)
            if z < foreground_value_2d[x,y]:
                foreground_index_2d[x,y] = i
                foreground_value_2d[x,y] = z

        # check the hole 
        for t in range(3):
            for i in range(1,w-1):
                for j in range(1,h-1):
                    if mask[i,j] == 1:
                        continue
                    sum = mask[i-1,j-1] + mask[i,j-1] + mask[i-1,j] + mask[i-1,j+1] \
                         +mask[i+1,j+1] + mask[i,j+1] + mask[i+1,j] + mask[i+1,j-1]
                    if sum >= 6: 
                        mask[i, j] = 1
        mask = Image.fromarray(np.uint8(mask*255))
        mask.save('%s_mask.png'%os.path.basename(img_path))
        # foreground color mapping to the human back
        z_max = max(vert_3d[:, 2])- min(vert_3d[:, 2])
        for t in range(10):
            for i in range(1,w-1):
                for j in range(1,h-1):
                    center= foreground_value_2d[i,j]
                    if foreground_index_2d[i-1,j] != 999999 and foreground_value_2d[i-1,j]>center+0.05:
                         foreground_index_2d[i-1,j] = 999999
                         foreground_value_2d[i-1,j] = 999999
                         #print('find abnormal mapping. remove')
                    if foreground_index_2d[i,j-1] != 999999 and foreground_value_2d[i,j-1]>center+0.05:
                         foreground_index_2d[i,j-1] = 999999
                         foreground_value_2d[i,j-1] = 999999
                         #print('find abnormal mapping. remove')
        # Draw Color
        count = 0
        in_selected = np.linspace(0, 128*64-1, num= round(128*64*percent), dtype=int)
        for i in range(vert_2d.shape[0]):
            v2 = vert_2d[i,:]
            v3 = verts[i,:]
            z = v3[2]
            x = int(round( (v2[1]+1)*0.5*imgsize ))
            y = int(round( (v2[0]+1)*0.5*imgsize ))
            if w<h:
                x = int(round(x -h/2 + w/2))
            else:
                y = int(round(y - w/2 + h/2))
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            if i == foreground_index_2d[x,y]: # if is the matched closet vertex draw color. 
                c = img[x, y, :]/255.0
                img_copy[x,y,:] = 0
                count +=1
                if not count in in_selected:
                    c = [1,1,1]
            else:
                c = [1,1,1] 
            fp.write( 'v %f %f %f %f %f %f\n' % ( v3[0], v3[1], v3[2], c[0], c[1], c[2]) )
        print(count)
        # 2D to 3D mapping
        for i in range(w):
            for j in range(h):
                vx, vy = i, j
                #if foreground_index_2d[i,j] < 99999:
                #    continue
                if w<h:
                    vx = vx + h/2 - w/2
                else:
                    vy = vy + w/2 - h/2
                vx = vx/imgsize *2 - 1 
                vy = vy/imgsize *2 - 1 
                
                vy /= camera[0,0] 
                vy -= camera[:, 1]
                vx /= camera[0,0] 
                vx -= camera[:, 2]
                vz = np.mean(verts[:,2])
                c = img[i,j,:]/255.0
                if not count in in_selected: 
                    c = [1,1,1]
                fp.write( 'v %f %f %f %f %f %f\n' % ( vy, vx, vz, c[0], c[1], c[2]) )
                count +=1
                background[i,j] = index
                index +=1

        for f in faces: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )
        for i in range(1,w):
            for j in range(1,h): 
                fp.write( 'f %d %d %d %d\n' % (background[i,j], background[i-1,j] ,background[i,j-1] , background[i-1, j-1]))

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, config.json_path)
