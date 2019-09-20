from pyrep import PyRep
from pyrep.const import RenderMode
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from pyrep.objects.vision_sensor import VisionSensor
from random import random, randrange, uniform, choice
import numpy as np
from os.path import dirname, join, abspath, expanduser
import os
from pyrep.const import TextureMappingMode
from PIL import Image
from pyrep.const import PrimitiveShape
from youBot_all import youBotAll

def _get_texture(pr):
    texture_file = choice(os.listdir('/home/georges/robotics_drl/data/textures/obj_textures/'))
    texture_object, texture_id = pr.create_texture(filename='/home/georges/robotics_drl/data/textures/obj_textures/%s'%(texture_file), resolution=[512,512])
    texture_object.set_renderable(0)
    return texture_id, texture_object

def add_object(boundary):
    while True:
        obj_dir = choice([name for name in os.listdir('/home/georges/Downloads/models') if name[-3:] == 'obj'])
        obj_mesh_file = join('/home/georges/Downloads/models', obj_dir)
        a = Shape.import_mesh(filename=obj_mesh_file,scaling_factor=0.01)
        box_size = a.get_bounding_box()
        height = box_size[1]
        width = box_size[-1]
        length = box_size[3]
        if height > 1 or width > 1 or length > 1 or height < 0.05 or width < 0.05 or length < 0.05:
            a.remove()
        else:
            try:
                # Create convex collidable object
                b = a.get_convex_decomposition(use_vhacd=True, vhacd_res=100)
            except:
                a.remove()
                continue
            break

    # b = Shape.create(PrimitiveShape.CUBOID,[box_size[-1],box_size[3],box_size[1]],orientation=[0,0,0], static=True)

    a.set_parent(b,keep_in_place=True)
    a.set_position([0,0,0],relative_to=b)

    b.set_collidable(1)
    b.set_measurable(0)
    b.set_detectable(0)
    b.set_renderable(0)

    b.set_respondable(1)
    b.set_dynamic(1)

    b.reset_dynamic_object()

    iter = 0
    while a.check_collision():
        iter += 1
        pos_2d = [uniform(-boundary, boundary) for _ in range(2)]
        b.set_position(pos_2d + [3])
        if iter > 12:
            break

    a_all = a.ungroup()

    for idx, j in enumerate(a_all):
        a_all[idx] = Shape(j)
        # a_all[idx].set_collidable(1)
        # a_all[idx].set_measurable(1)
        # a_all[idx].set_detectable(1)
        # a_all[idx].set_renderable(1)
        # a_all[idx].set_dynamic(0)
    return a_all, b

# pr = PyRep()
# SCENE_FILE = join(dirname(abspath(__file__)), 'test_rcan.ttt')
# pr.launch(SCENE_FILE,headless=True)
# pr.start()
# pr.set_simulation_timestep(0.5)
env = youBotAll(scene_name='scene1_5x5.ttt', boundary=2)
pr = env.pr
camera_arm = env.camera_arm
camera_arm = VisionSensor('Vision_sensor1')
camera_arm.set_render_mode(RenderMode.OPENGL3_WINDOWED)
camera_arm.set_resolution([512,512])

# obj1 = Shape('Shape0')
# obj2 = Shape('Shape')
# pos = obj2.get_position()
# obj2.remove()
obj2, obj2_coll = add_object(1)
obj2_coll.set_position([1,0,0])

floor = Shape('floor')
walls = [Shape('wall%s'%i) for i in range(4)]

texture_id, texture_object = _get_texture(pr)
floor.set_texture(texture = texture_id, mapping_mode = TextureMappingMode.PLANE, interpolate=True, decal_mode=False, repeat_along_u=True,
                repeat_along_v=True, uv_scaling=[1,1])
floor.set_color([1,1,1])

texture_id, texture_object = _get_texture(pr)
for j in walls:
    j.set_texture(texture = texture_id, mapping_mode = TextureMappingMode.PLANE, interpolate=True, decal_mode=False, repeat_along_u=True,
                    repeat_along_v=True, uv_scaling=[1,1])
    j.set_color([1,1,1])

texture_id, texture_object = _get_texture(pr)
color = [random() for _ in range(3)]
for i in obj2:
    i.set_texture(texture = texture_id, mapping_mode = TextureMappingMode.PLANE, interpolate=True, decal_mode=False)
    i.set_color([1,1,1])

for _ in range(1): pr.step()

for _ in range(100): pr.step()

for _ in range(10):
    for k in range(2):
        if k == 0:
            texture_id, texture_object = _get_texture(pr)
            floor.set_texture(texture = texture_id, mapping_mode = TextureMappingMode.PLANE, interpolate=True, decal_mode=False, repeat_along_u=True,
                            repeat_along_v=True, uv_scaling=[1,1])
            floor.set_color([1,1,1])

            texture_id, texture_object = _get_texture(pr)
            for j in walls:
                j.set_texture(texture = texture_id, mapping_mode = TextureMappingMode.PLANE, interpolate=True, decal_mode=False, repeat_along_u=True,
                                repeat_along_v=True, uv_scaling=[1,1])
                j.set_color([1,1,1])

            texture_id, texture_object = _get_texture(pr)
            color = [random() for _ in range(3)]
            for i in obj2:
                i.set_texture(texture = texture_id, mapping_mode = TextureMappingMode.PLANE, interpolate=True, decal_mode=False)
                i.set_color([1,1,1])

            for _ in range(30): pr.step()
        else:
            [j.remove_texture() for j in walls]
            for i in obj2:
                i.remove_texture()
                i.set_color([176/255, 58/255, 46/255])
            floor.remove_texture()
            for _ in range(30): pr.step()

        rgb_img_rand = camera_arm.capture_rgb()
        img = Image.fromarray(np.uint8(rgb_img_rand*255),'RGB')
        img.save('test_rcan%s.png'%(k))
