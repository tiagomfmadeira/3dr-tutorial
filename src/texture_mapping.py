import random

import cv2
import numpy as np
import open3d as o3d
from numba import jit
from tqdm import tqdm

from utils import project_to_camera, get_cameras_data


@jit(nopython=True)
def map_texture(faces, project_func,
                extrinsics_list, intrinsics_list, width_list, height_list, mesh_depth_image_list,
                depth_image_list, img_select_alg, img_filter, image_scores_array):
    text_coords = np.zeros((len(faces), 2, 3))
    camera_idxs = np.full(len(faces), fill_value=-1)
    curr_prob = np.ones(len(faces))
    curr_proj_area = np.zeros(len(faces))
    curr_score = np.full(len(faces), fill_value=-1)

    for face_idx, face in enumerate(faces):
        for cam_idx, extrinsics in enumerate(extrinsics_list):

            # Ignore images in explicit filter
            if cam_idx in img_filter:
                continue

            points_hom = np.ones((4, 3), dtype=np.float64)
            points_hom[:3, :3] = np.transpose(face)

            # Project points to camera coordinate system
            points_in_camera = np.dot(np.linalg.inv(extrinsics), points_hom)

            pixels, valid_pixels = project_func(intrinsics_list[cam_idx],
                                                np.array([0, 0, 0, 0, 0]),
                                                width_list[cam_idx], height_list[cam_idx],
                                                points_in_camera)
            # Make sure all vertices project within the image
            if not np.all(valid_pixels):
                continue

            x1, x2, x3 = pixels[0, :]
            y1, y2, y3 = pixels[1, :]

            # Don't use projection on triangles facing away from camera
            vertex_mat = np.ones((3, 3), dtype=np.float64)
            vertex_mat[0, :] = (height_list[cam_idx] - 1) - pixels[0, :]
            vertex_mat[1, :] = pixels[1, :]
            if np.linalg.det(vertex_mat) <= 0:
                continue

            # If any of the triangle vertices are far from their projection in Z
            depth_image = mesh_depth_image_list[cam_idx]
            # depth_image = depth_image_list[cam_idx]

            if abs(depth_image[int(y1), int(x1)] - points_in_camera[2][0]) > 0.02:
                continue
            if abs(depth_image[int(y2), int(x2)] - points_in_camera[2][1]) > 0.02:
                continue
            if abs(depth_image[int(y3), int(x3)] - points_in_camera[2][2]) > 0.02:
                continue

            ###################################
            # Choose an image for the triangle

            # OPTION 0 - Random triangle
            if img_select_alg == 'random':
                die = random.uniform(0, 1)
                if die <= (1 / curr_prob[face_idx]):
                    ############################
                    # Normalize pixel values
                    pixels[0, :] = pixels[0, :] / width_list[cam_idx]
                    pixels[1, :] = (height_list[cam_idx] - pixels[1, :]) / height_list[cam_idx]
                    text_coords[face_idx] = pixels
                    camera_idxs[face_idx] = cam_idx
                    curr_prob[face_idx] += 1

            # OPTION 1 - Area of projected triangle
            elif img_select_alg == 'area':
                area = (1 / 2) * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                # Keep triangle with largest projected area
                if area > curr_proj_area[face_idx]:
                    ############################
                    # Normalize pixel values
                    pixels[0, :] = pixels[0, :] / width_list[cam_idx]
                    pixels[1, :] = (height_list[cam_idx] - pixels[1, :]) / height_list[cam_idx]
                    text_coords[face_idx] = pixels
                    camera_idxs[face_idx] = cam_idx
                    curr_proj_area[face_idx] = area

            # OPTION 2 - Image quality score
            elif img_select_alg == 'score':
                score = image_scores_array[cam_idx]
                # Keep new texture or texture with a better score
                if curr_score[face_idx] == -1 or score < curr_score[face_idx]:
                    ############################
                    # Normalize pixel values
                    pixels[0, :] = pixels[0, :] / width_list[cam_idx]
                    pixels[1, :] = (height_list[cam_idx] - pixels[1, :]) / height_list[cam_idx]
                    text_coords[face_idx] = pixels
                    camera_idxs[face_idx] = cam_idx
                    curr_score[face_idx] = score

    return text_coords, camera_idxs


def compute_texture(mesh, cameras_data, img_select_alg, img_filter=[]):

    for camera in cameras_data.values():
        mask = np.all(camera['imageBGR'] == [0, 0, 0], axis=-1)
        camera['mesh_depth'][mask] = np.inf

    extrinsics_array = np.array([camera['extrinsics'] for camera in cameras_data.values()])
    intrinsics_array = np.array([camera['intrinsics'] for camera in cameras_data.values()])
    width_array = np.array([camera['width'] for camera in cameras_data.values()])
    height_array = np.array([camera['height'] for camera in cameras_data.values()])
    scan_depth_image_array = np.array([camera['scan_depth_image'] for camera in cameras_data.values()])
    mesh_depth_image_array = np.array([camera['mesh_depth'] for camera in cameras_data.values()])
    img_filter = np.array(img_filter)
    if img_select_alg == 'score':
        image_scores_array = np.array([camera['score'] for camera in cameras_data.values()])
    else:
        image_scores_array = np.array([])

    # Project mesh faces to images
    UV_list, camera_idxs = map_texture(np.asarray(mesh.vertices)[np.asarray(mesh.triangles)], project_to_camera,
                                       extrinsics_array, intrinsics_array, width_array, height_array, mesh_depth_image_array,
                                       scan_depth_image_array, img_select_alg, img_filter, image_scores_array)

    # Place images in open3D material list
    textures = []
    used_cameras = []

    for camera_idx, camera in enumerate(cameras_data.keys()):
        if camera_idx in camera_idxs:
            img = cv2.cvtColor(cameras_data[camera]['imageBGR'], cv2.COLOR_BGR2RGB)
            textures.append(o3d.geometry.Image(cv2.flip(img, 0)))
            used_cameras.append(camera)
    mesh.textures = textures

    # Assign each triangle of the mesh to a material list index
    triangle_material_ids = []
    for idx in camera_idxs:
        if idx == -1:
            triangle_material_ids.append(-1)
        else:
            triangle_material_ids.append(used_cameras.index(list(cameras_data.keys())[idx]))
    mesh.triangle_material_ids = o3d.utility.IntVector(triangle_material_ids)

    print("Total number of triangles: " + str(len(triangle_material_ids)))
    print("Triangles with no texture: " + str(triangle_material_ids.count(-1)))
    print("Used textures: " + str(used_cameras))

    # Assign uvs to triangles
    triangle_uvs = []
    for triangle in UV_list:
        triangle_uvs.extend(list(np.transpose(triangle)))
    mesh.triangle_uvs = o3d.utility.Vector2dVector(np.asarray(triangle_uvs))

    return mesh


def get_image_scores(cameras_data):

    # Per scan approach
    for i in tqdm(range(0, len(cameras_data), 6)):
        batch_score = 0 
        for j in range(i, min(i + 6, len(cameras_data))):
            camera = list(cameras_data.keys())[j]
            image = cameras_data[camera]['imageBGR']

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            score = ((np.max(gray_image) - np.min(gray_image)) * 100) / 255
            
            batch_score += score

        for j in range(i, min(i + 6, len(cameras_data))):
            camera = list(cameras_data.keys())[j]
            cameras_data[camera]['score'] = batch_score
