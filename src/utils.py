import os

import cv2
import numpy as np
import open3d as o3d
import pye57
from numba import jit
from scipy.spatial.transform import Rotation
from tqdm import tqdm


@jit(nopython=True)
def mse(target, source, mask=None):
    mse_map = (target.astype(np.float64) - source.astype(np.float64)) ** 2

    if mask is None:
        mask = np.ones_like(target)

    return (mask * mse_map).sum() / mask.sum(), mse_map * mask


@jit(nopython=True)
def rmse(target, source, mask=None):
    result, mse_map = mse(target, source, mask)
    return np.sqrt(result), mse_map


@jit(nopython=True)
def project_to_camera(intrinsic_matrix, distortion, width, height, pts):
    _, n_pts = pts.shape

    pixs = np.zeros((2, n_pts), dtype=np.float64)

    k1, k2, p1, p2, k3 = distortion
    # fx, _, cx, _, fy, cy, _, _, _ = intrinsic_matrix
    # print('intrinsic=\n' + str(intrinsic_matrix))
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    x = pts[0, :]
    y = pts[1, :]
    z = pts[2, :]

    xl = np.divide(x, z)
    yl = np.divide(y, z)
    r2 = xl ** 2 + yl ** 2
    xll = xl * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) + 2 * p1 * xl * yl + p2 * (r2 + 2 * xl ** 2)
    yll = yl * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) + p1 * (r2 + 2 * yl ** 2) + 2 * p2 * xl * yl
    pixs[0, :] = fx * xll + cx
    pixs[1, :] = fy * yll + cy

    # Compute mask of valid projections
    valid_z = z > 0
    valid_xpix = np.logical_and(pixs[0, :] >= 0, pixs[0, :] < width)
    valid_ypix = np.logical_and(pixs[1, :] >= 0, pixs[1, :] < height)
    valid_pixs = np.logical_and(valid_z, np.logical_and(valid_xpix, valid_ypix))
    return pixs, valid_pixs


@jit(nopython=True)
def filter_points_in_view(range_sparse, dists, width, height, indexes):
    depth_image = np.full((height, width), fill_value=-1, dtype=np.float64)
    result = np.full((height, width), fill_value=-1, dtype=np.int32)

    for idx, dist in enumerate(dists):
        x, y = range_sparse[idx]

        # If a point closer to the camera has been projected to this pixel
        if depth_image[y, x] != -1 and depth_image[y, x] < dist:
            continue

        depth_image[y, x] = dist

        result[y, x] = indexes[idx]

    return depth_image, result


def get_viewpoint_depth(pts_hom, cam_matrix, K, width, height):
    """Get depth image with points visible from a camera viewpoint
    and their indexes in the input array

    Parameters:
    pts_hom: vector of 3D points to project to camera view;
    cam_matrix: Extrinsic matrix of camera;
    K: Intrinsic matrix of camera;
    width:;
    height:;

    Returns:
    depth_image: matrix with depth values in x,y;
    point_indexes: matrix with index of corresponding point in x,y;

   """
    # Project points to camera coordinate system
    pts_in_camera = np.dot(np.linalg.inv(cam_matrix), pts_hom)

    pixels, valid_pixels = project_to_camera(K,
                                             np.array([0, 0, 0, 0, 0]),
                                             width,
                                             height,
                                             pts_in_camera)

    range_sparse = np.transpose(np.vstack((pixels[0, valid_pixels],
                                           pixels[1, valid_pixels])).astype(int))
    dists = pts_in_camera[2, valid_pixels]
    indexes = np.where(valid_pixels)[0]

    depth_image, point_indexes = filter_points_in_view(range_sparse, dists, width, height, indexes)

    return depth_image, point_indexes


def get_cameras_data_from_conf(conf_file_path, gt_path=None):
    working_dir = os.path.dirname(conf_file_path)
    cameras_data = {}
    header_info = ''

    data = open(conf_file_path).readlines()

    print("\nLoading data from config file...")
    for line in tqdm(data):

        # Split by spaces
        content = line.split()

        if not content:
            continue

        # Header information
        if content[0] == 'dataset':
            title = content[1] + " dataset"
            header_info += "\n#####" + "#" * len(title) + "#####"
            header_info += "\n#    " + title + "    #"
            header_info += "\n#####" + "#" * len(title) + "#####\n"

        if content[0] == 'n_images':
            header_info += "\nContains " + content[1] + " captures"

        if content[0] == 'depth_directory':
            depth_dir = content[1]
            header_info += "\nDepth directory is " + depth_dir

        if content[0] == 'color_directory':
            rgb_dir = content[1]
            header_info += "\nRGB directory is " + rgb_dir

        # Scan data
        if content[0] == 'intrinsics_matrix':
            intrinsics_list = content[1:]
            # intrinsics matrix
            K = np.zeros((3, 3))
            K[0, :] = intrinsics_list[0:3]
            K[1, :] = intrinsics_list[3:6]
            K[2, :] = intrinsics_list[6:9]

        if content[0] == 'scan':
            depth_file_name = content[1]
            rgb_file_name = content[2]
            extrinsics_list = content[3:]

            if gt_path:
                # If there is no ground truth, don't create instance in dict
                gt_file_path = gt_path + "/" + rgb_file_name.split('.')[0] + ".png"
                if not os.path.isfile(gt_file_path):
                    continue

            scan_depth_image = o3d.io.read_image(working_dir + '/' + depth_dir + '/' + depth_file_name)
            # 0.25mm per unit, divide by 4000 to get meters
            scan_depth_image = np.asarray(scan_depth_image, dtype=np.float32) / 4000

            image = o3d.io.read_image(working_dir + '/' + rgb_dir + '/' + rgb_file_name)
            image = np.asarray(image)

            height, width, _ = image.shape

            # extrinsics matrix
            cam_matrix = np.zeros((4, 4))
            cam_matrix[0, :] = extrinsics_list[0:4]
            cam_matrix[1, :] = extrinsics_list[4:8]
            cam_matrix[2, :] = extrinsics_list[8:12]
            cam_matrix[3, :] = extrinsics_list[12:16]

            opengl_to_opencv = np.zeros((4, 4))
            opengl_to_opencv[0, 0] = 1
            opengl_to_opencv[1, 1] = -1
            opengl_to_opencv[2, 2] = -1
            opengl_to_opencv[3, 3] = 1
            cam_matrix = np.dot(cam_matrix, opengl_to_opencv)

            cameras_data[rgb_file_name.split('.')[0]] = {'extrinsics': cam_matrix,
                                                         'intrinsics': K,
                                                         'height': height,
                                                         'width': width,
                                                         'imageBGR': cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                                         'scan_depth_image': scan_depth_image
                                                         }
    print(header_info)
    return cameras_data


def get_cameras_data_from_e57(e57):
    cameras_data = {}

    imf = e57.image_file
    root = imf.root()

    print("\nExtracting camera data...")
    for image_idx, image2D in enumerate(tqdm(root['images2D'])):
        # Get extrinsic matrix
        tx = image2D['pose']['translation']['x'].value()
        ty = image2D['pose']['translation']['y'].value()
        tz = image2D['pose']['translation']['z'].value()

        t = np.array([tx, ty, tz])

        rx = image2D['pose']['rotation']['x'].value()
        ry = image2D['pose']['rotation']['y'].value()
        rz = image2D['pose']['rotation']['z'].value()
        rw = image2D['pose']['rotation']['w'].value()

        r = Rotation.from_quat(np.array([rx, ry, rz, rw]))

        cam_matrix = np.zeros((4, 4))
        cam_matrix[3, 3] = 1
        cam_matrix[:-1, -1] = t
        cam_matrix[:3, :3] = r.as_matrix()

        opengl_to_opencv = np.zeros((4, 4))
        opengl_to_opencv[0, 0] = 1
        opengl_to_opencv[1, 1] = -1
        opengl_to_opencv[2, 2] = -1
        opengl_to_opencv[3, 3] = 1
        cam_matrix = np.dot(cam_matrix, opengl_to_opencv)

        # Get intrinsic matrix
        pinhole = image2D['pinholeRepresentation']

        focal_length = pinhole['focalLength'].value()
        pixel_height = pinhole['pixelHeight'].value()
        pixel_width = pinhole['pixelWidth'].value()
        principal_point_x = pinhole['principalPointX'].value()
        principal_point_y = pinhole['principalPointY'].value()

        K = np.zeros((3, 3))
        K[2, 2] = 1
        K[0, 0] = focal_length / pixel_width
        K[1, 1] = focal_length / pixel_height
        K[0, 2] = principal_point_x
        K[1, 2] = principal_point_y

        # Get picture from blob
        jpeg_image = pinhole['jpegImage']
        jpeg_image_data = np.zeros(shape=jpeg_image.byteCount(), dtype=np.uint8)
        jpeg_image.read(jpeg_image_data, 0, jpeg_image.byteCount())
        image = cv2.imdecode(jpeg_image_data, cv2.IMREAD_COLOR)

        height, width, channels = image.shape

        ##############################################
        # Generate depth image from laser scan (assume 6 images per scan for now)
        pc_data = e57.read_scan(image_idx // 6)
        pts = [pc_data['cartesianX'], pc_data['cartesianY'], pc_data['cartesianZ']]
        scan_pcd = o3d.geometry.PointCloud()
        scan_pcd.points = o3d.utility.Vector3dVector(np.vstack(pts).transpose())

        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mtl.shader = "defaultUnlit"
        renderer.scene.add_geometry("textured_mesh", scan_pcd, mtl)
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        renderer.setup_camera(o3d_intrinsics, np.linalg.inv(cam_matrix))
        scan_depth_image = np.array(renderer.render_to_depth_image(z_in_view_space=True))

        cameras_data[image_idx] = {'extrinsics': cam_matrix,
                                   'intrinsics': K,
                                   'height': height,
                                   'width': width,
                                   'imageBGR': image,
                                   'scan_depth_image': scan_depth_image
                                   }
    return cameras_data


def get_cameras_data(input_path, input_type):
    # E57 FILE
    if input_type == 'e57':
        e57 = pye57.E57(input_path)
        return get_cameras_data_from_e57(e57)

    # CONF FILE
    elif input_type == 'conf':
        return get_cameras_data_from_conf(input_path)


@jit(nopython=True)
def vertex_filter_by_viewpoint(vertices, extrinsics_list, intrinsics_list,
                               width_list, height_list, depth_image_array, project_func):
    vertex_validity = np.zeros(len(vertices.T))

    for cam_idx, extrinsics in enumerate(extrinsics_list):

        # World to camera coordinate system
        points_in_camera = np.dot(np.linalg.inv(extrinsics), vertices)

        # Project to camera
        pixels, valid_pixels = project_func(intrinsics_list[cam_idx],
                                            np.array([0, 0, 0, 0, 0]),
                                            width_list[cam_idx], height_list[cam_idx],
                                            points_in_camera)

        for vertex_idx, vertex in enumerate(points_in_camera.T):

            # It was already validated
            if vertex_validity[vertex_idx] == 1:
                continue

            # It is not within this camera viewpoint
            if not valid_pixels[vertex_idx]:
                continue

            # Z buffering is not consistent
            x, y = pixels.T[vertex_idx]
            if abs(depth_image_array[cam_idx][int(y), int(x)] - vertex[2]) > 0.02:
                continue

            # Vertex is valid
            vertex_validity[vertex_idx] = 1

    return vertex_validity
