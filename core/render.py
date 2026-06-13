import trimesh
import pyrender
import numpy as np
import cv2
import scipy.spatial.transform as st

def auto_render_mesh(vertices, faces, width=800, height=800):
    # 1. 加载并修复 Mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()

    # 居中处理
    mesh.vertices -= mesh.centroid
    
    # 2. 自动获取半径 (修正此处报错)
    try:
        # 尝试从 primitive 属性获取半径
        radius = mesh.bounding_sphere.primitive.radius
    except AttributeError:
        # 如果获取失败，使用包围盒对角线长度的一半作为近似半径
        radius = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) / 2.0
    
    # 3. 创建场景
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
    
    # 使用双面渲染，增强材质感
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2, 
        roughnessFactor=0.6, 
        baseColorFactor=(0.7, 0.7, 0.9, 1.0),
        doubleSided=True
    )
    
    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene.add(render_mesh)

    # 4. 相机与光照
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)

    # 创建一个绕 Y 轴旋转 45 度的旋转矩阵
    rotation = st.Rotation.from_euler('y', -45, degrees=True).as_matrix()
    camera_pose[:3, :3] = rotation

    camera_distance = radius * 2.5


    # 更新相机位置：旋转后的 Z 轴方向
    camera_pose[:3, 3] = rotation @ np.array([0, 0, camera_distance])

    scene.add(camera, pose=camera_pose)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)

    # 5. 渲染并释放资源
    renderer = None
    try:
        renderer = pyrender.OffscreenRenderer(width, height)
        color, _ = renderer.render(scene)
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    finally:
        if renderer:
            renderer.delete() # 确保释放显存
