import sys
from pathlib import Path
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import multiprocessing


def load_scene(file_path: Path):
    scene = pyrender.Scene()

    mesh = trimesh.load(file_path)
    if Path(file_path).suffix == ".glb":
        for k in mesh.geometry.keys():
            mesh_k = mesh.geometry[k]
            scene.add(pyrender.Mesh.from_trimesh(mesh_k))
    else:
        scene.add(pyrender.Mesh.from_trimesh(mesh))
    return scene

def viewer(file_path: Path):
    scene = load_scene(file_path)
    pyrender.Viewer(scene, use_raymond_lighting=True)

def offscreen_render(file_path: Path):
    scene = load_scene(file_path)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    s = np.sqrt(2)/2
    camera_pose = np.array([
        [0.0, -s,   s,   0.3],
        [1.0,  0.0, 0.0, 0.0],
        [0.0,  s,   s,   0.35],
        [0.0,  0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi/16.0,
                               outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(400, 400)
    color, depth = r.render(scene)
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(color)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(depth, cmap=plt.cm.gray_r)
    plt.show()

def render_in_process(file_path: Path):
    p = multiprocessing.Process(target=offscreen_render, args=(file_path,))
    p.start()
    p.join()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        # file_path = "examples/models/fuze.obj"
        # file_path = "examples/models/drill.obj"
        file_path = "examples/models/WaterBottle.glb"
        # file_path = "examples/models/wood.obj"
    
    # viewer(file_path)
    # offscreen_render(file_path)
    render_in_process(file_path)
