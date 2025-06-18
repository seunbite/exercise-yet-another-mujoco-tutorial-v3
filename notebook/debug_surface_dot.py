import sys, mujoco, time, os, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../package/helper/')
sys.path.append('../package/mujoco_usage/')
sys.path.append('../package/gpt_usage/')
sys.path.append('../package/detection_module/')
from mujoco_parser import *
from utility import *
from transformation import *
from gpt_helper import *
from owlv2 import *

np.set_printoptions(precision=2, suppress=True, linewidth=100)
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)
print("Ready.")

# Joint configuration
joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
print("Ready.")

# Functions to work with mesh and surface points
def load_mesh_vertices(mesh_file_path):
    """
    Load vertices from an OBJ file
    """
    vertices = []
    
    with open(mesh_file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Vertex line
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
    
    return np.array(vertices)

def get_random_surface_point(env, body_name, mesh_file_path):
    """
    Get a random surface point from the mesh file
    """
    # Load mesh vertices
    mesh_vertices = load_mesh_vertices(mesh_file_path)
    print(f"Loaded {len(mesh_vertices)} vertices from mesh")
    print(f"Mesh vertices range: X[{mesh_vertices[:,0].min():.3f}, {mesh_vertices[:,0].max():.3f}], "
          f"Y[{mesh_vertices[:,1].min():.3f}, {mesh_vertices[:,1].max():.3f}], "
          f"Z[{mesh_vertices[:,2].min():.3f}, {mesh_vertices[:,2].max():.3f}]")
    
    # Get object pose
    obj_position, obj_rotation = env.get_pR_body(body_name=body_name)
    print(f"Object position: {obj_position}")
    print(f"Object rotation:\n{obj_rotation}")
    
    # Transform mesh vertices to world coordinates
    world_vertices = []
    for vertex in mesh_vertices:
        # Apply rotation and translation
        world_vertex = obj_rotation @ vertex + obj_position
        world_vertices.append(world_vertex)
    
    world_vertices = np.array(world_vertices)
    print(f"World vertices range: X[{world_vertices[:,0].min():.3f}, {world_vertices[:,0].max():.3f}], "
          f"Y[{world_vertices[:,1].min():.3f}, {world_vertices[:,1].max():.3f}], "
          f"Z[{world_vertices[:,2].min():.3f}, {world_vertices[:,2].max():.3f}]")
    
    # Randomly select a vertex (surface point)
    random_idx = np.random.randint(0, len(world_vertices))
    surface_point = world_vertices[random_idx]
    print(f"Selected vertex {random_idx}: {surface_point}")
    
    return surface_point, world_vertices

def place_red_dot_on_surface(env, body_name, mesh_file_path, dot_radius=0.02):
    """
    Place a red dot on a random surface point of the object
    """
    # Get random surface point
    surface_point, all_vertices = get_random_surface_point(env, body_name, mesh_file_path)
    
    # Place red dot
    print(f"Placing red dot at {surface_point} with radius {dot_radius}")
    env.plot_sphere(
        p=surface_point,
        r=dot_radius,
        rgba=[1, 0, 0, 1],  # Red color
        label='Surface Dot'
    )
    
    print(f"Red dot placed at surface point: {surface_point}")
    return surface_point

def main():
    # Initialize environment
    xml_path = '../asset/makeup_frida/scene_table.xml'
    env = MuJoCoParserClass(name='Tabletop', rel_xml_path=xml_path, verbose=True)
    print("Done.")
    
    # Solve IK to get the initial position 
    env.reset()
    env.set_p_body(body_name='ur_base', p=np.array([0, 0, 0.5]))  # move UR
    q_init, ik_err_stack, ik_info = solve_ik(
        env=env,
        joint_names_for_ik=joint_names,
        body_name_trgt='ur_camera_center',
        q_init=np.deg2rad([0, 0, 0, 0, 0, 0]),  # ik from zero pose
        p_trgt=np.array([0.41, 0.0, 1.2]),
        R_trgt=rpy2r(np.deg2rad([-135.22, -0., -90])),
        max_ik_tick=5000,
        ik_err_th=1e-4,
        ik_stepsize=0.1,
        ik_eps=1e-2,
        ik_th=np.radians(1.0),
        verbose=False,
        reset_env=False,
        render=False,
        render_every=1,
    )
    print("Done.")

    # Initialize viewer
    env.reset()
    env.init_viewer(
        transparent=False,
        azimuth=105,
        distance=3.12,
        elevation=-29,
        lookat=[0.39, 0.25, 0.43],
    )

    env.set_p_body(body_name='ur_base', p=np.array([0, 0, 0.5]))  # move UR
    env.set_p_body(body_name='object_table', p=np.array([1.0, 0, 0]))  # move table
    
    # obj_names = env.get_body_names(prefix='obj_') # object names
    obj_names = ['obj_head']
    n_obj = len(obj_names)
    obj_xyzs = sample_xyzs(
        n_obj,
        x_range=[0.75, 1.25],
        y_range=[-0.4, +0.4],
        z_range=[0.51, 0.51],
        min_dist=0.1,
        xy_margin=0.0
    )
    obj_xyzs = np.array([[1, 0, 0.51]])
    R = rpy2r(np.radians([0, 0, 270]))
    print("Object list:")
    for obj_idx in range(n_obj):
        print(" [%d] obj_name:[%s]" % (obj_idx, obj_names[obj_idx]))
        env.set_p_base_body(body_name=obj_names[obj_idx], p=obj_xyzs[obj_idx, :])
        env.set_R_base_body(body_name=obj_names[obj_idx], R=R)
    
    # Move
    qpos = np.radians([0, -90, 60, 75, 90, 0])
    idxs_step = env.get_idxs_step(joint_names=joint_names)
    env.set_qpos_joints(joint_names=joint_names, qpos=q_init)

    # Place red dot on obj_head surface
    mesh_file_path = '../asset/makeup_frida/mesh/head.obj'
    red_dot_position = place_red_dot_on_surface(env, 'obj_head', mesh_file_path)

    # Main loop
    env_state = env.get_state()
    while env.is_viewer_alive():
        # Step
        env.step()
        
        # Render
        if env.loop_every(tick_every=10):
            env.plot_time()
            env.render()

    env.close_viewer()
    print("Done.")

if __name__ == "__main__":
    main() 