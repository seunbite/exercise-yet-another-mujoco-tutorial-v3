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
def load_mesh_vertices(mesh_file_path, mesh_scale=0.1):
    """
    Load vertices from an OBJ file and apply mesh scale
    """
    vertices = []
    
    with open(mesh_file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Vertex line
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    # Apply mesh scale
                    x *= mesh_scale
                    y *= mesh_scale
                    z *= mesh_scale
                    vertices.append([x, y, z])
    
    return np.array(vertices)

def visualize_mesh_vertices(env, world_vertices, vertex_radius=0.005, vertex_color=[0.8, 0.8, 0.8, 0.3]):
    """
    Visualize mesh vertices for debugging (very limited for performance)
    """
    # Use a very small limit to prevent segmentation faults
    max_vertices = min(len(world_vertices), 10)  # Maximum 10 vertices for safety
    print(f"Visualizing {max_vertices} vertices out of {len(world_vertices)} total (very limited for performance)...")
    
    # Plot limited vertices as small spheres
    for i in range(max_vertices):
        vertex = world_vertices[i]
        env.plot_sphere(
            p=vertex,
            r=vertex_radius,
            rgba=vertex_color,
            label=''
        )
    
    print(f"Mesh vertices visualized with radius {vertex_radius}")

def visualize_mesh_subset(env, world_vertices, subset_size=10, vertex_radius=0.008, vertex_color=[0.2, 0.8, 0.2, 0.5]):
    """
    Visualize a subset of mesh vertices for debugging (useful for dense meshes)
    """
    # Use a very small subset to prevent segmentation faults
    safe_subset_size = min(subset_size, 10)  # Maximum 10 vertices for safety
    
    if len(world_vertices) > safe_subset_size:
        # Randomly sample vertices
        indices = np.random.choice(len(world_vertices), safe_subset_size, replace=False)
        subset_vertices = world_vertices[indices]
        print(f"Visualizing {safe_subset_size} randomly sampled vertices out of {len(world_vertices)} total vertices...")
    else:
        subset_vertices = world_vertices
        print(f"Visualizing all {len(world_vertices)} vertices...")
    
    # Plot subset vertices as spheres (very limited to prevent performance issues)
    for i in range(len(subset_vertices)):
        vertex = subset_vertices[i]
        env.plot_sphere(
            p=vertex,
            r=vertex_radius,
            rgba=vertex_color,
            label=''
        )
    
    print(f"Mesh subset visualized with radius {vertex_radius} (plotted {len(subset_vertices)} vertices)")

def visualize_mesh_boundary(env, world_vertices, vertex_radius=0.01, vertex_color=[0.8, 0.2, 0.2, 0.8]):
    """
    Visualize mesh boundary vertices (extreme points) for debugging
    """
    print("Visualizing mesh boundary vertices...")
    
    # Find extreme points in each dimension
    x_min_idx = np.argmin(world_vertices[:, 0])
    x_max_idx = np.argmax(world_vertices[:, 0])
    y_min_idx = np.argmin(world_vertices[:, 1])
    y_max_idx = np.argmax(world_vertices[:, 1])
    z_min_idx = np.argmin(world_vertices[:, 2])
    z_max_idx = np.argmax(world_vertices[:, 2])
    
    boundary_indices = [x_min_idx, x_max_idx, y_min_idx, y_max_idx, z_min_idx, z_max_idx]
    boundary_vertices = world_vertices[boundary_indices]
    
    # Plot boundary vertices as larger spheres
    for i, vertex in enumerate(boundary_vertices):
        env.plot_sphere(
            p=vertex,
            r=vertex_radius,
            rgba=vertex_color,
            label=f'Boundary_{i}'
        )
    
    print(f"Mesh boundary visualized with {len(boundary_vertices)} vertices")

def visualize_mesh_simple(env, world_vertices, vertex_radius=0.01, vertex_color=[0.2, 0.8, 0.2, 0.8]):
    """
    Visualize just a few key mesh vertices for debugging (safest option)
    """
    print("Visualizing simple mesh representation...")
    
    # Just show the center and a few key points
    center = np.mean(world_vertices, axis=0)
    
    # Plot center point
    env.plot_sphere(
        p=center,
        r=vertex_radius,
        rgba=vertex_color,
        label='Mesh_Center'
    )
    
    # Plot just 3 random vertices
    if len(world_vertices) >= 3:
        indices = np.random.choice(len(world_vertices), 3, replace=False)
        for i, idx in enumerate(indices):
            vertex = world_vertices[idx]
            env.plot_sphere(
                p=vertex,
                r=vertex_radius * 0.7,
                rgba=[0.8, 0.2, 0.2, 0.8],  # Red color for sample points
                label=f'Sample_{i}'
            )
    
    print(f"Simple mesh visualization complete (center + 3 sample points)")

def get_random_surface_point(env, body_name, mesh_file_path, mesh_scale=0.1):
    """
    Get a random surface point from the mesh file
    """
    # Load mesh vertices with scale
    mesh_vertices = load_mesh_vertices(mesh_file_path, mesh_scale)
    print(f"Loaded {len(mesh_vertices)} vertices from mesh (scaled by {mesh_scale})")
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

def place_red_dot_on_surface(env, body_name, mesh_file_path, mesh_scale=0.1, dot_radius=0.02, visualize_mesh=True, debug_mode='simple'):
    """
    Place a red dot on a random surface point of the object
    """
    # Get random surface point
    surface_point, all_vertices = get_random_surface_point(env, body_name, mesh_file_path, mesh_scale)
    
    # Visualize mesh vertices if requested
    if visualize_mesh:
        if debug_mode == 'all':
            visualize_mesh_vertices(env, all_vertices, vertex_radius=0.003, vertex_color=[0.8, 0.8, 0.8, 0.2])
        elif debug_mode == 'subset':
            visualize_mesh_subset(env, all_vertices, subset_size=10, vertex_radius=0.005, vertex_color=[0.2, 0.8, 0.2, 0.4])
        elif debug_mode == 'boundary':
            visualize_mesh_boundary(env, all_vertices, vertex_radius=0.01, vertex_color=[0.8, 0.2, 0.2, 0.8])
        elif debug_mode == 'simple':
            visualize_mesh_simple(env, all_vertices, vertex_radius=0.01, vertex_color=[0.2, 0.8, 0.2, 0.8])
        elif debug_mode == 'none':
            pass
        else:
            print(f"Unknown debug_mode: {debug_mode}. Using 'simple' instead.")
            visualize_mesh_simple(env, all_vertices, vertex_radius=0.01, vertex_color=[0.2, 0.8, 0.2, 0.8])
    
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

    # Place red dot on obj_head surface (with correct mesh scale)
    mesh_file_path = '../asset/makeup_frida/mesh/head.obj'
    
    # Debug modes: 'all' (all vertices), 'subset' (random sample), 'boundary' (extreme points), 'simple' (center + 3 sample points), 'none' (no mesh)
    red_dot_position = place_red_dot_on_surface(
        env, 'obj_head', mesh_file_path, 
        mesh_scale=0.1, 
        dot_radius=0.02, 
        visualize_mesh=True, 
        debug_mode='simple'  # Try 'all' for all vertices, 'subset' for random sample, 'boundary' for extreme points, 'simple' for center + 3 sample points, 'none' for no mesh
    )

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