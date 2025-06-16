import sys,mujoco,time,os,json
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
np.set_printoptions(precision=2,suppress=True,linewidth=100)
plt.rc('xtick',labelsize=6); plt.rc('ytick',labelsize=6)
# %config InlineBackend.figure_format = 'retina'
print ("Ready.")


# Initialize environment
xml_path = '../asset/makeup_frida/scene_table.xml'
env = MuJoCoParserClass(name='MakeupFrida', rel_xml_path=xml_path, verbose=True)
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

# Get object names
obj_names = env.get_body_names(prefix='obj_')
n_obj = len(obj_names)
print("Object list:")
for obj_idx in range(n_obj):
    print(f" [{obj_idx}] obj_name:[{obj_names[obj_idx]}]")

# Main loop
env_state = env.get_state()
while env.is_viewer_alive():
    # Step
    env.step()
    
    # Render
    if env.loop_every(tick_every=10):
        env.plot_time()
        env.render()
    
    # Plot
    if env.loop_every(tick_every=500):  # every 1 second
        # Grab current view
        render_img = env.grab_image(rsz_rate=1/4)
        # Plot
        plt.figure(figsize=(8, 3))
        plt.imshow(render_img)
        plt.title(f"Time:[{env.get_sim_time():.2f}]sec", fontsize=10)
        plt.show()
        

env.close_viewer()
print ("Done.")
