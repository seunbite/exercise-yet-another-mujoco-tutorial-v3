from setuptools import setup, find_packages

setup(
    name="mjct",
    version="0.1.0",
    description="MuJoCo tutorial with RL and robotics components",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "numpy",
        "matplotlib",
        "torch",
        "mujoco",
        "opencv-python",
        "tqdm",
        "imageio",
        "gym",
        "cvxpy",
        "shapely",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    package_data={
        "": ["*.xml", "*.png", "*.jpg", "*.gif"],
    },
) 