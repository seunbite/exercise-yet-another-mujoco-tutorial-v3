import mujoco
# from mujoco import viewer as mj_viewer
import mujoco_viewer as mj_viewer
import numpy as np

class MujocoPythonViewer:

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        title: str = "MuJoCo Python Viewer",
        width: int = 1024,
        height: int = 768,
    ) -> None:
        self._viewer = mj_viewer.MujocoViewer(model, data, title=title, width=width, height=height, hide_menus=True)
        self.model = model
        self.data = data
        self.cam = self._viewer.cam
        self.vopt = self._viewer.vopt
        self.scn = self._viewer.scn
        self.ctx = self._viewer.ctx
        self.is_alive = True

    def add_marker(self, **marker_params):
        self._viewer.add_marker(**marker_params)

    def plot_marker(self, **kwargs):
        self.add_marker(**kwargs)

    # ------------------------------------------------------------------
    # Core API expected by MuJoCoParserClass
    # ------------------------------------------------------------------
    def render(self):
        self._viewer.render()

    # Aliases so that mujoco_parser can call viewer.is_alive as property
    @property
    def is_running(self):
        return self.is_alive

    # Required by MuJoCoParserClass.render() when viewer closed.
    def close(self):
        if not self.is_alive:
            return
        try:
            self._viewer.close()
        except Exception:
            pass
        self.is_alive = False

    # Context-manager convenience (optional)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 