"""Interactive viser editor for workspace-constraint boxes.

Lets you visually place, move, rotate, resize, and delete box obstacles around
the iiwa, then export them to JSON. The same JSON is consumed by
``fabric_box_avoidance.py`` (via ``--constraints``) and can be shipped to the
real hardware -- the boxes are defined in the robot's world frame, so they mean
the same thing in sim and on the real robot.

Each box gets a viser transform gizmo (drag to translate/rotate) plus GUI number
fields for its size. The robot URDF is shown for reference (read-only). Nothing
here talks to hardware or runs a fabric; it only edits geometry.

Usage:
    python -m odyssey.examples.constraint_editor
    python -m odyssey.examples.constraint_editor --load my_constraints.json
    python -m odyssey.examples.constraint_editor --output my_constraints.json
"""

import argparse
import os

import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy

from odyssey.fabrics import ObstacleSet, BoxObstacle

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(os.getcwd(), "constraints.json")


class _BoxEntry:
    """One editable box: its transform gizmo, mesh, size controls, and GUI folder."""

    def __init__(self, editor, box, index):
        self.editor = editor
        self.index = index
        server = editor.server
        self.name = box.name or f"box_{index}"

        # Transform gizmo drives the box pose (translate + rotate).
        self.controls = server.scene.add_transform_controls(
            f"/constraints/{self.name}", scale=0.3, position=tuple(box.center),
            wxyz=tuple(box.wxyz))
        # The box mesh is a child of the gizmo so it follows the pose.
        self.mesh = server.scene.add_box(
            f"/constraints/{self.name}/mesh", color=(200, 60, 60),
            dimensions=tuple(box.size), opacity=0.45)
        self._size = np.asarray(box.size, dtype=float)

        # GUI folder with per-box size fields and a delete button.
        self.folder = server.gui.add_folder(self.name)
        with self.folder:
            self.sx = server.gui.add_number("size x", float(self._size[0]), min=0.01, step=0.01)
            self.sy = server.gui.add_number("size y", float(self._size[1]), min=0.01, step=0.01)
            self.sz = server.gui.add_number("size z", float(self._size[2]), min=0.01, step=0.01)
            self.delete_btn = server.gui.add_button("delete", color="red", icon=viser.Icon.TRASH)

        for handle in (self.sx, self.sy, self.sz):
            handle.on_update(lambda _: self._on_size_change())
        self.delete_btn.on_click(lambda _: editor.remove_entry(self))
        self.controls.on_update(lambda _: editor.mark_dirty())

    def _on_size_change(self):
        self._size = np.array([self.sx.value, self.sy.value, self.sz.value])
        self.mesh.dimensions = tuple(self._size)
        self.editor.mark_dirty()

    def to_box(self):
        """Read the current gizmo pose + size back into a BoxObstacle."""
        return BoxObstacle.from_pose(
            center=np.array(self.controls.position),
            size=self._size,
            wxyz=np.array(self.controls.wxyz),
            name=self.name)

    def remove(self):
        self.controls.remove()   # removes children (the mesh) too
        self.folder.remove()


class ConstraintEditor:
    def __init__(self, output_path, load_path=None, host="127.0.0.1", port=8080):
        self.output_path = output_path
        self.server = viser.ViserServer(host=host, port=port)
        self._entries = []
        self._counter = 0

        self._show_robot()
        self._build_toolbar()

        # Seed from an existing file, else one starter box.
        if load_path and os.path.exists(load_path):
            for box in ObstacleSet.load_json(load_path).obstacles:
                self._add_entry(box)
            self._status(f"Loaded {len(self._entries)} boxes from {load_path}")
        else:
            self._add_entry(BoxObstacle([0.5, 0.0, 0.5], [0.2, 0.2, 0.2], name="box_0"))

    # -- scene setup --------------------------------------------------------
    def _show_robot(self):
        urdf = yourdfpy.URDF.load(
            os.path.join(_PKG_DIR, "urdf", "med.urdf"),
            mesh_dir=os.path.join(_PKG_DIR, "urdf", "lbr_description/"),
            load_meshes=True, build_scene_graph=True,
            load_collision_meshes=False, build_collision_scene_graph=False)
        self._viser_urdf = ViserUrdf(self.server, urdf_or_path=urdf, load_meshes=True)
        self._viser_urdf.update_cfg(np.zeros(7))
        self.server.scene.add_grid("/grid", width=2, height=2, position=(0.0, 0.0, 0.0))

    def _build_toolbar(self):
        gui = self.server.gui
        with gui.add_folder("Workspace constraints"):
            self._add_btn = gui.add_button("add box", icon=viser.Icon.PLUS)
            self._save_btn = gui.add_button("save JSON", icon=viser.Icon.DEVICE_FLOPPY)
            self._path_text = gui.add_text("output path", self.output_path)
            self._status_text = gui.add_text("status", "ready", disabled=True)
        self._add_btn.on_click(lambda _: self._add_default_box())
        self._save_btn.on_click(lambda _: self.save())

    # -- box management -----------------------------------------------------
    def _add_entry(self, box):
        entry = _BoxEntry(self, box, self._counter)
        self._counter += 1
        self._entries.append(entry)
        return entry

    def _add_default_box(self):
        self._add_entry(BoxObstacle([0.5, 0.0, 0.5], [0.2, 0.2, 0.2],
                                    name=f"box_{self._counter}"))
        self._status(f"{len(self._entries)} boxes")

    def remove_entry(self, entry):
        entry.remove()
        self._entries.remove(entry)
        self._status(f"{len(self._entries)} boxes")

    def mark_dirty(self):
        # Placeholder hook for live-preview features; pose reads are pull-based.
        pass

    # -- IO -----------------------------------------------------------------
    def to_obstacle_set(self):
        return ObstacleSet([e.to_box() for e in self._entries])

    def save(self):
        path = self._path_text.value or self.output_path
        self.to_obstacle_set().save_json(path)
        self._status(f"Saved {len(self._entries)} boxes -> {path}")
        print(f"Saved {len(self._entries)} constraint boxes to {path}")

    def _status(self, msg):
        self._status_text.value = msg

    def run(self):
        print(f"Constraint editor at http://{self.server.get_host()}:{self.server.get_port()}")
        print("Add/move/resize boxes, then click 'save JSON'. Ctrl-C to quit.")
        try:
            while True:
                import time
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Exiting constraint editor.")


def main():
    parser = argparse.ArgumentParser(description="Viser workspace-constraint box editor.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Where 'save JSON' writes (default: ./constraints.json).")
    parser.add_argument("--load", default=None,
                        help="Existing constraints JSON to open for editing.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    editor = ConstraintEditor(output_path=args.output, load_path=args.load,
                              host=args.host, port=args.port)
    editor.run()


if __name__ == "__main__":
    main()
