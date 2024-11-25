"""Object class."""

import trimesh


class Object:
    """Object class."""

    def __init__(self,
                 obj_path: str,
                 scale: list[float] = [1.0, 1.0, 1.0],
                 eular_angles: list[float] = [0.0, 0.0, 0.0],
                 label: str = None) -> None:
        """
        Initialize the object.

        Args:
            obj_path: The path to the object.
            scale: The scale of the object.
            eular_angles: The rotation of the object in Euler angles.
            label: The label of the object.
            name: The name of the object.

        Returns:
            None
        """
        # Load with trimesh
        self.obj_path = obj_path
        self.scale = scale
        self.eular_angles = eular_angles
        self.label = label
        self._load_mesh()

    def get_points(self):
        """
        Get the points of the object.

        Returns:
            The points of the object.
        """
        points = self._mesh.vertices
        return points

    def get_label(self):
        """
        Get the label of the object.

        Returns:
            The label of the object.
        """
        return self.label

    def _load_mesh(self):
        """
        Load the mesh from the object path.

        Returns:
            None
        """
        # Load mesh
        self._mesh = trimesh.load(self.obj_path)

        # If it's a Scene object, extract the geometries
        if isinstance(self._mesh, trimesh.Scene):
            self._mesh = self._mesh.dump(concatenate=True)
        self._mesh.apply_scale(self.scale)

        # Rotate Mesh
        self._rotate_mesh(self.eular_angles)

    def _rotate_mesh(self,
                     rot_rpy: list[float]):
        """
        Rotate the mesh.

        Args:
            rot_rpy: The rotation in Euler angles.

        Returns:
            None
        """
        center = [0, 0, 0]
        angle = rot_rpy[0]
        direction = [1, 0, 0]
        rot_matrix_r = trimesh.transformations.rotation_matrix(angle,
                                                               direction,
                                                               center)
        self._mesh.apply_transform(rot_matrix_r)
        angle = rot_rpy[1]
        direction = [0, 1, 0]
        rot_matrix_p = trimesh.transformations.rotation_matrix(angle,
                                                               direction,
                                                               center)
        self._mesh.apply_transform(rot_matrix_p)
        angle = rot_rpy[2]
        direction = [0, 0, 1]
        rot_matrix_y = trimesh.transformations.rotation_matrix(angle,
                                                               direction,
                                                               center)
        self._mesh.apply_transform(rot_matrix_y)
