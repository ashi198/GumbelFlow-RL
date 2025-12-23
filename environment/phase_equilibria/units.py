import os
import numpy as np 

class distillation_region:
    def __init__(self, boundary_1, boundary_2, index_high_boiler, index_low_boiler, s_p_indices_contained):
        """
        boundaries are given in cartesian coordinates, going from heavy boiler to low boiler
        """
        self.boundaries = [boundary_1, boundary_2]
        self.index_high_boiler = index_high_boiler
        self.index_low_boiler = index_low_boiler
        self.s_p_indices_contained = s_p_indices_contained


class singular_point:
    def __init__(self, molar_fractions, cart_coords, boiling_point, role, index):
        """
        class to store singular points in the ternary
        """
        self.molar_fractions = molar_fractions
        self.cart_coords = cart_coords
        self.boiling_point = boiling_point
        self.role = role  # -1 low boiler, 0 saddle, 1 high boiler
        self.index = index  # index inside a complete singular_points list (initialized later)

class ternary_vle:
    def __init__(self, index, name, path):
        """
        Stores the distillation regions, singular points and boiling points
        of the considered example.
        """
        self.index = index  # refers to outer list, where all ternaries are stored
        self.name = name
        self.standard_path = os.path.join(path, name)

        # indices of components
        self.comp_indices = np.load(os.path.join(self.standard_path, "sorted_indices.npy"))
        self.num_comp = len(self.comp_indices)

        # get pressure
        self.pressure = np.load(os.path.join(self.standard_path, "pressure.npy"))

        # transformation matrices
        matrices = np.load(os.path.join(self.standard_path, "coord_trafo_matrices.npy"))
        self.matrix_mfr_to_cart = matrices[0]
        self.matrix_cart_to_mfr = matrices[1]

        # number of distillation regions
        self.num_dis_regions = np.load(os.path.join(self.standard_path, "num_dis_reg.npy"))

        # get distillation regions
        self.distillation_regions = []
        for i in range(self.num_dis_regions):
            high_low_indices = np.load(os.path.join(self.standard_path, "dis_reg_" + str(i) + "_high_low.npy"))
            high_index = round(high_low_indices[0])
            low_index = round(high_low_indices[1])

            # boundaries and indices of singular points contained
            boundaries = []
            s_p_indices = []
            for j in range(2):
                bound_matrix = np.load(os.path.join(
                    self.standard_path, "dis_reg_" + str(i) + "_bound_" + str(j) + ".npy"))
                boundary = []
                for k in range(len(bound_matrix)):
                    boundary.append(bound_matrix[k])

                boundaries.append(boundary)

                index_pairs = np.load(os.path.join(
                    self.standard_path, "dis_reg_" + str(i) + "_bound_indices_" + str(j) + ".npy"))

                for k in range(len(index_pairs)):
                    for u in range(2):
                        candidate = round(index_pairs[k][u])
                        if candidate not in s_p_indices:
                            s_p_indices.append(candidate)

            self.distillation_regions.append(distillation_region(boundaries[0], boundaries[1], high_index, low_index,
                                                                 s_p_indices))

        # get singular points
        self.singular_points = []
        s_p_mfr = np.load(os.path.join(self.standard_path, "molar_fractions_s_p.npy"))
        s_p_cart_coords = np.load(os.path.join(self.standard_path, "cart_coords_s_p.npy"))
        s_p_bps = np.load(os.path.join(self.standard_path, "boiling_point_s_p.npy"))
        s_p_roles = np.load(os.path.join(self.standard_path, "role_s_p.npy"))

        for i in range(len(s_p_bps)):
            self.singular_points.append(singular_point(s_p_mfr[i], s_p_cart_coords[i], s_p_bps[i],
                                                       round(s_p_roles[i]), i))