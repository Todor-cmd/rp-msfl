
class Topology:
    def __init__(self, topology):
        self.topology = topology
        if topology == 'multi-line':
            # Keep track of the clients each server reaches
            self.server_control_dict = {0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [6, 7, 8, 9, 10, 11, 12, 13],
                                        2: [12, 13, 14, 15, 16, 17, 18, 19]}
            # Keep track of weights where keys refer to server and lists contain weights associated with overlap of clients with
            # same index in server_control_dict.
            self.overlap_weight_index = {
                0: [1, 1, 1, 1, 1, 1, 2, 2],
                1: [2, 2, 1, 1, 1, 1, 2, 2],
                2: [2, 2, 1, 1, 1, 1, 1, 1]
            }
        else:
            self.server_control_dict = {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                1: [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17],
                2: [0, 1, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19]
            }
            self.overlap_weight_index = {
                0: [3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
                1: [3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
                2: [3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
            }

    def get_topology(self):
        return self.topology
    def get_server_control(self):
        return self.server_control_dict


    def get_overlap_index(self):
        return self.overlap_weight_index


    def get_set_0(self):
        if self.topology == 'multi-line':
            return [0, 1, 2, 3, 4, 5]
        else:
            return [10, 11]


    def get_set_1(self):
        if self.topology == 'multi-line':
            return [8, 9, 10, 11]
        else:
            return [16, 17]


    def get_set_2(self):
        if self.topology == 'multi-line':
            return [14, 15, 16, 17, 18, 19]
        else:
            return [18, 19]


    def get_set_0_1(self):
        if self.topology == 'multi-line':
            return [6, 7]
        else:
            return [2, 3, 4, 5]


    def get_set_1_2(self):
        if self.topology == 'multi-line':
            return [12, 13]
        else:
            return [12, 13, 14, 15]


    def get_set_0_2(self):
        if self.topology == 'multi-line':
            return []
        else:
            return [6, 7, 8, 9]


    def get_set_0_1_2(self):
        if self.topology == 'multi-line':
            return []
        else:
            return [0, 1]
