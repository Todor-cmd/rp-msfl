from unittest import TestCase
import torch

import aggregations


class Test(TestCase):

    def test_fedmes_mean(self):
        # Example input data
        all_updates = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        overlap_weight_index = [1, 1, 2]

        # Call the function
        result = aggregations.fedmes_mean(all_updates, overlap_weight_index)

        # Define the expected result manually based on the input
        expected_result = torch.tensor([4.75, 5.75, 6.75])

        # Check if the result matches the expected result
        self.assertTrue(torch.allclose(result, expected_result))

    def test1_fedmes_tr_mean(self):
        # Example input data
        all_updates = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        n_attackers = 1
        overlap_weight_index = [1, 2, 3]

        # Call the function
        result = aggregations.fedmes_tr_mean_v2(all_updates, n_attackers, overlap_weight_index)

        # Define the expected result manually based on the input
        expected_result = torch.tensor([4.0, 5.0, 6.0])

        # Check if the result matches the expected result
        self.assertTrue(torch.allclose(result, expected_result))

    def test2_fedmes_tr_mean(self):
        # Example input data
        all_updates = torch.tensor([[7.0, 5.0, 3.0], [4.0, 2.0, 6.0], [1.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        n_attackers = 1
        overlap_weight_index = [1, 2, 3, 4]

        # Call the function
        result = aggregations.fedmes_tr_mean_v2(all_updates, n_attackers, overlap_weight_index)

        # Define the expected result manually based on the input
        expected_result = torch.tensor([5.0, 7.25, 7.8])

        # Check if the result matches the expected result
        self.assertTrue(torch.allclose(result, expected_result))

    def test_bulyan(self):
        # Example input data
        all_updates = torch.tensor([[7.0, 5.0, 3.0], [4.0, 2.0, 6.0], [1.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        n_attackers = 1
        overlap_weight_index = {0: 1, 1: 2, 2: 3, 3: 4}

        # Call the function
        result = aggregations.fedmes_bulyan(all_updates, n_attackers, overlap_weight_index)



