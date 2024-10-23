import torch
import unittest
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from bound_vanilla_rnn import RNN

class TestRNN(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 5
        self.time_step = 3
        self.activation = 'relu'
        self.rnn = RNN(self.input_size, self.hidden_size, self.output_size, self.time_step, self.activation)

    def test_zeroSplit(self):
        l = torch.tensor([[-1.0, -2.0]])
        u = torch.tensor([[ 1.0,  1.0]])
        dimensions = [0, 1]
        abstraction = (l, u)
        
        results = self.rnn.zeroSplit(abstraction, dimensions)
        print("Results: ", results)
        
        expected_results = [
            (torch.tensor([[-1.0, -2.0], [0.0, 1.0]]), torch.tensor([[0.0, -2.0], [1.0, 1.0]])),
            (torch.tensor([[0.0, -2.0], [1.0, 1.0]]), torch.tensor([[0.0, -2.0], [1.0, 1.0]])),
 
    ]
    
        # (torch.tensor([[2.0, 0.0],  [0.0, 0.0]]),  torch.tensor([[4.0, 5.0],[1.0, 0.5]]))
        self.assertEqual(len(results), len(expected_results))
        for res, exp in zip(results, expected_results):
            self.assertTrue(torch.equal(res[0], exp[0]))
            self.assertTrue(torch.equal(res[1], exp[1]))

if __name__ == '__main__':
    unittest.main()