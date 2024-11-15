import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

class TestConv2dEdgeCases(unittest.TestCase):
    def test_zero_batch_size(self):
        """Test Conv2d with a batch size of zero."""
        conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        input = torch.randn(0, 3, 32, 32) 
        output = conv(input)
        self.assertEqual(output.shape[0], 0)

    def test_single_channel_input(self):
        """Test Conv2d with single-channel input."""
        conv = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        input = torch.randn(10, 1, 32, 32)
        output = conv(input)
        self.assertEqual(output.shape[1], 6) 

    def test_non_square_input(self):
        """Test Conv2d with non-square input dimensions."""
        conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        input = torch.randn(10, 3, 28, 32)
        output = conv(input)
        self.assertEqual(output.shape[2], 26)
        self.assertEqual(output.shape[3], 30) 

    def test_integer_input(self):
        """Test Conv2d with integer input data type."""
        conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        input = torch.randint(0, 256, (10, 3, 32, 32), dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            conv(input)

    def test_extra_dimensions(self):
        """Test Conv2d with extra unnecessary dimensions."""
        conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        input = torch.randn(10, 3, 32, 32, 1) 
        with self.assertRaises(RuntimeError):
            conv(input)

class TestLSTMEdgeCases(unittest.TestCase):
    def test_zero_length_sequence(self):
        """Test LSTM with a sequence length of zero."""
        lstm = nn.LSTM(input_size=10, hidden_size=20)
        input = torch.randn(0, 5, 10) 
        with self.assertRaises(RuntimeError):
            lstm(input)

    def test_varying_sequence_lengths(self):
        """Test LSTM with sequences of varying lengths using packing."""
        lstm = nn.LSTM(input_size=10, hidden_size=20)
        sequences = [torch.randn(l, 10) for l in [5, 3, 2]]
        inputs_padded = nn.utils.rnn.pad_sequence(sequences)
        lengths = torch.tensor([5, 3, 2])
        packed_input = nn.utils.rnn.pack_padded_sequence(
            inputs_padded, lengths, enforce_sorted=False
        )
        packed_output, (hn, cn) = lstm(packed_input)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        self.assertEqual(output.shape[1], 3)

    def test_bidirectional_lstm(self):
        """Test bidirectional LSTM outputs."""
        lstm = nn.LSTM(input_size=10, hidden_size=20, bidirectional=True)
        input = torch.randn(5, 3, 10)
        output, (hn, cn) = lstm(input)
        self.assertEqual(output.shape[-1], 40) 

    def test_integer_input(self):
        """Test LSTM with integer input data type."""
        lstm = nn.LSTM(input_size=10, hidden_size=20)
        input = torch.randint(0, 256, (5, 3, 10), dtype=torch.int32)
        with self.assertRaises(ValueError):
            lstm(input) 


class CustomLoss(nn.Module):
    def forward(self, input, target):
        return torch.mean((input - target) ** 2)

class TestCustomLoss(unittest.TestCase):
    def test_zero_length_input(self):
        """Test custom loss with zero-length tensors."""
        loss_fn = CustomLoss()
        input = torch.randn(0)
        target = torch.randn(0)
        loss = loss_fn(input, target)
        self.assertTrue(torch.isnan(loss))  

    def test_mismatched_shapes(self):
        """Test custom loss with mismatched input and target shapes."""
        loss_fn = CustomLoss()
        input = torch.randn(10)
        target = torch.randn(12)
        with self.assertRaises(RuntimeError):
            loss_fn(input, target)

    def test_integer_input(self):
        """Test custom loss with integer tensors."""
        loss_fn = CustomLoss()
        input = torch.randint(0, 10, (10,), dtype=torch.int32).float()
        target = torch.randint(0, 10, (10,), dtype=torch.int32).float()
        loss = loss_fn(input, target)
        self.assertIsInstance(loss.item(), float)

    def test_gradients(self):
        """Test that gradients are computable in the custom loss."""
        loss_fn = CustomLoss()
        input = torch.randn(10, requires_grad=True)
        target = torch.randn(10)
        loss = loss_fn(input, target)
        loss.backward()
        self.assertIsNotNone(input.grad)

if __name__ == '__main__':
    unittest.main()
