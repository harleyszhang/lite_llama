import torch


class DummyInputGenerator:
    """A class to generate dummy inputs for memory profiling."""
    
    def __init__(self, device="cuda:5"):
        self.device = device

    
    def generate_dummy_input(self, model_config, batch_size=1, seq_len=32):
        """
        Generate dummy input tensors for memory profiling.
        
        Args:
            model_config: The model configuration object.
            batch_size (int): Batch size for dummy input.
            seq_len (int): Sequence length for dummy input.
            
        Returns:
            tuple: (dummy_input, dummy_position_ids)
        """
        dummy_input = torch.randint(
            0, model_config.vocab_size, 
            (batch_size, seq_len), 
            device=self.device
        )
        dummy_position_ids = torch.arange(
            0, seq_len, 
            dtype=torch.long, 
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        return dummy_input, dummy_position_ids