"""pytorchfi.fault_detector provides fault detection functionality"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Callable, Optional

# logging.basicConfig(
#     level=logging.INFO,     
#     format="%(levelname)s: %(message)s"
# )
THRESHOLD = 8e-6
class FaultDetector:
    def __init__(
        self,
        model: nn.Module,
        layer_types: List = None,
        #detector_function: Optional[Callable] = None,
        use_cuda: bool = False,
        remove_bias: bool = False,
        total_faults: int = 0,
        is_tvlsi: bool = False
    ):
        """
        Initialize Fault Detector
        Args:
            model: PyTorch model to monitor
            layer_types: List of layer types to monitor (default: [nn.Conv2d])
            detector_function: Custom detection function(input_tensor, output_tensor) -> bool
                              If None, uses default detector (can be implemented later)
            use_cuda: Whether to use CUDA
            remove_bias: If True, remove bias from output before detection (default: False)
        """
        if layer_types is None:
            layer_types = [nn.Conv2d]
        
        self.model = model
        self.layer_types = layer_types
        #self.detector_function = detector_function
        self.use_cuda = use_cuda
        self.remove_bias = remove_bias
        self.is_tvlsi = is_tvlsi
        
        # Store captured data
        self.layer_inputs = []
        self.layer_outputs = []
        self.layer_names = []
        self.layer_weights = []
        self.detection_results = []
        self.total_errors = 0
        self.total_faults_injected = total_faults
        self.layer_indices = {}
        self.detected_dict = {"layer": [],
                              "batch": [],
                              "channel": []
                              }
        # Hooks
        self.handles = []
            
    def _hook_fn(self, layer_name: str):
        """
        Create a forward hook function for a layer
        Args:
            layer_name: Name of the layer 
        Returns:
            Hook function for each layer
        """
        def hook(module, input_val, output):
            # Remove bias from output (if requested)
            output_to_process = output
            if self.remove_bias and hasattr(module, 'bias') and module.bias is not None:
                # Reshape bias for broadcasting: [out_channels] -> [1, out_channels, 1, 1]
                bias_reshaped = module.bias.view(1, -1, 1, 1)
                output_to_process = output - bias_reshaped
            
            # Capture input and output
            self.layer_inputs.append((layer_name, input_val))
            self.layer_outputs.append((layer_name, output_to_process))
            
            # Capture weight information if the layer has weights
            if hasattr(module, 'weight') and module.weight is not None:
                self.layer_weights.append((layer_name, module.weight.data.clone()))
            else:
                self.layer_weights.append((layer_name, None))
            # Get weight data
            weight_data = module.weight.data.clone() if hasattr(module, 'weight') and module.weight is not None else None
            # Set fault detector
            is_error = self.default_detector(input_val, output_to_process, weight_data, layer_name)
            self.detection_results.append({
                'layer_name': layer_name,
                'is_error': is_error,
                'input_shape': [t.shape for t in input_val] if isinstance(input_val, tuple) else input_val.shape,
                'output_shape': output_to_process.shape,
                'weight_shape': module.weight.shape if hasattr(module, 'weight') and module.weight is not None else None,
            })                    

        return hook
    
    def get_layer_idx(self, layer_name):
        layer_idx = 0
        for name, layer in self.model.named_modules():
            if self._is_target_layer(layer):
                if name == layer_name:
                    return layer_idx
                layer_idx += 1

        # If not found among target layers, return -1 as fallback
        return -1

    
    def register_hooks(self):
        """
        Register forward hooks on all target layers
        """
        self._clear_hooks()
        
        for name, layer in self.model.named_modules():
            if self._is_target_layer(layer):
                handle = layer.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle) # handles is a list of hook handles for every registered hook on a layer
                logging.info(f"Registered hook on layer: {name}")
    
    def _is_target_layer(self, layer: nn.Module) -> bool:
        for layer_type in self.layer_types:
            if isinstance(layer, layer_type):
                return True
        return False
    
    def _clear_hooks(self):
        """Remove all registered hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def clear_data(self):
        """Clear captured data"""
        self.layer_inputs = []
        self.layer_outputs = []
        self.layer_weights = []
        self.layer_names = []
        self.detection_results = []
        self.total_errors = 0
    
    # Not used
    def get_layer_data(self, layer_name: str = None) -> dict:
        if layer_name is None:
            return {
                'inputs': self.layer_inputs,
                'outputs': self.layer_outputs,
                'weights': self.layer_weights,
                'detection_results': self.detection_results,
            }
        else:
            layer_input = next((inp for name, inp in self.layer_inputs if name == layer_name), None)
            layer_output = next((out for name, out in self.layer_outputs if name == layer_name), None)
            layer_weight = next((w for name, w in self.layer_weights if name == layer_name), None)
            layer_detection = next((det for det in self.detection_results if det['layer_name'] == layer_name), None)
            
            return {
                'input': layer_input,
                'output': layer_output,
                'weight': layer_weight,
                'detection': layer_detection,
            }
    
    def get_detection_results(self) -> List[dict]:
        return self.detection_results

    def get_error_layers(self) -> List[str]:
        return [det['layer_name'] for det in self.detection_results if det['is_error']]
    
    def default_detector(self, input_tensors, output_tensor, weight_tensor, layer_name):
        """    
        Args:
            input_tensors: Input tensors
            output_tensor: Output tensors
            weight_tensor: Weight tensors
            
        Returns:
            True if error detected, False otherwise
        """
        layer_idx = self.get_layer_idx(layer_name)
        inp = input_tensors[0] if isinstance(input_tensors, (tuple, list)) else input_tensors
        inp_pad = F.pad(inp, (1, 1, 1, 1), mode="constant", value=0)
        out = output_tensor[0] if isinstance(output_tensor, (tuple, list)) else output_tensor     
        # Input checksum computation
        no_ker, ker_size = weight_tensor.shape[0], weight_tensor.shape[2] 
        batch, channels, rows, cols = inp_pad.shape[0], inp_pad.shape[1], inp_pad.shape[2], inp_pad.shape[3]
        input_checksum = torch.zeros(no_ker, 1, dtype=torch.float32, device=weight_tensor.device)
        sum = torch.zeros(channels, ker_size, ker_size)
        mul = torch.zeros(no_ker, channels, ker_size, ker_size)
        if ker_size == 3: # 3x3 convolution
            if not self.is_tvlsi:
                #sum = torch.zeros(channels, ker_size, ker_size) 
                for ch in range(channels):
                    for r in range(ker_size):
                        for c in range(ker_size):
                            window = inp_pad[0, ch, r:r+rows-2, c:c+cols-2]
                            sum[ch, r, c] = window.sum()

                #mul = torch.zeros(no_ker, channels, ker_size, ker_size) 
                for no in range(no_ker):
                    for ch in range(channels):
                        for r in range(ker_size):
                            for c in range(ker_size):
                                mul[no, ch, r, c] = weight_tensor[no, ch, r, c] * sum[ch, r, c]
                
                for no in range(no_ker):
                    input_checksum[no] = mul[no, :, :, :].sum()
            else:
                print("TVLSI")
                IN_CH = channels
                W = cols
                H = rows
                M = ker_size
                in_channel_sum = torch.zeros(IN_CH, 1, 1)

                for ch in range(IN_CH):
                    ch_sum = inp_pad[0, ch, :, :]
                    in_channel_sum[ch] = ch_sum.sum()
                    
                #sum = torch.zeros(channels, ker_size, ker_size)
                sub = torch.zeros(channels, ker_size, ker_size)
                for ch in range(channels):
                    for i in range(cols):
                        for j in range(rows):
                            for m in range(ker_size):
                                for n in range(ker_size):
                                    if m>i or i>H-M+m or n>j or j>W-M+n:
                                        sub[ch, m, n] += inp_pad[0, ch, i, j]
                
                for ch in range(IN_CH):
                    for i in range(ker_size):
                        for j in range(ker_size):
                            sum[ch, i, j] = in_channel_sum[ch] - sub[ch, i, j]
                
                #mul = torch.zeros(no_ker, channels, ker_size, ker_size)
                for no in range(no_ker):
                    for ch in range(channels):
                        for r in range(ker_size):
                            for c in range(ker_size):
                                mul[no, ch, r, c] = weight_tensor[no, ch, r, c] * sum[ch, r, c]
                
                for no in range(no_ker):
                    input_checksum[no] = mul[no, :, :, :].sum()
                    
        elif ker_size == 1: # 1x1 convolution
            sum = torch.zeros(channels, 1)
            for ch in range(channels):
                window = inp[0, ch, :, :]
                sum[ch] = window.sum()
            
            mul = torch.zeros(no_ker, channels)
            for no in range(no_ker):
                for ch in range(channels):
                    mul[no, ch] = weight_tensor[no, ch, 0, 0] * sum[ch, 0]
            
            for no in range(no_ker):
                input_checksum[no] = mul[no, :].sum()
        else:
            return False # Unsupported kernel size
                  
        # Output checksum computation
        output_checksum = torch.zeros(no_ker, 1, dtype=torch.float32, device=weight_tensor.device)
        for no in range(no_ker):
            output_checksum[no] = out[0, no, :, :].sum()
        
        # fault detection
        errors = torch.zeros(no_ker, 1, dtype=torch.bool)

        for b in range(batch):
            for no in range(no_ker):
                abs_diff = torch.abs(input_checksum[no] - output_checksum[no])     
                # Relative error: diff / magnitude
                magnitude = torch.abs(output_checksum[no])
                relative_error = abs_diff / magnitude
                # Convert to scalar for printing
                if relative_error > THRESHOLD:
                    errors[no] = True
                    print(f"[DETECTION] ERROR in layer={layer_idx} channel={no}: abs_diff={abs_diff.item():.2e}, rel_error={relative_error.item():.2e}")
                    self.detected_dict["layer"].append(layer_idx)
                    self.detected_dict["batch"].append(b)
                    self.detected_dict["channel"].append(no)
        # Calculate total errors
        false_count = [i for i, x in enumerate(errors) if x]
        self.total_errors += len(false_count)
        return errors.any().item()
    
    def print_detection_detailed_summary(self) -> str:
        """
        Print summary of detection results
        
        Returns:
            Summary string
        """
        summary_str = "============================ FAULT DETECTION SUMMARY ============================\n\n"
        summary_str += f" • Model: {self.model.__class__.__name__}\n"
        summary_str += f" • Total layers monitored: {len(set([name for name, _ in self.layer_inputs]))}\n"
        summary_str += f" • Layer name: {[name for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d)]}\n"
        summary_str += f" • Total faults injected: {self.total_faults_injected}\n" # Need to be changed
        summary_str += f" • Total faults detected: {self.total_errors}\n"
        summary_str += f" • Error layer: {self.get_error_layers()}\n\n"
        
        if not self.detection_results:
            summary_str += "No errors detected.\n"
        summary_str += "=" * 80 + "\n"
        logging.info(summary_str)
        return summary_str
    
    def __enter__(self):
        """Context manager entry"""
        self.register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self._clear_hooks()
        return False