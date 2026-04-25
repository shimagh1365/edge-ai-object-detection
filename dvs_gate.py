import torch
import numpy as np

class DVSTemporalGate:
    """
    Simulates a DVS (Dynamic Vision Sensor) event camera gate.
    
    Real DVS cameras fire asynchronous events when pixel brightness 
    changes exceed a threshold. Here we simulate this from frame 
    differences — a standard approach in the literature.
    
    Key idea: only trigger full RGB inference when meaningful 
    motion/change is detected. Static scenes skip inference entirely,
    dramatically reducing energy consumption.
    """
    def __init__(self, threshold=0.05, min_event_density=0.02):
        self.threshold = threshold          # pixel change threshold
        self.min_event_density = min_event_density  # min % pixels that must fire
        self.prev_frame = None
        self.skip_count = 0
        self.total_count = 0
        self.event_densities = []
    
    def should_process(self, frame: torch.Tensor) -> bool:
        """
        Returns True if RGB inference should run this frame.
        Returns False if scene is static — skip inference.
        """
        self.total_count += 1
        
        # Always process first frame
        if self.prev_frame is None:
            self.prev_frame = frame.clone()
            return True
        
        # Compute pixel-wise absolute difference (simulate DVS events)
        diff = torch.abs(frame.float() - self.prev_frame.float())
        
        # Event density = fraction of pixels that "fired"
        event_density = (diff > self.threshold).float().mean().item()
        self.event_densities.append(event_density)
        
        # Update previous frame
        self.prev_frame = frame.clone()
        
        if event_density >= self.min_event_density:
            return True  # Motion detected — run inference
        else:
            self.skip_count += 1
            return False  # Static scene — skip inference
    
    def reset(self):
        self.prev_frame = None
        self.skip_count = 0
        self.total_count = 0
        self.event_densities = []
    
    @property
    def skip_rate(self):
        if self.total_count == 0:
            return 0.0
        return self.skip_count / self.total_count
    
    @property
    def avg_event_density(self):
        if not self.event_densities:
            return 0.0
        return np.mean(self.event_densities)


def simulate_video_sequence(n_frames=200, img_size=640, 
                             motion_probability=0.3):
    """
    Simulate a realistic street scene video:
    - 70% of frames: static background (pedestrian not moving much)
    - 30% of frames: motion (pedestrian walking, car passing)
    """
    frames = []
    base_frame = torch.rand(1, 3, img_size, img_size)
    
    for i in range(n_frames):
        if torch.rand(1).item() < motion_probability:
            # Motion frame: add significant change
            noise_level = torch.rand(1).item() * 0.3 + 0.1
            frame = base_frame + torch.randn_like(base_frame) * noise_level
            frame = torch.clamp(frame, 0, 1)
            base_frame = frame.clone()
        else:
            # Static frame: minimal change
            noise_level = 0.01
            frame = base_frame + torch.randn_like(base_frame) * noise_level
            frame = torch.clamp(frame, 0, 1)
        frames.append(frame)
    
    return frames


if __name__ == '__main__':
    import time
    from ultralytics import YOLO
    
    print("=" * 55)
    print("DVS Temporal Gate Simulation")
    print("=" * 55)
    
    model = YOLO('yolov8n.pt')
    gate = DVSTemporalGate(threshold=0.05, min_event_density=0.02)
    
    # Simulate 200 frames of street scene
    print("Generating simulated street scene (200 frames)...")
    frames = simulate_video_sequence(n_frames=200, motion_probability=0.3)
    
    # Run with DVS gate
    inference_times = []
    inferences_run = 0
    
    for frame in frames:
        if gate.should_process(frame):
            start = time.perf_counter()
            model(frame, verbose=False)
            inference_times.append((time.perf_counter() - start) * 1000)
            inferences_run += 1
    
    # Results
    baseline_energy = 100 * (65.0 / 1000) * 1000  # from baseline
    effective_energy = baseline_energy * (1 - gate.skip_rate)
    
    print(f"\nResults over 200 frames:")
    print(f"  Total frames:        {gate.total_count}")
    print(f"  Inferences run:      {inferences_run}")
    print(f"  Frames skipped:      {gate.skip_count}")
    print(f"  Skip rate:           {gate.skip_rate*100:.1f}%")
    print(f"  Avg event density:   {gate.avg_event_density*100:.2f}%")
    print(f"\nEnergy Analysis:")
    print(f"  Baseline energy/frame:    {baseline_energy:.1f} mJ")
    print(f"  EG-YOLO effective:        {effective_energy:.1f} mJ")
    print(f"  Energy saving:            {gate.skip_rate*100:.1f}%")
    print(f"\nLatency (when inference runs):")
    if inference_times:
        print(f"  Mean: {np.mean(inference_times):.1f}ms")
        print(f"  P95:  {np.percentile(inference_times, 95):.1f}ms")
    print("=" * 55)
