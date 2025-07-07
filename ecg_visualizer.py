import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def visualize_segmentation(signals, r_peaks, beats, labels, sample_indexes=None, fs=360):
    """
    Visualize ECG beat segmentation results
    
    Parameters:
        signals (np.array): Raw ECG signal
        r_peaks (list): Array of R-peak positions
        beats (np.array): Array of segmented heartbeats
        labels (list): Array of beat labels
        sample_indexes (list): Indexes of specific beats to display
        fs (int): Sampling rate (Hz)
    """
    # Set default to show first 5 beats
    if sample_indexes is None:
        sample_indexes = range(min(5, len(beats)))
    
    # Create figure layout
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Full Signal View: Show all R-peaks
    ax1 = fig.add_subplot(gs[0, :])
    time = np.arange(len(signals)) / fs  # Convert to seconds
    ax1.plot(time, signals, 'b-', linewidth=0.8, alpha=0.7)
    
    # Mark R-peak positions and labels
    for i, peak in enumerate(r_peaks):
        peak_time = peak / fs
        ax1.plot(peak_time, signals[peak], 'ro', markersize=4)
        # Label every 10th beat to avoid overlap
        if i % 10 == 0:  
            ax1.text(peak_time, signals[peak]+0.2, labels[i], 
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    ax1.set_title('ECG Signal Overview', fontsize=14)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Amplitude (mV)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    
    # 2. Single Beat View: Show details of selected beats
    ax2 = fig.add_subplot(gs[1, 0])
    for idx in sample_indexes:
        if idx < len(beats):
            beat_time = np.arange(len(beats[idx])) / fs - 0.1  # Center at R-peak
            ax2.plot(beat_time, beats[idx], label=f'Beat {idx} ({labels[idx]})')
    
    # Mark typical ECG components (adjusted for physiological reality)
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='R-peak')
    ax2.axvspan(-0.16, -0.06, color='orange', alpha=0.2, label='P-wave region')
    ax2.axvspan(-0.04, 0.06, color='green', alpha=0.2, label='QRS complex')
    ax2.axvspan(0.08, 0.40, color='purple', alpha=0.2, label='T-wave region')
    
    ax2.set_title('Single Beat Detailed View', fontsize=14)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Amplitude (mV)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Beat Comparison View: Compare different beat types
    ax3 = fig.add_subplot(gs[1, 1])
    unique_labels = sorted(set(labels))  # Sort labels
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        label_beats = [b for b, l in zip(beats, labels) if l == label]
        if label_beats:
            # Calculate average beat
            avg_beat = np.mean(label_beats, axis=0)
            beat_time = np.arange(len(avg_beat)) / fs - 0.1
            
            # Plot average waveform and standard deviation range
            std_beat = np.std(label_beats, axis=0)
            ax3.plot(beat_time, avg_beat, color=colors[i], label=label, linewidth=2)
            ax3.fill_between(beat_time, 
                             avg_beat - std_beat, 
                             avg_beat + std_beat, 
                             color=colors[i], alpha=0.2)
    
    ax3.set_title('Comparison of Beat Types', fontsize=14)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Amplitude (mV)', fontsize=12)
    ax3.legend(title='Beat Type', fontsize=10, title_fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('ecg_segmentation_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# Test case
if __name__ == "__main__":
    # Simulated data - replace with real data
    fs = 360  # Sampling rate
    duration = 10  # Signal duration (seconds)
    num_points = fs * duration
    signals = np.random.randn(num_points) * 0.5  # Gaussian noise
    
    # Add simulated heartbeats
    for i in range(0, duration, 1):
        peak_pos = int(i * fs + fs/2)
        if peak_pos < num_points - 150:
            # Add QRS complex
            signals[peak_pos-10:peak_pos] = np.linspace(-0.2, 1.0, 10)
            signals[peak_pos:peak_pos+10] = np.linspace(1.0, -0.2, 10)
    
    # Simulate R-peak positions and labels
    r_peaks = [int(i * fs + fs/2) for i in range(duration)]
    beat_types = ['N', 'V', 'N', 'A', 'N', 'V', 'N', 'F', 'N', 'N']
    
    # Call beat segmentation function (assume implemented)
    # For demonstration, we'll create dummy beats
    beats = []
    labels = []
    for i, peak in enumerate(r_peaks):
        if peak - 100 >= 0 and peak + 199 < len(signals):
            beat = signals[peak-100:peak+200]
            beats.append(beat)
            labels.append(beat_types[i])
    beats = np.array(beats)
    
    # Visualize segmentation
    visualize_segmentation(signals, r_peaks, beats, labels, fs=fs)
