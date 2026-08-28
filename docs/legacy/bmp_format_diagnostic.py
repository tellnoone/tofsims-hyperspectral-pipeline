import numpy as np
import struct
import matplotlib.pyplot as plt
# Raw dataset file path
filepath = r"Data/raw/YOUR_FILE.bmp"  # point this at a file on your machine

# Open file, skip the 54-byte BMP header, grab the raw pixel bytes
with open(filepath, 'rb') as f:
    header = f.read(54)
    f.seek(54)
    raw = f.read()

H, W = 640, 640
total_pixels = H * W

print(f"Data bytes: {len(raw)}")
print(f"Expected:   {total_pixels * 8}")
print(f"First 16 bytes hex: {raw[:16].hex()}")
print("=" * 60)

# Try every valid way to split 8 bytes per pixel
# We only use shapes that are guaranteed to fit
tests = {
    "uint64  (1 channel)":  np.frombuffer(raw, dtype=np.uint64).reshape(H, W),
    "int64   (1 channel)":  np.frombuffer(raw, dtype=np.int64).reshape(H, W),
    "uint32  (2 channels)": np.frombuffer(raw, dtype=np.uint32).reshape(H, W, 2),
    "int32   (2 channels)": np.frombuffer(raw, dtype=np.int32).reshape(H, W, 2),
    "float32 (2 channels)": np.frombuffer(raw, dtype=np.float32).reshape(H, W, 2),
    "uint16  (4 channels)": np.frombuffer(raw, dtype=np.uint16).reshape(H, W, 4),
    "int16   (4 channels)": np.frombuffer(raw, dtype=np.int16).reshape(H, W, 4),    
}

# Create a big figure to preview every option
fig, axes = plt.subplots(len(tests), 3, figsize=(14, 3.5 * len(tests)))
fig.suptitle("BMP Format Diagnostic — looking for the fly", fontsize=14, fontweight='bold')

for row, (name, arr) in enumerate(tests.items()):
    print(f"\n>>> {name}")
    
    if arr.ndim == 2:
        # Single channel
        data = arr
        print(f"    Shape: {data.shape} | dtype: {data.dtype}")
        print(f"    Zeros: {np.count_nonzero(data == 0)} / {data.size}")
        print(f"    Min: {data.min()} | Max: {data.max()} | Mean: {data.mean():.2f}")
        
        # Plot
        axes[row, 0].imshow(data, cmap='viridis')
        axes[row, 0].set_title(f"{name}\nmin={data.min()}, max={data.max()}")
        axes[row, 0].axis('off')
        
        axes[row, 1].hist(data.flatten(), bins=100, color='steelblue')
        axes[row, 1].set_title("Histogram (log scale)")
        axes[row, 1].set_yscale('log')
        
        axes[row, 2].imshow(np.log1p(data), cmap='hot')
        axes[row, 2].set_title("Log(1 + data) — reveals faint structure")
        axes[row, 2].axis('off')
        
    else:
        # Multi-channel: test each channel separately
        best_ch = 0
        best_score = -1
        
        for ch in range(arr.shape[2]):
            ch_data = arr[:, :, ch]
            nz = np.count_nonzero(ch_data == 0)
            score = nz  # More zeros = more likely real data
            
            print(f"    Channel {ch}: zeros={nz}, max={ch_data.max()}, mean={ch_data.mean():.2f}")
            
            if score > best_score:
                best_score = score
                best_ch = ch
        
        # Plot the best channel (the one with the most zeros)
        data = arr[:, :, best_ch]
        print(f"    *** Plotting Channel {best_ch} (most zeros) ***")
        
        axes[row, 0].imshow(data, cmap='viridis')
        axes[row, 0].set_title(f"{name}\nCh{best_ch}: min={data.min()}, max={data.max()}")
        axes[row, 0].axis('off')
        
        axes[row, 1].hist(data.flatten(), bins=100, color='steelblue')
        axes[row, 1].set_title("Histogram (log scale)")
        axes[row, 1].set_yscale('log')
        
        axes[row, 2].imshow(np.log1p(data), cmap='hot')
        axes[row, 2].set_title("Log(1 + data)")
        axes[row, 2].axis('off')

plt.tight_layout()
plt.savefig("format_diagnostic.png", dpi=150)
plt.show()

print("\n" + "=" * 60)
print("INSTRUCTIONS:")
print("Look at the saved image 'format_diagnostic.png'.")
print("The correct format is the row where:")
print("  1. You can SEE the fly shape in the left or right panel")
print("  2. The histogram has a huge spike at zero")
print("  3. The max value is reasonable (hundreds to thousands, not millions)")
print("=" * 60)

#--------------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt

filepath = r"Data/raw/YOUR_FILE.bmp"  # point this at a file on your machine

# Load raw image payload
with open(filepath, 'rb') as f:
    f.seek(54)
    raw = f.read()

# Reshape into 4 channels
data = np.frombuffer(raw, dtype=np.uint16).reshape(640, 640, 4)

# Extract Channel 2 (Primary Image Data)
fly_image = data[:, :, 2]

# Display
plt.figure(figsize=(8, 8))
plt.imshow(fly_image, cmap='viridis')
plt.colorbar(label='Intensity Count')
plt.title("Fossil Fly — ToF-SIMS Signal (Channel 2)", fontsize=12, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig("fossil_fly_channel2.png", dpi=300)
plt.show()