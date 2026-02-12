# Complete Architecture Documentation

This document provides comprehensive documentation of **ALL** modules in the Smooth Visualization of Delayed Satellite Imagery project, including client-side, server-side, model training, and data acquisition components.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Data Acquisition](#data-acquisition)
3. [Model Architecture & Training](#model-architecture--training)
4. [Server-Side Processing](#server-side-processing)
5. [Client-Side Processing](#client-side-processing)
6. [Deployment & Integration](#deployment--integration)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SATELLITE DATA SOURCES                        │
│              (MOSDAC WMS Service - INSAT-3R)                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                              │
│  • WMS GetMap API requests                                       │
│  • Bounding box specification                                    │
│  • Timestamp-based image retrieval                               │
└──────────────┬──────────────────────┬────────────────────────────┘
               │                      │
               │                      │
     ┌─────────▼────────┐   ┌────────▼──────────┐
     │  CLIENT-SIDE     │   │  SERVER-SIDE      │
     │  PROCESSING      │   │  PROCESSING       │
     │  (Browser)       │   │  (Flask API)      │
     └─────────┬────────┘   └────────┬──────────┘
               │                      │
               │                      │
     ┌─────────▼────────┐   ┌────────▼──────────┐
     │  TensorFlow.js   │   │  PyTorch          │
     │  SepConv Model   │   │  IFRNet Model     │
     └─────────┬────────┘   └────────┬──────────┘
               │                      │
               │                      │
     ┌─────────▼────────┐   ┌────────▼──────────┐
     │  Video Overlay   │   │  MP4 Video        │
     │  on Leaflet Map  │   │  Generation       │
     └──────────────────┘   └───────────────────┘
```

---

## Data Acquisition

### WMS (Web Map Service) Integration

#### Location
- Implemented in both client and server modules

#### Data Source
- **Provider**: MOSDAC (Meteorological & Oceanographic Satellite Data Archival Centre)
- **Satellite**: INSAT-3R (Indian National Satellite System)
- **Service Type**: OGC-compatible WMS
- **Data Product**: 3R_IMG (visible imagery)
- **Temporal Resolution**: 30-minute intervals

#### WMS Request Structure

**Base URL Pattern:**
```
https://mosdac.gov.in/live_data/wms/live3RL1BSTD1km/products/Insat3r/3R_IMG/
2024/{DATE}/3RIMG_{DATE}2024_{TIMESTAMP}_L1B_STD_V01R00.h5
```

**Query Parameters:**
- `SERVICE=WMS` - Web Map Service protocol
- `VERSION=1.3.0` - WMS version
- `REQUEST=GetMap` - Request type
- `FORMAT=image/png` - Output format
- `LAYERS=IMG_VIS` - Visible spectrum layer
- `COLORSCALERANGE=20,489` (client) or `46,538` (server) - Data value range
- `STYLES=boxfill/greyscale` - Rendering style
- `WIDTH=256&HEIGHT=256` - Image dimensions
- `CRS=EPSG:3857` - Web Mercator projection
- `BBOX={minX},{minY},{maxX},{maxY}` - Bounding box in Web Mercator

#### Data Acquisition Modules

**Client-Side (index.html, lines 42-99):**
- Uses Leaflet.js for map interaction
- Geographic coordinate to Web Mercator conversion
- User-selected bounding box
- CORS proxy for cross-origin requests

**Server-Side (app.py, lines 42-62):**
```python
def load_image(timestamps, bbox):
    """
    Fetches satellite images from WMS service
    
    Args:
        timestamps: List of time strings (e.g., ["0015", "0045"])
        bbox: Bounding box string in Web Mercator coordinates
        
    Returns:
        List of numpy arrays (BGR images)
    """
```

**Data Flow:**
1. User selects geographic area on map
2. Coordinates converted to EPSG:3857 (Web Mercator)
3. Bounding box formatted: `minX,minY,maxX,maxY`
4. WMS GetMap request constructed with bbox + timestamp
5. PNG image retrieved and decoded
6. Image stored as numpy array for processing

---

## Model Architecture & Training

### IFRNet (Intermediate Feature Refine Network)

#### Overview
IFRNet is the primary model used for **server-side** frame interpolation. It's a deep learning model specifically designed for video frame interpolation.

#### Model Components

**Location:** `/server/model.py`

#### Architecture Layers

**1. Encoder (Lines 51-76)**
```python
class Encoder(nn.Module):
    - pyramid1: 3→24 channels (stride 2)
    - pyramid2: 24→36 channels (stride 2)
    - pyramid3: 36→54 channels (stride 2)
    - pyramid4: 54→72 channels (stride 2)
```
Creates a feature pyramid with 4 resolution levels for multi-scale processing.

**2. Decoder Modules (Lines 79-144)**
- **Decoder4**: Processes lowest resolution (72×2 + 1 = 145 channels)
- **Decoder3**: Processes mid-low resolution with optical flow warping
- **Decoder2**: Processes mid-high resolution with optical flow warping
- **Decoder1**: Processes full resolution, outputs final frame + mask

**3. ResBlock (Lines 19-48)**
- Residual blocks with side channels
- PReLU activation
- Skip connections for gradient flow

#### Key Features

**Optical Flow Warping:**
- Estimates bidirectional flow (frame0→t, frame1→t)
- Warps input frames to target time
- Uses `utils.warp()` function (lines 14-22)

**Multi-Scale Refinement:**
- Processes at 4 different scales (1/16, 1/8, 1/4, 1/1)
- Upsamples and refines at each level
- Progressive flow refinement

**Adaptive Blending:**
- Learns blending mask (sigmoid activation)
- Combines warped frames: `mask * img0_warp + (1-mask) * img1_warp`
- Adds residual correction for fine details

#### Training Components

**Loss Functions** (`/server/loss.py`):

**1. Charbonnier L1 Loss (Lines 77-86)**
```python
class Charbonnier_L1:
    # Robust L1 loss: sqrt(diff^2 + epsilon)
    # Less sensitive to outliers than MSE
```

**2. Ternary Loss (Lines 9-39)**
```python
class Ternary:
    # Patch-based loss comparing local structure
    # Robust to illumination changes
    # Uses 7×7 patches by default
```

**3. Geometry Loss (Lines 42-74)**
```python
class Geometry:
    # Ensures spatial consistency in feature space
    # Preserves geometric structure
```

**4. Charbonnier Adaptive Loss (Lines 89-97)**
```python
class Charbonnier_Ada:
    # Adaptive loss with robust weighting
    # Used for optical flow supervision
```

#### Training Process

**Model Training** (Lines 206-254 in model.py):

1. **Forward Pass:**
   - Encode both input frames
   - Encode ground truth intermediate frame
   - Decode with flow estimation
   - Generate interpolated frame

2. **Loss Computation:**
   - `loss_rec`: Reconstruction loss (L1 + Ternary)
   - `loss_geo`: Geometry consistency loss
   - `loss_dis`: Flow distillation loss (if ground truth flow available)

3. **Optimizer:**
   - Configuration not specified in code
   - Externally configured during training

**Pre-trained Weights:**
- File: `IFRNet_S_Vimeo90K.pth`
- Model variant: IFRNet-S (Small version)
- Based on filename, likely trained on Vimeo90K dataset (a video frame interpolation benchmark)

#### Inference

**Inference Function** (Lines 161-203):
```python
def inference(img0, img1, embt, scale_factor=1.0):
    """
    Args:
        img0, img1: Input frames [B, 3, H, W], normalized [0, 1]
        embt: Time embedding (0.5 for middle frame)
        scale_factor: Optional downsampling for speed
        
    Returns:
        imgt_pred: Interpolated frame [B, 3, H, W]
    """
```

**Normalization:**
- Subtracts mean across both frames
- Mean is added back after prediction
- Ensures temporal consistency

---

### SepConv Model (Client-Side)

#### Overview
Separable Convolution model used for **browser-based** interpolation (TensorFlow.js).

#### Location
- Model definition: `/sepconv-model/model.json`
- Weights: `/sepconv-model/group1-shard1of1.bin`
- Training notebook: `/sepconv-model/sep-conv-implementation.ipynb`

#### Model Architecture

**Framework:** TensorFlow.js (converted from original implementation)

**Key Concept:**
- Learns separable convolution kernels for pixel synthesis
- Adaptive kernels based on local image content
- More lightweight than IFRNet for browser deployment

**Input:**
- Two RGB images [256, 256, 3]
- Normalized to [0, 1]

**Output:**
- Single interpolated RGB image [256, 256, 3]

**Model Hosting:**
- GitHub repository: `Latent-Space-Interpolation`
- Loaded dynamically in browser via CDN

#### Training

**Training Notebook:** `sep-conv-implementation.ipynb`
- Contains model training code
- Dataset preparation
- Conversion to TensorFlow.js format

---

## Server-Side Processing

### Flask API Server

**Location:** `/server/app.py`

#### Key Modules

**1. Image Fetching (`get_image`, lines 14-40)**
```python
def get_image(url):
    """
    Fetches image from URL using requests library
    
    Process:
    1. HTTP GET request with streaming
    2. Convert response bytes to numpy array
    3. Decode with OpenCV (BGR format)
    4. Return as numpy array
    
    Error Handling:
    - HTTP errors via raise_for_status()
    - Decode failures return None
    """
```

**2. Batch Image Loading (`load_image`, lines 42-62)**
```python
def load_image(timestamps, bbox):
    """
    Loads multiple satellite images for given timestamps
    
    Args:
        timestamps: List of time strings
        bbox: URL-encoded bounding box
        
    Returns:
        List of BGR images (numpy arrays)
        
    Handles:
    - Failed downloads (appends None)
    - Constructs full WMS URLs
    - Iterates through all timestamps
    """
```

**3. Interpolation Endpoint (`/interpolate/`, lines 65-107)**

**HTTP Method:** POST

**Request Body (JSON):**
```json
{
    "timestamp": ["0015", "0045", "0115"],
    "bbox": "8734567.12,2345678.90,8745678.23,2356789.01",
    "req_id": "unique-uuid-string"
}
```

**Response Body (JSON):**
```json
{
    "message": "Video generated successfully",
    "video_path": "static/unique-uuid-string.mp4"
}
```

**Processing Pipeline:**
1. Validate input (minimum 2 timestamps)
2. Load images from WMS service
3. For each consecutive pair of images:
   - Run interpolation (`interpolate(img1, img2)`)
   - Generate 30 frames per pair
   - Collect all frames
4. Save frames as MP4 video (30 FPS)
5. Return video path

**4. Health Check Endpoint (`/hello`, lines 110-112)**
```python
@app.route('/hello', methods=['GET'])
def hello():
    return 'hi'  # Simple health check
```

#### Dependencies

**Key Libraries** (from `requirements.txt`):
- `Flask==3.1.0` - Web framework
- `Flask-Cors==5.0.0` - CORS support for browser access
- `torch==2.5.1+cpu` - PyTorch (CPU version)
- `opencv-python==4.10.0.84` - Image processing
- `numpy==2.2.0` - Numerical operations
- `imageio==2.36.1` - Video file I/O
- `imageio-ffmpeg==0.5.1` - FFmpeg integration
- `requests==2.32.3` - HTTP requests

---

### Interpolation Generator

**Location:** `/server/generator.py`

#### Key Functions

**1. Model Initialization (Lines 9-10)**
```python
model = Model().eval()
model.load_state_dict(torch.load('IFRNet_S_Vimeo90K.pth', weights_only=True))
```
- Loads pre-trained IFRNet model
- Sets to evaluation mode (disables dropout, batch norm updates)
- CPU-only inference

**2. Single Frame Interpolation (`interpolate_frame`, lines 13-22)**
```python
def interpolate_frame(model, img0, img1):
    """
    Generates single intermediate frame
    
    Input Processing:
    - Convert BGR (OpenCV) to RGB
    - Transpose to [C, H, W] format
    - Normalize to [0, 1] range
    - Add batch dimension
    - Create time embedding (0.5 for middle)
    
    Output:
    - Numpy array [H, W, 3], range [0, 255], uint8
    """
```

**3. Hierarchical Interpolation (`interpolate`, lines 26-82)**

**Interpolation Tree Structure:**
```python
interpolation_tree = {
    15: (1, 30),   # Frame 15 from frames 1 and 30
    7: (1, 15),    # Frame 7 from frames 1 and 15
    23: (15, 30),  # Frame 23 from frames 15 and 30
    # ... 28 total intermediate frames
}
```

**Algorithm:**
1. Initialize with input images as frames 1 and 30
2. Use recursive memoization:
   ```python
   def get_interpolated_image(index):
       if index in images:
           return images[index]  # Already computed
       
       # Get parent frames from tree
       img_a_index, img_b_index = interpolation_tree[index]
       img_a = get_interpolated_image(img_a_index)
       img_b = get_interpolated_image(img_b_index)
       
       # Compute and cache
       images[index] = interpolate_frame(model, img_a, img_b)
       return images[index]
   ```
3. Generate all 30 frames (2 original + 28 interpolated)
4. Return dictionary: `{1: img1, 2: img2, ..., 30: img30}`

**Optimization:**
- Memoization prevents redundant computation
- Binary subdivision minimizes error accumulation
- Hierarchical approach ensures temporal coherence

---

### Utility Functions

**Location:** `/server/utils.py`

#### Core Functions

**1. Optical Flow Warping (`warp`, lines 14-22)**
```python
def warp(img, flow):
    """
    Warps image according to optical flow field
    
    Args:
        img: [B, C, H, W] image tensor
        flow: [B, 2, H, W] optical flow (dx, dy)
        
    Returns:
        Warped image [B, C, H, W]
        
    Implementation:
    - Creates normalized coordinate grid [-1, 1]
    - Adds flow offsets
    - Uses grid_sample for bilinear interpolation
    - Border padding mode for out-of-bounds pixels
    """
```

**2. Robust Weighting (`get_robust_weight`, lines 25-28)**
```python
def get_robust_weight(flow_pred, flow_gt, beta):
    """
    Computes robust weight for flow supervision
    
    Used in training to downweight outliers
    Weight = exp(-beta * EPE)
    EPE = Endpoint Error between predicted and GT flow
    """
```

**3. File I/O Functions (lines 48-228)**
- `read()` - Universal file reader (images, flow, PFM, etc.)
- `write()` - Universal file writer
- `readImage()`, `writeImage()` - Image I/O with PIL/imageio
- `readFlow()`, `writeFlow()` - Optical flow file handling
- `readPFM()`, `writePFM()` - PFM format support

**4. AverageMeter (lines 31-45)**
```python
class AverageMeter:
    """
    Utility for tracking running averages
    Used in training for metrics tracking
    """
```

---

## Client-Side Processing

### Overview
Complete browser-based implementation using JavaScript and TensorFlow.js.

**Location:** `/frontend/client-processing/`

### Key Modules

For detailed client-side documentation, see [CLIENT_SIDE_MODULES.md](CLIENT_SIDE_MODULES.md).

**Summary:**

**1. Image Processing (`main.js`)**
- `loadImageAsTensor()` - Loads images from URLs
- `imgToTensor()` - Preprocesses for model input

**2. Interpolation Engine (`main.js`)**
- TensorFlow.js SepConv model
- Hierarchical frame generation (same tree as server)
- 28 intermediate frames from 2 inputs

**3. Video Generation (`main.js`)**
- Canvas rendering at 30 FPS
- MediaRecorder API
- WebM output format

**4. Map Integration (`index.html`)**
- Leaflet.js for interactive maps
- WMS layer display
- Video overlay on geographic bounds
- Playback controls

---

## Deployment & Integration

### Two Processing Modes

#### Mode 1: Client-Side Processing
**Path:** `/frontend/client-processing/`

**Advantages:**
- No server required
- Instant processing (no network latency)
- Privacy (data stays in browser)

**Limitations:**
- Limited by browser performance
- Smaller model (SepConv vs IFRNet)
- Single device compute resources

**Use Case:**
- Quick previews
- Low-latency applications
- Offline capability (with service workers)

#### Mode 2: Server-Side Processing
**Path:** `/frontend/server-processing/` + `/server/`

**Advantages:**
- More powerful model (IFRNet)
- GPU acceleration possible
- Better quality interpolation

**Limitations:**
- Requires server infrastructure
- Network latency
- Server compute costs

**Use Case:**
- High-quality video generation
- Batch processing
- Resource-constrained clients

### Integration Points

**1. WMS Service Integration**
- Both modes fetch from same MOSDAC WMS
- Common bounding box format (EPSG:3857)
- Common timestamp format (HHMM)

**2. Video Output**
- Client: WebM Blob URLs for in-browser playback
- Server: MP4 files served from `/static/` directory

**3. User Interface**
- Leaflet.js map (both modes)
- Bounding box selection
- Timestamp selection
- Video overlay with playback controls

### System Requirements

**Client-Side:**
- Modern browser (Chrome, Firefox, Edge)
- WebGL support for TensorFlow.js
- MediaRecorder API support
- Minimum 4GB RAM recommended

**Server-Side:**
- Python 3.8+
- Flask web server
- PyTorch (CPU or GPU)
- FFmpeg for video encoding
- 8GB+ RAM recommended
- Optional: CUDA GPU for faster processing

---

## Complete Data Flow

### End-to-End Pipeline

```
1. USER INTERACTION
   └─> User selects area on Leaflet map
   └─> Selects start/end timestamps

2. DATA ACQUISITION
   └─> Convert lat/lon to Web Mercator (EPSG:3857)
   └─> Format bounding box: minX,minY,maxX,maxY
   └─> Construct WMS GetMap URLs for each timestamp
   └─> Fetch PNG images (256×256)

3. IMAGE PROCESSING
   ┌─> CLIENT PATH:
   │   └─> Load images as tensors (TensorFlow.js)
   │   └─> Normalize to [0, 1]
   │   └─> Run SepConv model inference
   │
   └─> SERVER PATH:
       └─> POST to /interpolate/ endpoint
       └─> Load images as numpy arrays (OpenCV)
       └─> Normalize and convert to PyTorch tensors
       └─> Run IFRNet model inference

4. FRAME INTERPOLATION
   └─> Apply hierarchical binary interpolation
   └─> Generate 28 intermediate frames per image pair
   └─> Total frames = (num_pairs × 28) + num_images

5. VIDEO GENERATION
   ┌─> CLIENT PATH:
   │   └─> Render frames to Canvas sequentially
   │   └─> Record with MediaRecorder API
   │   └─> Output WebM Blob URL
   │
   └─> SERVER PATH:
       └─> Collect all frames as numpy arrays
       └─> Save as MP4 with imageio (30 FPS)
       └─> Return video file path

6. VISUALIZATION
   └─> Create L.videoOverlay on Leaflet map
   └─> Set geographic bounds (lat/lon)
   └─> Enable playback controls (play/pause, frame step)
   └─> Display on map with proper geo-registration
```

---

## Performance Characteristics

### Processing Time Estimates

**Client-Side (TensorFlow.js):**
- Model loading: 2-5 seconds (first time)
- Per-frame interpolation: ~0.5-2 seconds (CPU)
- 28 frames: ~15-60 seconds total
- Video encoding: ~2-5 seconds

**Server-Side (PyTorch):**
- Model loading: 1-2 seconds (first time)
- Per-frame interpolation: ~0.1-0.5 seconds (CPU), ~0.01-0.05 seconds (GPU)
- 28 frames: ~3-15 seconds (CPU), ~0.3-1.5 seconds (GPU)
- Video encoding: ~1-3 seconds
- Network latency: variable

### Memory Requirements

**Client:**
- Model: ~2-5 MB
- Input images: ~0.5 MB each
- Intermediate tensors: ~50-100 MB
- Video buffer: ~5-20 MB

**Server:**
- Model: ~5-10 MB
- Per-request memory: ~200-500 MB
- Concurrent requests limited by RAM

---

## Summary

This project implements a complete pipeline for smooth satellite imagery visualization:

1. ✅ **Data Acquisition**: WMS service integration for INSAT-3R satellite data
2. ✅ **Model Training**: Pre-trained IFRNet and SepConv models for frame interpolation
3. ✅ **Server Processing**: Flask API with PyTorch for high-quality interpolation
4. ✅ **Client Processing**: TensorFlow.js for browser-based interpolation
5. ✅ **Visualization**: Leaflet.js with geo-registered video overlays
6. ✅ **Dual Mode**: Flexible deployment (client-only or client-server)

**Key Innovation:** Hierarchical binary interpolation generates smooth 30-frame sequences from just 2 satellite images, enabling fluid visualization of delayed satellite data.
