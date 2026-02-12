# Client-Side Video Generation and Interpolation Modules

This document describes the client-side modules involved in video generation and interpolation for the Smooth Visualization of Delayed Satellite Imagery project. **This excludes server-side processing components.**

## Overview

The client-side processing is implemented entirely in the browser using JavaScript and TensorFlow.js, allowing for frame interpolation and video generation without requiring server-side computation.

## Location

All client-side modules are located in:
```
/frontend/client-processing/
```

## Core Modules

### 1. **main.js** - Primary Client-Side Processing Module

This is the main JavaScript file that handles all client-side video generation and interpolation logic.

#### Key Components:

#### A. Image Loading and Tensor Conversion
- **`loadImageAsTensor(url)`**: Asynchronously loads an image from a URL and converts it to a TensorFlow.js tensor
  - Uses HTML5 Image API with CORS support
  - Converts pixels to tensor format using `tf.browser.fromPixels()`
  
- **`imgToTensor(url1, url2)`**: Processes two images for interpolation
  - Normalizes pixel values by dividing by 255.0
  - Expands dimensions for batch processing
  - Returns two preprocessed tensors ready for the model

#### B. Interpolation Engine
- **`interpolation_tree`**: A lookup table defining the hierarchical interpolation structure
  - Generates 28 intermediate frames from 2 input frames (30 frames total)
  - Uses hierarchical binary subdivision approach that recursively halves intervals
  - Maps target frame indices to their parent frame pairs

- **`interpolate(url1, url2)`**: Main interpolation function
  - Recursively generates intermediate frames between two input images
  - Uses memoization to avoid redundant computations
  - Implements hierarchical frame generation using the interpolation tree
  - Returns an object containing all interpolated frames (indices 1-30)

- **`get_interpolated_image(index)`**: Recursive helper function
  - Computes interpolated frames on-demand
  - Uses parent frames from the interpolation tree
  - Caches results to optimize performance

#### C. TensorFlow.js Model Integration
- **Model Loading**: Uses TensorFlow.js to load the pre-trained SepConv model
  - Model URL: Hosted on GitHub (Latent-Space-Interpolation repository)
  - Model format: TensorFlow.js Graph Model (model.json)
  - Loads asynchronously on page initialization

- **`predict(img1, img2)`**: Inference function
  - Takes two image tensors as input
  - Runs the loaded SepConv model to generate intermediate frame
  - Returns interpolated frame tensor

#### D. Post-Processing
- **`postprocess(img)`**: Prepares model output for display
  - Removes batch dimension using squeeze
  - Clips values to valid range [0, 1]
  - Scales back to 0-255 pixel range
  - Converts to integer format

- **`call(url1, url2)`**: Complete processing pipeline
  - Orchestrates interpolation and post-processing
  - Processes all generated frames
  - Returns array of displayable frame tensors

#### E. Video Generation
- **`tensorToVideo(url1, url2)`**: Converts interpolated frames to browser-playable video
  - Creates an HTML5 canvas (256x256)
  - Uses MediaRecorder API to record canvas stream
  - Renders frames sequentially at 30 FPS
  - Outputs WebM video format
  - Returns a Blob URL for video playback
  - Uses `requestAnimationFrame` for smooth rendering
  - Implements proper memory management with tensor disposal

### 2. **index.html** - User Interface and Integration

While primarily a UI file, it contains important client-side integration logic:

#### Key Integration Components:

#### A. External Libraries
- **Leaflet.js (v1.9.4)**: Interactive map display
  - Renders WMS layers
  - Handles video overlays on map
  - Provides geographic coordinate system support

- **TensorFlow.js**: Machine learning framework
  - Loaded from CDN: `@tensorflow/tfjs`
  - Enables browser-based deep learning inference
  - No server-side computation required

- **Boxicons**: UI icons for video controls

#### B. Video Overlay Management
- **Rectangle Selection**: User draws bounding box on map
- **WMS Image Fetching**: Retrieves satellite images for selected area
- **Video Overlay Creation**: Uses `L.videoOverlay()` to display generated video on map
- **Geographic Projection**: Converts lat/lon to Web Mercator (EPSG:3857)

#### C. Video Playback Controls
- **Play/Pause**: Toggle video playback
- **Frame Navigation**: Step forward/backward by individual frames
- **Frame Duration Calculation**: Based on 30 FPS (1/30 second per frame)

### 3. **style.css** - Visual Styling

Provides styling for the UI components but does not contain processing logic.

## Technical Dependencies

### Required Browser APIs:
1. **Canvas API**: For rendering frames
2. **MediaRecorder API**: For video recording
3. **Blob API**: For video URL generation
4. **Web Mercator Projection**: For geographic calculations

### External Libraries:
1. **TensorFlow.js**: ML model inference engine
2. **Leaflet.js**: Map rendering and overlay management

## Data Flow

```
1. User selects area on map
2. Fetch two satellite images from WMS service (via CORS proxy)
3. Load images as tensors (loadImageAsTensor)
4. Normalize and prepare tensors (imgToTensor)
5. Run interpolation to generate 30 frames (interpolate)
6. Post-process each frame (postprocess)
7. Render frames to canvas sequentially
8. Record canvas stream as WebM video (tensorToVideo)
9. Create video overlay on map
10. Enable playback controls
```

## Interpolation Algorithm

The client uses a **hierarchical binary interpolation approach**:

1. Start with 2 frames (frame 1 and frame 30)
2. Generate middle frame (frame 15) first
3. Recursively generate intermediate frames:
   - Between 1-15: generate frame 7
   - Between 15-30: generate frame 23
   - Continue subdividing until all 30 frames are generated

This approach minimizes error accumulation compared to sequential interpolation.

## Performance Considerations

- **Memory Management**: Uses `tf.tidy()` to prevent memory leaks
- **Tensor Disposal**: Explicitly disposes of tensors after use
- **Memoization**: Caches computed frames to avoid redundant calculations
- **Asynchronous Processing**: Uses async/await for non-blocking operations
- **Frame Rate**: Fixed at 30 FPS for smooth playback

## Model Information

- **Model Type**: Separable Convolution for Video Frame Interpolation
- **Framework**: TensorFlow.js (converted from original model)
- **Input**: Two RGB images (normalized 0-1)
- **Output**: Single interpolated RGB image
- **Architecture**: Based on "Adaptive Separable Convolution for Video Frame Interpolation" paper

## Output Format

- **Video Format**: WebM
- **Resolution**: 256x256 pixels
- **Frame Rate**: 30 FPS
- **Frame Count**: 30 frames (from 2 input images)
- **Color Space**: RGB

## Browser Compatibility

Requires modern browsers with support for:
- ES6+ JavaScript features (async/await, Promise, etc.)
- TensorFlow.js WebGL backend
- MediaRecorder API
- Canvas API
- CORS-enabled image loading

## Summary

The client-side implementation provides a complete, browser-based solution for:
1. ✅ Loading satellite imagery
2. ✅ Deep learning-based frame interpolation
3. ✅ Video generation from interpolated frames
4. ✅ Geographic video overlay on interactive maps
5. ✅ Video playback controls

All processing happens in the browser using TensorFlow.js, eliminating the need for server-side computation for the interpolation and video generation tasks.
