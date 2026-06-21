# openpilot Model Beginner Codewalk

Date started: 2026-06-21

Repo:

`commaai/openpilot`

This is a running note for learning openpilot's driving model from scratch. It avoids car ports, OBD, panda, CAN, message-process architecture, and hardware integration unless they are necessary for understanding the model.

## 0. The Core Mental Model

The openpilot driving model is best understood as:

```text
camera images
  + short image history
  + learned feature history
  + small driving context
  -> neural network
  -> future motion plan
  -> desired curvature and acceleration
```

It does not directly output:

```text
steering wheel angle
gas pedal percentage
brake pedal percentage
```

Instead, it outputs a future plan. Runtime code later turns that plan into:

```text
desired curvature
desired acceleration
should stop
```

The repo describes this model as a:

```text
Driving Model (vision model + temporal policy model)
```

Code reference:

[models/README.md](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/models/README.md#L4)

Meaning:

```text
vision model:
  understands the current images

temporal policy model:
  uses current understanding + recent history to predict future driving motion
```

## 1. The Main Model Files

The driving model files live here:

`selfdrive/modeld/models`

Important files:

```text
driving_supercombo.onnx      normal driving model
big_driving_supercombo.onnx  larger driving model variant
dmonitoring_model.onnx       driver monitoring model, separate topic
README.md                    model input/output explanation
```

The repo recommends Netron to inspect ONNX graphs:

[models/README.md](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/models/README.md#L1)

Netron:

[https://netron.app](https://netron.app)

## 2. What Is ONNX?

ONNX is a saved neural-network file format.

Think:

```text
training framework / model code
  -> export
  -> ONNX file
```

An ONNX file contains:

```text
model graph
trained weights
input names and shapes
output names and shapes
metadata
```

It usually does not contain:

```text
training data
optimizer state
training loop
full loss setup
why the model learned what it learned
```

For an engineer analyzing a deployed ONNX model, the useful questions are:

```text
What are the inputs?
What are the outputs?
What are the main layer/operator families?
How does runtime code preprocess inputs?
How does runtime code interpret outputs?
```

For openpilot, the ONNX is not the whole story. The runtime preprocessing and output parsing are equally important.

## 3. The Model Inputs

The current model contract includes:

```text
img
big_img
features_buffer
desire_pulse
traffic_convention
action_t
```

The repo README describes the visual inputs as:

```text
image stream
wide image stream
```

Code reference:

[models/README.md image stream](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/models/README.md#L6)

[models/README.md wide image stream](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/models/README.md#L12)

Practical mapping:

```text
image stream      -> img
wide image stream -> big_img
```

The model README says each stream uses two consecutive images:

[models/README.md two images](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/models/README.md#L7)

So:

```text
img     = two recent packed frames from one visual stream
big_img = two recent packed frames from another/wider visual stream
```

## 4. What Is `big_img`?

`big_img` is not "a bigger tensor" in this exported model contract.

Both visual inputs have the same packed tensor shape:

```text
img     = (1, 12, 128, 256)
big_img = (1, 12, 128, 256)
```

Simple mental model:

```text
img:
  normal road visual stream

big_img:
  wide / alternate road visual stream
```

Together they give the model more visual context.

## 5. YUV, Channels, And Why There Are 6 Channels

RGB image:

```text
R = red
G = green
B = blue
```

YUV image:

```text
Y = brightness / luma
U = color difference channel
V = color difference channel
```

YUV is common in camera/video systems because brightness detail matters a lot, and color can often be stored more compactly.

The model README says each image is represented as YUV420 with 6 channels:

[models/README.md YUV420 channels](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/models/README.md#L8)

The 6 channels are:

```text
Y split into 4 smaller brightness sheets
U as channel 5
V as channel 6
```

The README defines the exact Y splits:

[models/README.md Y splits](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/models/README.md#L9)

Expanded:

```text
Y[::2, ::2]   -> brightness sheet 1
Y[::2, 1::2]  -> brightness sheet 2
Y[1::2, ::2]  -> brightness sheet 3
Y[1::2, 1::2] -> brightness sheet 4
U             -> color sheet 5
V             -> color sheet 6
```

The beginner-friendly way to say this:

```text
channel = one image sheet/layer

1 frame = 6 sheets
2 frames = 12 sheets
```

That is why the visual tensor has 12 channels.

## 6. What Is NV12?

NV12 is a specific memory layout for YUV image data.

It stores:

```text
Y plane first
then interleaved UV plane
```

Picture:

```text
YYYYYYYYYYYYYYYY
YYYYYYYYYYYYYYYY
YYYYYYYYYYYYYYYY

UVUVUVUVUVUVUVUV
UVUVUVUVUVUVUVUV
```

Meaning:

```text
Y:
  brightness for pixels

UV:
  color information, lower resolution, U and V alternating
```

openpilot receives camera buffers in this kind of efficient video format, then prepares them for the model.

## 7. Model Timing Constants

The constants file defines the model's time and history setup:

[constants.py ModelConstants](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/constants.py#L6)

Important values:

[IDX_N](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/constants.py#L8)

[MODEL_RUN_FREQ and MODEL_CONTEXT_FREQ](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/constants.py#L16)

[FEATURE_LEN](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/constants.py#L20)

Key translation:

```text
IDX_N = 33
  predict 33 future points

MODEL_RUN_FREQ = 20
  model runs around 20 times per second

MODEL_CONTEXT_FREQ = 5
  history/context is sampled at around 5 Hz

FEATURE_LEN = 512
  one learned hidden/memory feature has 512 numbers
```

## 8. The Future Plan

The model predicts a future plan.

The plan has:

```text
33 future time points
15 values per future time point
```

The 15 values are defined here:

[Plan slices](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/constants.py#L67)

Code:

```python
class Plan:
  POSITION = slice(0, 3)
  VELOCITY = slice(3, 6)
  ACCELERATION = slice(6, 9)
  T_FROM_CURRENT_EULER = slice(9, 12)
  ORIENTATION_RATE = slice(12, 15)
```

Translation:

```text
position:
  x, y, z

velocity:
  x, y, z

acceleration:
  x, y, z

orientation:
  3 orientation values

orientation rate:
  3 orientation-rate values
```

The plan is later converted into:

```text
desired curvature
desired acceleration
```

## 9. Real Camera vs Virtual Model Camera

There are two ideas:

```text
real camera:
  actual camera frame from the device

virtual model camera:
  imaginary standard camera view the neural network expects
```

The model was trained expecting a particular camera geometry. So before pixels go into the network, openpilot warps the real camera image into the virtual model camera view.

The virtual model camera is defined in:

[common/transformations/model.py](https://github.com/commaai/openpilot/blob/master/common/transformations/model.py#L9)

Important code:

```python
MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_CY = 47.6

medmodel_fl = 910.0
medmodel_intrinsics = np.array([
  [medmodel_fl,  0.0,  0.5 * MEDMODEL_INPUT_SIZE[0]],
  [0.0,  medmodel_fl,                   MEDMODEL_CY],
  [0.0,  0.0,                                   1.0]])
```

Code reference:

[MEDMODEL_INPUT_SIZE](https://github.com/commaai/openpilot/blob/master/common/transformations/model.py#L10)

[MEDMODEL_CY](https://github.com/commaai/openpilot/blob/master/common/transformations/model.py#L12)

[medmodel_fl and intrinsics](https://github.com/commaai/openpilot/blob/master/common/transformations/model.py#L14)

Translation:

```text
MEDMODEL_INPUT_SIZE:
  model image size, 512 wide x 256 high

medmodel_fl:
  focal length of the virtual model camera, measured in pixels

MEDMODEL_CY:
  vertical principal point / center point of the virtual model camera
```

The intrinsics matrix has the common camera form:

```text
[ fx   0  cx ]
[  0  fy  cy ]
[  0   0   1 ]
```

For this virtual model camera:

```text
fx = 910
fy = 910
cx = 256
cy = 47.6
```

Pixel coordinates are measured from the top-left of the image:

```text
(0, 0)       top-left
x increases  to the right
y increases  downward
```

So:

```text
cx = 256:
  horizontal center of a 512-wide image

cy = 47.6:
  vertical principal point near the top of the 256-high image
```

Focal length intuition:

```text
lower focal length:
  wider-angle view

higher focal length:
  narrower / more zoomed-in view
```

## 10. What Is The Warp Matrix?

The warp matrix is a coordinate translator.

It answers:

```text
For this pixel in the virtual model image,
which pixel in the real camera image should I sample?
```

Example:

```text
model image pixel:
  (100, 50)

warp matrix maps it to real camera pixel:
  (372, 220)

openpilot samples/copies that source pixel value.
```

The warp matrix is computed here:

[get_warp_matrix](https://github.com/commaai/openpilot/blob/master/common/transformations/model.py#L64)

Code:

```python
def get_warp_matrix(device_from_calib_euler: np.ndarray, intrinsics: np.ndarray, bigmodel_frame: bool = False) -> np.ndarray:
  calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
  device_from_calib = rot_from_euler(device_from_calib_euler)
  camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
  warp_matrix: np.ndarray = camera_from_calib @ calib_from_model
  return warp_matrix
```

Plain English:

```text
1. Start with the virtual model camera geometry.
2. Use calibration to understand how the device/camera is rotated.
3. Use the real camera intrinsics.
4. Build one 3x3 matrix that maps model-view coordinates to real-camera coordinates.
```

The purpose:

```text
make the real camera frame look like the standard model-camera view
before the neural network sees it
```

## 11. Why Use The Inverse Warp?

When creating the model image, openpilot usually thinks backward:

```text
for every output/model pixel:
  find where it came from in the real image
```

This avoids holes.

Bad way:

```text
for every real camera pixel:
  throw it forward into the model image
```

That can leave empty pixels in the output.

Better way:

```text
for every model pixel:
  ask the matrix where to sample in the real camera image
```

This is the idea behind the inverse matrix input named `M_inv`.

## 12. `warp_perspective_tinygrad()` Inputs

The actual pixel sampling function starts here:

[compile_modeld.py warp_perspective_tinygrad](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L50)

Code:

```python
def warp_perspective_tinygrad(src_flat, M_inv, dst_shape, src_shape, stride_pad, border_fill_val=None):
```

Inputs:

```text
src_flat:
  the real camera image plane flattened into one long array

M_inv:
  inverse warp matrix
  maps output/model pixels back to real-camera source pixels

dst_shape:
  size of the output/model image we are creating

src_shape:
  size of the source/real camera image plane

stride_pad:
  extra padding bytes at the end of camera-buffer rows

border_fill_val:
  optional value used when the mapped source pixel is outside the real image
```

Beginner translation:

```text
src_flat = real image data
M_inv    = recipe for where to sample from
dst      = model image size
src      = real image size
```

## 13. What `warp_perspective_tinygrad()` Does

Inside the function, it creates output pixel coordinates:

[x and y grid](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L54)

Code:

```python
x = Tensor.arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst).reshape(-1)
y = Tensor.arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst).reshape(-1)
```

Meaning:

```text
create all pixel coordinates in the model image
```

Then it applies the inverse 3x3 matrix:

[matrix application](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L57)

Code:

```python
src_x = M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2]
src_y = M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2]
src_w = M_inv[2, 0] * x + M_inv[2, 1] * y + M_inv[2, 2]
```

Then it divides by `w`:

[perspective divide](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L62)

Code:

```python
src_x = src_x / src_w
src_y = src_y / src_w
```

Meaning:

```text
convert perspective/projective coordinates back into normal image x,y coordinates
```

Then it rounds and clips:

[round and clip](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L65)

Code:

```python
x_round = Tensor.round(src_x)
y_round = Tensor.round(src_y)
x_nn_clipped = x_round.clip(0, w_src - 1).cast('int')
y_nn_clipped = y_round.clip(0, h_src - 1).cast('int')
```

Meaning:

```text
nearest-neighbor sample from a valid source pixel
```

Then it samples:

[source sample](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L69)

Code:

```python
idx = y_nn_clipped * (w_src + stride_pad) + x_nn_clipped
sampled = src_flat[idx]
```

Meaning:

```text
turn source x,y into a flat memory index
read that pixel value
```

## 14. Preparing A Full Frame

The function that prepares a frame starts here:

[make_frame_prepare](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L92)

It handles NV12/YUV layout.

Inside it, openpilot separates the UV plane:

[UV plane handling](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L100)

Code:

```python
uv = input_frame[uv_offset:uv_offset + uv_height * stride].reshape(uv_height, stride)
```

Then it warps Y:

[warp Y](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L103)

It warps U:

[warp U](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L106)

It warps V:

[warp V](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L109)

Then it joins them:

[join YUV and pack tensor](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L112)

Code:

```python
yuv = y.cat(u).cat(v).reshape((model_h * 3 // 2, model_w))
tensor = frames_to_tensor(yuv)
return tensor
```

## 15. Packing YUV Into Model Tensor

The packing function starts here:

[frames_to_tensor](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L80)

Code:

```python
def frames_to_tensor(frames):
  H = (frames.shape[0] * 2) // 3
  W = frames.shape[1]
  in_img1 = Tensor.cat(frames[0:H:2, 0::2],
                       frames[1:H:2, 0::2],
                       frames[0:H:2, 1::2],
                       frames[1:H:2, 1::2],
                       frames[H:H+H//4].reshape((H//2, W//2)),
                       frames[H+H//4:H+H//2].reshape((H//2, W//2)), dim=0).reshape((6, H//2, W//2))
  return in_img1
```

Translation:

```text
take warped YUV
split brightness Y into 4 sheets
add U
add V
return 6-channel tensor
```

## 16. The Full Preprocessing Chain

The full preprocessing chain is:

```text
real camera NV12 frame
  -> separate Y and UV memory layout
  -> deinterleave UV into U and V
  -> warp Y into model camera view
  -> warp U into model camera view
  -> warp V into model camera view
  -> rejoin as YUV
  -> pack into 6 channels
  -> queue two frames
  -> feed neural network as img / big_img
```

Another version:

```text
real camera image
  -> virtual model-camera image
  -> model-ready YUV tensor
```

## 17. Where We Are In The Repo Walk

Already covered:

```text
models/README.md
  model input contract

constants.py
  model timing, plan size, feature size

common/transformations/model.py
  virtual camera geometry and warp matrix

compile_modeld.py
  actual pixel warp and YUV tensor packing
```

Next repo topic:

```text
ModelState in modeld.py
```

That will explain:

```text
how openpilot loads the compiled model
how it keeps image/feature/desire queues
how it runs inference
how it saves hidden_state for the next frame
```

## 18. Clarification: Why Reverse Warp?

This part is easy to misunderstand.

The goal is to create a clean output image:

```text
model-view image, 512 x 256
```

Every pixel in that model-view image must get a value.

There are two possible ways to warp:

```text
Forward warp:
  for every real camera pixel:
    ask where it should land in the model image

Reverse warp:
  for every model image pixel:
    ask where it should sample from in the real camera image
```

openpilot uses the reverse style.

Why? Because forward warp can leave holes.

Example:

```text
real pixel A maps to model x = 10.2
real pixel B maps to model x = 11.8
```

After rounding:

```text
A may fill output pixel 10
B may fill output pixel 12
output pixel 11 may get nothing
```

That is a hole.

Reverse warp avoids this because it starts from the output image:

```text
for output pixel 0:
  find source pixel

for output pixel 1:
  find source pixel

for output pixel 2:
  find source pixel
```

Every output pixel gets deliberately filled.

So the mental model is:

```text
Do not throw source pixels forward.
Instead, pull source pixels backward into every output slot.
```

That is why `warp_perspective_tinygrad()` uses `M_inv`.

## 19. Clarification: NV12 vs YUV

YUV is the color representation:

```text
Y = brightness
U = color difference
V = color difference
```

NV12 is the memory layout used to store that YUV data.

So:

```text
YUV = what the image values mean
NV12 = how those values are arranged in memory
```

NV12 layout:

```text
Y plane:
  one brightness value for each pixel

UV plane:
  lower-resolution color values, interleaved as U,V,U,V,...
```

Picture:

```text
Y Y Y Y Y Y Y Y
Y Y Y Y Y Y Y Y
Y Y Y Y Y Y Y Y
Y Y Y Y Y Y Y Y

U V U V U V U V
U V U V U V U V
```

The model ultimately wants separate packed channels:

```text
Y split into four brightness channels
U as one channel
V as one channel
```

So openpilot has to:

```text
read NV12 memory
separate Y
deinterleave UV into U and V
warp Y, U, V
pack them into model channels
```

## 20. Clarification: Planes, Not Plans

When this note says:

```text
real Y -> model-view Y
real U -> model-view U
real V -> model-view V
```

it means image planes, not driving plans.

Image plane:

```text
a 2D sheet of image values
```

So:

```text
Y plane:
  2D brightness sheet

U plane:
  2D color sheet

V plane:
  2D color sheet
```

The driving `plan` is a later neural-network output:

```text
future trajectory over time
```

These are unrelated words that sound similar:

```text
image plane:
  preprocessing image sheet

driving plan:
  model output trajectory
```

## 21. `warp_perspective_tinygrad()` In Plain English

Code reference:

[warp_perspective_tinygrad](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L50)

The function:

```python
def warp_perspective_tinygrad(src_flat, M_inv, dst_shape, src_shape, stride_pad, border_fill_val=None):
```

Plain meaning:

```text
Given one source image plane from the real camera,
create one output image plane in the model camera view.
```

Inputs:

```text
src_flat:
  source image plane as a flat array

M_inv:
  matrix that maps output/model pixel coordinates to source/real pixel coordinates

dst_shape:
  output size we want, for example model Y size

src_shape:
  source size we are sampling from, for example real camera Y size

stride_pad:
  row-padding in the real camera buffer

border_fill_val:
  optional fill value if sample location is outside the source image
```

The algorithm:

```text
1. Create every output pixel coordinate.
2. Use M_inv to map each output coordinate into source-image coordinates.
3. Divide by perspective depth w.
4. Round source x,y to nearest pixel.
5. Clip x,y so they stay inside the source image.
6. Convert x,y into a flat memory index.
7. Read source pixel values.
8. Return the warped output plane.
```

Important:

```text
This function warps one image plane at a time.
```

It is called separately for:

```text
Y
U
V
```

Then those warped planes are packed into the model tensor.

## 22. More Precise: What A Warp Matrix Is

A warp matrix is a 3x3 matrix used to map pixel coordinates from one image view to another.

For this model path, the useful mental direction is:

```text
model-view pixel coordinate
  -> real-camera pixel coordinate
```

A pixel coordinate is written in homogeneous form:

```text
[x, y, 1]
```

The warp does:

```text
[sx_raw]     [m00 m01 m02] [x]
[sy_raw]  =  [m10 m11 m12] [y]
[sw]         [m20 m21 m22] [1]
```

Then:

```text
source_x = sx_raw / sw
source_y = sy_raw / sw
```

That divide by `sw` is what makes it a perspective warp instead of just a simple rotate/scale/shift.

In the code:

[matrix apply](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L57)

```python
src_x = M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2]
src_y = M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2]
src_w = M_inv[2, 0] * x + M_inv[2, 1] * y + M_inv[2, 2]

src_x = src_x / src_w
src_y = src_y / src_w
```

Here:

```text
x, y:
  every pixel coordinate in the output/model image

src_x, src_y:
  the corresponding coordinate in the real camera image
```

So `M_inv` answers:

```text
For this output pixel, where do I read from in the input image?
```

## 23. More Precise: Why Reverse Mapping Avoids Holes

Images are discrete grids.

Example output image:

```text
pixel 0
pixel 1
pixel 2
pixel 3
pixel 4
```

Forward warp says:

```text
for every source pixel:
  compute where it lands in output
```

Suppose source pixels land at these continuous output positions:

```text
source A -> output x = 0.2
source B -> output x = 1.1
source C -> output x = 3.0
source D -> output x = 4.2
```

After rounding:

```text
source A fills output pixel 0
source B fills output pixel 1
source C fills output pixel 3
source D fills output pixel 4
```

Output pixel 2 receives no source pixel:

```text
0 filled
1 filled
2 empty   <- hole
3 filled
4 filled
```

This happens because the mapping is continuous, but the output image is a discrete grid. Perspective transforms can stretch, shrink, rotate, and skew the grid, so source samples do not land perfectly on every output pixel.

Reverse mapping says:

```text
for every output pixel:
  compute where to sample from in source
```

So:

```text
output pixel 0 samples source coordinate ...
output pixel 1 samples source coordinate ...
output pixel 2 samples source coordinate ...
output pixel 3 samples source coordinate ...
output pixel 4 samples source coordinate ...
```

Every output pixel is visited exactly because the loop is over the output grid.

That is the precise reason:

```text
forward mapping:
  source coverage does not guarantee output coverage

reverse mapping:
  output coverage is guaranteed because every output pixel is explicitly computed
```

In openpilot's code, the sampling is nearest-neighbor:

[nearest pixel](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L65)

```python
x_round = Tensor.round(src_x)
y_round = Tensor.round(src_y)
```

So if the computed source coordinate is:

```text
(372.4, 220.7)
```

it samples approximately:

```text
(372, 221)
```

Reverse mapping does not mean samples can never repeat.

Two output pixels may still sample the same source pixel if the warp compresses the image:

```text
output pixel 10 -> source pixel 50.2 -> samples source 50
output pixel 11 -> source pixel 50.4 -> samples source 50
```

That is allowed. It means the source image is being compressed in that region.

The important difference is:

```text
forward mapping:
  some output pixels may never get assigned

reverse mapping:
  every output pixel computes a source coordinate and gets assigned a value
```

So reverse mapping guarantees output coverage, not perfect uniqueness.

## 23.1 How The Sampling Estimate Works

The warp matrix usually maps an output pixel to a non-integer source coordinate:

```text
output pixel:
  (100, 50)

source coordinate:
  (372.4, 220.7)
```

But real image pixels live at integer grid positions:

```text
(372, 220)
(373, 220)
(372, 221)
(373, 221)
```

So the code must estimate the pixel value at `(372.4, 220.7)`.

There are two common ways:

```text
nearest-neighbor:
  pick the closest source pixel

bilinear interpolation:
  blend the four nearest source pixels
```

openpilot's tinygrad warp uses nearest-neighbor here:

```python
x_round = Tensor.round(src_x)
y_round = Tensor.round(src_y)
```

That means:

```text
(372.4, 220.7)
  -> round
  -> (372, 221)
```

Then it reads that source pixel.

This is the "estimation":

```text
continuous source coordinate
  -> nearest integer source pixel
  -> sampled value
```

If the source coordinate goes outside the real image, the code clips it:

```python
x_nn_clipped = x_round.clip(0, w_src - 1).cast('int')
y_nn_clipped = y_round.clip(0, h_src - 1).cast('int')
```

So:

```text
source x < 0:
  use x = 0

source x > image width:
  use last valid x
```

That keeps memory access valid.

## 23.2 Why Forward Mapping Can Mix Or Collide

Forward mapping can also have collisions.

Example:

```text
source pixel A -> output pixel 10
source pixel B -> output pixel 10
```

Now two source pixels want to write into the same output pixel.

The code would need a rule:

```text
which one wins?
last write?
average them?
blend based on area?
```

At the same time, another output pixel may receive nothing:

```text
output pixel 11:
  no source pixel landed here
```

So forward mapping has two problems:

```text
holes:
  output pixels with no value

collisions:
  multiple source pixels fighting for one output pixel
```

Reverse mapping is simpler:

```text
each output pixel reads exactly one estimated source value
```

There can still be repeated reads from the same source pixel, but there are no write collisions because every output pixel writes only to itself.

## 23.3 Warp Matrix Formula Broken Down

This formula can look intimidating:

```text
[sx_raw]     [m00 m01 m02] [x]
[sy_raw]  =  [m10 m11 m12] [y]
[sw]         [m20 m21 m22] [1]
```

Break it into three equations:

```text
sx_raw = m00*x + m01*y + m02
sy_raw = m10*x + m11*y + m12
sw     = m20*x + m21*y + m22
```

Then:

```text
source_x = sx_raw / sw
source_y = sy_raw / sw
```

What each part means:

```text
x, y:
  pixel coordinate in the output/model image

m00..m22:
  the 9 numbers inside the warp matrix

sx_raw, sy_raw:
  temporary transformed coordinates

sw:
  perspective scale/depth term

source_x, source_y:
  final coordinate to sample from in the real camera image
```

If this were only rotate/scale/shift, `sw` would usually stay constant.

Perspective warp allows `sw` to change with `x,y`, which lets straight image grids bend/skew like camera perspective.

Simple mental model:

```text
normal 2D transform:
  shift / rotate / zoom the image flatly

perspective transform:
  shift / rotate / zoom / skew as if the camera viewpoint changed
```

For openpilot:

```text
output/model pixel coordinate
  -> 3x3 matrix
  -> real-camera source coordinate
  -> nearest-neighbor sample
  -> output pixel value
```

## 23.4 Important Naming: Output Image vs Neural Network Output

In the warp explanation, "output pixel" does not mean the neural network's final driving output.

It means:

```text
output of the warp operation
```

That warped image is created before the neural network runs.

So the order is:

```text
real camera frame
  -> warp operation
  -> warped output image
  -> packed tensor
  -> neural network input
  -> neural network output plan
```

The "output pixel" in reverse mapping is a pixel in this intermediate image:

```text
warped output image = model-ready camera image
```

Maybe clearer names:

```text
source image:
  real camera image

destination image:
  virtual model-camera image
```

Reverse mapping means:

```text
for every destination pixel:
  compute where to read from in the source image
```

## 23.5 Why Not Just Make Forward Mapping Approximate Too?

You can do forward mapping, but it becomes more complicated.

Forward mapping:

```text
for every source pixel:
  compute where it lands in destination image
```

If the landing coordinate is non-integer:

```text
source pixel -> destination coordinate (10.3, 20.8)
```

you have choices:

```text
round to nearest destination pixel
spread/splat into neighboring pixels
accumulate weighted values
normalize afterward
run a hole-filling pass
```

If you simply round, you get holes and collisions:

```text
holes:
  destination pixels nobody wrote to

collisions:
  destination pixels multiple source pixels wrote to
```

If you "splat" each source pixel across nearby destination pixels, you need extra bookkeeping:

```text
destination_value_sum
destination_weight_sum
final_value = destination_value_sum / destination_weight_sum
```

And even then, some destination pixels can still have:

```text
weight_sum = 0
```

which means:

```text
no source information landed there
```

Reverse mapping is simpler for image resampling:

```text
for each destination pixel:
  compute source coordinate
  sample source image
```

This naturally creates one value for every destination pixel.

That is why reverse mapping is the standard approach for resize/warp/remap operations.

## 23.6 Sampling From Source: Point 3 Re-explained

At this stage, the real camera image already exists.

The thing we are trying to create is:

```text
destination image:
  the model-ready warped camera image
```

For one destination pixel:

```text
destination pixel = (x, y)
```

The warp matrix computes:

```text
source coordinate = (source_x, source_y)
```

Usually that source coordinate is decimal:

```text
(372.4, 220.7)
```

But source pixels are stored at integer positions:

```text
(372, 220)
(373, 220)
(372, 221)
(373, 221)
```

So the code estimates the value.

openpilot's tinygrad warp does nearest-neighbor:

```text
round source_x
round source_y
read that pixel
```

Example:

```text
source coordinate:
  (372.4, 220.7)

nearest integer pixel:
  (372, 221)

destination pixel value:
  real_camera_image[221, 372]
```

So the destination image is built by repeatedly doing:

```text
destination[y, x] = source[round(source_y), round(source_x)]
```

This happens for every destination pixel.

## 23.7 The Estimation Problem

The estimation problem exists because the warp matrix works in continuous coordinates, but images are stored on a discrete pixel grid.

The matrix can return:

```text
source coordinate:
  (372.4, 220.7)
```

But the source image only has stored values at:

```text
integer pixel centers:
  (372, 220)
  (373, 220)
  (372, 221)
  (373, 221)
```

So the code has to answer:

```text
What pixel value should represent the image at (372.4, 220.7)?
```

That is the estimation problem.

### Nearest-Neighbor Estimate

Nearest-neighbor is the simplest estimate:

```text
pick the closest real pixel
```

Example:

```text
(372.4, 220.7)
  -> nearest integer pixel
  -> (372, 221)
```

Pros:

```text
fast
simple
hardware-friendly
does not blend values
```

Cons:

```text
can look blocky
can alias fine details
less smooth than interpolation
```

openpilot's tinygrad warp uses nearest-neighbor in this code path:

```python
x_round = Tensor.round(src_x)
y_round = Tensor.round(src_y)
```

### Bilinear Estimate

Bilinear interpolation is smoother.

It looks at four nearby source pixels:

```text
(372, 220)   (373, 220)
(372, 221)   (373, 221)
```

Then it blends them based on how close the decimal coordinate is to each one.

If the coordinate is:

```text
(372.4, 220.7)
```

then the result is a weighted mix of those four pixels.

Pros:

```text
smoother
less blocky
often better for visual quality
```

Cons:

```text
more compute
more memory reads
more complicated kernel
blends sharp edges
```

For this model-preprocessing path, openpilot is using nearest-neighbor because this operation is part of a real-time inference pipeline.

## 23.8 What If The Warp Stretches Or Compresses The Image?

A perspective warp can stretch one region and compress another.

### If It Compresses

Multiple destination pixels may read the same source pixel:

```text
destination 10 -> source 50.2 -> source pixel 50
destination 11 -> source 50.4 -> source pixel 50
```

This means:

```text
some source detail is being compressed
```

That is not a collision problem because both destination pixels are only reading from source. No two pixels are fighting to write to the same destination slot.

### If It Stretches

Adjacent destination pixels may sample source pixels farther apart:

```text
destination 10 -> source 50
destination 11 -> source 53
```

This means:

```text
some source detail may be skipped
```

This is a normal resampling tradeoff. Any resize/warp operation has to choose how to estimate pixels when changing geometry.

## 23.9 What If The Source Coordinate Is Outside The Real Image?

Sometimes a destination pixel maps to a source coordinate outside the source image:

```text
source_x = -12
source_y = 3000
```

That is invalid memory. The code must handle it.

openpilot clips coordinates:

```python
x_nn_clipped = x_round.clip(0, w_src - 1).cast('int')
y_nn_clipped = y_round.clip(0, h_src - 1).cast('int')
```

Meaning:

```text
if source_x < 0:
  use 0

if source_x > last column:
  use last valid column

if source_y < 0:
  use 0

if source_y > last row:
  use last valid row
```

There is also optional border-fill logic:

```python
if border_fill_val is None:
  return sampled
```

If a border value is provided, out-of-bounds pixels can be filled with a chosen value instead of clipped sampling.

## 23.10 Why This Is Not A Model Decision

This warp estimate is not the driving model "thinking."

It is deterministic image preprocessing:

```text
same source image
same warp matrix
same sampling rule
  -> same destination image
```

The neural network comes after this.

The point of the warp is to provide a stable image format:

```text
real camera view
  -> standardized virtual model-camera view
```

Then the model handles the actual learned driving task:

```text
model-ready image
  -> visual features
  -> temporal policy
  -> future plan
```

So the warp's estimation problem is a computer-vision resampling problem, not an end-to-end driving decision.

## 24. More Precise: NV12 Memory Layout

NV12 is YUV 4:2:0 semi-planar format.

For an image of width `W` and height `H`:

```text
Y plane size:
  W * H bytes

UV plane size:
  W * H / 2 bytes

total:
  W * H * 1.5 bytes
```

Why only half as much UV?

Because U and V are shared across 2x2 pixel blocks.

One 2x2 block:

```text
Y00 Y01
Y10 Y11
```

has:

```text
4 brightness values
1 U value
1 V value
```

So brightness is full resolution:

```text
one Y per pixel
```

Color is lower resolution:

```text
one U,V pair for a 2x2 pixel block
```

NV12 memory looks like:

```text
Y plane:
  Y Y Y Y Y Y Y Y
  Y Y Y Y Y Y Y Y
  Y Y Y Y Y Y Y Y
  Y Y Y Y Y Y Y Y

UV plane:
  U V U V U V U V
  U V U V U V U V
```

In code, openpilot reads the UV plane:

[UV plane](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L100)

```python
uv = input_frame[uv_offset:uv_offset + uv_height * stride].reshape(uv_height, stride)
```

Then it separates U and V:

[U sample](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L106)

```python
uv[:cam_h//2, :cam_w:2]
```

[V sample](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L109)

```python
uv[:cam_h//2, 1:cam_w:2]
```

Meaning:

```text
even UV bytes:
  U values

odd UV bytes:
  V values
```

So NV12 is not a different color "scale" from YUV.

It is:

```text
YUV values arranged in a specific compact byte layout.
```
