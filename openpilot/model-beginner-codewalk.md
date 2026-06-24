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

## 25. What Comes Next: Runtime Model State

So far, we understand the preprocessing path:

```text
real camera NV12 frame
  -> warp into virtual model-camera image
  -> pack YUV into model tensor
```

The next repo concept is:

```text
ModelState
```

Code reference:

[ModelState class](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L75)

`ModelState` is the object that owns the model runtime.

It answers:

```text
Where is the compiled model loaded?
Where are image/history queues stored?
Where is the output parser stored?
How does one camera tick become one model inference?
Where is hidden_state saved for the next tick?
```

The mental transition is:

```text
compile_modeld.py:
  defines preprocessing and queue helpers

modeld.py / ModelState:
  uses those helpers at runtime
```

## 26. `ModelState.__init__`: What Gets Created At Startup

Constructor reference:

[ModelState.__init__](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L78)

Important code:

```python
jits = pickle.loads(read_file_chunked(modeld_pkl_path(usbgpu)))
metadata = jits['metadata']
self.input_shapes = metadata['input_shapes']
self.output_slices = metadata['output_slices']
```

Meaning:

```text
load the compiled tinygrad model bundle
read the model input shapes
read the output slicing table
```

The compiled model bundle contains more than the neural net:

```text
metadata
compiled run_policy function
compiled warp function for this camera resolution
```

Then:

```python
self.frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
```

Code reference:

[frame_skip](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L89)

Given:

```text
MODEL_RUN_FREQ = 20
MODEL_CONTEXT_FREQ = 5
```

Then:

```text
frame_skip = 4
```

Meaning:

```text
model can run at 20 Hz,
but context history is sampled every 4 ticks,
which gives 5 Hz history spacing.
```

Then:

```python
self.input_queues, self.npy = make_input_queues(...)
```

Code reference:

[make_input_queues call](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L90)

Meaning:

```text
create rolling buffers for image frames, feature history, desire history,
and small numpy-backed inputs like traffic convention and action timing.
```

Then:

```python
self.parser = Parser()
self.run_policy = jits['run_policy']
self.warp_enqueue = jits[(cam_w,cam_h)]
```

Code references:

[Parser setup](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L93)

[compiled functions](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L95)

Meaning:

```text
Parser:
  turns raw output vector into named arrays

run_policy:
  runs the neural network

warp_enqueue:
  warps current camera frames and updates image queues
```

So startup creates:

```text
compiled model functions
input queues
metadata
output parser
state memory
```

## 27. Runtime Queues: Why They Exist

The model does not receive only one isolated image.

It receives:

```text
current images
recent image history
feature history
desire history
small context inputs
```

The queue helper starts here:

[make_input_queues](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L145)

Important code:

```python
input_queues.update({
  'feat_q': Tensor(np.zeros((frame_skip * fb[1], fb[0], fb[2]), dtype=np.float32), device=device).contiguous().realize(),
  'desire_q': Tensor(np.zeros((frame_skip * dp[1], dp[0], dp[2]), dtype=np.float32), device=device).contiguous().realize(),
  'packed_npy_inputs': Tensor(packed_npy_inputs, device='NPY').realize(),
})
```

Code reference:

[feature/desire queues](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L155)

Meaning:

```text
feat_q:
  rolling buffer for previous hidden_state vectors

desire_q:
  rolling buffer for desire pulses

packed_npy_inputs:
  compact storage for small non-image inputs
```

The generic queue update is:

[shift_and_sample](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L163)

```python
def shift_and_sample(buf, new_val, sample_fn):
  buf.assign(buf[1:].cat(new_val, dim=0).contiguous())
  return sample_fn(buf)
```

Plain English:

```text
drop the oldest item
append the newest item
sample the buffer into the exact model input shape
```

## 28. `ModelState.run`: One Tick Of Runtime

Runtime method reference:

[ModelState.run](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L102)

This function is the per-frame model path.

High-level:

```text
wrap camera buffers
update small inputs
warp images
run neural network
parse outputs
save hidden_state
return parsed outputs
```

### Step 1: Wrap Camera Buffer Without Copying

Code reference:

[Tensor.from_blob](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L104)

```python
ptr = np.frombuffer(bufs[key].data, dtype=np.uint8).ctypes.data
...
Tensor.from_blob(ptr, (yuv_size,), dtype='uint8', device=self.WARP_DEV)
```

Meaning:

```text
point tinygrad at the existing camera buffer memory
instead of copying the full image into a new array
```

Why:

```text
real-time system
large camera frames
avoid unnecessary memory copies
```

### Step 2: Convert Desire Into A Pulse

Code reference:

[desire pulse logic](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L113)

```python
self.npy['desire'][:] = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
self.prev_desire[:] = inputs['desire_pulse']
```

Meaning:

```text
only send the desire when it newly turns on
```

This is why it is called a pulse:

```text
lane-change-left starts:
  send pulse once

lane-change-left remains active:
  send zero after the rising edge
```

### Step 3: Store Context Inputs And Warp Matrices

Code reference:

[context and transforms](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L117)

```python
self.npy['traffic_convention'][:] = inputs['traffic_convention']
self.npy['action_t'][:] = inputs['action_t']
self.npy['tfm'][:,:] = transforms['img'][:,:]
self.npy['big_tfm'][:,:] = transforms['big_img'][:,:]
```

Meaning:

```text
traffic convention:
  left-hand / right-hand traffic context

action_t:
  timing context

tfm / big_tfm:
  warp matrices for img and big_img
```

### Step 4: Warp Current Frames And Update Image Queues

Code reference:

[warp_enqueue call](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L122)

```python
img, big_img = self.warp_enqueue(..., frame=self.full_frames['img'], big_frame=self.full_frames['big_img'])
```

This calls the compiled warp function created by `compile_modeld.py`.

The helper logic is:

[warp_enqueue helper](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L180)

```python
warped_frame = frame_prepare(frame, tfm).unsqueeze(0)
warped_big_frame = frame_prepare(big_frame, big_tfm).unsqueeze(0)
...
img = shift_and_sample(img_q, warped[0:1], sample_skip_fn)
big_img = shift_and_sample(big_img_q, warped[1:2], sample_skip_fn)
```

Meaning:

```text
warp current img frame
warp current big_img frame
append them to image queues
sample queues into model-ready img and big_img tensors
```

### Step 5: Run The Neural Network

Code reference:

[run_policy call](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L127)

```python
outs, = self.run_policy(...)
model_output = outs.numpy()[0]
```

At this point, the neural network runs.

Inputs include:

```text
img
big_img
features_buffer
desire_pulse
traffic_convention
action_t
```

The helper constructs those inputs here:

[run_policy inputs](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/compile_modeld.py#L199)

```python
inputs = {
  'img': img,
  'big_img': big_img,
  'features_buffer': feat_buf,
  'desire_pulse': desire_buf,
  'traffic_convention': traffic_convention,
  'action_t': action_t,
}
```

### Step 6: Slice And Parse The Raw Output

Code reference:

[parse output](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L131)

```python
outputs_dict = self.parser.parse_outputs(self.slice_outputs(model_output, self.output_slices))
```

Meaning:

```text
raw vector from model
  -> split into named slices
  -> parse each slice into usable arrays/probabilities/distributions
```

We will cover this parser next.

### Step 7: Save Hidden State For Next Tick

Code reference:

[hidden_state save](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L132)

```python
self.npy['prev_feat'][:] = model_output[self.output_slices['hidden_state']]
```

Meaning:

```text
take the model's 512-number hidden_state output
save it as prev_feat
feed it into the feature queue on the next tick
```

This closes the temporal loop:

```text
current frame
  -> model
  -> hidden_state
  -> next frame's features_buffer
```

## 29. ModelState Summary

`ModelState` is the bridge between preprocessing and inference:

```text
camera buffer
  -> tinygrad tensor view
  -> warp and image queue
  -> feature/desire queues
  -> neural network run
  -> raw output vector
  -> parsed outputs
  -> hidden_state saved for next tick
```

The next repo topic after `ModelState` is:

```text
output parsing
```

That means:

```text
How does one raw 2576-value model output vector become:
  plan
  lane_lines
  road_edges
  lead
  meta
  desire_state
  hidden_state
```

## 30. Output Parsing: The Next Model Boundary

The neural network returns one raw vector.

For the normal driving model we inspected:

```text
outputs = (1, 2576)
```

That vector is not directly useful as-is.

It is just:

```text
[number, number, number, number, ...]
```

Runtime needs to convert it into named outputs:

```text
meta
desire_pred
pose
lane_lines
road_edges
lead
hidden_state
plan
desire_state
```

This happens in two steps:

```text
1. slice the raw vector into named chunks
2. parse each chunk into usable values/probabilities/distributions
```

The handoff happens here:

[parse output call](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L131)

```python
outputs_dict = self.parser.parse_outputs(self.slice_outputs(model_output, self.output_slices))
```

Read that as:

```text
slice_outputs:
  raw vector -> named raw chunks

parse_outputs:
  named raw chunks -> interpreted outputs
```

## 31. Step 1: `slice_outputs`

Code reference:

[slice_outputs](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L98)

```python
def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
  parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
  return parsed_model_outputs
```

This is simple dictionary slicing.

If metadata says:

```text
plan = slice(1576, 2566)
```

then:

```text
raw_output[1576:2566] -> outs["plan"]
```

If metadata says:

```text
hidden_state = slice(1064, 1576)
```

then:

```text
raw_output[1064:1576] -> outs["hidden_state"]
```

So:

```text
slice_outputs does not understand meaning.
It only cuts the raw vector into named sections.
```

The meaning comes from the parser.

## 32. Step 2: `Parser`

Parser code:

[Parser class](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/parse_model_outputs.py#L20)

The parser has three main tools:

```text
sigmoid
softmax
MDN parser
```

These match different kinds of model outputs.

## 33. Sigmoid: Binary Probabilities

Code reference:

[parse_binary_crossentropy](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/parse_model_outputs.py#L38)

```python
def parse_binary_crossentropy(self, name, outs):
  raw = outs[name]
  outs[name] = sigmoid(raw)
```

Sigmoid turns raw numbers into probabilities between 0 and 1.

Use it for yes/no style outputs:

```text
lane line exists?
lead exists?
hard brake likely?
driver may disengage?
```

Examples in code:

[binary parses](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/parse_model_outputs.py#L101)

```python
self.parse_binary_crossentropy('lane_lines_prob', outs)
self.parse_binary_crossentropy('meta', outs)
self.parse_binary_crossentropy('lead_prob', outs)
```

Mental model:

```text
raw score:
  can be any number

sigmoid output:
  probability-like value from 0 to 1
```

## 34. Softmax: Category Probabilities

Code reference:

[parse_categorical_crossentropy](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/parse_model_outputs.py#L30)

```python
def parse_categorical_crossentropy(self, name, outs, out_shape=None):
  raw = outs[name]
  if out_shape is not None:
    raw = raw.reshape((raw.shape[0],) + out_shape)
  outs[name] = softmax(raw, axis=-1)
```

Softmax turns a list of raw scores into probabilities that sum to 1.

Use it for category choices:

```text
which desire state?
which predicted desire class?
```

Examples in code:

[desire parsing](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/parse_model_outputs.py#L102)

```python
self.parse_categorical_crossentropy('desire_pred', outs, out_shape=(ModelConstants.DESIRE_PRED_LEN,ModelConstants.DESIRE_PRED_WIDTH))
```

[desire_state parsing](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/parse_model_outputs.py#L114)

```python
self.parse_categorical_crossentropy('desire_state', outs, out_shape=(ModelConstants.DESIRE_PRED_WIDTH,))
```

Mental model:

```text
raw scores:
  [2.1, -0.4, 0.7]

softmax:
  [0.73, 0.06, 0.21]
```

## 35. MDN: Prediction Plus Uncertainty

MDN means:

```text
Mixture Density Network
```

The parser starts here:

[parse_mdn](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/parse_model_outputs.py#L44)

This is used for outputs where the model predicts continuous values:

```text
future plan positions
lane line coordinates
road edge coordinates
lead trajectory
pose
road transform
```

Why not just output one number?

Because continuous driving predictions have uncertainty.

Example:

```text
lane marking is faded
lead car is partially occluded
future path is uncertain
road edge is unclear
```

So MDN-style outputs include:

```text
mean prediction
standard deviation / uncertainty
possibly multiple hypotheses
```

In code:

[MDN mean/std split](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/parse_model_outputs.py#L50)

```python
n_values = (raw.shape[2] - out_N)//2
pred_mu = raw[:,:,:n_values]
pred_std = safe_exp(raw[:,:,n_values: 2*n_values])
```

Translation:

```text
first part:
  predicted values / means

second part:
  uncertainty values / standard deviations
```

`safe_exp` is used because the model stores std-like values in a form that needs exponentiation:

[safe_exp](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/parse_model_outputs.py#L4)

```python
return np.exp(np.clip(x, -np.inf, 11), out=out)
```

The clip prevents exponent values from exploding numerically.

## 36. Which Outputs Use MDN?

Vision outputs:

[parse_vision_outputs](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/parse_model_outputs.py#L95)

```python
self.parse_mdn('pose', outs, ...)
self.parse_mdn('wide_from_device_euler', outs, ...)
self.parse_mdn('road_transform', outs, ...)
self.parse_mdn('lane_lines', outs, ...)
self.parse_mdn('road_edges', outs, ...)
self.parse_mdn('lead', outs, ...)
```

Policy output:

[parse_policy_outputs](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/parse_model_outputs.py#L112)

```python
self.parse_mdn('plan', outs, in_N=0, out_N=0, out_shape=(ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH))
```

This means the `plan` is parsed as:

```text
33 future points
15 values per point
plus corresponding uncertainty/std values
```

So after parsing, the output dictionary has:

```text
plan
plan_stds
```

## 37. Parser Summary

The parser is not the neural network.

It is the decoder for the neural network's raw output.

Clean flow:

```text
raw output vector
  -> slice by metadata
  -> sigmoid for binary probabilities
  -> softmax for category probabilities
  -> MDN parser for continuous predictions and uncertainty
  -> output dictionary
```

After this, openpilot has usable model outputs:

```text
plan
lane_lines
road_edges
lead
meta
desire_state
hidden_state
```

The next repo topic after parsing is:

```text
plan-to-action conversion
```

That answers:

```text
How does the parsed future plan become desired curvature and desired acceleration?
```

## 38. Plan-To-Action Conversion

After parsing, openpilot has a usable `plan`.

Now it needs to convert that plan into the compact action target:

```text
desiredCurvature
desiredAcceleration
shouldStop
```

This happens here:

[get_action_from_model](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L37)

Important code:

```python
def get_action_from_model(model_output, prev_action, lat_action_t, long_action_t, v_ego):
  if 'action' not in model_output:
    plan = model_output['plan'][0]
    desired_accel, should_stop = get_accel_from_plan(...)
    desired_curvature = get_curvature_from_plan(...)
  else:
    desired_accel = model_output['action'][0,1]
    desired_curvature = model_output['action'][0,0] / (max(1.0, v_ego))**2
    should_stop = (v_ego < 0.3 and desired_accel < 0.1)
```

There are two possible paths:

```text
Path A:
  model directly outputs action

Path B:
  model outputs plan, and runtime derives action from the plan
```

For the model metadata we inspected, the important path is:

```text
plan -> action
```

So focus on this:

```text
plan velocity/acceleration -> desired acceleration
plan yaw/yaw-rate          -> desired curvature
```

## 39. What Is `desiredAcceleration`?

Acceleration is the longitudinal target:

```text
speed up
maintain speed
slow down
stop
```

The conversion happens here:

[get_accel_from_plan](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/drive_helpers.py#L43)

Call site:

[desired_accel call](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L41)

```python
desired_accel, should_stop = get_accel_from_plan(
  plan[:,Plan.VELOCITY][:,0],
  plan[:,Plan.ACCELERATION][:,0],
  ModelConstants.T_IDXS,
  action_t=long_action_t
)
```

Breakdown:

```text
plan[:, Plan.VELOCITY][:, 0]:
  future x-direction speeds from the plan

plan[:, Plan.ACCELERATION][:, 0]:
  future x-direction accelerations from the plan

ModelConstants.T_IDXS:
  the future time points

long_action_t:
  the future time where we want the acceleration target
```

Inside `get_accel_from_plan`:

```python
v_now = speeds[0]
a_now = accels[0]
v_target = np.interp(action_t, t_idxs, speeds)
a_target = 2 * (v_target - v_now) / (action_t) - a_now
```

Plain English:

```text
look at current planned speed
look at planned speed at the action time
compute the acceleration needed to get there smoothly
```

`np.interp` matters because the model plan has 33 discrete future points, but `action_t` may fall between two of those points.

So:

```text
interpolation estimates the planned speed at exactly long_action_t
```

## 40. Why The Acceleration Formula Looks Weird

The formula:

```python
a_target = 2 * (v_target - v_now) / action_t - a_now
```

comes from simple constant-jerk-ish motion reasoning.

Do not overfocus on the derivation yet. The useful intuition is:

```text
given:
  current speed
  current acceleration
  target future speed
  time until that target

compute:
  acceleration command that points us toward the model's future speed
```

So the model does not say:

```text
press gas 12%
```

It says through the plan:

```text
the car should be moving at this future speed
```

Runtime converts that into:

```text
desired acceleration
```

## 41. `shouldStop`

Inside `get_accel_from_plan`:

[should_stop](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/drive_helpers.py#L56)

```python
should_stop = (v_now < vEgoStopping and a_target < 0.1)
```

Meaning:

```text
if we are already very slow
and the target acceleration is not asking us to move meaningfully,
then mark should_stop
```

This is a compact stop/hold-style signal derived from the longitudinal plan.

## 42. What Is `desiredCurvature`?

Curvature is the lateral target:

```text
how sharply the vehicle path should bend
```

Straight road:

```text
curvature = 0
```

Gentle turn:

```text
small curvature
```

Sharp turn:

```text
larger curvature
```

This is better than outputting steering angle directly because steering angle depends on vehicle-specific behavior.

The model/runtime uses:

```text
desired path shape
```

not:

```text
raw actuator command
```

## 43. Curvature From The Plan

Call site:

[desired_curvature call](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L45)

```python
desired_curvature = get_curvature_from_plan(
  plan[:,Plan.T_FROM_CURRENT_EULER][:,2],
  plan[:,Plan.ORIENTATION_RATE][:,2],
  ModelConstants.T_IDXS,
  v_ego,
  lat_action_t
)
```

Breakdown:

```text
plan[:, Plan.T_FROM_CURRENT_EULER][:, 2]:
  future yaw values from the plan

plan[:, Plan.ORIENTATION_RATE][:, 2]:
  current/future yaw-rate values from the plan

v_ego:
  current ego speed

lat_action_t:
  future time where we want the lateral target
```

Yaw means:

```text
heading angle of the car/path
```

Yaw rate means:

```text
how quickly heading is changing
```

If heading is changing, the path is curving.

## 44. The Curvature Formula

Helper code:

[get_curvature_from_plan](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/drive_helpers.py#L64)

```python
def get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
  if action_t < MIN_STABLE_DELAY:
    psi_target = (action_t / MIN_STABLE_DELAY) * np.interp(MIN_STABLE_DELAY, t_idxs, yaws)
  else:
    psi_target = np.interp(action_t, t_idxs, yaws)
  psi_rate = yaw_rates[0]
  return curv_from_psis(psi_target, psi_rate, vego, action_t)
```

Then:

[curv_from_psis](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/drive_helpers.py#L59)

```python
def curv_from_psis(psi_target, psi_rate, vego, action_t):
  vego = np.clip(vego, MIN_SPEED, np.inf)
  curv_from_psi = psi_target / (vego * action_t)
  return 2*curv_from_psi - psi_rate / vego
```

Plain English:

```text
look at where the plan says the heading/yaw should be at lat_action_t
look at the current yaw rate
use speed to convert heading change into path curvature
```

Why speed matters:

```text
the same heading change over the same time implies different curvature depending on speed
```

At higher speed, small curvature can create high lateral acceleration.

Core relation:

```text
lateral acceleration ≈ curvature * speed^2
```

This is why the direct-action fallback divides by speed squared:

[direct action curvature](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L51)

```python
desired_curvature = model_output['action'][0,0] / (max(1.0, v_ego))**2
```

That means:

```text
if direct model action is lateral acceleration-like,
convert it to curvature by dividing by speed^2
```

## 45. Smoothing The Action

After acceleration and curvature are computed:

[smoothing](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L54)

```python
desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)
if v_ego > MIN_LAT_CONTROL_SPEED:
  desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, LAT_SMOOTH_SECONDS)
else:
  desired_curvature = prev_action.desiredCurvature
```

Acceleration is smoothed:

```text
avoid sudden jumps in desired acceleration
```

Curvature is only updated above a minimum speed:

```text
at very low speed, curvature math becomes unstable or less meaningful
```

The smoothing helper:

[smooth_value](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/drive_helpers.py#L22)

```python
alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1
return alpha * val + (1 - alpha) * prev_val
```

Meaning:

```text
blend new target with previous target
```

## 46. Final Action Object

Return code:

[Action return](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/modeld.py#L60)

```python
return log.ModelDataV2.Action(
  desiredCurvature=float(desired_curvature),
  desiredAcceleration=float(desired_accel),
  shouldStop=bool(should_stop)
)
```

So the model path has now become:

```text
parsed plan
  -> desired acceleration
  -> desired curvature
  -> should stop
```

This is the boundary between:

```text
model prediction:
  what motion should happen

control system:
  how to make the car follow that motion
```

## 47. Plan-To-Action Summary

Full flow:

```text
model output vector
  -> parser
  -> plan
  -> get_accel_from_plan
  -> desiredAcceleration
  -> get_curvature_from_plan
  -> desiredCurvature
  -> Action object
```

The most important distinction:

```text
The model does not output steering wheel angle.
The model predicts future motion.
Runtime converts future motion into curvature and acceleration targets.
```

The next repo topic is:

```text
controls boundary
```

That means:

```text
Where does modeld stop?
Where do controllers begin?
Why is the comma controls challenge related but separate?
```

## 48. Controls Boundary: Where The Model Stops

After `get_action_from_model`, the model side has produced:

```text
desiredCurvature
desiredAcceleration
shouldStop
```

This is still not:

```text
steering command
gas command
brake command
```

It is a motion target.

The boundary is:

```text
modeld:
  predicts desired motion

controls:
  tracks desired motion
```

So:

```text
modeld says:
  "the path should curve this much"
  "the acceleration should be this much"

controllers answer:
  "what steering/torque/accel output achieves that?"
```

## 49. Lateral Boundary: Curvature To Steering/Lataccel

The model-side lateral target is:

```text
desiredCurvature
```

Different lateral controllers can use it differently.

### Angle Controller

Code reference:

[latcontrol_angle update](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/latcontrol_angle.py#L16)

Important line:

```python
angle_steers_des = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
```

Meaning:

```text
desired curvature
  -> vehicle model
  -> desired steering angle
```

This is where vehicle behavior starts to matter.

The model did not output steering angle directly. The controller converts path curvature into a steering target using a vehicle model.

### Torque Controller

Code reference:

[latcontrol_torque update](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/latcontrol_torque.py#L59)

Important lines:

```python
measured_curvature = -VM.calc_curvature(...)
measurement = measured_curvature * CS.vEgo ** 2
future_desired_lateral_accel = desired_curvature * CS.vEgo ** 2
...
error = setpoint - measurement
...
output_lataccel = self.pid.update(...)
output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)
```

Meaning:

```text
desired curvature
  -> desired lateral acceleration
  -> compare against measured lateral acceleration
  -> PID/feedforward controller
  -> steering torque output
```

The important relation:

```text
lateral acceleration ≈ curvature * speed^2
```

So the torque controller works in lateral-acceleration space because that maps more naturally to steering torque behavior.

## 50. Longitudinal Boundary: Acceleration To Output Accel

The model-side longitudinal target is:

```text
desiredAcceleration
```

Longitudinal controller code:

[longcontrol update](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/longcontrol.py#L59)

Important lines:

```python
error = a_target - CS.aEgo
output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target)
```

Meaning:

```text
desired acceleration
  -> compare with actual acceleration
  -> PID/feedforward controller
  -> output acceleration target
```

Again:

```text
model predicts target motion
controller tracks target motion
```

## 51. Where E2E Acceleration Fits

In longitudinal planning, openpilot can compare/use:

```text
MPC-derived acceleration target
E2E model acceleration target
```

Code reference:

[longitudinal_planner e2e action](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/longitudinal_planner.py#L149)

Important lines:

```python
output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
output_should_stop_e2e = sm['modelV2'].action.shouldStop
```

Then in experimental mode:

```python
output_a_target = min(output_a_target_e2e, output_a_target_mpc)
self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc
```

Conceptual meaning:

```text
the model can provide an end-to-end acceleration/stop target,
but planner/controller logic still bounds and integrates it with other control logic.
```

This reinforces the boundary:

```text
model output is a target,
not raw pedal/brake actuation.
```

## 52. Why The Controls Challenge Is Related

The comma controls challenge is about the lower half of this stack.

openpilot model stack:

```text
camera/context/history
  -> model
  -> desired curvature / desired acceleration
```

controls challenge:

```text
desired lateral behavior
  -> controller
  -> simulated vehicle response
  -> tracking cost
```

So the challenge is not asking:

```text
Can you train openpilot's vision model?
```

It asks something closer to:

```text
Given a target lateral acceleration/path behavior,
can you produce steering actions that track it well?
```

That is why leaderboard methods include:

```text
PID + feedforward
MPC
optimized action sequences
PPO
evolution / Bayesian optimization
```

Those are controller strategies.

For an RL/PufferLib project, this is useful because:

```text
observation:
  vehicle state + target/future plan

action:
  steering command

environment:
  learned tiny physics simulator

reward/cost:
  tracking error + smoothness/jerk penalties
```

## 53. End-To-End Model Stack Summary

The model stack we walked through is:

```text
1. Real camera frame arrives as NV12/YUV.

2. openpilot separates Y, U, V image planes.

3. A warp matrix maps the real camera view into the virtual model-camera view.

4. Reverse mapping fills every destination/model-view pixel by sampling from the real camera image.

5. The warped YUV image is packed into 6 channels per frame.

6. Two frames become 12 channels for img and big_img.

7. ModelState maintains rolling image, feature, and desire queues.

8. run_policy feeds img, big_img, features_buffer, desire_pulse, traffic_convention, and action_t into the ONNX/tinygrad model.

9. The model returns one raw vector.

10. slice_outputs cuts the raw vector into named output heads.

11. Parser converts raw heads using sigmoid, softmax, and MDN parsing.

12. The parsed plan becomes desiredAcceleration and desiredCurvature.

13. Controls code tracks those targets.
```

The most important split:

```text
model:
  decides desired future motion

controller:
  makes the vehicle follow that desired motion
```

## 54. What To Study Next

At this point, the model runtime stack is coherent.

Next learning options:

```text
Option A:
  go deeper into curvature/lateral acceleration/controllers

Option B:
  go into comma controls challenge and build a controller/RL environment

Option C:
  go into training theory: imitation learning, world models, learned simulation, policy training

Option D:
  inspect ONNX architecture visually with Netron
```

For your stated interest in comma.ai and RL/PufferLib, the strongest next path is:

```text
controls challenge
  -> understand target tracking
  -> implement PID/MPC baseline
  -> wrap as RL environment
  -> train PPO with PufferLib
```
