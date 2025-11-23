# Gesture Feature Vector Specification (138 Elements)

## 1. Overview

This document specifies the structure and meaning of the 138-element floating-point feature vector used to represent a single frame of one- or two-handed gesture data. The vector is derived from MediaPipe Hand landmarks and is designed to be fed into a sequence model (like a VQ-VAE or Transformer).

The core design philosophy is to separate the *shape* of the hands (Normalized Pose) from their *movement* through space (Velocity and Acceleration).

## 2. Top-Level Structure

The 138-element vector is a concatenation of data for the Left Hand and the Right Hand.

-   **Indices `0` to `68` (69 elements):** Left Hand Data
-   **Indices `69` to `137` (69 elements):** Right Hand Data

```python
feature_vector = [
    # Left Hand Block 
    left_hand_pose_x0, left_hand_pose_y0, left_hand_pose_z0, ..., # 63 elements
    left_wrist_velocity_x, left_wrist_velocity_y, left_wrist_velocity_z, # 3 elements
    left_wrist_acceleration_x, left_wrist_acceleration_y, left_wrist_acceleration_z, # 3 elements
    
    # Right Hand Block 
    right_hand_pose_x0, right_hand_pose_y0, right_hand_pose_z0, ..., # 63 elements
    right_wrist_velocity_x, right_wrist_velocity_y, right_wrist_velocity_z, # 3 elements
    right_wrist_acceleration_x, right_wrist_acceleration_y, right_wrist_acceleration_z, # 3 elements
]
```
## 3. Detailed Block Structure
Each 69-element hand block is structured identically.
### 3.1. Left Hand Data (Indices 0-68)
#### Indices 0 - 62 (63 elements): Normalized Pose
Represents the 3D shape of the hand, made invariant to its position and scale.
Consists of 21 landmarks, each with (X, Y, Z) coordinates, flattened into a single array.
Order: [Landmark0_X, Landmark0_Y, Landmark0_Z, Landmark1_X, Landmark1_Y, Landmark1_Z, ...]
The landmark order is consistent with MediaPipe Hands:
```
WRIST
THUMB_CMC
THUMB_MCP
THUMB_IP
THUMB_TIP
INDEX_FINGER_MCP
INDEX_FINGER_PIP
INDEX_FINGER_DIP
INDEX_FINGER_TIP
MIDDLE_FINGER_MCP
MIDDLE_FINGER_PIP
MIDDLE_FINGER_DIP
MIDDLE_FINGER_TIP
RING_FINGER_MCP
RING_FINGER_PIP
RING_FINGER_DIP
RING_FINGER_TIP
PINKY_MCP
PINKY_PIP
PINKY_DIP
PINKY_TIP
```
#### Indices 63 - 65 (3 elements): Wrist Velocity
Represents the global motion of the hand.
Calculated as position(t) - position(t-1) using the un-normalized wrist coordinates.
Order: [Velocity_X, Velocity_Y, Velocity_Z]
#### Indices 66 - 68 (3 elements): Wrist Acceleration
Represents the change in the hand's motion.
Calculated as velocity(t) - velocity(t-1).
Order: [Acceleration_X, Acceleration_Y, Acceleration_Z]
### 3.2. Right Hand Data (Indices 69-137)
The structure is identical to the Left Hand data, simply offset by 69.
#### Indices 69 - 131 (63 elements) 
Normalized Pose (Right Hand)
#### Indices 132 - 134 (3 elements) 
Wrist Velocity (Right Hand)
#### Indices 135 - 137 (3 elements) 
Wrist Acceleration (Right Hand)
## 4. Normalization and Coordinate System
The meaning of the vector values is defined by the following rules.
### 4.1. Coordinate System
The coordinate system is relative to the camera's view:
+X: To the right
+Y: Downwards
+Z: Away from the camera (deeper into the scene)
### 4.2. "Anchor Hand" Normalization Method
This method is necessary for correctly representing two-handed interactions (like clapping).

#### Determining the Anchor Hand
The `right` hand is preferred as the anchor. If only the `left` hand is visible, it becomes the anchor.

#### Calculate Anchor Position
The 3D coordinate of the anchor hand's wrist is stored as anchor_wrist_pos.
#### Calculate Stable Scaling Factor
A stable measure of the anchor hand's size is calculated (e.g., the distance between its wrist and middle finger MCP joint). This is the scale_factor.
#### Normalizing Both Hands:
For every visible hand (both left and right), its 21 landmarks are first centered by subtracting the anchor_wrist_pos.
The resulting centered coordinates are then scaled by dividing by the scale_factor.
This ensures that when hands touch, their normalized coordinates converge to the same location, preserving the "touching" information.
#### Single-Hand Case
 If only one hand is visible, it becomes the anchor, and this process simplifies to normalizing the hand relative to its own wrist and size.
### 4.3. Handling Missing Data
If a hand is not detected in a given frame, its entire 69-element block in the feature vector is filled with zeros.
## 5. Final Summary Table
Here is the data converted into a Markdown table. I have formatted the "parent" blocks (Left/Right Hand Data Block) in bold to distinguish the hierarchy.

| Index Range | Size | Content Description | Notes |
| --- | --- | --- | --- |
| **0 - 68** | **69** | **Left Hand Data Block** | Padded with zeros if left hand is not detected. |
| 0 - 62 | 63 | Normalized Pose (21 x 3D landmarks) | Normalized using the "Anchor Hand" method. |
| 63 - 65 | 3 |  Wrist Velocity (X, Y, Z) | Calculated from un-normalized coordinates. |
| 66 - 68 | 3 |  Wrist Acceleration (X, Y, Z) | Calculated from velocity. |
| **69 - 137** | **69** | **Right Hand Data Block** | Padded with zeros if right hand is not detected. |
| 69 - 131 | 63 |  Normalized Pose (21 x 3D landmarks) | Normalized using the "Anchor Hand" method. |
| 132 - 134 | 3 |  Wrist Velocity (X, Y, Z) | Calculated from un-normalized coordinates. |
| 135 - 137 | 3 |  Wrist Acceleration (X, Y, Z) | Calculated from velocity. | 