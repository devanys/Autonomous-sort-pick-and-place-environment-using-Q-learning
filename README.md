# Autonomous-sort-pick-and-place-environment-using-Q-learning


https://github.com/user-attachments/assets/9afd2524-1fbd-4b8b-a962-585d8bf67b2d


## Equations

> All equations are rendered as images, so they display in any Markdown viewer.
> Section 9 (Q-learning) is a **proposed extension**. The current `main.py` uses a fixed color-to-slot mapping.

**Contents**

1. [Inverse Kinematics](#1-inverse-kinematics-2-link-planar--prismatic-z)
2. [Forward Kinematics](#2-forward-kinematics-verification)
3. [Trajectory Generation](#3-trajectory-generation-smoothstep)
4. [Conveyor Motion](#4-conveyor-motion)
5. [Camera Projection](#5-camera-projection-3d-to-pixel)
6. [Color Detection](#6-color-detection-hsv-thresholding)
7. [Slot Verification](#7-slot-verification-point-in-polygon)
8. [Grasping](#8-grasping-fixed-constraint)
9. [Q-Learning Extension](#9-q-learning-extension-slot-selection-policy)

### Notation

| Symbol | Meaning | Value / Unit |
|---|---|---|
| L1, L2 | Arm link lengths | 0.28 m, 0.20 m |
| (x0, y0) | SCARA base position | (0.02, 0.02) m |
| q1, q2 | Shoulder and elbow angles | rad |
| q3 | Prismatic (vertical) joint extension | 0 to 0.14 m |
| z_up | Gripper tip height at q3 = 0 | 0.227 m |
| dt | Simulation time step | 1/240 s |
| W, H | Camera resolution | 240 x 180 px |
| V, P | Camera view and projection matrices | 4 x 4 |

**Pipeline summary:** camera (Sec. 5, 6) → color decision → slot choice (Sec. 6, or 9 with RL) → inverse kinematics (Sec. 1) → smooth trajectory (Sec. 3) → grasp and place (Sec. 8) → slot check (Sec. 7).

---

### 1. Inverse Kinematics (2-link planar + prismatic Z)

Converts a target tip position (x, y, z) into joint values (q1, q2, q3).

Relative target position (from the arm base):

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}\Delta&space;x=x-x_0\qquad\Delta&space;y=y-y_0)

Elbow angle (law of cosines):

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}\cos&space;q_2=\frac{\Delta&space;x^2%2B\Delta&space;y^2-L_1^2-L_2^2}{2L_1L_2})

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}q_2=\pm\arccos%28\cos&space;q_2%29)

cos q2 is clamped to [-1, 1] so out-of-reach targets don't crash.

Shoulder angle:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}q_1=\mathrm{atan2}%28\Delta&space;y,&space;\Delta&space;x%29-\mathrm{atan2}%28L_2\sin&space;q_2,&space;L_1%2BL_2\cos&space;q_2%29)

Wrapped to [-pi, pi):

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}q_1\leftarrow%28%28q_1%2B\pi%29\bmod&space;2\pi%29-\pi)

Of the two elbow solutions (+q2 and -q2), the code picks the one **closest to the current elbow angle**, which prevents the arm from flipping between elbow-up and elbow-down.

Prismatic (vertical) joint:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}q_3=\mathrm{clip}%28z_{up}-z,&space;0,&space;q_{3,max}%29)

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}z_{up}=Z_{arm}-\mathrm{TIP}_{off}=0.227~\mathrm{m},\quad&space;q_{3,max}=0.14~\mathrm{m})

| Motion height | z (m) | q3 (m) |
|---|---|---|
| Hover | 0.200 | 0.027 |
| Pick (belt) | 0.112 | 0.115 |
| Place (slot) | 0.146 | 0.081 |

Reachable workspace (annulus), 0.08 m <= r <= 0.48 m:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}%7CL_1-L_2%7C\le\sqrt{\Delta&space;x^2%2B\Delta&space;y^2}\le&space;L_1%2BL_2)

**Worked example:** target above the IR sensor stop point, (x, y) = (-0.38, 0.10), hover height z = 0.20.

```
dx = -0.40,  dy = 0.08
cos(q2) = (0.1664 - 0.1184) / 0.112 = 0.4286   ->  q2 = +1.128 rad  (64.6°)
q1      = 2.944 - 0.459             = 2.485 rad  (142.4°)
q3      = 0.227 - 0.200             = 0.027 m
```

Check with forward kinematics (Section 2): x = -0.380, y = 0.100.

---

### 2. Forward Kinematics (verification)

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}x=x_0%2BL_1\cos&space;q_1%2BL_2\cos%28q_1%2Bq_2%29)

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}y=y_0%2BL_1\sin&space;q_1%2BL_2\sin%28q_1%2Bq_2%29)

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}z=z_{up}-q_3)

---

### 3. Trajectory Generation (smoothstep)

Every arm move is interpolated in joint space. For a move of duration T:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}N=\left\lfloor\frac{T}{\Delta&space;t}\right\rfloor,\quad\tau_i=\frac{i}{N},\quad&space;i=1,\dots,N)

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}s_i=\tau_i^2%283-2\tau_i%29)

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}\mathbf{q}_i=\mathbf{q}_{start}%2B%28\mathbf{q}_{target}-\mathbf{q}_{start}%29s_i)

Because s'(0) = s'(1) = 0, each move starts and ends with zero velocity, which avoids jerky motion and protects the grasp. For example, a 1.0 s move gives N = 240 steps.

---

### 4. Conveyor Motion

**IR belt** (objects move in -y at v = 0.22 m/s):

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}y_{k%2B1}=y_k-v\Delta&space;t)

The belt stops, and the IR LED turns on, when the object reaches the sensor:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}y\le&space;y_{sensor}=0.10~\mathrm{m})

**Color feeder queue** (speed 0.06 m/s, spacing d = 0.10 m). Each box cannot pass the junction or the box ahead:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}x_i^{k%2B1}=\min%28x_i^k%2Bv\Delta&space;t,&space;x_{junction},&space;x_{i-1}-d%29)

**Empty tray feeder** (moves in -x, speed 0.06 m/s, spacing 0.30 m):

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}x_i^{k%2B1}=\max%28x_i^k-v\Delta&space;t,&space;x_{junction},&space;x_{i-1}%2Bd%29)

**Tray belt** (moves in +y at 0.15 m/s, stops at y >= y_fill):

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}y_{k%2B1}=y_k%2Bv_{tray}\Delta&space;t)

---

### 5. Camera Projection (3D to pixel)

Used to project the tray slot corners and the scan area into the 2D camera image.
Given view matrix V and projection matrix P (FOV 55 deg, aspect 240/180):

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}%5Bx_c,y_c,z_c,w_c%5D=%5BX,Y,Z,1%5D&space;V&space;P)

Normalized device coordinates:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}n_x=\frac{x_c}{w_c},\quad&space;n_y=\frac{y_c}{w_c},\quad&space;n_z=\frac{z_c}{w_c})

Pixel coordinates (origin top-left, W = 240, H = 180):

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}u=%28\frac{n_x}{2}%2B\frac{1}{2}%29W)

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}v=%281-%28\frac{n_y}{2}%2B\frac{1}{2}%29%29H)

A point is valid only if 0 <= u < W and 0 <= v < H.

The code also tests both V and V transpose (matrix layout differs between libraries) and keeps the one that places the target closest to the image center:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}\mathrm{score}=%7Cu-\frac{W}{2}%7C%2B%7Cv-\frac{H}{2}%7C)

---

### 6. Color Detection (HSV thresholding)

The RGB frame is converted to HSV (OpenCV: H in [0,179], S and V in [0,255]). A binary mask is produced for each color c (1 if inside the range, otherwise 0):

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}M_c%28u,v%29=1\quad\text{if}\quad\mathbf{lo}_c\le\mathrm{HSV}%28u,v%29\le\mathbf{hi}_c)

| Color | H range | S range | V range |
|---|---|---|---|
| Red | 0-10 and 170-179 | 80-255 | 60-255 |
| Yellow | 20-35 | 80-255 | 60-255 |
| Green | 40-85 | 80-255 | 60-255 |
| Blue | 95-130 | 80-255 | 60-255 |

Red wraps around the hue circle, so its mask is the union of two ranges:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}M_{red}=M_{%5B0,10%5D}\cup&space;M_{%5B170,179%5D})

**Post-processing**

1. Restrict the mask to the ROI (scan region around the sensor stop point).
2. Morphological opening with a 3x3 kernel K (removes noise):

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}M\leftarrow%28M\ominus&space;K%29\oplus&space;K)

3. Find external contours and discard those with area A < 40 px.
4. Take the bounding box (x, y, w, h) of each remaining contour.

**Final decision:** the color with the largest bounding-box area:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}c^*=\arg\max_{c}%28w_c\cdot&space;h_c%29)

| Color | Target slot |
|---|---|
| Red | 1 |
| Blue | 2 |
| Green | 3 |
| Yellow | 4 |

---

### 7. Slot Verification (point-in-polygon)

Centroid of each detected blob:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}c_x=x%2B\frac{w}{2},\quad&space;c_y=y%2B\frac{h}{2})

The Tray camera checks whether the centroid lies inside the tray outline and inside each slot quadrilateral using `cv2.pointPolygonTest`. The overlay label is:

| Label | Meaning |
|---|---|
| `COLOR OK` | Blob in the slot has the same color as the slot |
| `COLOR! OTHER` | Wrong color in the slot |
| `COLOR -` | Slot empty |

---

### 8. Grasping (fixed constraint)

When the gripper reaches the object, the relative transform between gripper frame G and object frame C is stored:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}T_{rel}=T_G^{-1}&space;T_C)

A `JOINT_FIXED` constraint with this relative pose is created (max force 200 N). The object then follows the gripper rigidly until the constraint is removed above the slot.

---

### 9. Q-Learning Extension (slot-selection policy)

> **Proposed extension.** In the current code the slot comes from a fixed color-to-slot table. Here the agent learns that choice from rewards instead.

#### 9.1 Problem formulation

After the IR camera detects a color, the agent chooses which tray slot to place the box into.

| Element | Definition |
|---|---|
| State s | (c, m): detected color c and a 4-bit mask m of slots already filled |
| Action a | Slot index, a in {0, 1, 2, 3} |
| Reward r | +10 for the correct and empty slot, -10 otherwise |
| Episode | One tray (4 placements) |

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}s=%28c,m%29,\quad&space;c\in\{\mathrm{red},\mathrm{blue},\mathrm{green},\mathrm{yellow}\},\quad&space;m\in\{0,1\}^4)

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}N_S=4\times2^4=64,\quad&space;N_A=4)

Reward function:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}r=%2B10\quad\text{if}\quad&space;a=\mathrm{slot}%28c%29~\text{and~the~slot~is~empty})

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}r=-10\quad\text{otherwise})

A perfect tray gives a total return of +40.

#### 9.2 Action selection (epsilon-greedy)

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}a=\text{random~action}\quad\text{with~probability}\quad\varepsilon)

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}a=\arg\max_{a%27}Q%28s,a%27%29\quad\text{with~probability}\quad1-\varepsilon)

Epsilon decays after each episode:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}\varepsilon_{t%2B1}=\max%28\varepsilon_{min},\;\varepsilon_t\cdot\lambda%29)

#### 9.3 Q-value update (Bellman / temporal difference)

Temporal-difference error:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}\delta=r%2B\gamma\max_{a%27}Q%28s%27,a%27%29-Q%28s,a%29)

Update rule:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}Q%28s,a%29\leftarrow&space;Q%28s,a%29%2B\alpha\,\delta)

Equivalent expanded form:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}Q%28s,a%29\leftarrow%281-\alpha%29\,Q%28s,a%29%2B\alpha\left%5Br%2B\gamma\max_{a%27}Q%28s%27,a%27%29\right%5D)

#### 9.4 Optimal policy

At convergence, Q approaches the optimal action-value function:

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}Q^*%28s,a%29=\mathbb{E}\left%5Br%2B\gamma\max_{a%27}Q^*%28s%27,a%27%29\right%5D)

![eq](https://latex.codecogs.com/png.image?\dpi{130}\bg{white}\pi^*%28s%29=\arg\max_{a}Q^*%28s,a%29)

#### 9.5 Hyperparameters

| Symbol | Meaning | Suggested value |
|---|---|---|
| alpha | Learning rate | 0.1 |
| gamma | Discount factor | 0.9 |
| epsilon (start) | Initial exploration rate | 1.0 |
| epsilon_min | Minimum exploration rate | 0.05 |
| lambda | Epsilon decay per episode | 0.995 |

```python
slot_i = int(np.argmax(Q[state_id(color, mask)]))
```

and call `update(...)` after the tray camera verifies the placement (`COLOR OK` gives +10, `COLOR! OTHER` gives -10).
