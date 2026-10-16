import math, time, random
import numpy as np
import pybullet as p
import pybullet_data

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[!] opencv not installed -> snapshots saved as .ppm")

DT = 1/240
SLOWMO = 0

PLATE_TOP  = 0.03
CONV_X     = -0.38
BELT_TOP   = 0.095
SENSOR_Y   = 0.10
BELT_SPEED = 0.22
NUM_OBJ    = 4

TRAY_CONV_X = 0.38
TRAY_FILL_Y = -0.17
TRAY_XY     = (TRAY_CONV_X, TRAY_FILL_Y)
TRAY_IR_Y   = 0.32
TRAY_Z      = 0.124
TRAY_SPEED  = 0.15
TRAY_BACK_Y = -0.55
TRAY_END_Y  = 0.68

IR_BELT_Y_BACK   = 1.25
IR_BELT_Y_FRONT  = -0.55
IR_BELT_Y_HALF   = (IR_BELT_Y_BACK - IR_BELT_Y_FRONT) / 2
IR_BELT_Y_CENTER = (IR_BELT_Y_BACK + IR_BELT_Y_FRONT) / 2

TRAY_BELT_Y_HALF   = 0.65
TRAY_BELT_Y_CENTER = 0.05

COLOR_CONVEYOR_Y = {"red": 0.42, "blue": 0.66, "green": 0.90, "yellow": 1.14}
COLOR_CONV_X_BACK   = CONV_X - 1.20
COLOR_CONV_X_FRONT  = CONV_X
COLOR_CONV_LENGTH   = COLOR_CONV_X_FRONT - COLOR_CONV_X_BACK
COLOR_CONV_HALF_W   = 0.062
COLOR_CONV_SPACING  = 0.10
COLOR_CONV_COUNT    = 5
COLOR_CONV_SPEED    = 0.06
COLOR_CONV_JUNCTION_X = COLOR_CONV_X_FRONT - 0.04
COLOR_CONV_BACK_SPAWN_X = COLOR_CONV_X_BACK + 0.05

TRAY_FEED_Y       = TRAY_BACK_Y
TRAY_FEED_X_FRONT = TRAY_CONV_X
TRAY_FEED_X_BACK  = TRAY_CONV_X + 1.20
TRAY_FEED_LENGTH  = TRAY_FEED_X_BACK - TRAY_FEED_X_FRONT
TRAY_FEED_HALF_W  = 0.062
TRAY_FEED_SPACING = 0.30
TRAY_FEED_COUNT   = 4
TRAY_FEED_SPEED   = 0.06
TRAY_FEED_JUNCTION_X = TRAY_FEED_X_FRONT + 0.04
TRAY_FEED_BACK_SPAWN_X = TRAY_FEED_X_BACK - 0.10

SLOT_Z     = 0.146
PICK_Z     = BELT_TOP + 0.017
HOVER_Z    = 0.20

HOME_Q = [math.pi/2, 0.0, 0.0, 0.0]

BEAM_Y, BEAM_Z = 0.10, 0.88

L1, L2  = 0.28, 0.20
BASE_XY = (0.02, 0.02)
ARM_Z   = PLATE_TOP + 0.34
TIP_OFF = 0.143
TIP_UP  = ARM_Z - TIP_OFF
Q3_MAX  = 0.14
URDF_PATH = "scara.urdf"

CAM_W, CAM_H = 240, 180
IR_ROI_HALF_W = 16
IR_ROI_HALF_H = 14
TRAY_ROI_MARGIN = 8

TEAL    = [0.05, 0.45, 0.55, 1]; ORANGE = [0.93, 0.45, 0.06, 1]
OLIVE   = [0.58, 0.52, 0.22, 1]; GBASE  = [0.72, 0.78, 0.50, 1]
SILVER  = [0.80, 0.80, 0.82, 1]; DARK   = [0.22, 0.22, 0.25, 1]
GRAY    = [0.55, 0.55, 0.58, 1]; MAGENTA= [0.78, 0.08, 0.45, 1]
GREEN   = [0.15, 0.60, 0.42, 1]; BEIGE  = [0.88, 0.78, 0.55, 1]
WHITE   = [0.92, 0.92, 0.92, 1]; TRAYC  = [0.72, 0.73, 0.76, 1]

OBJ_DEFS = {"red":   [0.85,0.12,0.10,1],
            "blue":  [0.12,0.30,0.85,1],
            "green": [0.12,0.70,0.30,1],
            "yellow":[0.90,0.75,0.10,1]}
SLOT_COLORS  = ["red", "blue", "green", "yellow"]
COLOR_BGR    = {"red":(0,0,255), "green":(0,255,0),
                "blue":(255,0,0), "yellow":(0,255,255)}
COLOR_HSV = {
    "red":    [((0,80,60),(10,255,255)), ((170,80,60),(179,255,255))],
    "yellow": [((20,80,60),(35,255,255))],
    "green":  [((40,80,60),(85,255,255))],
    "blue":   [((95,80,60),(130,255,255))],
}
MIN_AREA = 40

SCARA_URDF = '''<?xml version="1.0"?>
<robot name="scara_pick_place">
  <material name="orange"><color rgba="0.93 0.45 0.06 1"/></material>
  <material name="olive"><color rgba="0.58 0.52 0.22 1"/></material>
  <material name="gbase"><color rgba="0.72 0.78 0.50 1"/></material>
  <material name="silver"><color rgba="0.80 0.80 0.82 1"/></material>
  <material name="dark"><color rgba="0.25 0.25 0.28 1"/></material>
  <link name="base_link">
    <inertial><origin xyz="0 0 0.17"/><mass value="4.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial>
    <collision><origin xyz="0 0 0.0125"/><geometry><cylinder radius="0.09" length="0.025"/></geometry></collision>
    <collision><origin xyz="0 0 0.185"/><geometry><cylinder radius="0.048" length="0.32"/></geometry></collision>
    <visual><origin xyz="0 0 0.0125"/><geometry><cylinder radius="0.09" length="0.025"/></geometry><material name="gbase"/></visual>
    <visual><origin xyz="0 0 0.185"/><geometry><cylinder radius="0.048" length="0.32"/></geometry><material name="olive"/></visual>
    <visual><origin xyz="0 0.037 0.19"/><geometry><cylinder radius="0.006" length="0.28"/></geometry><material name="silver"/></visual>
    <visual><origin xyz="-0.032 -0.0185 0.19"/><geometry><cylinder radius="0.006" length="0.28"/></geometry><material name="silver"/></visual>
    <visual><origin xyz="0.032 -0.0185 0.19"/><geometry><cylinder radius="0.006" length="0.28"/></geometry><material name="silver"/></visual>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.34"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="80" velocity="3.0"/>
    <dynamics damping="0.8"/>
  </joint>
  <link name="link1">
    <inertial><origin xyz="0.14 0 0"/><mass value="1.4"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.004"/></inertial>
    <collision><geometry><cylinder radius="0.085" length="0.04"/></geometry></collision>
    <visual><geometry><cylinder radius="0.085" length="0.04"/></geometry><material name="orange"/></visual>
    <visual><origin xyz="0 0 0.032"/><geometry><cylinder radius="0.045" length="0.025"/></geometry><material name="orange"/></visual>
    <collision><origin xyz="0.165 0 0" rpy="0 1.5708 0"/><geometry><cylinder radius="0.026" length="0.20"/></geometry></collision>
    <visual><origin xyz="0.165 0 0" rpy="0 1.5708 0"/><geometry><cylinder radius="0.026" length="0.20"/></geometry><material name="orange"/></visual>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0.28 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-2.7" upper="2.7" effort="80" velocity="3.0"/>
    <dynamics damping="0.8"/>
  </joint>
  <link name="link2">
    <inertial><origin xyz="0.10 0 0"/><mass value="0.9"/>
      <inertia ixx="0.006" ixy="0" ixz="0" iyy="0.006" iyz="0" izz="0.002"/></inertial>
    <collision><geometry><cylinder radius="0.055" length="0.045"/></geometry></collision>
    <visual><geometry><cylinder radius="0.055" length="0.045"/></geometry><material name="orange"/></visual>
    <collision><origin xyz="0.12 0 0" rpy="0 1.5708 0"/><geometry><cylinder radius="0.021" length="0.14"/></geometry></collision>
    <visual><origin xyz="0.12 0 0" rpy="0 1.5708 0"/><geometry><cylinder radius="0.021" length="0.14"/></geometry><material name="orange"/></visual>
    <collision><origin xyz="0.20 0 0"/><geometry><box size="0.075 0.052 0.040"/></geometry></collision>
    <visual><origin xyz="0.20 0 0"/><geometry><box size="0.075 0.052 0.040"/></geometry><material name="orange"/></visual>
  </link>
  <joint name="joint3" type="prismatic">
    <parent link="link2"/><child link="link3"/>
    <origin xyz="0.20 0 0"/><axis xyz="0 0 -1"/>
    <limit lower="0" upper="0.14" effort="100" velocity="1.0"/>
    <dynamics damping="2.0"/>
  </joint>
  <link name="link3">
    <inertial><origin xyz="0 0 -0.03"/><mass value="0.35"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.0004"/></inertial>
    <collision><origin xyz="0 0 -0.025"/><geometry><cylinder radius="0.008" length="0.05"/></geometry></collision>
    <visual><origin xyz="0 0 -0.025"/><geometry><cylinder radius="0.008" length="0.05"/></geometry><material name="silver"/></visual>
    <collision><origin xyz="0 0 -0.062"/><geometry><box size="0.032 0.032 0.024"/></geometry></collision>
    <visual><origin xyz="0 0 -0.062"/><geometry><box size="0.032 0.032 0.024"/></geometry><material name="dark"/></visual>
  </link>
  <joint name="joint4" type="revolute">
    <parent link="link3"/><child link="gripper"/>
    <origin xyz="0 0 -0.075"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2.0"/>
    <dynamics damping="0.3"/>
  </joint>
  <link name="gripper">
    <inertial><origin xyz="0 0 -0.03"/><mass value="0.22"/>
      <inertia ixx="0.0006" ixy="0" ixz="0" iyy="0.0006" iyz="0" izz="0.0003"/></inertial>
    <collision><origin xyz="0 0 -0.012"/><geometry><box size="0.052 0.028 0.024"/></geometry></collision>
    <visual><origin xyz="0 0 -0.012"/><geometry><box size="0.052 0.028 0.024"/></geometry><material name="dark"/></visual>
    <collision><origin xyz="0.023 0 -0.046"/><geometry><box size="0.008 0.010 0.044"/></geometry></collision>
    <visual><origin xyz="0.023 0 -0.046"/><geometry><box size="0.008 0.010 0.044"/></geometry><material name="silver"/></visual>
    <collision><origin xyz="-0.023 0 -0.046"/><geometry><box size="0.008 0.010 0.044"/></geometry></collision>
    <visual><origin xyz="-0.023 0 -0.046"/><geometry><box size="0.008 0.010 0.044"/></geometry><material name="silver"/></visual>
  </link>
  <joint name="tip_joint" type="fixed">
    <parent link="gripper"/><child link="tip_link"/><origin xyz="0 0 -0.068"/>
  </joint>
  <link name="tip_link">
    <inertial><mass value="0.001"/><inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>
  </link>
</robot>
'''

def write_urdf(path=URDF_PATH):
    with open(path, "w") as f:
        f.write(SCARA_URDF)
    return path

def add_box(half, pos, rgba, orn=(0,0,0,1), mass=0):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos, baseOrientation=orn)

def add_cyl(radius, h, pos, rgba, axis="z", mass=0):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=h)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=h, rgbaColor=rgba)
    if axis == "x":
        orn = p.getQuaternionFromEuler([0, math.pi/2, 0])
    elif axis == "y":
        orn = p.getQuaternionFromEuler([math.pi/2, 0, 0])
    else:
        orn = (0,0,0,1)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos, baseOrientation=orn)

def add_vis_box(half, pos, rgba, orn=(0,0,0,1)):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                             baseVisualShapeIndex=vis,
                             basePosition=pos, baseOrientation=orn)

def add_vis_cyl(radius, h, pos, rgba, orn=(0,0,0,1)):
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=h, rgbaColor=rgba)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                             baseVisualShapeIndex=vis,
                             basePosition=pos, baseOrientation=orn)

def add_perp_conveyor(x_start, x_end, y, belt_z, color=GRAY, accent=None, half_w=0.062):
    length = x_end - x_start
    cx = (x_start + x_end) / 2
    add_box([length/2, half_w, 0.005], [cx, y, belt_z], color)
    sw_z = belt_z - 0.030
    side_color = accent if accent is not None else DARK
    for sy in (-half_w, half_w):
        add_box([length/2, 0.008, 0.03], [cx, y+sy, sw_z], side_color)
    roller_z = belt_z - 0.045
    roller_len = half_w * 2 + 0.04
    for rx in (x_start + 0.04, x_end - 0.04):
        add_cyl(0.045, roller_len, [rx, y, roller_z], [0.16,0.16,0.20,1], axis="y")
    leg_top = sw_z - 0.03
    if leg_top <= 0:
        leg_top = 0.04
    leg_half_z = leg_top / 2
    for lx in (x_start + 0.08, x_end - 0.08):
        for ly_sign in (-1, 1):
            add_box([0.012, 0.012, leg_half_z], [lx, y + (half_w+0.01)*ly_sign, leg_half_z], DARK)

def quat_look_at(d):
    f = np.array([0.0, 0.0, -1.0])
    d = np.array(d, float); d /= np.linalg.norm(d)
    c = np.cross(f, d); s = np.linalg.norm(c)
    dot = max(-1.0, min(1.0, float(np.dot(f, d))))
    if s < 1e-8:
        return (0,0,0,1) if dot > 0 else p.getQuaternionFromAxisAngle([1,0,0], math.pi)
    return p.getQuaternionFromAxisAngle((c/s).tolist(), math.acos(dot))

def _mat4(flat):
    return np.array(flat, dtype=float).reshape(4, 4)

def _project_with(vm, pm, pt):
    clip = np.array([pt[0], pt[1], pt[2], 1.0]) @ vm @ pm
    if abs(clip[3]) < 1e-9:
        return None
    n = clip[:3] / clip[3]
    return (n[0]*0.5+0.5)*CAM_W, (1.0-(n[1]*0.5+0.5))*CAM_H

CAMS = {}

def add_gantry_camera(name, clamp_x, clamp_rgba, target, label):
    add_box([0.026, 0.026, 0.036], [clamp_x, BEAM_Y, BEAM_Z], clamp_rgba)
    body = np.array([clamp_x, BEAM_Y, 0.54 if name == "ir" else 0.46])
    tgt  = np.array(target, float)
    d = tgt - body; d /= np.linalg.norm(d)
    q = quat_look_at(d)
    wire_top, wire_bot = 0.844, body[2] + 0.030
    add_vis_cyl(0.0018, wire_top - wire_bot,
                [clamp_x, BEAM_Y, (wire_top + wire_bot)/2], [0.3, 0.3, 0.32, 1])
    add_vis_box([0.020, 0.020, 0.024], list(body), [0.05, 0.05, 0.06, 1], orn=q)
    add_vis_cyl(0.008, 0.020, list(body + d*0.030), [0.12, 0.12, 0.14, 1], orn=q)
    p.addUserDebugText(label, [clamp_x, BEAM_Y, 0.96],
                       textColorRGB=[0.05, 0.05, 0.05], textSize=1.0)
    eye_f = list(body + d*0.06)
    vm = _mat4(p.computeViewMatrix(cameraEyePosition=eye_f,
                                   cameraTargetPosition=list(tgt),
                                   cameraUpVector=[0, 1, 0]))
    pm = _mat4(p.computeProjectionMatrixFOV(fov=55, aspect=CAM_W/CAM_H,
                                            nearVal=0.01, farVal=3.0))
    def score(V, M):
        r = _project_with(V, M, tgt)
        if r is None:
            return 1e9
        return abs(r[0]-CAM_W/2) + abs(r[1]-CAM_H/2)
    if score(vm, pm) <= score(vm.T, pm.T):
        V, M = vm, pm
    else:
        V, M = vm.T, pm.T
    CAMS[name] = {"eye": eye_f, "target": list(tgt), "up": [0,1,0],
                 "vm": V, "pm": M, "quads": {}, "roi": None, "outline": None}

def project(cam, pt):
    c = CAMS[cam]
    r = _project_with(c["vm"], c["pm"], pt)
    if r is None:
        return None
    u, v = r
    if u < 0 or u >= CAM_W or v < 0 or v >= CAM_H:
        return None
    return int(u), int(v)

def _roi_from_points(pts, margin):
    xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
    x0 = max(0, min(xs)-margin); y0 = max(0, min(ys)-margin)
    x1 = min(CAM_W, max(xs)+margin); y1 = min(CAM_H, max(ys)+margin)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1-x0, y1-y0)

tray_slots, belt_objects = [], []
belt_color = {}
attached = set()
belt_running = True
placed_count = 0
led_id, beam_id = None, None
led2_id, beam2_id = None, None
led2_state = False
active_trays = []
color_feeder_queues = {c: [] for c in OBJ_DEFS.keys()}
empty_tray_feeder_queue = []
cycle_count = 0
current_filling_tray = None

def slot_positions():
    cx, cy = TRAY_XY
    return [(cx-0.044, cy+0.034), (cx+0.044, cy+0.034),
            (cx-0.044, cy-0.034), (cx+0.044, cy-0.034)]

def spawn_tray(x, y):
    halves = [[0.095, 0.075, 0.006]]
    poss   = [[0, 0, 0]]
    for dx, dy, hw, hh in [(0,  0.071, 0.099, 0.004), (0, -0.071, 0.099, 0.004),
                           (-0.091, 0, 0.004, 0.075), ( 0.091, 0, 0.004, 0.075),
                           (0, 0, 0.004, 0.064),      (0, 0, 0.084, 0.004)]:
        halves.append([hw, hh, 0.012])
        poss.append([dx, dy, 0.018])
    col = p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX]*7,
                                      halfExtents=halves,
                                      collisionFramePositions=poss)
    vis = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX]*7,
                                   halfExtents=halves,
                                   visualFramePositions=poss,
                                   rgbaColors=[TRAYC]*7)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[x, y, TRAY_Z])

def spawn_empty_tray_visual(x, y):
    halves = [[0.095, 0.075, 0.006]]
    poss   = [[0, 0, 0]]
    for dx, dy, hw, hh in [(0,  0.071, 0.099, 0.004), (0, -0.071, 0.099, 0.004),
                           (-0.091, 0, 0.004, 0.075), ( 0.091, 0, 0.004, 0.075),
                           (0, 0, 0.004, 0.064),      (0, 0, 0.084, 0.004)]:
        halves.append([hw, hh, 0.012])
        poss.append([dx, dy, 0.018])
    vis = p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX]*7,
                                   halfExtents=halves,
                                   visualFramePositions=poss,
                                   rgbaColors=[TRAYC]*7)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                             baseVisualShapeIndex=vis,
                             basePosition=[x, y, TRAY_Z])

def spawn_color_box_visual(cname, x):
    rgba = OBJ_DEFS[cname]
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.017]*3, rgbaColor=rgba)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                             baseVisualShapeIndex=vis,
                             basePosition=[x, COLOR_CONVEYOR_Y[cname], BELT_TOP+0.019])

def init_color_feeders():
    for cname in color_feeder_queues:
        color_feeder_queues[cname].clear()
    for cname in COLOR_CONVEYOR_Y:
        for i in range(COLOR_CONV_COUNT):
            x_pos = COLOR_CONV_BACK_SPAWN_X + i * COLOR_CONV_SPACING
            body = spawn_color_box_visual(cname, x_pos)
            color_feeder_queues[cname].append(body)

def animate_color_conveyors():
    for cname, q in color_feeder_queues.items():
        if not q:
            continue
        for i, body in enumerate(q):
            pos, orn = p.getBasePositionAndOrientation(body)
            cur_x = pos[0]
            new_x = cur_x + COLOR_CONV_SPEED * DT
            new_x = min(new_x, COLOR_CONV_JUNCTION_X)
            if i > 0:
                ahead_body = q[i-1]
                ahead_pos, _ = p.getBasePositionAndOrientation(ahead_body)
                max_x = ahead_pos[0] - COLOR_CONV_SPACING
                new_x = min(new_x, max_x)
            if new_x != cur_x:
                p.resetBasePositionAndOrientation(body, [new_x, pos[1], pos[2]], orn)
        last_body = q[-1]
        last_pos, _ = p.getBasePositionAndOrientation(last_body)
        if last_pos[0] > COLOR_CONV_BACK_SPAWN_X + COLOR_CONV_SPACING and len(q) < COLOR_CONV_COUNT:
            body = spawn_color_box_visual(cname, COLOR_CONV_BACK_SPAWN_X)
            q.append(body)

def transfer_color_box_to_ir(cname):
    q = color_feeder_queues.get(cname, [])
    if not q:
        return None
    front_body = q.pop(0)
    front_pos, _ = p.getBasePositionAndOrientation(front_body)
    p.removeBody(front_body)
    rgba = OBJ_DEFS[cname]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.017]*3)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.017]*3, rgbaColor=rgba)
    obj = p.createMultiBody(baseMass=0.04, baseCollisionShapeIndex=col,
                            baseVisualShapeIndex=vis,
                            basePosition=[CONV_X, front_pos[1], BELT_TOP+0.019])
    belt_objects.append(obj)
    belt_color[obj] = cname
    return obj

def wait_color_box_at_junction(cname, timeout=30.0):
    q = color_feeder_queues.get(cname, [])
    t0 = time.time()
    while time.time() - t0 < timeout:
        _step()
        if q:
            front_body = q[0]
            fp, _ = p.getBasePositionAndOrientation(front_body)
            if fp[0] >= COLOR_CONV_JUNCTION_X - 0.005:
                return True
    return False

def init_empty_tray_feeder():
    empty_tray_feeder_queue.clear()
    for i in range(TRAY_FEED_COUNT):
        x_pos = TRAY_FEED_BACK_SPAWN_X - i * TRAY_FEED_SPACING
        body = spawn_empty_tray_visual(x_pos, TRAY_FEED_Y)
        empty_tray_feeder_queue.append(body)

def animate_empty_tray_feeder():
    if not empty_tray_feeder_queue:
        return
    for i, body in enumerate(empty_tray_feeder_queue):
        pos, orn = p.getBasePositionAndOrientation(body)
        cur_x = pos[0]
        new_x = cur_x - TRAY_FEED_SPEED * DT
        new_x = max(new_x, TRAY_FEED_JUNCTION_X)
        if i > 0:
            ahead_body = empty_tray_feeder_queue[i-1]
            ahead_pos, _ = p.getBasePositionAndOrientation(ahead_body)
            min_x = ahead_pos[0] + TRAY_FEED_SPACING
            new_x = max(new_x, min_x)
        if new_x != cur_x:
            p.resetBasePositionAndOrientation(body, [new_x, pos[1], pos[2]], orn)
    last_body = empty_tray_feeder_queue[-1]
    last_pos, _ = p.getBasePositionAndOrientation(last_body)
    if last_pos[0] < TRAY_FEED_BACK_SPAWN_X - TRAY_FEED_SPACING and len(empty_tray_feeder_queue) < TRAY_FEED_COUNT:
        body = spawn_empty_tray_visual(TRAY_FEED_BACK_SPAWN_X, TRAY_FEED_Y)
        empty_tray_feeder_queue.append(body)

def wait_empty_tray_at_junction(timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _step()
        if empty_tray_feeder_queue:
            front_body = empty_tray_feeder_queue[0]
            fp, _ = p.getBasePositionAndOrientation(front_body)
            if fp[0] <= TRAY_FEED_JUNCTION_X + 0.005:
                return True
    return False

def transfer_empty_tray_to_belt():
    if not empty_tray_feeder_queue:
        return None
    front_body = empty_tray_feeder_queue.pop(0)
    p.removeBody(front_body)
    return spawn_tray(TRAY_CONV_X, TRAY_BACK_Y)

def build_world():
    global led_id, beam_id, led2_id, beam2_id
    p.resetDebugVisualizerCamera(2.2, -55, -35, [0.0, 0.10, 0.25])
    ground = p.loadURDF("plane.urdf")
    p.changeVisualShape(ground, -1, rgbaColor=[0.78, 0.86, 0.88, 1])
    add_box([1.00, 0.80, 0.015], [0.0, 0.10, 0.015], BEIGE)

    for x in (-0.58, 0.58):
        add_box([0.05, 0.05, 0.015], [x, BEAM_Y, 0.015], DARK)
        add_box([0.025, 0.025, 0.425], [x, BEAM_Y, 0.425], TEAL)
    add_box([0.63, 0.028, 0.03], [0.0, BEAM_Y, BEAM_Z], TEAL)

    add_gantry_camera("ir",   CONV_X,      GREEN,   [CONV_X, SENSOR_Y, BELT_TOP],
                      "IR CAMERA")
    add_gantry_camera("tray", TRAY_CONV_X, MAGENTA, [TRAY_CONV_X, TRAY_FILL_Y, SLOT_Z],
                      "TRAY CAMERA")

    for sx in (-0.062, 0.062):
        add_box([0.008, IR_BELT_Y_HALF+0.02, 0.03],
                [CONV_X+sx, IR_BELT_Y_CENTER, 0.065], DARK)
    add_box([0.052, IR_BELT_Y_HALF, 0.005],
            [CONV_X, IR_BELT_Y_CENTER, 0.09], GRAY)
    for ry in (IR_BELT_Y_FRONT + 0.05, IR_BELT_Y_BACK - 0.05):
        add_cyl(0.045, 0.16, [CONV_X, ry, 0.05], [0.16,0.16,0.20,1], axis="x")
    for ly in (0.55, 0.25, -0.15, -0.45):
        if IR_BELT_Y_FRONT < ly < IR_BELT_Y_BACK:
            add_box([0.07, 0.012, 0.033], [CONV_X, ly, 0.0325], DARK)

    sx = CONV_X - 0.084
    add_box([0.010, 0.016, 0.014], [sx, SENSOR_Y, 0.098], MAGENTA)
    add_box([0.008, 0.014, 0.014], [CONV_X+0.078, SENSOR_Y, 0.098], GRAY)
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.008)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.008, rgbaColor=[0, 0.8, 0.2, 1])
    led_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=[sx, SENSOR_Y, 0.122])
    beam_id = p.addUserDebugLine([sx+0.010, SENSOR_Y, 0.102],
                                 [CONV_X+0.068, SENSOR_Y, 0.102],
                                 [0.55, 0.1, 0.1], lineWidth=1)

    for cname, y_pos in COLOR_CONVEYOR_Y.items():
        rgba = OBJ_DEFS[cname]
        accent = [min(1, rgba[0]*0.6), min(1, rgba[1]*0.6), min(1, rgba[2]*0.6), 1]
        add_perp_conveyor(COLOR_CONV_X_BACK, COLOR_CONV_X_FRONT, y_pos,
                          belt_z=0.09, color=GRAY, accent=accent, half_w=COLOR_CONV_HALF_W)
        p.addUserDebugText(f"{cname.upper()} FEEDER",
                           [COLOR_CONV_X_BACK-0.06, y_pos, 0.18],
                           textColorRGB=[rgba[0], rgba[1], rgba[2]])

    for sx in (-0.062, 0.062):
        add_box([0.008, TRAY_BELT_Y_HALF+0.02, 0.03],
                [TRAY_CONV_X+sx, TRAY_BELT_Y_CENTER, 0.065], DARK)
    add_box([0.052, TRAY_BELT_Y_HALF, 0.005],
            [TRAY_CONV_X, TRAY_BELT_Y_CENTER, 0.113], GRAY)
    for ry in (TRAY_BELT_Y_CENTER + TRAY_BELT_Y_HALF - 0.05,
               TRAY_BELT_Y_CENTER - TRAY_BELT_Y_HALF + 0.05):
        add_cyl(0.045, 0.16, [TRAY_CONV_X, ry, 0.05], [0.16,0.16,0.20,1], axis="x")
    for ly in (0.55, 0.25, -0.15, -0.50):
        add_box([0.07, 0.012, 0.033], [TRAY_CONV_X, ly, 0.0325], DARK)
    add_box([0.06, 0.004, 0.02], [TRAY_CONV_X, TRAY_END_Y + 0.02, 0.135], DARK)

    tx = TRAY_CONV_X - 0.084
    add_box([0.010, 0.016, 0.014], [tx, TRAY_IR_Y, 0.132], MAGENTA)
    add_box([0.008, 0.014, 0.014], [TRAY_CONV_X+0.078, TRAY_IR_Y, 0.132], GRAY)
    col2 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.008)
    vis2 = p.createVisualShape(p.GEOM_SPHERE, radius=0.008, rgbaColor=[0, 0.8, 0.2, 1])
    led2_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col2,
                                baseVisualShapeIndex=vis2,
                                basePosition=[tx, TRAY_IR_Y, 0.156])
    beam2_id = p.addUserDebugLine([tx+0.010, TRAY_IR_Y, 0.132],
                                  [TRAY_CONV_X+0.068, TRAY_IR_Y, 0.132],
                                  [0.55, 0.1, 0.1], lineWidth=1)

    add_perp_conveyor(TRAY_FEED_X_FRONT, TRAY_FEED_X_BACK, TRAY_FEED_Y,
                      belt_z=0.113, color=GRAY, accent=OLIVE, half_w=TRAY_FEED_HALF_W)
    p.addUserDebugText("EMPTY TRAY FEEDER",
                       [TRAY_FEED_X_BACK-0.06, TRAY_FEED_Y, 0.18],
                       textColorRGB=[0.4, 0.4, 0.2])

    p.addUserDebugText("TRAY", [TRAY_CONV_X, TRAY_FILL_Y, 0.30], textColorRGB=[0.1,0.1,0.1])
    p.addUserDebugText("IR SENSOR", [CONV_X-0.02, SENSOR_Y+0.10, 0.18],
                       textColorRGB=[0.6,0.05,0.35])
    p.addUserDebugText("TRAY EXIT (IR-2)", [TRAY_CONV_X-0.06, TRAY_IR_Y, 0.22],
                       textColorRGB=[0.6,0.05,0.35])
    p.addUserDebugText("TRAY BELT TOP", [TRAY_CONV_X-0.06, TRAY_END_Y, 0.22],
                       textColorRGB=[0.6,0.05,0.35])
    p.addUserDebugText("TRAY ENTRY (back)", [TRAY_CONV_X-0.06, TRAY_BACK_Y, 0.22],
                       textColorRGB=[0.6,0.05,0.35])
    tray_slots.extend(slot_positions())

    stop_pt = [CONV_X, SENSOR_Y, BELT_TOP + 0.017]
    cpt = project("ir", stop_pt)
    if cpt is not None:
        cx, cy = cpt
        x0 = max(0, cx - IR_ROI_HALF_W); y0 = max(0, cy - IR_ROI_HALF_H)
        x1 = min(CAM_W, cx + IR_ROI_HALF_W); y1 = min(CAM_H, cy + IR_ROI_HALF_H)
        CAMS["ir"]["roi"] = (x0, y0, x1-x0, y1-y0)
    print(f"[INIT] IR scan ROI: {CAMS['ir']['roi']}")

    corners3d = [(TRAY_CONV_X-0.100, TRAY_FILL_Y-0.082, TRAY_Z+0.050),
                 (TRAY_CONV_X+0.100, TRAY_FILL_Y-0.082, TRAY_Z+0.050),
                 (TRAY_CONV_X+0.100, TRAY_FILL_Y+0.082, TRAY_Z+0.050),
                 (TRAY_CONV_X-0.100, TRAY_FILL_Y+0.082, TRAY_Z+0.050)]
    pts = [project("tray", c) for c in corners3d]
    if all(pp is not None for pp in pts):
        CAMS["tray"]["outline"] = [(int(u), int(v)) for (u, v) in pts]
        CAMS["tray"]["roi"] = _roi_from_points(CAMS["tray"]["outline"], TRAY_ROI_MARGIN)

    quads = {}
    for i, cname in enumerate(SLOT_COLORS):
        cx, cy = tray_slots[i]
        corners = [(cx-0.040, cy-0.028, SLOT_Z+0.001), (cx+0.040, cy-0.028, SLOT_Z+0.001),
                   (cx+0.040, cy+0.028, SLOT_Z+0.001), (cx-0.040, cy+0.028, SLOT_Z+0.001)]
        pts = [project("tray", c) for c in corners]
        if all(pp is not None for pp in pts):
            quads[cname] = [(int(u), int(v)) for (u, v) in pts]
    CAMS["tray"]["quads"] = quads

    init_color_feeders()
    init_empty_tray_feeder()
    print(f"[INIT] 4 color feeders (animated belts): {sum(len(q) for q in color_feeder_queues.values())} boxes total")
    print(f"[INIT] Empty tray feeder (animated belt): {len(empty_tray_feeder_queue)} trays queued")

def on_belt(obj):
    (x, y, z), _ = p.getBasePositionAndOrientation(obj)
    return obj not in attached and abs(x-CONV_X) < 0.06 and IR_BELT_Y_FRONT < y < IR_BELT_Y_BACK and z < 0.18

def set_led(detected):
    global beam_id
    p.changeVisualShape(led_id, -1, rgbaColor=[0.9,0.1,0.1,1] if detected else [0,0.8,0.2,1])
    p.removeUserDebugItem(beam_id)
    sx = CONV_X - 0.084
    beam_id = p.addUserDebugLine([sx+0.010, SENSOR_Y, 0.102],
                                 [CONV_X+0.068, SENSOR_Y, 0.102],
                                 [0.9,0.1,0.1] if detected else [0.55,0.1,0.1],
                                 lineWidth=2 if detected else 1)

def set_led2(detected):
    global beam2_id
    tx = TRAY_CONV_X - 0.084
    p.changeVisualShape(led2_id, -1, rgbaColor=[0.9,0.1,0.1,1] if detected else [0,0.8,0.2,1])
    p.removeUserDebugItem(beam2_id)
    beam2_id = p.addUserDebugLine([tx+0.010, TRAY_IR_Y, 0.132],
                                  [TRAY_CONV_X+0.068, TRAY_IR_Y, 0.132],
                                  [0.9,0.1,0.1] if detected else [0.55,0.1,0.1],
                                  lineWidth=2 if detected else 1)

def _belt():
    global belt_running
    for obj in belt_objects:
        if not on_belt(obj):
            continue
        y = p.getBasePositionAndOrientation(obj)[0][1]
        if belt_running:
            p.resetBaseVelocity(obj, [0, -BELT_SPEED, 0], [0, 0, 0])
            if y <= SENSOR_Y:
                belt_running = False
                set_led(True)
                print("[IR-1] Object reached sensor -> IR belt STOPPED")
        else:
            p.resetBaseVelocity(obj, [0, 0, 0], [0, 0, 0])

def _tray_conv():
    global led2_state, current_filling_tray
    if not active_trays:
        if led2_state:
            set_led2(False)
            led2_state = False
        return
    trays_to_remove = []
    led2_should_be_on = False
    for tray in active_trays:
        body = tray["body"]
        pos, orn = p.getBasePositionAndOrientation(body)
        if tray["state"] == "incoming":
            ny = pos[1] + TRAY_SPEED * DT
            if ny >= TRAY_FILL_Y:
                ny = TRAY_FILL_Y
                tray["state"] = "filling"
                print(f"[TRAY] Tray {tray['cycle']} reached fill position -> STOP for filling")
            d = ny - pos[1]
            p.resetBasePositionAndOrientation(body, [pos[0], ny, pos[2]], orn)
            for c in tray["cubes"]:
                cp, co = p.getBasePositionAndOrientation(c)
                p.resetBasePositionAndOrientation(c, [cp[0], cp[1]+d, cp[2]], co)
                p.resetBaseVelocity(c, [0, 0, 0], [0, 0, 0])
        elif tray["state"] == "filling":
            pass
        elif tray["state"] in ("outgoing", "exiting"):
            ny = pos[1] + TRAY_SPEED * DT
            if tray["state"] == "outgoing" and not tray["ir2_triggered"] and ny >= TRAY_IR_Y:
                tray["ir2_triggered"] = True
                tray["state"] = "exiting"
                print(f"[IR-2] Tray {tray['cycle']} crossed exit sensor -> continuing to top")
            if ny >= TRAY_END_Y:
                ny = TRAY_END_Y
                trays_to_remove.append(tray)
                print(f"[TRAY] Tray {tray['cycle']} reached top -> removed (loop)")
                continue
            d = ny - pos[1]
            p.resetBasePositionAndOrientation(body, [pos[0], ny, pos[2]], orn)
            for c in tray["cubes"]:
                cp, co = p.getBasePositionAndOrientation(c)
                p.resetBasePositionAndOrientation(c, [cp[0], cp[1]+d, cp[2]], co)
                p.resetBaseVelocity(c, [0, 0, 0], [0, 0, 0])
            if tray["state"] == "exiting":
                led2_should_be_on = True
    if led2_should_be_on != led2_state:
        set_led2(led2_should_be_on)
        led2_state = led2_should_be_on
    for tray in trays_to_remove:
        for c in tray["cubes"]:
            p.removeBody(c)
        p.removeBody(tray["body"])
        if tray in active_trays:
            active_trays.remove(tray)
        if current_filling_tray is tray:
            current_filling_tray = None

def _render(eye, target, up):
    vm = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target,
                             cameraUpVector=up)
    pm = p.computeProjectionMatrixFOV(fov=55, aspect=CAM_W/CAM_H, nearVal=0.01, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(CAM_W, CAM_H, viewMatrix=vm, projectionMatrix=pm,
                                       renderer=p.ER_TINY_RENDERER)
    return np.reshape(rgb, (CAM_H, CAM_W, 4))[:, :, :3].astype(np.uint8)

def detect_objects(img, roi=None):
    if not HAS_CV2:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    results = []
    for name, ranges in COLOR_HSV.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for (lo, hi) in ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        if roi is not None:
            rx, ry, rw, rh = roi
            cut = np.zeros_like(mask)
            cut[ry:ry+rh, rx:rx+rw] = mask[ry:ry+rh, rx:rx+rw]
            mask = cut
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            results.append((name, (x, y, w, h)))
    return results

def identify_color_at_ir():
    if not HAS_CV2:
        return None
    img = _render(CAMS["ir"]["eye"], CAMS["ir"]["target"], CAMS["ir"]["up"])
    roi = CAMS["ir"]["roi"]
    best, best_area = None, 0
    for name, (x, y, w, h) in detect_objects(img, roi=roi):
        if w*h > best_area:
            best, best_area = name, w*h
    return best

_last_cam_t = 0.0

def update_cams(force=False):
    global _last_cam_t
    now = time.time()
    if not force and now - _last_cam_t < 0.15:
        return
    _last_cam_t = now

    tray_img = _render(CAMS["tray"]["eye"], CAMS["tray"]["target"], CAMS["tray"]["up"])
    ir_img   = _render(CAMS["ir"]["eye"],   CAMS["ir"]["target"],   CAMS["ir"]["up"])

    if HAS_CV2:
        t = cv2.resize(tray_img, (CAM_W*2, CAM_H*2), interpolation=cv2.INTER_NEAREST)
        i = cv2.resize(ir_img,   (CAM_W*2, CAM_H*2), interpolation=cv2.INTER_NEAREST)

        ir_roi = CAMS["ir"]["roi"]
        if ir_roi is not None:
            rx, ry, rw, rh = ir_roi
            cv2.rectangle(i, (rx*2, ry*2), ((rx+rw)*2, (ry+rh)*2), (220, 220, 220), 1)
            cv2.putText(i, "SCAN", (rx*2, ry*2-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        ir_dets = detect_objects(ir_img, roi=ir_roi)
        if ir_dets:
            name, (x, y, w, h) = max(ir_dets, key=lambda d: d[1][2]*d[1][3])
            cv2.rectangle(i, (x*2, y*2), ((x+w)*2, (y+h)*2), COLOR_BGR[name], 2)
            cv2.putText(i, name.upper(), (x*2, max(12, y*2-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BGR[name], 2)
            cv2.putText(i, f"OBJECT: {name.upper()}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BGR[name], 2)
        else:
            cv2.putText(i, "OBJECT: -", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

        det = (not belt_running)
        cv2.putText(i, f"BELT: {'STOPPED' if det else 'RUNNING'}",
                    (8, CAM_H*2-16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if det else (0, 255, 0), 1)

        tray_roi  = CAMS["tray"]["roi"]
        outline   = CAMS["tray"]["outline"]
        quads     = CAMS["tray"]["quads"]
        active    = any(t["state"] == "filling" for t in active_trays)

        if active and outline is not None:
            oq = np.array([[u*2, v*2] for (u, v) in outline], np.int32)
            cv2.polylines(t, [oq], True, (255, 255, 255), 1)

        if active and tray_roi is not None:
            dets = detect_objects(tray_img, roi=tray_roi)
            slot_status = {c: None for c in SLOT_COLORS}
            for name, (x, y, w, h) in dets:
                ccx, ccy = x + w/2, y + h/2
                inside = False
                if outline is not None:
                    inside = cv2.pointPolygonTest(
                        np.array(outline, np.int32), (float(ccx), float(ccy)), False) >= 0
                if not inside:
                    continue
                cv2.rectangle(t, (x*2, y*2), ((x+w)*2, (y+h)*2), COLOR_BGR[name], 1)
                for qname, quad in quads.items():
                    if cv2.pointPolygonTest(np.array(quad, np.int32),
                                            (float(ccx), float(ccy)), False) >= 0:
                        slot_status[qname] = name

            for qname, quad in quads.items():
                pts = np.array([[u*2, v*2] for (u, v) in quad], np.int32)
                cv2.polylines(t, [pts], True, COLOR_BGR[qname], 2)
                ux = min(q[0] for q in quad)*2 + 3
                vy = min(q[1] for q in quad)*2 - 5
                if slot_status[qname] == qname:
                    lbl = f"{qname.upper()} OK"
                elif slot_status[qname] is not None:
                    lbl = f"{qname.upper()}! {slot_status[qname].upper()}"
                else:
                    lbl = f"{qname.upper()} -"
                cv2.putText(t, lbl, (ux, max(12, vy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BGR[qname], 1)

        n_active = len(active_trays)
        n_incoming = sum(1 for t_ in active_trays if t_["state"] == "incoming")
        n_filling = sum(1 for t_ in active_trays if t_["state"] == "filling")
        n_exiting = sum(1 for t_ in active_trays if t_["state"] == "exiting")
        cv2.putText(t, f"TRAY: {placed_count}/{NUM_OBJ} slots filled",
                    (8, CAM_H*2-80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(t, f"ACTIVE: {n_active} | IN: {n_incoming} | FILL: {n_filling} | OUT: {n_exiting}",
                    (8, CAM_H*2-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(t, f"CYCLE: {cycle_count} | LOOPING -> TOP",
                    (8, CAM_H*2-16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        cv2.imshow("Tray Camera (gantry beam - magenta)", t[:, :, ::-1])
        cv2.imshow("IR Camera (gantry beam - green)",     i[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == 27:
            p.disconnect()
    else:
        for name, img in (("tray_cam", tray_img), ("ir_cam", ir_img)):
            with open(f"{name}.ppm", "wb") as f:
                f.write(f"P6\n{CAM_W} {CAM_H}\n255\n".encode())
                f.write(img.tobytes())

robot, JOINTS, GRIP = None, [0, 1, 2, 3], 3
FORCES = [120, 120, 120, 20]

def load_robot():
    global robot
    robot = p.loadURDF(write_urdf(), basePosition=[BASE_XY[0], BASE_XY[1], PLATE_TOP],
                       useFixedBase=True)

def _step(n=1):
    for _ in range(n):
        animate_color_conveyors()
        animate_empty_tray_feeder()
        _belt()
        _tray_conv()
        p.stepSimulation()
        if SLOWMO > 0:
            time.sleep(DT * SLOWMO)
    update_cams()

def scara_ik(x, y, z):
    dx, dy = x - BASE_XY[0], y - BASE_XY[1]
    q3 = max(0.0, min(Q3_MAX, TIP_UP - z))
    c2 = max(-1.0, min(1.0, (dx*dx + dy*dy - L1*L1 - L2*L2) / (2*L1*L2)))
    a2 = math.acos(c2)
    cur = p.getJointState(robot, JOINTS[1])[0]
    best = None
    for q2 in (a2, -a2):
        q1 = math.atan2(dy, dx) - math.atan2(L2*math.sin(q2), L1 + L2*math.cos(q2))
        q1 = (q1 + math.pi) % (2*math.pi) - math.pi
        if best is None or abs(q2 - cur) < abs(best[1] - cur):
            best = (q1, q2)
    return best[0], best[1], q3

def goto_joint(target, duration=1.0):
    start = [p.getJointState(robot, j)[0] for j in JOINTS]
    N = max(1, int(duration / DT))
    for i in range(1, N+1):
        s = i / N
        s = s*s*(3 - 2*s)
        q = [start[k] + (target[k]-start[k])*s for k in range(4)]
        p.setJointMotorControlArray(robot, JOINTS, p.POSITION_CONTROL,
                                    targetPositions=q, forces=FORCES)
        _step(1)

def move_tip(x, y, z, duration=1.0):
    q1, q2, q3 = scara_ik(x, y, z)
    q4 = p.getJointState(robot, JOINTS[3])[0]
    goto_joint([q1, q2, q3, q4], duration)

def go_home(duration=1.0):
    goto_joint(HOME_Q, duration)

def settle(t=0.3):
    tgt = [p.getJointState(robot, j)[0] for j in JOINTS]
    for _ in range(max(1, int(t / DT))):
        p.setJointMotorControlArray(robot, JOINTS, p.POSITION_CONTROL,
                                    targetPositions=tgt, forces=FORCES)
        _step(1)

def pause(dur):
    for _ in range(max(1, int(dur / DT))):
        _step(1)

def grasp(cube):
    ls = p.getLinkState(robot, GRIP, computeForwardKinematics=True)
    gp, go = ls[4], ls[5]
    cp, co = p.getBasePositionAndOrientation(cube)
    ip, io = p.invertTransform(gp, go)
    rel_pos, rel_orn = p.multiplyTransforms(ip, io, cp, co)
    cid = p.createConstraint(robot, GRIP, cube, -1, p.JOINT_FIXED, [0,0,0],
                             rel_pos, [0,0,0,1], [0,0,0,1], rel_orn)
    p.changeConstraint(cid, maxForce=200)
    attached.add(cube)
    return cid

def wait_object_at_sensor(timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _step()
        if belt_objects and not belt_running:
            return belt_objects[0]
    raise RuntimeError("Timeout: object never reached the sensor")

def wait_tray_at_fill(timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _step()
        if current_filling_tray is not None and current_filling_tray["state"] == "filling":
            return current_filling_tray
    raise RuntimeError("Timeout: tray never reached fill position")

def run_task():
    global belt_running, placed_count, cycle_count, current_filling_tray
    color_to_slot = {c: i for i, c in enumerate(SLOT_COLORS)}
    print("=== SYSTEM START ===")
    go_home(1.0)
    try:
        while p.getConnectionInfo()["isConnected"]:
            cycle_count += 1
            print(f"\n===== CYCLE {cycle_count}: waiting for empty tray at feeder junction =====")
            wait_empty_tray_at_junction()
            new_tray_body = transfer_empty_tray_to_belt()
            new_tray_cubes = []
            new_tray = {
                "body": new_tray_body,
                "cubes": new_tray_cubes,
                "state": "incoming",
                "ir2_triggered": False,
                "cycle": cycle_count
            }
            active_trays.append(new_tray)
            placed_count = 0
            set_led2(False)
            print(f"[TRAY FEEDER] empty tray transferred to tray belt -> moving to fill position")
            current_filling_tray = new_tray
            wait_tray_at_fill()

            order = list(OBJ_DEFS.keys())
            random.shuffle(order)
            print(f"[CYCLE] color order this cycle: {order}")

            for cname in order:
                print(f"[{cname.upper()} FEEDER] waiting for box at junction...")
                wait_color_box_at_junction(cname)
                obj = transfer_color_box_to_ir(cname)
                belt_running = True
                set_led(False)
                print(f"[{cname.upper()} FEEDER] box transferred to IR belt at y={COLOR_CONVEYOR_Y[cname]:.2f} -> running to IR sensor")

                obj = wait_object_at_sensor()
                seen = identify_color_at_ir()
                color = seen if seen else belt_color[obj]
                slot_i = color_to_slot[color]
                print(f"[IRCAM] scan result: {color.upper()} -> target slot {slot_i+1}")

                pos_obj = p.getBasePositionAndOrientation(obj)[0]
                px, py = pos_obj[0], pos_obj[1]
                print(f"[SCARA] picking {color.upper()} at ({px:.2f}, {py:.2f})")
                move_tip(px, py, HOVER_Z, 1.0)
                move_tip(px, py, PICK_Z, 0.7)
                cid = grasp(obj); settle(0.3)
                move_tip(px, py, HOVER_Z, 0.7)

                sx, sy = tray_slots[slot_i]
                print(f"[SCARA] placing into {color.upper()} slot ({sx:.2f}, {sy:.2f})")
                move_tip(sx, sy, HOVER_Z, 1.2)
                move_tip(sx, sy, SLOT_Z, 0.7)
                p.removeConstraint(cid); attached.discard(obj)
                belt_objects.remove(obj)
                new_tray_cubes.append(obj)
                placed_count += 1
                settle(0.4)
                move_tip(sx, sy, HOVER_Z, 0.6)

                print(f"[TRAY] filled {placed_count}/{NUM_OBJ}")
                go_home(1.0)

            print(f"[TRAY] Tray {cycle_count} full -> starts moving toward top (loop)")
            new_tray["state"] = "outgoing"
            current_filling_tray = None
            print(f"[TRAY] Cycle {cycle_count} done. Tray continues on belt; next empty tray incoming (loop)")
    except RuntimeError as e:
        print(f"[EXIT] {e}")
    update_cams(force=True)

def manual_mode():
    sliders = [p.addUserDebugParameter("Base (q1)", -3.14, 3.14, 0),
               p.addUserDebugParameter("Elbow (q2)", -2.7, 2.7, 0),
               p.addUserDebugParameter("Down Z (q3)", 0, Q3_MAX, 0.05),
               p.addUserDebugParameter("Gripper rotate (q4)", -3.14, 3.14, 0)]
    while p.getConnectionInfo()["isConnected"]:
        target = [p.readUserDebugParameter(s) for s in sliders]
        p.setJointMotorControlArray(robot, JOINTS, p.POSITION_CONTROL,
                                    targetPositions=target, forces=FORCES)
        _step(1)

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(DT)
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    load_robot()
    build_world()
    for j, q0 in zip(JOINTS, HOME_Q):
        p.resetJointState(robot, j, q0)
    settle(0.5)
    update_cams(force=True)
    run_task()
    manual_mode()
    if HAS_CV2:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()