import cv2
import time
import json
import socket
import sys
import threading
import numpy as np
from pathlib import Path
from pidnet_road_segmenter import PIDNetRoadSegmenter
import config as cfg

# =========================================================
# CONFIGURATION
# =========================================================
# Sử dụng cấu hình từ config.py và checkpoints_floor
CAM_ID = cfg.CAM_ID
WEIGHT_PATH = Path("checkpoints_floor/pidnet_floor_best.pt")
PIDNET_ROOT = cfg.PIDNET_ROOT
MODEL_SIZE = cfg.PIDNET_MODEL_SIZE

FRAME_WIDTH = cfg.FRAME_WIDTH
FRAME_HEIGHT = cfg.FRAME_HEIGHT
SEG_WIDTH = cfg.SEG_WIDTH
SEG_HEIGHT = cfg.SEG_HEIGHT

SEND_TO_ESP_SERIAL = getattr(cfg, "SEND_TO_ESP_SERIAL", False)
ESP_SERIAL_PORT = getattr(cfg, "ESP_SERIAL_PORT", "COM5")
ESP_SERIAL_BAUD = getattr(cfg, "ESP_SERIAL_BAUD", 115200)
ESP_SERIAL_HZ = getattr(cfg, "ESP_SERIAL_HZ", 10.0)
ESP_SERIAL_DEBUG = getattr(cfg, "ESP_SERIAL_DEBUG", False)
ESP_SERIAL_RX_DEBUG = getattr(cfg, "ESP_SERIAL_RX_DEBUG", False)
ESP_SERIAL_KEEPALIVE_SEC = max(0.1, float(getattr(cfg, "ESP_SERIAL_KEEPALIVE_SEC", 0.5)))
OUTPUT_DIR = cfg.OUTPUT_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

# PIDNet Custom Model Config (0 = background, 1 = floor)
NUM_CLASSES = 2
ROAD_CLASS_ID = 1  # Floor is class 1

# ROI for Navigation (bottom half of the image)
NAV_ROI_TOP = FRAME_HEIGHT // 2
NAV_ROI_BOTTOM = FRAME_HEIGHT

# Bumper ROI for obstacle detection (bottom-center area)
BUMPER_Y_START = int(FRAME_HEIGHT * getattr(cfg, "GROUND_ROI_Y_START_RATIO", 0.75))
BUMPER_Y_END = FRAME_HEIGHT - 1
BUMPER_TOP_X_START = int(FRAME_WIDTH * getattr(cfg, "GROUND_ROI_TOP_X_START_RATIO", 0.10))
BUMPER_TOP_X_END = int(FRAME_WIDTH * getattr(cfg, "GROUND_ROI_TOP_X_END_RATIO", 0.90))
BUMPER_BOTTOM_X_START = int(FRAME_WIDTH * getattr(cfg, "GROUND_ROI_BOTTOM_X_START_RATIO", 0.25))
BUMPER_BOTTOM_X_END = int(FRAME_WIDTH * getattr(cfg, "GROUND_ROI_BOTTOM_X_END_RATIO", 0.75))
BUMPER_TRAPEZOID_POINTS = np.array([
    [BUMPER_TOP_X_START, BUMPER_Y_START],
    [BUMPER_TOP_X_END, BUMPER_Y_START],
    [BUMPER_BOTTOM_X_END, BUMPER_Y_END],
    [BUMPER_BOTTOM_X_START, BUMPER_Y_END],
], dtype=np.int32)
BUMPER_TRAPEZOID_MASK = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
cv2.fillPoly(BUMPER_TRAPEZOID_MASK, [BUMPER_TRAPEZOID_POINTS], 255)
BUMPER_TRAPEZOID_AREA = cv2.countNonZero(BUMPER_TRAPEZOID_MASK)
BUMPER_OBSTACLE_THRESHOLD = float(getattr(cfg, "BUMPER_OBSTACLE_THRESHOLD", 0.45))
BUMPER_CENTER_X = FRAME_WIDTH // 2
BUMPER_LEFT_AREA = cv2.countNonZero(BUMPER_TRAPEZOID_MASK[:, :BUMPER_CENTER_X])
BUMPER_RIGHT_AREA = cv2.countNonZero(BUMPER_TRAPEZOID_MASK[:, BUMPER_CENTER_X:])
DEAD_END_SIDE_THRESHOLD = float(getattr(cfg, "DEAD_END_SIDE_THRESHOLD", 0.45))
POLE_DETECT_COLUMNS = max(1, int(getattr(cfg, "POLE_DETECT_COLUMNS", 15)))
POLE_DETECT_ROWS = max(1, int(getattr(cfg, "POLE_DETECT_ROWS", 3)))
POLE_DETECT_Y_START_RATIO = float(getattr(cfg, "POLE_DETECT_Y_START_RATIO", 0.45))
POLE_DETECT_X_START_RATIO = float(getattr(cfg, "POLE_DETECT_X_START_RATIO", 0.05))
POLE_DETECT_X_END_RATIO = float(getattr(cfg, "POLE_DETECT_X_END_RATIO", 0.95))
POLE_FLOOR_THRESHOLD = float(getattr(cfg, "POLE_FLOOR_THRESHOLD", 0.65))
POLE_MIN_CELL_AREA = max(1, int(getattr(cfg, "POLE_MIN_CELL_AREA", 150)))
POLE_Y_START = BUMPER_Y_START + int(
    (BUMPER_Y_END + 1 - BUMPER_Y_START) * POLE_DETECT_Y_START_RATIO
)
POLE_X_START = max(BUMPER_BOTTOM_X_START, int(FRAME_WIDTH * POLE_DETECT_X_START_RATIO))
POLE_X_END = min(BUMPER_BOTTOM_X_END, int(FRAME_WIDTH * POLE_DETECT_X_END_RATIO))
POLE_CELL_ROIS = []
for pole_row in range(POLE_DETECT_ROWS):
    pole_cell_y_start = POLE_Y_START + (
        (BUMPER_Y_END + 1 - POLE_Y_START) * pole_row // POLE_DETECT_ROWS
    )
    pole_cell_y_end = POLE_Y_START + (
        (BUMPER_Y_END + 1 - POLE_Y_START) * (pole_row + 1) // POLE_DETECT_ROWS
    )
    for pole_col in range(POLE_DETECT_COLUMNS):
        pole_x_start = POLE_X_START + (
            (POLE_X_END - POLE_X_START) * pole_col // POLE_DETECT_COLUMNS
        )
        pole_x_end = POLE_X_START + (
            (POLE_X_END - POLE_X_START) * (pole_col + 1) // POLE_DETECT_COLUMNS
        )
        pole_mask = BUMPER_TRAPEZOID_MASK[
            pole_cell_y_start:pole_cell_y_end,
            pole_x_start:pole_x_end,
        ]
        pole_area = cv2.countNonZero(pole_mask)
        if pole_area >= POLE_MIN_CELL_AREA:
            POLE_CELL_ROIS.append(
                (pole_x_start, pole_x_end, pole_cell_y_start, pole_cell_y_end, pole_mask, pole_area)
            )

# Speed scaling for obstacle behavior
NORMAL_SPEED_SCALE = float(getattr(cfg, "NORMAL_SPEED_SCALE", 1.0))
OBSTACLE_SLOW_SPEED_SCALE = float(getattr(cfg, "OBSTACLE_SLOW_SPEED_SCALE", 0.35))
AVOIDANCE_SPEED_SCALE = float(getattr(cfg, "AVOIDANCE_SPEED_SCALE", 0.45))
EMERGENCY_SPEED_SCALE = float(getattr(cfg, "EMERGENCY_SPEED_SCALE", 0.0))
SLOWDOWN_STEERING_SCALE = float(getattr(cfg, "SLOWDOWN_STEERING_SCALE", 0.25))
TANK_TURN_STEERING_VALUE = float(getattr(cfg, "TANK_TURN_STEERING_VALUE", 2.0))
SLOWDOWN_DURATION_SEC = float(getattr(cfg, "SLOWDOWN_DURATION_SEC", 0.5))
AVOIDANCE_MIN_LOCK_SEC = float(getattr(cfg, "AVOIDANCE_MIN_LOCK_SEC", 1.2))
AVOIDANCE_TIMEOUT_SEC = float(getattr(cfg, "AVOIDANCE_TIMEOUT_SEC", 2.5))
RECOVERY_BACK_SEC = float(getattr(cfg, "RECOVERY_BACK_SEC", 0.7))
RECOVERY_TURN_SEC = float(getattr(cfg, "RECOVERY_TURN_SEC", 1.0))
RECOVERY_BACK_SPEED_SCALE = float(getattr(cfg, "RECOVERY_BACK_SPEED_SCALE", 0.45))
RECOVERY_TURN_SPEED_SCALE = float(getattr(cfg, "RECOVERY_TURN_SPEED_SCALE", 0.50))
OBSTACLE_CLEAR_FRAMES = int(getattr(cfg, "OBSTACLE_CLEAR_FRAMES", 6))
TURN_DIRECTION_HOLD_SEC = float(getattr(cfg, "TURN_DIRECTION_HOLD_SEC", 4.0))
ESCAPE_CLEAR_FRAMES = int(getattr(cfg, "ESCAPE_CLEAR_FRAMES", 8))
CORNER_REENTRY_WINDOW_SEC = float(getattr(cfg, "CORNER_REENTRY_WINDOW_SEC", 2.0))
OBSTACLE_CONFIRMATION_SEC = float(getattr(cfg, "OBSTACLE_CONFIRMATION_SEC", 0.5))
NAV_MIN_FLOOR_PIXELS = max(1, int(getattr(cfg, "NAV_MIN_FLOOR_PIXELS", 100)))
NO_GROUND_BACK_SEC = float(getattr(cfg, "NO_GROUND_BACK_SEC", 0.8))
NO_GROUND_BACK_SPEED_SCALE = float(getattr(cfg, "NO_GROUND_BACK_SPEED_SCALE", 0.40))


# Exponential Moving Average coefficient for smoothing the centroid
EMA_ALPHA = 0.15

# =========================================================
# PID CONTROLLER CLASS
# =========================================================
class PIDController:
    def __init__(self, Kp=2.0, Ki=0.02, Kd=0.5, dt=0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error):
        self.integral += error * self.dt
        # Clamp integral to prevent windup
        self.integral = max(min(self.integral, 5.0), -5.0)
        
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        
        p_term = self.Kp * error
        i_term = self.Ki * self.integral
        d_term = self.Kd * derivative
        
        output = p_term + i_term + d_term
        # Clamp output steering value to [-1.0, 1.0]
        output = max(min(output, 1.0), -1.0)
        
        return output, p_term, i_term, d_term

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

# =========================================================
# VISUALIZATION HELPERS
# =========================================================
def draw_title(image, title):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    return out

def draw_steering_needle(image, steering_val, status_text, status_color):
    """
    Draws a steering dial gauge at the bottom center of the image.
    steering_val: float between -1.0 (left) and 1.0 (right)
    """
    out = image.copy()
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT - 35
    r = 60
    
    # Draw dial background semi-circle
    cv2.ellipse(out, (cx, cy), (r, r), 0, 180, 360, (50, 50, 50), -1)
    cv2.ellipse(out, (cx, cy), (r, r), 0, 180, 360, (200, 200, 200), 2)
    
    # Tick marks for -1.0, -0.5, 0.0, 0.5, 1.0
    for angle_deg in [180, 225, 270, 315, 360]:
        angle_rad = np.deg2rad(angle_deg)
        x1 = int(cx + (r - 8) * np.cos(angle_rad))
        y1 = int(cy + (r - 8) * np.sin(angle_rad))
        x2 = int(cx + r * np.cos(angle_rad))
        y2 = int(cy + r * np.sin(angle_rad))
        cv2.line(out, (x1, y1), (x2, y2), (200, 200, 200), 2)

    # Calculate needle angle: steering_val is [-1.0, 1.0] -> angle is [180, 360] deg
    # Center (straight) is 270 deg
    needle_angle_deg = 270 + (steering_val * 90)
    needle_angle_rad = np.deg2rad(needle_angle_deg)
    
    nx = int(cx + (r - 10) * np.cos(needle_angle_rad))
    ny = int(cy + (r - 10) * np.sin(needle_angle_rad))
    
    # Draw needle shadow
    cv2.line(out, (cx, cy), (nx, ny), (0, 0, 0), 4)
    # Draw needle
    cv2.line(out, (cx, cy), (nx, ny), (0, 165, 255), 2) # Orange needle
    cv2.circle(out, (cx, cy), 6, (0, 165, 255), -1)
    cv2.circle(out, (cx, cy), 3, (255, 255, 255), -1)
    
    # Text tags for Left/Right
    cv2.putText(out, "L", (cx - r - 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(out, "R", (cx + r + 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display rotation value
    val_text = f"Steer: {steering_val:+.2f}"
    cv2.putText(out, val_text, (cx - 45, cy - r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    
    return out

def open_esp_serial():
    if not SEND_TO_ESP_SERIAL:
        return None

    try:
        import serial
    except ImportError:
        print("[ERROR] Chua cai pyserial. Chay: py -m pip install pyserial")
        return None

    try:
        esp = serial.Serial(
            ESP_SERIAL_PORT,
            ESP_SERIAL_BAUD,
            timeout=0.02,
            write_timeout=0.5
        )
        esp.setDTR(False)
        esp.setRTS(False)
        time.sleep(2.0)
        drain_esp_serial(esp)
        print(f"[ESP] Serial connected: {ESP_SERIAL_PORT} @ {ESP_SERIAL_BAUD}")
        send_esp_serial_line(esp, '{"cmd":"json_ping"}')
        print('[ESP_TX] {"cmd":"json_ping"}')
        time.sleep(0.1)
        drain_esp_serial(esp)
        return esp
    except Exception as e:
        print(f"[ERROR] Khong mo duoc ESP serial {ESP_SERIAL_PORT}: {e}")
        try:
            from serial.tools import list_ports
            available_ports = [port.device for port in list_ports.comports()]
            ports_text = ", ".join(available_ports) if available_ports else "khong co cong COM nao"
            print(f"[ESP] Cong COM dang co: {ports_text}")
        except Exception:
            pass
        return None

def drain_esp_serial(esp_serial, force_print=False):
    if esp_serial is None:
        return

    try:
        while esp_serial.in_waiting > 0:
            line = esp_serial.readline().decode("utf-8", errors="replace").strip()
            if line:
                print(f"[ESP_RX] {line}")
    except Exception as e:
        print(f"[WARN] Doc ESP serial loi: {e}")

def read_esp_serial_for(esp_serial, seconds):
    t_end = time.time() + seconds
    while time.time() < t_end:
        drain_esp_serial(esp_serial)
        time.sleep(0.02)

def send_esp_serial_line(esp_serial, line):
    esp_serial.write((line.rstrip("\r\n") + "\n").encode("utf-8"))
    esp_serial.flush()

def clamp_unit(value):
    return max(-1.0, min(1.0, float(value)))

def calculate_motor_outputs(steering_value, speed_scale, motion_mode):
    speed = max(0.0, min(1.0, float(speed_scale)))
    steering = clamp_unit(steering_value)

    if motion_mode == "STOP":
        return 0.0, 0.0

    if motion_mode == "RECOVERY_BACK":
        return -speed, -speed

    if motion_mode in ("AVOIDANCE_TURN", "RECOVERY_TURN"):
        if steering >= 0:
            return speed, -speed
        return -speed, speed

    turn_amount = abs(steering)
    if steering >= 0:
        return speed, speed * (1.0 - turn_amount)
    return speed * (1.0 - turn_amount), speed

def format_esp_camera_control(control_data):
    esp_data = {
        "frame_idx": control_data["frame_idx"],
        "steering_value": control_data["steering_value"],
        "speed_scale": control_data["speed_scale"],
        "left_motor": control_data["left_motor"],
        "right_motor": control_data["right_motor"],
        "turn_type": control_data["turn_type"],
        "motion_mode": control_data["motion_mode"],
    }
    return json.dumps(esp_data, separators=(",", ":"))

def send_esp_camera_control(esp_serial, control_data):
    if esp_serial is None:
        return True, ""

    try:
        line = format_esp_camera_control(control_data)
        send_esp_serial_line(esp_serial, line)
        return True, line
    except Exception as e:
        print(f"[ERROR] Gui ESP serial loi: {e}")
        return False, ""

class ESPControlStreamer:
    def __init__(self, esp_serial):
        self.esp_serial = esp_serial
        self.latest_line = ""
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.last_debug = 0.0
        self.last_sent_line = ""
        self.last_send_at = 0.0
        self.is_failed = False

    def start(self):
        self.thread.start()

    def update(self, control_data):
        line = format_esp_camera_control(control_data)
        with self.lock:
            self.latest_line = line

    def _run(self):
        period = 1.0 / max(ESP_SERIAL_HZ, 1.0)

        while not self.stop_event.is_set():
            try:
                drain_esp_serial(self.esp_serial)

                with self.lock:
                    line = self.latest_line

                now = time.monotonic()
                should_send = (
                    line
                    and (
                        line != self.last_sent_line
                        or now - self.last_send_at >= ESP_SERIAL_KEEPALIVE_SEC
                    )
                )
                if should_send:
                    send_esp_serial_line(self.esp_serial, line)
                    self.last_sent_line = line
                    self.last_send_at = now

                    debug_now = time.time()
                    if ESP_SERIAL_DEBUG and debug_now - self.last_debug >= 1.0:
                        print(f"[ESP_TX] {line}")
                        self.last_debug = debug_now
            except Exception as e:
                self.is_failed = True
                print(f"[ERROR] ESP serial streamer loi: {e}")
                self.stop_event.set()

            time.sleep(period)

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

        try:
            send_esp_serial_line(self.esp_serial, '{"cmd":"stop"}')
            self.esp_serial.close()
        except Exception:
            pass

def run_esp_serial_test():
    esp_serial = open_esp_serial()
    if esp_serial is None:
        return 1

    try:
        test_lines = [
            '{"cmd":"camera_test","value":0}',
            '{"frame_idx":9999,"steering_value":0,"speed_scale":1.0,"left_motor":1.0,"right_motor":1.0,"turn_type":"NORMAL_ADJUSTMENT","motion_mode":"NORMAL"}',
        ]

        for line in test_lines:
            print(f"[ESP_TX] {line}")
            send_esp_serial_line(esp_serial, line)
            read_esp_serial_for(esp_serial, 0.8)

            print('[ESP_TX] {"cmd":"stop"}')
            send_esp_serial_line(esp_serial, '{"cmd":"stop"}')
            read_esp_serial_for(esp_serial, 0.4)

        return 0
    finally:
        try:
            send_esp_serial_line(esp_serial, '{"cmd":"stop"}')
            esp_serial.close()
        except Exception:
            pass

# =========================================================
# MAIN PROCESS
# =========================================================
def main():
    if not WEIGHT_PATH.exists():
        print(f"Error: Weight file not found at {WEIGHT_PATH}")
        return

    print("Loading PIDNet Floor Segmenter...")
    segmenter = PIDNetRoadSegmenter(
        pidnet_root=PIDNET_ROOT,
        weight_path=WEIGHT_PATH,
        model_size=MODEL_SIZE,
        num_classes=NUM_CLASSES,
        seg_width=SEG_WIDTH,
        seg_height=SEG_HEIGHT
    )

    print(f"Opening live camera ID={CAM_ID}...")
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Thử lại bằng driver mặc định nếu CAP_DSHOW lỗi
        cap = cv2.VideoCapture(CAM_ID)
        
    if not cap.isOpened():
        print(f"Error: Không mở được camera ID={CAM_ID}")
        return

    # Thiết lập kích thước khung hình
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    # Đặt BUFFERSIZE = 1 để tránh trễ ảnh (luôn lấy frame mới nhất)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Camera opened successfully.")
    print("Press 'q' to exit.")

    # Thiết lập UDP socket truyền tin tới phần cứng (mặc định localhost port 5005)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_target = ("127.0.0.1", 5005)
    esp_serial = open_esp_serial()
    esp_streamer = ESPControlStreamer(esp_serial) if esp_serial is not None else None
    if esp_streamer is not None:
        esp_streamer.start()
        print(f"[ESP] Control streamer started at {ESP_SERIAL_HZ:.1f} Hz")

    # Đặt tần số xác thực mặc định dựa trên ước lượng 20 FPS cho camera chạy PIDNet
    ESTIMATED_FPS = 20.0
    CONFIRMATION_FRAMES = max(1, int(ESTIMATED_FPS * OBSTACLE_CONFIRMATION_SEC))
    
    # Khởi tạo bộ điều khiển PID với dt mặc định
    pid = PIDController(Kp=2.0, Ki=0.02, Kd=0.5, dt=1.0/ESTIMATED_FPS)
    
    frame_idx = 0
    t_last = time.time()
    
    # Tracking variables
    x_c_smoothed = FRAME_WIDTH / 2.0
    obstacle_frame_counter = 0
    clear_frame_counter = 0
    escape_clear_counter = 0
    is_avoidance_active = False
    is_slowdown_active = False
    nav_state = "NORMAL"
    state_started_at = time.time()
    locked_turn_sign = 1
    turn_lock_until = 0.0
    turn_until_clear_active = False
    last_turn_clear_at = 0.0
    
    try:
        while True:
            t_loop_start = time.time()
            
            # Đọc frame từ camera
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Không đọc được frame từ camera. Thử lại...")
                continue
                
            frame_idx += 1
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # 1. Segment Floor
            seg_map = segmenter.predict_seg_map(frame)
            road_mask = segmenter.create_road_mask(
                seg_map,
                road_class_id=ROAD_CLASS_ID,
                sidewalk_class_id=1,
                use_sidewalk=False
            )
            
            # 2. Check inverted-trapezoid Bumper ROI for obstacles
            bumper_floor_mask = cv2.bitwise_and(road_mask, BUMPER_TRAPEZOID_MASK)
            bumper_area = BUMPER_TRAPEZOID_AREA
            floor_pixels = cv2.countNonZero(bumper_floor_mask)
            floor_ratio = floor_pixels / max(bumper_area, 1)
            left_bumper_floor_ratio = (
                cv2.countNonZero(bumper_floor_mask[:, :BUMPER_CENTER_X])
                / max(BUMPER_LEFT_AREA, 1)
            )
            right_bumper_floor_ratio = (
                cv2.countNonZero(bumper_floor_mask[:, BUMPER_CENTER_X:])
                / max(BUMPER_RIGHT_AREA, 1)
            )
            pole_min_floor_ratio = 1.0
            blocked_pole_cells = []
            for pole_x_start, pole_x_end, pole_cell_y_start, pole_cell_y_end, pole_mask, pole_area in POLE_CELL_ROIS:
                pole_road_mask = road_mask[pole_cell_y_start:pole_cell_y_end, pole_x_start:pole_x_end]
                pole_floor_pixels = cv2.countNonZero(cv2.bitwise_and(pole_road_mask, pole_mask))
                pole_floor_ratio = pole_floor_pixels / pole_area
                pole_min_floor_ratio = min(pole_min_floor_ratio, pole_floor_ratio)
                if pole_floor_ratio < POLE_FLOOR_THRESHOLD:
                    blocked_pole_cells.append(
                        (pole_x_start, pole_x_end, pole_cell_y_start, pole_cell_y_end, pole_floor_ratio)
                    )

            large_obstacle_detected = floor_ratio < BUMPER_OBSTACLE_THRESHOLD
            pole_obstacle_detected = len(blocked_pole_cells) > 0
            dead_end_detected = (
                left_bumper_floor_ratio < DEAD_END_SIDE_THRESHOLD
                and right_bumper_floor_ratio < DEAD_END_SIDE_THRESHOLD
            )
            obstacle_detected_in_bumper = large_obstacle_detected or pole_obstacle_detected
            
            # 3. Update temporal confirmation counter
            if obstacle_detected_in_bumper:
                obstacle_frame_counter += 1
                if obstacle_frame_counter > CONFIRMATION_FRAMES:
                    obstacle_frame_counter = CONFIRMATION_FRAMES
                clear_frame_counter = 0
            else:
                obstacle_frame_counter = max(0, obstacle_frame_counter - 2)
                if obstacle_frame_counter == 0:
                    clear_frame_counter += 1

            obstacle_seen = obstacle_frame_counter > 0
            obstacle_confirmed = obstacle_frame_counter >= CONFIRMATION_FRAMES

            # 4. Calculate Floor Centroid (Approach A)
            nav_roi_mask = road_mask[NAV_ROI_TOP:NAV_ROI_BOTTOM, :]
            floor_coords = np.argwhere(nav_roi_mask == 255) # returns [y, x]

            emergency_no_ground = floor_coords.shape[0] <= NAV_MIN_FLOOR_PIXELS
            if not emergency_no_ground:
                x_c = np.mean(floor_coords[:, 1])
                x_c_smoothed = EMA_ALPHA * x_c + (1.0 - EMA_ALPHA) * x_c_smoothed
            
            # Compute steering error: normalized distance from center [-1.0, 1.0]
            steering_error = (x_c_smoothed - (FRAME_WIDTH / 2.0)) / (FRAME_WIDTH / 2.0)
            escape_clear_now = not emergency_no_ground and not obstacle_seen
            if escape_clear_now:
                escape_clear_counter += 1
                if escape_clear_counter > ESCAPE_CLEAR_FRAMES:
                    escape_clear_counter = ESCAPE_CLEAR_FRAMES
            else:
                escape_clear_counter = max(0, escape_clear_counter - 1)
            escape_path_clear = escape_clear_counter >= ESCAPE_CLEAR_FRAMES
            
            # If in avoidance active, boost steering error to turn sharply
            t_now = time.time()
            state_elapsed = t_now - state_started_at

            if nav_state == "NO_GROUND_BACK":
                if state_elapsed >= NO_GROUND_BACK_SEC:
                    if left_bumper_floor_ratio != right_bumper_floor_ratio:
                        locked_turn_sign = 1 if right_bumper_floor_ratio > left_bumper_floor_ratio else -1
                    else:
                        locked_turn_sign = 1 if steering_error >= 0 else -1
                    nav_state = "RECOVERY_TURN"
                    state_started_at = t_now
                    obstacle_frame_counter = 0
                    clear_frame_counter = 0
                    escape_clear_counter = 0
                    turn_until_clear_active = True
                    pid.reset()
            elif nav_state == "RECOVERY_TURN" and turn_until_clear_active:
                if state_elapsed >= RECOVERY_TURN_SEC and escape_path_clear:
                    turn_lock_until = t_now + TURN_DIRECTION_HOLD_SEC
                    nav_state = "NORMAL"
                    state_started_at = t_now
                    last_turn_clear_at = t_now
                    obstacle_frame_counter = 0
                    clear_frame_counter = 0
                    escape_clear_counter = 0
                    turn_until_clear_active = False
            elif emergency_no_ground or dead_end_detected:
                if nav_state != "NO_GROUND_BACK":
                    nav_state = "NO_GROUND_BACK"
                    state_started_at = t_now
                    turn_until_clear_active = True
                    pid.reset()
            else:
                if nav_state == "NORMAL":
                    if obstacle_seen:
                        if t_now - last_turn_clear_at <= CORNER_REENTRY_WINDOW_SEC:
                            nav_state = "NO_GROUND_BACK"
                            turn_until_clear_active = True
                            pid.reset()
                        else:
                            nav_state = "SLOWDOWN"
                        state_started_at = t_now
                elif nav_state == "SLOWDOWN":
                    if not obstacle_seen and clear_frame_counter >= OBSTACLE_CLEAR_FRAMES:
                        nav_state = "NORMAL"
                        state_started_at = t_now
                    elif obstacle_confirmed or state_elapsed >= SLOWDOWN_DURATION_SEC:
                        if t_now >= turn_lock_until:
                            locked_turn_sign = 1 if steering_error >= 0 else -1
                        turn_lock_until = t_now + TURN_DIRECTION_HOLD_SEC
                        turn_until_clear_active = False
                        nav_state = "AVOIDANCE_TURN_LOCKED"
                        state_started_at = t_now
                        pid.reset()
                elif nav_state == "AVOIDANCE_TURN_LOCKED":
                    if state_elapsed >= AVOIDANCE_MIN_LOCK_SEC and escape_path_clear:
                        nav_state = "NORMAL"
                        state_started_at = t_now
                        last_turn_clear_at = t_now
                        obstacle_frame_counter = 0
                    elif state_elapsed >= AVOIDANCE_TIMEOUT_SEC:
                        turn_lock_until = t_now + TURN_DIRECTION_HOLD_SEC
                        turn_until_clear_active = True
                        escape_clear_counter = 0
                        nav_state = "RECOVERY_BACK"
                        state_started_at = t_now
                        pid.reset()
                elif nav_state == "RECOVERY_BACK":
                    if state_elapsed >= RECOVERY_BACK_SEC:
                        escape_clear_counter = 0
                        nav_state = "RECOVERY_TURN"
                        state_started_at = t_now
                        pid.reset()
            state_elapsed = t_now - state_started_at
            is_avoidance_active = nav_state == "AVOIDANCE_TURN_LOCKED"
            is_slowdown_active = nav_state == "SLOWDOWN"

            if nav_state == "NO_GROUND_BACK":
                steering_error = 0.0
                status_text = f"DEAD END: BACK {state_elapsed:.1f}/{NO_GROUND_BACK_SEC:.1f}s"
                status_color = (255, 0, 255)
            elif nav_state == "SLOWDOWN":
                steering_error *= SLOWDOWN_STEERING_SCALE
                confirm_pct = int((obstacle_frame_counter / CONFIRMATION_FRAMES) * 100)
                status_text = f"SLOWDOWN BEFORE AVOIDANCE ({confirm_pct}%)"
                status_color = (0, 165, 255)
            elif nav_state == "AVOIDANCE_TURN_LOCKED":
                steering_error = locked_turn_sign * max(abs(steering_error), 0.6)
                locked_label = "RIGHT" if locked_turn_sign > 0 else "LEFT"
                status_text = f"AVOIDANCE LOCKED {locked_label} ({state_elapsed:.1f}s)"
                status_color = (0, 0, 255)
            elif nav_state == "RECOVERY_BACK":
                steering_error = 0.0
                recovery_label = "TURN UNTIL CLEAR" if turn_until_clear_active else "RECOVERY"
                status_text = f"{recovery_label}: BACK ({state_elapsed:.1f}s)"
                status_color = (255, 0, 255)
            elif nav_state == "RECOVERY_TURN":
                steering_error = float(locked_turn_sign)
                locked_label = "R" if locked_turn_sign > 0 else "L"
                recovery_label = "TURN CLEAR" if turn_until_clear_active else "RECOVERY"
                status_text = f"{recovery_label}: TURN {locked_label} UNTIL CLEAR {escape_clear_counter}/{ESCAPE_CLEAR_FRAMES}"
                status_color = (255, 0, 255)
            else:
                status_text = "PATH CLEAR - NORMAL"
                status_color = (0, 255, 0)
            
            # Tính dt thực tế của vòng lặp để cập nhật PID chính xác
            dt_real = t_now - t_last
            t_last = t_now
            
            # Giới hạn dt để tránh đột biến quá lớn
            pid.dt = max(min(dt_real, 0.5), 0.005)
            
            # 5. Update PID Controller
            steering_val, p_val, i_val, d_val = pid.update(steering_error)
            
            if not is_avoidance_active and obstacle_frame_counter == 0:
                pid.integral = 0.0

            # 5.5. Xác định chi tiết hướng xoay và kiểu xoay để phần cứng dễ đọc
            if abs(steering_val) < 0.05:
                direction = "STRAIGHT"
                turn_intensity = 0.0
            elif steering_val > 0:
                direction = "RIGHT"
                turn_intensity = float(steering_val)
            else:
                direction = "LEFT"
                turn_intensity = float(abs(steering_val))

            if nav_state in ("RECOVERY_BACK", "NO_GROUND_BACK"):
                direction = "BACK"
                turn_intensity = 0.0
            elif nav_state == "RECOVERY_TURN":
                direction = "RIGHT" if locked_turn_sign > 0 else "LEFT"
                turn_intensity = 1.0

            if nav_state == "NO_GROUND_BACK":
                turn_type = "NO_GROUND_BACK"
                motion_mode = "RECOVERY_BACK"
                speed_scale = NO_GROUND_BACK_SPEED_SCALE
            elif nav_state == "RECOVERY_BACK":
                turn_type = "RECOVERY_BACK"
                motion_mode = "RECOVERY_BACK"
                speed_scale = RECOVERY_BACK_SPEED_SCALE
            elif nav_state == "RECOVERY_TURN":
                turn_type = "RECOVERY_TURN"
                motion_mode = "RECOVERY_TURN"
                speed_scale = RECOVERY_TURN_SPEED_SCALE
            elif is_avoidance_active:
                turn_type = "SHARP_AVOIDANCE"
                motion_mode = "AVOIDANCE_TURN"
                speed_scale = AVOIDANCE_SPEED_SCALE
            elif "EMERGENCY" in status_text:
                turn_type = "EMERGENCY_STOP"
                motion_mode = "STOP"
                speed_scale = EMERGENCY_SPEED_SCALE
            elif is_slowdown_active:
                turn_type = "SLOWDOWN_BEFORE_AVOIDANCE"
                motion_mode = "SLOWDOWN"
                speed_scale = OBSTACLE_SLOW_SPEED_SCALE
            else:
                turn_type = "NORMAL_ADJUSTMENT"
                motion_mode = "NORMAL"
                speed_scale = NORMAL_SPEED_SCALE
            voice_motion_paused = False

            left_motor, right_motor = calculate_motor_outputs(steering_val, speed_scale, motion_mode)
            esp_steering_value = steering_val
            if motion_mode in ("AVOIDANCE_TURN", "RECOVERY_TURN"):
                esp_steering_value = TANK_TURN_STEERING_VALUE if locked_turn_sign >= 0 else -TANK_TURN_STEERING_VALUE
                turn_intensity = 1.0
            elif motion_mode in ("STOP", "RECOVERY_BACK"):
                esp_steering_value = 0.0

            # Tạo dữ liệu JSON đầu ra gửi tới phần cứng
            control_data = {
                "frame_idx": int(frame_idx),
                "timestamp": float(time.time()),
                "direction": direction,
                "turn_intensity": round(turn_intensity, 4),
                "turn_type": turn_type,
                "motion_mode": motion_mode,
                "nav_state": nav_state,
                "speed_scale": round(float(speed_scale), 4),
                "left_motor": round(float(left_motor), 4),
                "right_motor": round(float(right_motor), 4),
                "steering_value": round(float(esp_steering_value), 4),
                "pid_steering_value": round(float(steering_val), 4),
                "obstacle_detected": bool(obstacle_detected_in_bumper),
                "obstacle_counter": int(obstacle_frame_counter),
                "floor_ratio": round(float(floor_ratio), 4),
                "obstacle_threshold": round(float(BUMPER_OBSTACLE_THRESHOLD), 4),
                "large_obstacle_detected": bool(large_obstacle_detected),
                "pole_obstacle_detected": bool(pole_obstacle_detected),
                "dead_end_detected": bool(dead_end_detected),
                "left_bumper_floor_ratio": round(float(left_bumper_floor_ratio), 4),
                "right_bumper_floor_ratio": round(float(right_bumper_floor_ratio), 4),
                "pole_min_floor_ratio": round(float(pole_min_floor_ratio), 4),
                "pole_floor_threshold": round(float(POLE_FLOOR_THRESHOLD), 4),
                "blocked_pole_cells": int(len(blocked_pole_cells)),
                "emergency_no_ground": bool(emergency_no_ground),
                "turn_until_clear_active": bool(turn_until_clear_active),
                "escape_clear_counter": int(escape_clear_counter),
                "escape_path_clear": bool(escape_path_clear),
                "is_avoidance_active": bool(is_avoidance_active),
                "voice_motion_paused": False,
                "status": str(status_text)
            }

            # Cách 1: Ghi ra file JSON nguyên tử (atomic write) có hỗ trợ fallback cho Windows
            json_file_path = OUTPUT_DIR / "control_output.json"
            temp_file_path = json_file_path.with_suffix(".tmp")
            try:
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    json.dump(control_data, f, indent=4)
                
                import os
                if json_file_path.exists():
                    try:
                        os.remove(str(json_file_path))
                    except OSError:
                        pass
                
                try:
                    os.rename(str(temp_file_path), str(json_file_path))
                except OSError:
                    with open(json_file_path, "w", encoding="utf-8") as f:
                        json.dump(control_data, f, indent=4)
                    try:
                        os.remove(str(temp_file_path))
                    except OSError:
                        pass
            except Exception as e:
                print(f"[ERROR] Ghi file JSON loi: {e}")

            # Cách 2: In ra stdout
            print(f"[JSON_OUT] {json.dumps(control_data)}")

            # Cách 3: Gửi UDP socket
            try:
                udp_socket.sendto(json.dumps(control_data).encode("utf-8"), udp_target)
            except Exception:
                pass

            # Cách 4: Gửi JSON nhỏ qua Serial USB tới ESP8266
            if esp_streamer is not None and not esp_streamer.is_failed:
                esp_streamer.update(control_data)

            # 6. Build Visualization Outputs
            overlay = segmenter.overlay_road(frame, road_mask, alpha=0.35)
            cv2.line(overlay, (0, NAV_ROI_TOP), (FRAME_WIDTH, NAV_ROI_TOP), (255, 255, 0), 1)
            
            cv2.circle(overlay, (int(x_c_smoothed), NAV_ROI_TOP + 30), 8, (0, 255, 255), -1)
            cv2.line(overlay, (int(x_c_smoothed), NAV_ROI_TOP), (int(x_c_smoothed), FRAME_HEIGHT), (0, 255, 255), 2)
            cv2.line(overlay, (FRAME_WIDTH//2, NAV_ROI_TOP), (FRAME_WIDTH//2, FRAME_HEIGHT), (255, 255, 255), 1, cv2.LINE_AA)
            
            bumper_color = (0, 0, 255) if obstacle_detected_in_bumper else (0, 255, 0)
            cv2.polylines(overlay, [BUMPER_TRAPEZOID_POINTS], True, bumper_color, 2)
            for pole_x_start, pole_x_end, pole_cell_y_start, pole_cell_y_end, _ in blocked_pole_cells:
                cv2.rectangle(
                    overlay,
                    (pole_x_start, pole_cell_y_start),
                    (pole_x_end - 1, pole_cell_y_end - 1),
                    (0, 0, 255),
                    2,
                )
            cv2.putText(
                overlay,
                (
                    f"Floor: {floor_ratio:.0%}/{BUMPER_OBSTACLE_THRESHOLD:.0%} "
                    f"Pole: {pole_min_floor_ratio:.0%}/{POLE_FLOOR_THRESHOLD:.0%}"
                ),
                (BUMPER_TOP_X_START, max(BUMPER_Y_START - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bumper_color,
                2,
            )

            cv2.line(overlay, (int(FRAME_WIDTH * 0.33), NAV_ROI_TOP), (int(FRAME_WIDTH * 0.33), FRAME_HEIGHT), (150, 150, 150), 1, cv2.LINE_8)
            cv2.line(overlay, (int(FRAME_WIDTH * 0.67), NAV_ROI_TOP), (int(FRAME_WIDTH * 0.67), FRAME_HEIGHT), (150, 150, 150), 1, cv2.LINE_8)
            
            # 7. Create Information Panel (Frame 2)
            info_panel = np.zeros_like(frame)
            cv2.rectangle(info_panel, (15, 15), (FRAME_WIDTH - 15, 60), (30, 30, 30), -1)
            cv2.rectangle(info_panel, (15, 15), (FRAME_WIDTH - 15, 60), status_color, 2)
            cv2.putText(info_panel, status_text, (30, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            
            cv2.putText(info_panel, "Obstacle Confirmation Counter (1s):", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            bar_w = 400
            bar_h = 16
            bar_x = 25
            bar_y = 105
            cv2.rectangle(info_panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            filled_w = int((obstacle_frame_counter / CONFIRMATION_FRAMES) * bar_w)
            if filled_w > 0:
                cv2.rectangle(info_panel, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), status_color, -1)
            cv2.rectangle(info_panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
            cv2.putText(info_panel, f"{obstacle_frame_counter}/{CONFIRMATION_FRAMES} fr", (bar_x + bar_w + 10, bar_y + 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            # PID text readout
            fps_live = 1.0 / max(dt_real, 1e-6)
            cv2.putText(info_panel, "=== PID CONTROLLER STEERING ===", (25, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(info_panel, f"Live Loop FPS        : {fps_live:.1f} Hz", (25, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info_panel, f"Steering Error e(t) : {steering_error:+.4f}", (25, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info_panel, f"Speed Scale          : {speed_scale:.2f} ({motion_mode})", (25, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            cv2.putText(info_panel, f"Motor L/R            : {left_motor:+.2f} / {right_motor:+.2f}", (25, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            cv2.putText(info_panel, f"Proportional (Kp)    : {p_val:+.4f}", (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(info_panel, f"Integral     (Ki)    : {i_val:+.4f}", (25, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(info_panel, f"Derivative   (Kd)    : {d_val:+.4f}", (25, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(info_panel, f"Total PID Output (w) : {steering_val:+.4f}", (25, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            
            info_panel = draw_steering_needle(info_panel, steering_val, status_text, status_color)
            
            left_show = draw_title(overlay, "LIVE CAMERA FLOOR SEGMENTATION")
            right_show = draw_title(info_panel, "LIVE NAVIGATION STATUS")
            
            # Combine to form dashboard
            dashboard = cv2.hconcat([left_show, right_show])
            
            # Show live preview
            preview_img = cv2.resize(dashboard, (int(FRAME_WIDTH * 2 * 0.75), int(FRAME_HEIGHT * 0.75)))
            cv2.imshow("PIDNet + PID Steering Live Camera", preview_img)
            
            if frame_idx % 20 == 0:
                print(f"[LIVE] Frame: {frame_idx} | FPS: {fps_live:.1f} | State: {status_text} | Steer: {steering_val:.3f}")
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Live camera session stopped by user.")
                break
                
    finally:
        if esp_streamer is not None:
            esp_streamer.stop()
        elif esp_serial is not None:
            try:
                send_esp_serial_line(esp_serial, '{"cmd":"stop"}')
                esp_serial.close()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Exiting.")

if __name__ == "__main__":
    if "--esp-test" in sys.argv:
        raise SystemExit(run_esp_serial_test())
    main()
