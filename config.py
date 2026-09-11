from pathlib import Path

CAM_ID = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

OUTPUT_DIR = Path("outputs_pidnet")
OUTPUT_DIR.mkdir(exist_ok=True)

PIDNET_ROOT = Path("PIDNet")
PIDNET_MODEL_SIZE = "pidnet-s"
PIDNET_WEIGHT = PIDNET_ROOT / "pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt"

# Resize ảnh trước khi đưa vào PIDNet để tăng FPS
SEG_WIDTH = 320
SEG_HEIGHT = 240

# Không cần segment mọi frame, dùng lại mask cũ để mượt hơn
SEG_EVERY_N_FRAMES = 3

# Cityscapes class id
ROAD_CLASS_ID = 0
SIDEWALK_CLASS_ID = 1
USE_SIDEWALK = True

OVERLAY_ALPHA = 0.45
OVERLAY_ALPHA = 0.45

# Khung kiem tra mat dat/vat can o phan duoi anh.
# Hinh thang thuong: canh tren hep hon, canh duoi rong hon gan khung xe.
GROUND_ROI_TOP_X_START_RATIO = 0.25
GROUND_ROI_TOP_X_END_RATIO = 0.75
GROUND_ROI_BOTTOM_X_START_RATIO = 0.05
GROUND_ROI_BOTTOM_X_END_RATIO = 0.95
GROUND_ROI_Y_START_RATIO = 0.68

# Toc do gui xuong ESP khi dieu khien theo camera.
NORMAL_SPEED_SCALE = 1.0
OBSTACLE_SLOW_SPEED_SCALE = 0.35
AVOIDANCE_SPEED_SCALE = 0.60
EMERGENCY_SPEED_SCALE = 0.0
SLOWDOWN_STEERING_SCALE = 0.25
TANK_TURN_STEERING_VALUE = 2.0
SLOWDOWN_DURATION_SEC = 0.5
AVOIDANCE_MIN_LOCK_SEC = 1.2
AVOIDANCE_TIMEOUT_SEC = 2.5
RECOVERY_BACK_SEC = 0.7
RECOVERY_TURN_SEC = 1.0
RECOVERY_BACK_SPEED_SCALE = 0.45
RECOVERY_TURN_SPEED_SCALE = 0.65
OBSTACLE_CLEAR_FRAMES = 10
BUMPER_OBSTACLE_THRESHOLD = 0.45
DEAD_END_SIDE_THRESHOLD = 0.45
# Detect small vertical obstacles such as chair legs in the lower trapezoid.
POLE_DETECT_COLUMNS = 15
POLE_DETECT_ROWS = 3
POLE_DETECT_Y_START_RATIO = 0.45
POLE_DETECT_X_START_RATIO = 0.05
POLE_DETECT_X_END_RATIO = 0.95
POLE_FLOOR_THRESHOLD = 0.65
POLE_MIN_CELL_AREA = 150
OBSTACLE_CONFIRMATION_SEC = 0.50
TURN_DIRECTION_HOLD_SEC = 4.0
ESCAPE_CLEAR_FRAMES = 14
CORNER_REENTRY_WINDOW_SEC = 2.0
NAV_MIN_FLOOR_PIXELS = 100
NO_GROUND_BACK_SEC = 1.2
NO_GROUND_BACK_SPEED_SCALE = 0.40

# ESP8266 serial output. Change ESP_SERIAL_PORT if your board uses another COM port.
SEND_TO_ESP_SERIAL = True
ESP_SERIAL_PORT = "COM5"
ESP_SERIAL_BAUD = 115200
ESP_SERIAL_HZ = 15.0
ESP_SERIAL_KEEPALIVE_SEC = 0.5
ESP_SERIAL_DEBUG = False
ESP_SERIAL_RX_DEBUG = False

# Trang thai tam dung do mic lien tuc cua RAG ghi. Camera la tien trinh duy
# nhat gui lenh motor xuong ESP, nen khong xay ra tranh chap cong COM.
VOICE_MOTION_CONTROL_FILE = Path(__file__).resolve().parent / "voice_motion_control.json"
