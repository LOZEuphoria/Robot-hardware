import cv2
import time
import json
import socket
import numpy as np
from pathlib import Path
from pidnet_road_segmenter import PIDNetRoadSegmenter

# =========================================================
# CONFIGURATION
# =========================================================
VIDEO_PATH = Path("7882087290505.mp4")
WEIGHT_PATH = Path("checkpoints_floor/pidnet_floor_best.pt")
PIDNET_ROOT = Path("PIDNet")
MODEL_SIZE = "pidnet-l"

# Output display size
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Segmentation size (smaller for faster execution)
SEG_WIDTH = 320
SEG_HEIGHT = 240

# Output directories
OUTPUT_DIR = Path("outputs_pidnet")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "steering_simulation.mp4"

# PIDNet Custom Model Config (0 = background, 1 = floor)
NUM_CLASSES = 2
ROAD_CLASS_ID = 1  # Floor is class 1

# ROI for Navigation (bottom half of the image)
NAV_ROI_TOP = FRAME_HEIGHT // 2
NAV_ROI_BOTTOM = FRAME_HEIGHT

# Bumper ROI for obstacle detection (bottom-center area)
BUMPER_Y_START = int(FRAME_HEIGHT * 0.75)
BUMPER_Y_END = FRAME_HEIGHT
BUMPER_X_START = int(FRAME_WIDTH * 0.30)
BUMPER_X_END = int(FRAME_WIDTH * 0.70)

# Floor ratio threshold in Bumper (below this = obstacle detected)
BUMPER_OBSTACLE_THRESHOLD = 0.75

# Exponential Moving Average coefficient for smoothing the centroid
EMA_ALPHA = 0.15

# =========================================================
# PID CONTROLLER CLASS
# =========================================================
class PIDController:
    def __init__(self, Kp=1.8, Ki=0.05, Kd=0.4, dt=0.05):
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

# =========================================================
# MAIN PROCESS
# =========================================================
def main():
    if not VIDEO_PATH.exists():
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

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

    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1 second confirmation duration
    CONFIRMATION_FRAMES = int(fps_video)
    print(f"Video FPS: {fps_video:.1f} | 1-second debounce = {CONFIRMATION_FRAMES} frames")

    # Initialize VideoWriter (Dashboard is 2 frames horizontally: Original Overlay + Navigation Info)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dashboard_width = FRAME_WIDTH * 2
    dashboard_height = FRAME_HEIGHT
    
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        fourcc,
        fps_video,
        (dashboard_width, dashboard_height)
    )

    # Initialize PID controller
    pid = PIDController(Kp=2.0, Ki=0.02, Kd=0.5, dt=1.0/fps_video)
    
    # Thiết lập UDP socket truyền tin tới phần cứng (mặc định localhost port 5005)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_target = ("127.0.0.1", 5005)
    
    frame_idx = 0
    start_time = time.time()
    
    # Tracking variables
    x_c_smoothed = FRAME_WIDTH / 2.0
    obstacle_frame_counter = 0
    is_avoidance_active = False
    
    print("Processing video frames...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
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
            
            # 2. Check Bumper ROI for obstacles
            bumper_roi = road_mask[BUMPER_Y_START:BUMPER_Y_END, BUMPER_X_START:BUMPER_X_END]
            bumper_area = bumper_roi.size
            floor_pixels = cv2.countNonZero(bumper_roi)
            floor_ratio = floor_pixels / max(bumper_area, 1)
            
            obstacle_detected_in_bumper = floor_ratio < BUMPER_OBSTACLE_THRESHOLD
            
            # 3. Update temporal confirmation counter
            if obstacle_detected_in_bumper:
                obstacle_frame_counter += 1
                if obstacle_frame_counter > CONFIRMATION_FRAMES:
                    obstacle_frame_counter = CONFIRMATION_FRAMES
            else:
                # Slowly decay to prevent rapid oscillations
                obstacle_frame_counter = max(0, obstacle_frame_counter - 2)
            
            # Check state: is obstacle confirmed? (held for 1 second)
            if obstacle_frame_counter >= CONFIRMATION_FRAMES:
                is_avoidance_active = True
                status_text = "AVOIDANCE ACTIVE: ROTATE"
                status_color = (0, 0, 255) # Red
            elif obstacle_frame_counter > 0:
                is_avoidance_active = False
                confirm_pct = int((obstacle_frame_counter / CONFIRMATION_FRAMES) * 100)
                status_text = f"CONFIRMING OBSTACLE ({confirm_pct}%)"
                status_color = (0, 165, 255) # Orange
            else:
                is_avoidance_active = False
                status_text = "PATH CLEAR - NORMAL"
                status_color = (0, 255, 0) # Green

            # 4. Calculate Floor Centroid (Approach A)
            nav_roi_mask = road_mask[NAV_ROI_TOP:NAV_ROI_BOTTOM, :]
            floor_coords = np.argwhere(nav_roi_mask == 255) # returns [y, x]
            
            if floor_coords.shape[0] > 100:
                # Calculate mean column coordinate
                x_c = np.mean(floor_coords[:, 1])
                # Smooth with EMA
                x_c_smoothed = EMA_ALPHA * x_c + (1.0 - EMA_ALPHA) * x_c_smoothed
            else:
                # Emergency fallback if no floor is detected in ROI
                # Turn right strongly if no floor
                x_c_smoothed = FRAME_WIDTH * 0.8
                status_text = "EMERGENCY: NO WALKABLE GROUND"
                status_color = (0, 0, 255)
            
            # Compute steering error: normalized distance from center [-1.0, 1.0]
            steering_error = (x_c_smoothed - (FRAME_WIDTH / 2.0)) / (FRAME_WIDTH / 2.0)
            
            # If in avoidance active, boost steering output to rotate faster
            if is_avoidance_active:
                # Boost error to force maximum rotation in the direction of the centroid bias
                if steering_error >= 0:
                    steering_error = max(steering_error, 0.6)
                else:
                    steering_error = min(steering_error, -0.6)
            
            # 5. Update PID Controller
            steering_val, p_val, i_val, d_val = pid.update(steering_error)
            
            # Reset PID integral term if path becomes clear
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

            if is_avoidance_active:
                turn_type = "SHARP_AVOIDANCE"       # Xoay tránh vật cản gấp
            elif "EMERGENCY" in status_text:
                turn_type = "EMERGENCY_STOP"        # Dừng khẩn cấp
            else:
                turn_type = "NORMAL_ADJUSTMENT"     # Điều chỉnh bám sàn bình thường

            # Tạo dữ liệu JSON đầu ra gửi tới phần cứng
            control_data = {
                "frame_idx": int(frame_idx),
                "timestamp": float(time.time()),
                "direction": direction,
                "turn_intensity": round(turn_intensity, 4),
                "turn_type": turn_type,
                "steering_value": round(float(steering_val), 4),
                "obstacle_detected": bool(obstacle_detected_in_bumper),
                "obstacle_counter": int(obstacle_frame_counter),
                "is_avoidance_active": bool(is_avoidance_active),
                "status": str(status_text)
            }

            # Cách 1: Ghi ra file JSON nguyên tử (atomic write) có hỗ trợ fallback cho Windows
            json_file_path = OUTPUT_DIR / "control_output.json"
            temp_file_path = json_file_path.with_suffix(".tmp")
            try:
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    json.dump(control_data, f, indent=4)
                
                # Trên Windows, nếu file đích tồn tại ta nên xóa trước khi rename để tránh lỗi Access Denied
                import os
                if json_file_path.exists():
                    try:
                        os.remove(str(json_file_path))
                    except OSError:
                        pass
                
                # Thử rename, nếu thất bại (do file đang bị khóa đọc), ta sẽ ghi đè trực tiếp
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

            # Cách 2: In ra stdout (cho các ứng dụng đọc luồng subprocess stdout)
            print(f"[JSON_OUT] {json.dumps(control_data)}")

            # Cách 3: Gửi UDP socket (cho các thiết bị nhúng chạy song song qua mạng như ESP32/Raspberry Pi)
            try:
                udp_socket.sendto(json.dumps(control_data).encode("utf-8"), udp_target)
            except Exception as e:
                pass

            # 6. Build Visualization Outputs
            # Draw overlay on original image
            overlay = segmenter.overlay_road(frame, road_mask, alpha=0.35)
            
            # Draw Nav ROI boundary line
            cv2.line(overlay, (0, NAV_ROI_TOP), (FRAME_WIDTH, NAV_ROI_TOP), (255, 255, 0), 1)
            
            # Draw centroid marker
            cv2.circle(overlay, (int(x_c_smoothed), NAV_ROI_TOP + 30), 8, (0, 255, 255), -1) # Yellow dot
            cv2.line(overlay, (int(x_c_smoothed), NAV_ROI_TOP), (int(x_c_smoothed), FRAME_HEIGHT), (0, 255, 255), 2)
            cv2.line(overlay, (FRAME_WIDTH//2, NAV_ROI_TOP), (FRAME_WIDTH//2, FRAME_HEIGHT), (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw Bumper box
            bumper_color = (0, 0, 255) if obstacle_detected_in_bumper else (0, 255, 0)
            cv2.rectangle(overlay, (BUMPER_X_START, BUMPER_Y_START), (BUMPER_X_END, BUMPER_Y_END), bumper_color, 2)
            cv2.putText(overlay, f"Bumper: {floor_ratio:.1%}", (BUMPER_X_START + 5, BUMPER_Y_START + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, bumper_color, 1)

            # Left/Right Sector split lines
            cv2.line(overlay, (int(FRAME_WIDTH * 0.33), NAV_ROI_TOP), (int(FRAME_WIDTH * 0.33), FRAME_HEIGHT), (150, 150, 150), 1, cv2.LINE_8)
            cv2.line(overlay, (int(FRAME_WIDTH * 0.67), NAV_ROI_TOP), (int(FRAME_WIDTH * 0.67), FRAME_HEIGHT), (150, 150, 150), 1, cv2.LINE_8)
            
            # 7. Create Information Panel (Frame 2)
            info_panel = np.zeros_like(frame)
            
            # Status and Alert Title
            cv2.rectangle(info_panel, (15, 15), (FRAME_WIDTH - 15, 60), (30, 30, 30), -1)
            cv2.rectangle(info_panel, (15, 15), (FRAME_WIDTH - 15, 60), status_color, 2)
            cv2.putText(info_panel, status_text, (30, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            
            # Debounce Progress Bar
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
            
            # PID Outputs text readouts
            cv2.putText(info_panel, "=== PID CONTROLLER STEERING ===", (25, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(info_panel, f"Steering Error e(t) : {steering_error:+.4f}", (25, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info_panel, f"Proportional (Kp={pid.Kp:.2f})  : {p_val:+.4f}", (25, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(info_panel, f"Integral     (Ki={pid.Ki:.2f})  : {i_val:+.4f}", (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            cv2.putText(info_panel, f"Derivative   (Kd={pid.Kd:.2f})  : {d_val:+.4f}", (25, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(info_panel, f"Total PID Output (w) : {steering_val:+.4f}", (25, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            
            # Embed steering needle gauge in the info panel
            info_panel = draw_steering_needle(info_panel, steering_val, status_text, status_color)
            
            # Set titles on frames
            left_show = draw_title(overlay, "PIDNET SEGMENTATION & CENTROID")
            right_show = draw_title(info_panel, "NAVIGATION CONTROL SYSTEM STATUS")
            
            # Write Frame Index and processing FPS on display
            elapsed = time.time() - start_time
            fps_current = frame_idx / max(elapsed, 1e-6)
            cv2.putText(left_show, f"Frame: {frame_idx}/{total_frames} | FPS: {fps_current:.1f}", (15, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Combine horizontally to form the dashboard
            dashboard = cv2.hconcat([left_show, right_show])
            
            # Write dashboard to output video
            writer.write(dashboard)
            
            # Preview (resize to fit screen)
            # preview_img = cv2.resize(dashboard, (int(dashboard_width * 0.75), int(dashboard_height * 0.75)))
            # cv2.imshow("PIDNet + PID Steering Control Simulation", preview_img)
            
            if frame_idx % 20 == 0:
                print(f"Processed: {frame_idx}/{total_frames} frames (FPS: {fps_current:.1f}), Counter: {obstacle_frame_counter}, Steer: {steering_val:.3f}")
                
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord("q"):
            #     print("Stopped by user.")
            #     break
                
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
        duration = time.time() - start_time
        print("\nProcessing finished.")
        print(f"Total processed frames: {frame_idx}")
        print(f"Average FPS: {frame_idx / max(duration, 1e-6):.1f}")
        print(f"Simulation saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()
