import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import torchreid

# =========================
# 1. Feature Extractor: OSNet
# =========================
class OSNetReID:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[OSNet] Using device: {self.device}")
        
        # Load mô hình OSNet
        self.model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=0, pretrained=True
        )
        self.model.to(self.device).eval().float()
        
        # Chuẩn hóa ảnh theo ImageNet
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std  = np.array([0.229, 0.224, 0.225])

    def _preprocess(self, img_bgr):
        # Resize về chuẩn 128x256 của ReID
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 256))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1) # HWC -> CHW
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        return tensor.to(self.device)

    @torch.no_grad()
    def extract(self, img_bgr):
        tensor = self._preprocess(img_bgr)
        feat = self.model(tensor)
        # Normalize vector về độ dài 1 để tính cosine distance
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat[0].cpu().numpy()

# Hàm tính khoảng cách Cosine (0: giống hệt, 1: khác hoàn toàn)
def cosine_distance(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return 1.0 - (np.dot(a, b) / denom)

# =========================
# 2. Priority ReID Tracker
# =========================
class PriorityReIDTracker:
    def __init__(self):
        self.target_emb = None
        self.is_tracking = False
        self.lost_counter = 0
        self.TIMEOUT_FRAMES = 210  # Số frame mất dấu trước khi reset
        
        self.STRICT_REID_THRESH = 0.4 
        self.MAX_JUMP_RATIO = 0.40

        # --- KALMAN FILTER ---
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        
        # Tăng nhiễu hệ thống lên chút để Kalman ít tin vào vận tốc tức thời hơn
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1 

    def set_target(self, embedding, bbox):
        self.target_emb = embedding.copy()
        self.is_tracking = True
        self.lost_counter = 0
        
        x1, y1, x2, y2 = bbox
        self.last_w = x2 - x1
        self.last_h = y2 - y1
        cx, cy = (x1+x2)/2, (y1+y2)/2
        
        # Reset trạng thái: Vận tốc (index 2, 3) về 0 hết
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        
        print(f"[Tracker] Target locked! Size: {self.last_w}x{self.last_h}")

    def update(self, all_bboxes, all_embeddings):
        if not self.is_tracking or self.target_emb is None:
            return None, None, None

        # ============================================================
        # [NEW] ÁP DỤNG MA SÁT (FRICTION) TRƯỚC KHI DỰ ĐOÁN
        # ============================================================
        # Lấy vận tốc hiện tại (vx, vy)
        vx = self.kf.statePost[2, 0]
        vy = self.kf.statePost[3, 0]
        
        # Nhân với hệ số < 1.0 (ví dụ 0.85) để hãm phanh
        # Số càng nhỏ -> Dừng càng nhanh (ít trôi)
        # Số càng to -> Trôi càng xa
        FRICTION_FACTOR = 0.7 
        
        self.kf.statePost[2, 0] *= FRICTION_FACTOR
        self.kf.statePost[3, 0] *= FRICTION_FACTOR
        # ============================================================

        # 1. PREDICT
        prediction = self.kf.predict()
        pred_cx, pred_cy = prediction[0, 0], prediction[1, 0]

        # Tạo box dự đoán
        p_x1 = int(pred_cx - self.last_w / 2)
        p_y1 = int(pred_cy - self.last_h / 2)
        p_x2 = int(pred_cx + self.last_w / 2)
        p_y2 = int(pred_cy + self.last_h / 2)
        pred_bbox = [p_x1, p_y1, p_x2, p_y2]

        if len(all_embeddings) == 0:
            self.lost_counter += 1
            self._check_timeout()
            return pred_bbox, 0.0, -1

        # 2. MATCHING (Logic cũ)
        candidates = [] 
        for i, (emb, bbox) in enumerate(zip(all_embeddings, all_bboxes)):
            dist = cosine_distance(self.target_emb, emb)
            if dist < self.STRICT_REID_THRESH:
                candidates.append((dist, i, bbox))
        
        candidates.sort(key=lambda x: x[0])

        best_idx = None
        best_score = 999.0
        
        for dist, idx, bbox in candidates:
            # SANITY CHECK
            x1, y1, x2, y2 = bbox
            meas_cx, meas_cy = (x1+x2)/2, (y1+y2)/2
            
            jump_dist = np.sqrt((meas_cx - pred_cx)**2 + (meas_cy - pred_cy)**2)
            max_jump = self.last_h * self.MAX_JUMP_RATIO
            
            
            if jump_dist < max_jump:
                best_idx = idx
                best_score = dist
                break 
            else:
                continue

        # 3. RESULT
        if best_idx is not None:
            self.lost_counter = 0
            best_bbox = all_bboxes[best_idx]
            
            x1, y1, x2, y2 = best_bbox
            self.last_w = x2 - x1
            self.last_h = y2 - y1
            
            meas_cx, meas_cy = (x1+x2)/2, (y1+y2)/2
            self.kf.correct(np.array([[meas_cx], [meas_cy]], np.float32))
            
            return best_bbox, best_score, best_idx
        else:
            self.lost_counter += 1
            self._check_timeout()
            return pred_bbox, 0.0, -1

    def _check_timeout(self):
        if self.lost_counter > self.TIMEOUT_FRAMES:
            print("[Tracker] Timeout! Target lost. Resetting...")
            self.is_tracking = False
            self.target_emb = None

# =========================
# 3. Mouse Selector
# =========================
class MouseSelector:
    def __init__(self):
        self.clicked = False
        self.point = None 

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True
            self.point = (x, y)
            print(f"[Mouse] Selected at {self.point}")

# =========================
# 4. Main Program
# =========================
def main():
    # --- CONFIG ---
    video_path = r"D:\WORKSPACE\Notebook\yolo_person\testvideo2.mp4" 
    yolo_path = r"D:\WORKSPACE\Notebook\yolo_person\best2.pt"
    
    # Giới hạn độ phân giải để tăng tốc (960px là đẹp)
    MAX_WIDTH = 960 
    
    # Auto Device
    device_str = 0 if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Running YOLO on: {device_str}")
    
    # Load Models
    print("Loading Models...")
    model = YOLO(yolo_path)
    reid = OSNetReID(device="cuda" if torch.cuda.is_available() else "cpu")
    tracker = PriorityReIDTracker()
    mouse = MouseSelector()

    cap = cv2.VideoCapture(video_path)
    
    # Cửa sổ cho phép resize
    cv2.namedWindow("Target Tracking", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Target Tracking", mouse.callback)

    prev_time = 0 

    while True:
        # Tính FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        ret, frame = cap.read()
        if not ret: break
        
        # 1. Resize Frame (Tăng tốc)
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            scale = MAX_WIDTH / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (MAX_WIDTH, new_h))
        
        # 2. YOLO Detect
        results = model.predict(frame, conf=0.5, verbose=False, classes=[0], device=device_str)
        
        boxes = []
        embeddings = []
        confs = []
        
        if results[0].boxes:
            for box in results[0].boxes:
                xyxy = box.xyxy.cpu().numpy()[0].astype(int)
                conf_val = float(box.conf[0].item())
                
                x1, y1, x2, y2 = xyxy
                
                # Crop ảnh để ReID
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                # Extract Feature
                emb = reid.extract(crop)
                
                boxes.append([x1, y1, x2, y2])
                embeddings.append(emb)
                confs.append(conf_val)

        # 3. Logic Tracking
        if not tracker.is_tracking:
            # === CHẾ ĐỘ CHỌN TARGET ===
            cv2.putText(frame, "CLICK TO SELECT TARGET", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Vẽ các box ứng viên
            for box in boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 100, 0), 2)

            # Xử lý click chuột
            if mouse.clicked and mouse.point:
                mx, my = mouse.point
                selected_idx = None
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    if x1 < mx < x2 and y1 < my < y2:
                        selected_idx = i
                        break
                
                if selected_idx is not None:
                    tracker.set_target(embeddings[selected_idx], boxes[selected_idx])
                
                mouse.clicked = False # Reset click
        else:
            # === CHẾ ĐỘ TRACKING ===
            target_box, dist_score, idx = tracker.update(boxes, embeddings)
            
            if target_box is not None:
                x1, y1, x2, y2 = target_box
                
                if idx != -1:
                    # ---> TÌM THẤY (Màu Xanh Lá)
                    similarity = (1.0 - dist_score) * 100
                    str = f"[Conf: {confs[idx]:.2f}|Sim: {similarity:.1f}%]"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Info Tag
                    cv2.putText(frame, str, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # ---> DỰ ĐOÁN / BỊ CHE (Màu Vàng)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "OCCLUDED", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # ---> MẤT DẤU (Màu Đỏ)
                cv2.putText(frame, "LOST TARGET", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Hiển thị FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"LOST: {tracker.lost_counter}/{tracker.TIMEOUT_FRAMES}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 10, 255), 2)

        cv2.imshow("Target Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()