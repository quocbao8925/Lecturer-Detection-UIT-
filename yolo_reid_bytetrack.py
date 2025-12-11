import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import torchreid


# [TỐI ƯU 1] Bật chế độ Benchmark để PyTorch tự tìm thuật toán Conv nhanh nhất cho phần cứng
torch.backends.cudnn.benchmark = True

# =========================
# 1. Feature Extractor: OSNet (Đã nâng cấp Batch Processing)
# =========================
class OSNetReID:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[OSNet] Using device: {self.device}")
        
        self.model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=0, pretrained=True
        )
        # Thứ tự chuẩn: to Device -> eval -> half
        self.model.to(self.device).eval()
        
        if self.device.type == 'cuda':
            self.model.half()  # Ép FP16 để tăng tốc độ tính toán

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std  = np.array([0.229, 0.224, 0.225])

    def _preprocess(self, img_bgr):
        # Hàm này xử lý 1 ảnh lẻ (giữ lại để dùng nếu cần)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 256))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        return tensor

    def extract_batch(self, img_list):
        """
        [TỐI ƯU MỚI] Xử lý một lúc nhiều ảnh (Batch)
        """
        if len(img_list) == 0:
            return []

        # 1. Preprocess tất cả ảnh trong list (trên CPU)
        processed_imgs = []
        for img_bgr in img_list:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 256))
            img = img.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1)
            processed_imgs.append(torch.from_numpy(img))

        # 2. Stack lại thành 1 Tensor duy nhất: [N, 3, 256, 128]
        batch_tensor = torch.stack(processed_imgs)
        
        # 3. Đưa vào GPU 1 lần duy nhất
        batch_tensor = batch_tensor.to(self.device)
        
        if self.device.type == 'cuda':
            batch_tensor = batch_tensor.half()

        # 4. Inference
        with torch.no_grad():
            features = self.model(batch_tensor)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        # 5. Trả về numpy array
        return features.cpu().float().numpy()

def cosine_distance(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return 1.0 - (np.dot(a, b) / denom)

# =========================
# 2. Persistent Tracker (GIỮ NGUYÊN 100% LOGIC)
# =========================
class PersistentTracker:
    def __init__(self):
        self.target_emb = None      
        self.state = "IDLE"         # IDLE -> TRACKING -> LOST -> WAITING
        
        self.lost_counter = 0
        self.COASTING_LIMIT = 60    # 2 giây (30fps) dùng Kalman dự đoán
        
        # --- CẤU HÌNH ---
        self.STRICT_REID_THRESH = 0.4  # Độ giống nhau (>55%)
        self.MAX_JUMP_RATIO = 0.40      # Không nhảy quá 40% chiều cao 1 frame

        # Kalman Filter
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        # Process Noise cao hơn chút để Kalman linh hoạt
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05

    def set_target(self, embedding, bbox):
        self.target_emb = embedding.copy()
        self.state = "TRACKING"
        self.lost_counter = 0
        
        x1, y1, x2, y2 = bbox
        self.last_w = x2 - x1
        self.last_h = y2 - y1
        cx, cy = (x1+x2)/2, (y1+y2)/2
        
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        print("[Tracker] Target LOCKED. ID Saved.")

    def update(self, all_bboxes, all_embeddings):
        if self.state == "IDLE" or self.target_emb is None:
            return None, None, -1

        FRICTION = 0.7 
        SHAPE_DRIFT = 0.9 
        self.kf.statePost[2, 0] *= FRICTION
        self.kf.statePost[3, 0] *= FRICTION

        # 1. PREDICT
        pred_bbox = None
        pred_cx, pred_cy = 0, 0
        
        if self.state in ["TRACKING", "LOST"]:
            prediction = self.kf.predict()
            pred_cx, pred_cy = prediction[0, 0], prediction[1, 0]
            
            p_x1 = int(pred_cx - self.last_w / 2)
            p_y1 = int(pred_cy - self.last_h / 2)
            p_x2 = int(pred_cx*SHAPE_DRIFT + self.last_w / 2)
            p_y2 = int(pred_cy*SHAPE_DRIFT + self.last_h / 2)
            pred_bbox = [p_x1, p_y1, p_x2, p_y2]

        # 2. FILTERING
        candidates = []
        for i, (emb, bbox) in enumerate(zip(all_embeddings, all_bboxes)):
            dist = cosine_distance(self.target_emb, emb)
            if dist < self.STRICT_REID_THRESH:
                candidates.append((dist, i, bbox))
        
        candidates.sort(key=lambda x: x[0]) 

        # 3. LOGIC XỬ LÝ THEO TRẠNG THÁI
        if self.state == "WAITING":
            if len(candidates) > 0:
                dist, idx, bbox = candidates[0]
                x1, y1, x2, y2 = bbox
                cx, cy = (x1+x2)/2, (y1+y2)/2
                self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
                self.last_w = x2 - x1
                self.last_h = y2 - y1
                self.state = "TRACKING"
                self.lost_counter = 0
                return bbox, dist, idx
            else:
                return None, 0.0, -1

        else: 
            best_idx = None
            best_score = 999.0

            for dist, idx, bbox in candidates:
                x1, y1, x2, y2 = bbox
                meas_cx, meas_cy = (x1+x2)/2, (y1+y2)/2
                jump_dist = np.sqrt((meas_cx - pred_cx)**2 + (meas_cy - pred_cy)**2)
                max_jump = self.last_h * self.MAX_JUMP_RATIO
                
                if jump_dist < max_jump:
                    best_idx = idx
                    best_score = dist
                    break 
            
            if best_idx is not None:
                self.state = "TRACKING"
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
                if self.lost_counter > self.COASTING_LIMIT:
                    if self.state != "WAITING":
                        print("[Tracker] Lost too long -> Switch to WAITING mode.")
                    self.state = "WAITING"
                    return None, 0.0, -1
                else:
                    self.state = "LOST"
                    return pred_bbox, 0.0, -1

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
            print(f"[Mouse] Clicked at {self.point}")

# =========================
# 4. Main Program (Đã tối ưu Batch)
# =========================
def main():
    video_path = r"D:\WORKSPACE\Notebook\yolo_person\testvideo4.mp4" 
    yolo_path = r"D:\WORKSPACE\Notebook\yolo_person\best2.pt"
    using_cam = 0 # Set to 0 to use video file
    
    # [TỐI ƯU 2] Giữ nguyên resize vì đây là cách tăng FPS tốt nhất
    MAX_WIDTH = 960 
    
    device_str = 0 if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Running YOLO on: {device_str}")
    
    print("Loading Models...")
    model = YOLO(yolo_path)
    
    reid = OSNetReID(device="cuda" if torch.cuda.is_available() else "cpu")
    tracker = PersistentTracker()
    mouse = MouseSelector()

    cap = cv2.VideoCapture(0 if (using_cam) else video_path)
    cv2.namedWindow("Smart ReID Tracker", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Smart ReID Tracker", mouse.callback)

    prev_time = 0 

    while True:
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        ret, frame = cap.read()
        if not ret: break
        
        if (using_cam):
            frame = cv2.flip(frame, 1)  # mirror

        # 1. Resize Frame
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
        
        # [TỐI ƯU 3] Gom tất cả crop lại để xử lý 1 lần (Batch Inference)
        crops_to_extract = []
        meta_data_temp = [] # Lưu tạm để map lại sau khi extract

        if results[0].boxes:
            for box in results[0].boxes:
                xyxy = box.xyxy.cpu().numpy()[0].astype(int)
                conf_val = float(box.conf[0].item())
                
                x1, y1, x2, y2 = xyxy
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                # Thay vì extract ngay, ta gom lại
                crops_to_extract.append(crop)
                meta_data_temp.append((x1, y1, x2, y2, conf_val))

            # Nếu có crop nào thì extract một thể
            if len(crops_to_extract) > 0:
                # Chạy Batch Extract
                batch_embeddings = reid.extract_batch(crops_to_extract)
                
                # Bung ra lại các list
                for i in range(len(batch_embeddings)):
                    x1, y1, x2, y2, conf_val = meta_data_temp[i]
                    emb = batch_embeddings[i]
                    
                    boxes.append([x1, y1, x2, y2])
                    embeddings.append(emb)
                    confs.append(conf_val)

        # 3. Logic Hiển thị & Tương tác (GIỮ NGUYÊN LOGIC CŨ)
        if tracker.state == "IDLE":
            cv2.putText(frame, "CLICK TO SELECT TARGET", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            for box in boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 100, 0), 2)

            if mouse.clicked and mouse.point:
                mx, my = mouse.point
                selected_idx = None
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    if x1 < mx < x2 and y1 < my < y2:
                        selected_idx = i
                        break
                if selected_idx is not None:
                    tracker.set_target(embeddings[selected_idx], boxes[selected_idx])
                mouse.clicked = False
        else:
            target_box, score, idx = tracker.update(boxes, embeddings)
            
            if tracker.state == "WAITING":
                cv2.putText(frame, "WAITING RE-ENTRY...", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            elif target_box is not None:
                x1, y1, x2, y2 = target_box
                if idx != -1:
                    similarity = (1.0 - score) * 100
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"[Sim: {similarity:.1f}% | Conf: {confs[idx]:.2f}]", (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "OCCLUDED", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Smart ReID Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()