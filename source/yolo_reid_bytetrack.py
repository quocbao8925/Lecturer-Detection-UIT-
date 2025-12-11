import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import torchreid


# [TỐI ƯU 1] Bật chế độ Benchmark để PyTorch tự tìm thuật toán Conv nhanh nhất cho phần cứng
torch.backends.cudnn.benchmark = True

# =========================
# 1. Feature Extractor: OSNet
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

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea)
    return iou
# =========================
# 2. Persistent Tracker 
# =========================
class PersistentTracker:
    def __init__(self):
        self.target_emb = None      
        self.state = "IDLE"         
        self.lost_counter = 0
        self.COASTING_LIMIT = 90   
        
        # --- CẤU HÌNH ---
        self.STRICT_REID_THRESH = 0.38 
        self.IOU_THRESHOLD = 0.1

        # Kalman Filter
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

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

        # Ma sát
        FRICTION = 0.7
        self.kf.statePost[2, 0] *= FRICTION
        self.kf.statePost[3, 0] *= FRICTION

        # 1. PREDICT
        pred_bbox = None
        pred_cx, pred_cy = 0, 0
        
        if self.state in ["TRACKING", "LOST"]:
            prediction = self.kf.predict()
            pred_cx, pred_cy = prediction[0, 0], prediction[1, 0]
            
            # [FIX 1] ĐÓNG BĂNG KÍCH THƯỚC (SIZE FREEZE)
            # Khi dự đoán, luôn dùng kích thước cũ (self.last_w, self.last_h)
            # Không để Kalman tự tính toán kích thước -> Chặn đứng việc box bị teo
            w = self.last_w
            h = self.last_h
            
            p_x1 = int(pred_cx - w / 2)
            p_y1 = int(pred_cy - h / 2)
            p_x2 = int(pred_cx + w / 2)
            p_y2 = int(pred_cy + h / 2)
            pred_bbox = [p_x1, p_y1, p_x2, p_y2]

        # 2. FILTERING
        candidates = []
        for i, (emb, bbox) in enumerate(zip(all_embeddings, all_bboxes)):
            dist = cosine_distance(self.target_emb, emb)
            if dist < self.STRICT_REID_THRESH:
                candidates.append((dist, i, bbox))
        
        candidates.sort(key=lambda x: x[0]) 

        # 3. LOGIC XỬ LÝ
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

            # Duyệt các ứng viên tiềm năng
            for dist, idx, bbox in candidates:
                # [FIX 2] DÙNG IoU THAY VÌ KHOẢNG CÁCH TÂM
                # Tính độ chồng lấn giữa: Ghost Box (pred_bbox) và Ứng viên (bbox)
                
                iou = compute_iou(pred_bbox, bbox)
                
                # Logic: Nếu có chồng lấn (IoU > 0.1) -> Bắt luôn
                # Cách này cực nhạy, chỉ cần Ghost Box "chạm" vào người là dính
                if iou > self.IOU_THRESHOLD:
                    best_idx = idx
                    best_score = dist
                    break 
                
                # (Fallback) Nếu Ghost Box trôi quá xa không chạm, check khoảng cách tâm như cũ
                # Dành cho trường hợp object nhảy rất nhanh
                x1, y1, x2, y2 = bbox
                meas_cx, meas_cy = (x1+x2)/2, (y1+y2)/2
                jump_dist = np.sqrt((meas_cx - pred_cx)**2 + (meas_cy - pred_cy)**2)
                
                # Nới lỏng khoảng cách cho phép (0.6 chiều cao)
                if jump_dist < (self.last_h * 0.6): 
                    best_idx = idx
                    best_score = dist
                    break

            if best_idx is not None:
                self.state = "TRACKING"
                self.lost_counter = 0
                best_bbox = all_bboxes[best_idx]
                
                x1, y1, x2, y2 = best_bbox
                # Cập nhật lại kích thước chuẩn
                self.last_w = x2 - x1
                self.last_h = y2 - y1
                
                meas_cx, meas_cy = (x1+x2)/2, (y1+y2)/2
                self.kf.correct(np.array([[meas_cx], [meas_cy]], np.float32))
                return best_bbox, best_score, best_idx
            
            else:
                self.lost_counter += 1
                if self.lost_counter > self.COASTING_LIMIT:
                    self.state = "WAITING"
                    return None, 0.0, -1
                else:
                    self.state = "LOST"
                    # Trả về Ghost Box (kích thước đã được đóng băng ở trên)
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
    video_path = r"D:\WORKSPACE\Notebook\yolo_person\testvideo.mp4" 
    yolo_path = r"D:\WORKSPACE\Notebook\yolo_person\best3.pt"
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