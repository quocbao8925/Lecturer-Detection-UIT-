import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import torchreid

# [TỐI ƯU 1] Benchmark
torch.backends.cudnn.benchmark = True

# =========================
# 1. Feature Extractor: OSNet
# =========================

"""
class OSNetReID:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[OSNet] Using device: {self.device}")
        
        self.model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=0, pretrained=True
        )
        self.model.to(self.device).eval()
        
        if self.device.type == 'cuda':
            self.model.half() 

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std  = np.array([0.229, 0.224, 0.225])

    def _preprocess(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 256))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        return tensor

    def extract_batch(self, img_list):
        if len(img_list) == 0: return []
        processed_imgs = []
        for img_bgr in img_list:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 256))
            img = img.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1)
            processed_imgs.append(torch.from_numpy(img))

        batch_tensor = torch.stack(processed_imgs).to(self.device)
        
        if self.device.type == 'cuda':
            batch_tensor = batch_tensor.half()

        with torch.no_grad():
            features = self.model(batch_tensor)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
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
# 2. Persistent Tracker (ĐÃ FIX: DÙNG IOU)
# =========================
class PersistentTracker:
    def __init__(self):
        self.target_emb = None      
        self.state = "IDLE"         
        self.lost_counter = 0
        self.COASTING_LIMIT = 90    
        
        # --- CẤU HÌNH ---
        self.STRICT_REID_THRESH = 0.40 
        self.IOU_THRESHOLD = 0.1       # Chỉ cần chạm nhau 10% là bắt
        self.MAX_SCALE_CHANGE = 1.8 

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
        print("[Tracker] Target LOCKED.")

    def check_scale_consistency(self, current_bbox, ghost_bbox):
        w_new = current_bbox[2] - current_bbox[0]
        h_new = current_bbox[3] - current_bbox[1]
        area_new = w_new * h_new
        
        w_old = self.last_w
        h_old = self.last_h
        area_old = w_old * h_old
        
        ratio = area_new / (area_old + 1e-6)
        
        # Chống to lên quá nhanh
        if ratio > self.MAX_SCALE_CHANGE: return False 

        # Chống nhỏ đi quá nhanh (trừ khi tâm vẫn trùng khớp)
        if ratio < (1.0 / self.MAX_SCALE_CHANGE):
            cx_new = (current_bbox[0] + current_bbox[2]) / 2
            cy_new = (current_bbox[1] + current_bbox[3]) / 2
            cx_old = (ghost_bbox[0] + ghost_bbox[2]) / 2
            cy_old = (ghost_bbox[1] + ghost_bbox[3]) / 2
            dist = np.sqrt((cx_new - cx_old)**2 + (cy_new - cy_old)**2)
            
            if dist > (h_old * 0.3): return False
            return True
            
        return True

    def update(self, all_bboxes, all_embeddings):
        if self.state == "IDLE" or self.target_emb is None:
            return None, None, -1

        # Ma sát
        FRICTION = 0.8
        self.kf.statePost[2, 0] *= FRICTION
        self.kf.statePost[3, 0] *= FRICTION

        # 1. PREDICT
        pred_bbox = None
        pred_cx, pred_cy = 0, 0
        
        if self.state in ["TRACKING", "LOST"]:
            prediction = self.kf.predict()
            pred_cx, pred_cy = prediction[0, 0], prediction[1, 0]
            
            # [FIX] Đóng băng kích thước Ghost Box
            w, h = self.last_w, self.last_h
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

            for dist, idx, bbox in candidates:
                # Check Scale trước
                if not self.check_scale_consistency(bbox, pred_bbox):
                    continue 

                # ========================================================
                # [FIX QUAN TRỌNG] GỌI HÀM COMPUTE_IOU Ở ĐÂY
                # ========================================================
                # Ưu tiên 1: Nếu Box mới chạm vào Ghost Box (IoU > 0.1) -> LẤY LUÔN
                iou = compute_iou(pred_bbox, bbox)
                if iou > self.IOU_THRESHOLD:
                    best_idx = idx
                    best_score = dist
                    break 
                
                # Ưu tiên 2: Nếu không chạm nhưng tâm ở gần (Fallback)
                x1, y1, x2, y2 = bbox
                meas_cx, meas_cy = (x1+x2)/2, (y1+y2)/2
                jump_dist = np.sqrt((meas_cx - pred_cx)**2 + (meas_cy - pred_cy)**2)
                
                if jump_dist < (self.last_h * 0.6): 
                    best_idx = idx
                    best_score = dist
                    break

            if best_idx is not None:
                self.state = "TRACKING"
                self.lost_counter = 0
                best_bbox = all_bboxes[best_idx]
                
                # Update Size mượt
                new_w = best_bbox[2] - best_bbox[0]
                new_h = best_bbox[3] - best_bbox[1]
                self.last_w = 0.7 * self.last_w + 0.3 * new_w
                self.last_h = 0.7 * self.last_h + 0.3 * new_h
                
                meas_cx = (best_bbox[0] + best_bbox[2])/2
                meas_cy = (best_bbox[1] + best_bbox[3])/2
                self.kf.correct(np.array([[meas_cx], [meas_cy]], np.float32))
                return best_bbox, best_score, best_idx
            
            else:
                self.lost_counter += 1
                if self.lost_counter > self.COASTING_LIMIT:
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
# 4. Main Program
# =========================
def main():
    video_path = r"D:\WORKSPACE\Notebook\yolo_person\video\testvideo3.mp4" 
    yolo_path = r"D:\WORKSPACE\Notebook\yolo_person\model\best3.pt"
    using_cam = 0 
    
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

    while True:
        curr_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        if (using_cam): frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            scale = MAX_WIDTH / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (MAX_WIDTH, new_h))
        
        results = model.predict(frame, conf=0.5, verbose=False, classes=[0], device=device_str)
        
        boxes = []
        embeddings = []
        confs = []
        crops_to_extract = [] 
        meta_data_temp = []

        if results[0].boxes:
            for box in results[0].boxes:
                xyxy = box.xyxy.cpu().numpy()[0].astype(int)
                conf_val = float(box.conf[0].item())
                x1, y1, x2, y2 = xyxy
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                crops_to_extract.append(crop)
                meta_data_temp.append((x1, y1, x2, y2, conf_val))

            if len(crops_to_extract) > 0:
                batch_embeddings = reid.extract_batch(crops_to_extract)
                for i in range(len(batch_embeddings)):
                    x1, y1, x2, y2, conf_val = meta_data_temp[i]
                    boxes.append([x1, y1, x2, y2])
                    embeddings.append(batch_embeddings[i])
                    confs.append(conf_val)

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

        cv2.imshow("Smart ReID Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
"""