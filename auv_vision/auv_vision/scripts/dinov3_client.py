#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import zmq
import json
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Pose2D
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult
from cv_bridge import CvBridge, CvBridgeError
import time
import base64

class TrackerClient:
    def __init__(self):
        rospy.init_node('zmq_tracker_client', anonymous=True)
        
        # Parametreler
        self.camera_topic = rospy.get_param('~camera_topic', '/taluy/cameras/cam_bottom/image_raw') 
        self.zmq_port = rospy.get_param('~zmq_port', 5555)
        
        self.bridge = CvBridge()
        self.init_zmq()
        
        # Publisherlar
        self.pub_result = rospy.Publisher('/tracker/target_point', Point, queue_size=1)
        self.pub_debug = rospy.Publisher('/tracker/debug_image', Image, queue_size=1)
        self.pub_detections = rospy.Publisher('/tracker/detections', Detection2DArray, queue_size=1)
        self.pub_mask = rospy.Publisher('/tracker/mask', Image, queue_size=1)
        self.pub_obb = rospy.Publisher('/yolo_result_bottom', YoloResult, queue_size=1)
        
        self.sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        rospy.loginfo("Debug Yayini Hazir: /tracker/debug_image")

    def init_zmq(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 3000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect("tcp://127.0.0.1:{}".format(self.zmq_port))

    def image_callback(self, data):
        start_time = time.time()
        try:
            # 1. ROS Image -> OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            h, w = cv_img.shape[:2]
            
            # 2. ZMQ Iletisimi
            _, encoded = cv2.imencode('.jpg', cv_img)
            self.socket.send(encoded.tobytes())
            
            msg = self.socket.recv()
            res = json.loads(msg)
            
            if res['status'] == 'ok':
                # debug_img uzerinde cizim yapacagiz
                debug_img = cv_img.copy()
                
                # 3. Timings Gosterimi
                if 'timings' in res:
                    t = res['timings']
                    # Sol ust koseye yazalim
                    y_offset = 20
                    lines = [
                        f"FPS: {1.0/(time.time()-start_time):.1f} (ROS)",
                        f"Total Lag: {t.get('total', 0)*1000:.1f}ms",
                        f"DinoV3: {t.get('dino', 0)*1000:.1f}ms",
                        f"FastSAM: {t.get('fastsam', 0)*1000:.1f}ms",
                        f"Exact Match: {t.get('match', 0)*1000:.1f}ms",
                        f"  -> Prop: {t.get('propagate', 0)*1000:.1f}ms",
                        f"  -> IoU: {t.get('iou', 0)*1000:.1f}ms"
                    ]
                    for mn, line in enumerate(lines):
                        cv2.putText(debug_img, line, (10, y_offset + mn*20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                for r in res.get('results', []):
                    if r['found']:
                        # Koordinatlar
                        bx = r['bbox'] # [x1, y1, x2, y2]
                        cx, cy = r['center']
                        iou = r['iou']
                        
                        # A. Sayisal Veri Yayini
                        self.pub_result.publish(Point(x=cx, y=cy, z=iou))
                        
                        # B. Gorsel Annotation (Cizim)
                        # Maske varsa ciz
                        if 'mask' in r:
                            try:
                                mask_bytes = base64.b64decode(r['mask'])
                                mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
                                mask_img = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
                                
                                # Maskeyi renkli hale getirelim (Yari saydam kirmizi)
                                # debug_img ile ayni boyutta renkli maske olustur
                                color_mask = np.zeros_like(debug_img)
                                color_mask[mask_img > 0] = [0, 0, 255] # BGR: Kirmizi
                                
                                # Alpha blending
                                alpha = 0.5
                                debug_img = cv2.addWeighted(debug_img, 1, color_mask, alpha, 0)
                                
                                # Maske kenarlarini ciz (kontur)
                                contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 2)
                                
                            except Exception as e:
                                rospy.logwarn(f"Mask decode error: {e}")

                        # Nesne etrafina yesil kutu
                        cv2.rectangle(debug_img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
                        # Merkez noktaya kirmizi daire
                        cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)
                        # Bilgi metni
                        label = "Target IoU: {:.2f}".format(iou)
                        cv2.putText(debug_img, label, (bx[0], bx[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # C. Vision Msgs Yayini
                det_array = Detection2DArray()
                det_array.header = data.header # timestamp senkronizasyonu
                
                for r in res.get('results', []):
                    if r['found']:
                        bx = r['bbox']
                        iou = r['iou']
                        cx, cy = r['center']
                        
                        det = Detection2D()
                        det.header = data.header
                        
                        # Bounding Box (vision_msgs, merkez tabanli calisir)
                        # bx: [min_x, min_y, max_x, max_y]
                        w_box = bx[2] - bx[0]
                        h_box = bx[3] - bx[1]
                        
                        det.bbox.center.x = cx
                        det.bbox.center.y = cy
                        det.bbox.center.theta = 0
                        det.bbox.size_x = w_box
                        det.bbox.size_y = h_box
                        
                        # Hypothesis (Score/Class)
                        hyp = ObjectHypothesisWithPose()
                        hyp.id = 0 # Class ID (Target)
                        hyp.score = iou # Score olarak IoU kullaniyoruz
                        det.results.append(hyp)
                        
                        det_array.detections.append(det)
                
                self.pub_detections.publish(det_array)
                
                # D. Maske ve OBB Yayini
                # Eger maske varsa (tek bir hedef takip edildigi icin mask_img degiskenini kullanabiliriz)
                # Not: Dongu icinde mask_img olusturuluyordu. En son islenen (veya tek) hedefin maskesini yayinlayalim.
                # Birden fazla hedef varsa bu mantik sonuncuyu alir, ancak su an tracker tek hedef odakl.
                
                obb_array = Detection2DArray()
                obb_array.header = data.header

                if 'mask_img' in locals() and mask_img is not None:
                    # 1. Maskeyi Yayinla
                    mask_msg = self.bridge.cv2_to_imgmsg(mask_img, "mono8")
                    mask_msg.header = data.header
                    self.pub_mask.publish(mask_msg)
                    
                    # 2. OBB Hesapla ve Yayinla
                    # Konturlari bul (yukarida bulmustuk ama lokal olabilir, tekrar bulalim veya saklayalim)
                    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        if cv2.contourArea(cnt) < 10: continue # Cok kucuk gurultuleri atla
                        
                        # Rotated Rectangle (OBB)
                        rect = cv2.minAreaRect(cnt)
                        ((cx, cy), (w, h), angle) = rect
                        
                        # OBB Detection Mesaji
                        obb_det = Detection2D()
                        obb_det.header = data.header
                        
                        obb_det.bbox.center.x = cx
                        obb_det.bbox.center.y = cy
                        
                        # Theta: Radyan cinsinden. 
                        # cv2.minAreaRect angle donusu versiyona gore degisir ama genelde derece doner.
                        # Radyana cevirelim.
                        obb_det.bbox.center.theta = np.deg2rad(angle)
                        
                        obb_det.bbox.size_x = w
                        obb_det.bbox.size_y = h
                        
                        # Score ekleyelim (mevcutsa)
                        hyp = ObjectHypothesisWithPose()
                        hyp.id = 0
                        hyp.score = 1.0 # Maskeden uretildigi icin score 1.0 veya tracker score kullanilabilir
                        # Yukaridaki dongudeki score'u alamadik, basitlik icin 1.0 diyoruz veya globalde saklayabiliriz.
                        obb_det.results.append(hyp)
                        
                        obb_array.detections.append(obb_det)
                        
                        # Debug image'e OBB cizimi
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(debug_img, [box], 0, (255, 0, 0), 2) # Mavi renk OBB
                
                yolo_result = YoloResult()
                yolo_result.header = data.header
                yolo_result.detections = obb_array
                self.pub_obb.publish(yolo_result)

                # 3. Debug Goruntusunu Publish Et
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
                self.pub_debug.publish(debug_msg)
            
            # FPS Hesapla
            fps = 1.0 / (time.time() - start_time)
            rospy.loginfo_throttle(5, "Tracking Active | FPS: {:.1f}".format(fps))

        except zmq.error.Again:
            rospy.logwarn("ZMQ Timeout - Reconnecting...")
            self.socket.close()
            self.init_zmq()
        except Exception as e:
            rospy.logerr("Callback Error: {}".format(e))

if __name__ == '__main__':
    try:
        TrackerClient()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass