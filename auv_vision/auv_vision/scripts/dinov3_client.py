#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import zmq
import json
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import time

class TrackerClient:
    def __init__(self):
        rospy.init_node('zmq_tracker_client', anonymous=True)
        
        # Parametreler
        self.camera_topic = rospy.get_param('~camera_topic', '/taluy/cameras/cam_front/image_raw') 
        self.zmq_port = rospy.get_param('~zmq_port', 5555)
        
        self.bridge = CvBridge()
        self.init_zmq()
        
        # Publisherlar
        self.pub_result = rospy.Publisher('/tracker/target_point', Point, queue_size=1)
        self.pub_debug = rospy.Publisher('/tracker/debug_image', Image, queue_size=1)
        
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
                
                for r in res.get('results', []):
                    if r['found']:
                        # Koordinatlar
                        bx = r['bbox'] # [x1, y1, x2, y2]
                        cx, cy = r['center']
                        iou = r['iou']
                        
                        # A. Sayisal Veri Yayini
                        self.pub_result.publish(Point(x=cx, y=cy, z=iou))
                        
                        # B. Gorsel Annotation (Cizim)
                        # Nesne etrafina yesil kutu
                        cv2.rectangle(debug_img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
                        # Merkez noktaya kirmizi daire
                        cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)
                        # Bilgi metni
                        label = "Target IoU: {:.2f}".format(iou)
                        cv2.putText(debug_img, label, (bx[0], bx[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
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