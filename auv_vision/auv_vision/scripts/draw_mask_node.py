#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse

class DrawMaskNode:
    def __init__(self):
        self.bridge = CvBridge()
        
        # ROS parameters
        self.input_image_topic = rospy.get_param(
            "~input_image_topic", "/taluy/cameras/cam_bottom/image_rect_color"
        )
        self.output_image_topic = rospy.get_param(
            "~output_image_topic", "/taluy/vision/draw_mask/image"
        )
        self.output_mask_topic = rospy.get_param(
            "~output_mask_topic", "/taluy/vision/draw_mask/mask"
        )
        
        # Thread safety variables
        self.image_lock = threading.Lock()
        self.latest_msg = None
        
        # GUI coordination variables
        self.trigger_gui = False
        self.gui_event = threading.Event()
        self.gui_input_msg = None
        self.gui_status = None # 'saved', 'cancelled', 'error'
        
        # Saved state for continuous publishing
        self.saved_frame = None
        self.saved_mask = None
        self.saved_header = None
        
        # Publishers
        self.pub_image = rospy.Publisher(self.output_image_topic, Image, queue_size=1)
        self.pub_mask = rospy.Publisher(self.output_mask_topic, Image, queue_size=1)
        
        # Subscriber
        self.sub = rospy.Subscriber(
            self.input_image_topic, Image, self.image_callback, queue_size=1
        )
        
        # Service Server
        self.srv = rospy.Service("draw_mask", Trigger, self.handle_draw_mask_service)
        
        rospy.loginfo(f"[DrawMaskNode] Subscribed to input: {self.input_image_topic}")
        rospy.loginfo(f"[DrawMaskNode] Output image topic: {self.output_image_topic}")
        rospy.loginfo(f"[DrawMaskNode] Output mask topic: {self.output_mask_topic}")
        rospy.loginfo("[DrawMaskNode] Service 'draw_mask' is ready.")

    def image_callback(self, msg):
        with self.image_lock:
            self.latest_msg = msg

    def handle_draw_mask_service(self, req):
        with self.image_lock:
            if self.latest_msg is None:
                err_msg = f"No bottom camera image received yet on {self.input_image_topic}."
                rospy.logwarn(f"[DrawMaskNode] {err_msg}")
                return TriggerResponse(success=False, message=err_msg)
            img_msg = self.latest_msg

        rospy.loginfo("[DrawMaskNode] Service triggered. Opening drawing GUI in main thread...")
        
        # Pass image message and reset GUI state
        self.gui_input_msg = img_msg
        self.gui_status = None
        self.gui_event.clear()
        self.trigger_gui = True

        # Wait for the main thread to complete GUI execution
        self.gui_event.wait()

        if self.gui_status == "saved":
            return TriggerResponse(
                success=True,
                message=f"Mask and image successfully published to {self.output_mask_topic} and {self.output_image_topic}."
            )
        elif self.gui_status == "cancelled":
            return TriggerResponse(success=False, message="Mask drawing cancelled by user.")
        else:
            return TriggerResponse(success=False, message="An error occurred during GUI drawing.")

    def run_gui_loop(self):
        try:
            frame = self.bridge.imgmsg_to_cv2(self.gui_input_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"[DrawMaskNode] CvBridge conversion failed: {e}")
            self.gui_status = "error"
            return

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Painting state
        brush_size = 10
        drawing = False
        mouse_x, mouse_y = None, None
        legend_height = 40

        def draw_mask_cb(event, x, y, flags, param):
            nonlocal drawing, mask, brush_size, mouse_x, mouse_y
            mouse_x, mouse_y = x, y
            
            # Offset y by legend_height as we prepend it to the display
            y_img = y - legend_height
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                if 0 <= y_img < frame.shape[0] and 0 <= x < frame.shape[1]:
                    cv2.circle(mask, (x, y_img), brush_size, 255, -1)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    if 0 <= y_img < frame.shape[0] and 0 <= x < frame.shape[1]:
                        cv2.circle(mask, (x, y_img), brush_size, 255, -1)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False

        window_name = "Draw Mask"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, draw_mask_cb)

        rospy.loginfo("[DrawMaskNode] Paint window opened. Focus the window to paint.")
        
        while not rospy.is_shutdown():
            vis = frame.copy()
            
            # Render translucent red mask overlay where painted
            if np.any(mask == 255):
                roi = vis[mask == 255]
                red_layer = np.zeros_like(roi)
                red_layer[:] = [0, 0, 255]
                blended = cv2.addWeighted(roi, 0.5, red_layer, 0.5, 0)
                vis[mask == 255] = blended

            # Draw brush preview circle if cursor is inside the image
            if mouse_x is not None and mouse_y is not None:
                y_img = mouse_y - legend_height
                if 0 <= y_img < frame.shape[0] and 0 <= mouse_x < frame.shape[1]:
                    cv2.circle(vis, (mouse_x, y_img), brush_size, (255, 255, 255), 1)

            # Prepend legend strip at the top for controls information
            legend = np.zeros((legend_height, frame.shape[1], 3), dtype=np.uint8)
            controls_text = f"Brush: {brush_size}px | [a/s] Save & Pub | [Esc] Cancel | [c] Clear | [q/e] Brush Size"
            cv2.putText(
                legend, controls_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )

            # Assemble and show
            final_display = np.vstack((legend, vis))
            cv2.imshow(window_name, final_display)
            
            key = cv2.waitKey(10) & 0xFF

            # Key controls handling
            if key == ord('a') or key == ord('s'):
                if np.sum(mask) == 0:
                    rospy.logwarn("[DrawMaskNode] Mask is empty! Please paint a region first.")
                    continue
                
                # Save the frame, mask, and header for continuous publishing
                self.saved_frame = frame.copy()
                self.saved_mask = mask.copy()
                self.saved_header = self.gui_input_msg.header
                
                # Publish the original BGR8 image and the MONO8 mask
                self.publish_mask_and_image(self.saved_frame, self.saved_mask, self.saved_header)
                self.gui_status = "saved"
                break
                
            elif key == ord('c'):
                mask[:] = 0
                rospy.loginfo("[DrawMaskNode] Mask cleared.")
                
            elif key == ord('q'):
                brush_size = max(1, brush_size - 5)
                rospy.loginfo(f"[DrawMaskNode] Brush size: {brush_size}")
                
            elif key == ord('e'):
                brush_size = min(100, brush_size + 5)
                rospy.loginfo(f"[DrawMaskNode] Brush size: {brush_size}")
                
            elif key == 27: # Escape key
                rospy.loginfo("[DrawMaskNode] Drawing cancelled.")
                self.gui_status = "cancelled"
                break

            # Fallback check if the user clicks window 'X' button
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    rospy.loginfo("[DrawMaskNode] Window closed. Drawing cancelled.")
                    self.gui_status = "cancelled"
                    break
            except cv2.error:
                pass

        cv2.destroyWindow(window_name)
        for _ in range(5):
            cv2.waitKey(1)

    def publish_mask_and_image(self, frame, mask, header):
        # Construct and publish original image (BGR8)
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = header.frame_id
        self.pub_image.publish(img_msg)

        # Construct and publish mask image (MONO8)
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header.stamp = rospy.Time.now()
        mask_msg.header.frame_id = header.frame_id
        self.pub_mask.publish(mask_msg)

        rospy.loginfo_throttle(5.0, "[DrawMaskNode] Continuously publishing saved mask and image...")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.trigger_gui:
                self.run_gui_loop()
                self.trigger_gui = False
                self.gui_event.set()
            else:
                if self.saved_frame is not None and self.saved_mask is not None:
                    self.publish_mask_and_image(self.saved_frame, self.saved_mask, self.saved_header)
            rate.sleep()

def main():
    rospy.init_node("draw_mask_node")
    node = DrawMaskNode()
    node.run()

if __name__ == "__main__":
    main()
