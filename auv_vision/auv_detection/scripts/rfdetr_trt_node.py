#!/usr/bin/env python3

import ast
import threading
from pathlib import Path

import cv2
import numpy as np
import pycuda.driver as cuda
import rospy
import tensorrt as trt
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import CompressedImage, Image
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray
from vision_msgs.msg import ObjectHypothesisWithPose


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IDLE_RATE_HZ = 200.0


def parse_list_param(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    if isinstance(value, (int, float)):
        return [value]

    text = str(value).strip()
    if text in ("", "[]", "()"):
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = [item.strip() for item in text.split(",") if item.strip()]

    if isinstance(parsed, (list, tuple)):
        return list(parsed)

    if parsed in (None, ""):
        return []

    return [parsed]


def parse_int_set(value):
    parsed = parse_list_param(value)
    if not parsed:
        return None
    return set(int(item) for item in parsed)


def parse_string_list(value):
    return [str(item) for item in parse_list_param(value)]


def parse_dict_param(value):
    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    text = str(value).strip()
    if text in ("", "{}"):
        return {}

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError("Failed to parse dict parameter '{}': {}".format(text, exc))

    if not isinstance(parsed, dict):
        raise ValueError(
            "Expected dict parameter, got '{}'".format(type(parsed).__name__)
        )

    return dict(parsed)


def parse_int_dict(value):
    return {int(key): int(val) for key, val in parse_dict_param(value).items()}


def parse_class_name_lookup(value):
    if value is None:
        return {}

    if isinstance(value, dict):
        return {int(key): str(val) for key, val in value.items()}

    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return {}
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = [item.strip() for item in text.split(",") if item.strip()]

    if isinstance(parsed, dict):
        return {int(key): str(val) for key, val in parsed.items()}

    if isinstance(parsed, (list, tuple)):
        return {
            index: str(name) for index, name in enumerate(parsed) if str(name) != ""
        }

    return {}


def parse_device_index(device_value):
    if isinstance(device_value, int):
        return device_value

    device_text = str(device_value).strip()
    if not device_text:
        return 0

    if ":" in device_text:
        device_text = device_text.rsplit(":", 1)[1]

    return int(device_text)


def preprocess_image(image_bgr, input_hw):
    orig_h, orig_w = image_bgr.shape[:2]
    in_h, in_w = input_hw

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    image_float = image_resized.astype(np.float32) / 255.0
    image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD
    image_float = np.transpose(image_float, (2, 0, 1))
    image_float = np.expand_dims(image_float, axis=0)

    return np.ascontiguousarray(image_float), (orig_h, orig_w)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def box_cxcywh_to_xyxy(boxes):
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = np.clip(boxes[:, 2], 0.0, None)
    h = np.clip(boxes[:, 3], 0.0, None)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return np.stack([x1, y1, x2, y2], axis=1)


def postprocess_rfdetr(
    outputs,
    orig_hw,
    labels_output_name,
    boxes_output_name,
    topk=300,
    threshold=0.5,
):
    if labels_output_name not in outputs or boxes_output_name not in outputs:
        raise RuntimeError(
            "Expected outputs named '{}' and '{}', got {}".format(
                labels_output_name,
                boxes_output_name,
                list(outputs.keys()),
            )
        )

    logits = outputs[labels_output_name]
    boxes = outputs[boxes_output_name]

    if logits.ndim != 3 or boxes.ndim != 3:
        raise RuntimeError(
            "Unexpected shapes. {}={}, {}={}".format(
                labels_output_name,
                logits.shape,
                boxes_output_name,
                boxes.shape,
            )
        )

    logits = logits[0]
    boxes = boxes[0]
    probabilities = sigmoid(logits)

    if probabilities.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    num_classes = probabilities.shape[1]
    flat_scores = probabilities.reshape(-1)
    topk = max(1, min(int(topk), flat_scores.shape[0]))

    topk_indices = np.argpartition(-flat_scores, topk - 1)[:topk]
    topk_scores = flat_scores[topk_indices]

    order = np.argsort(-topk_scores)
    topk_indices = topk_indices[order]
    topk_scores = topk_scores[order]

    topk_boxes = topk_indices // num_classes
    topk_labels = topk_indices % num_classes

    selected_boxes = box_cxcywh_to_xyxy(boxes[topk_boxes])
    orig_h, orig_w = orig_hw
    selected_boxes[:, [0, 2]] *= float(orig_w)
    selected_boxes[:, [1, 3]] *= float(orig_h)
    selected_boxes[:, [0, 2]] = np.clip(selected_boxes[:, [0, 2]], 0.0, float(orig_w))
    selected_boxes[:, [1, 3]] = np.clip(selected_boxes[:, [1, 3]], 0.0, float(orig_h))

    keep = topk_scores >= float(threshold)

    selected_boxes = selected_boxes[keep]
    selected_scores = topk_scores[keep]
    selected_labels = topk_labels[keep]

    if selected_boxes.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    valid = np.logical_and(
        selected_boxes[:, 2] > selected_boxes[:, 0],
        selected_boxes[:, 3] > selected_boxes[:, 1],
    )

    return (
        selected_boxes[valid].astype(np.float32),
        selected_scores[valid].astype(np.float32),
        selected_labels[valid].astype(np.int32),
    )


def make_detection_array(boxes, scores, class_ids):
    detections = []

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [float(value) for value in box]
        detection = Detection2D()
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.id = int(class_id)
        hypothesis.score = float(score)
        detection.results.append(hypothesis)
        detection.bbox = BoundingBox2D()
        detection.bbox.center = Pose2D(
            x=(x1 + x2) * 0.5,
            y=(y1 + y2) * 0.5,
            theta=0.0,
        )
        detection.bbox.size_x = x2 - x1
        detection.bbox.size_y = y2 - y1
        detections.append(detection)

    return detections


def make_yolo_result(header, boxes, scores, class_ids):
    result_msg = YoloResult()
    result_msg.header = header
    result_msg.detections = Detection2DArray()
    result_msg.detections.header = header
    result_msg.detections.detections = make_detection_array(boxes, scores, class_ids)
    result_msg.masks = []
    return result_msg


def remap_class_ids(class_ids, class_id_map):
    if not class_id_map or class_ids.size == 0:
        return class_ids

    return np.array(
        [class_id_map.get(int(class_id), int(class_id)) for class_id in class_ids]
    )


def filter_detections_by_classes(boxes, scores, class_ids, allowed_classes):
    if not allowed_classes or class_ids.size == 0:
        return boxes, scores, class_ids

    keep = np.isin(class_ids, list(allowed_classes))
    return boxes[keep], scores[keep], class_ids[keep]


def draw_detections(
    image_bgr,
    boxes,
    scores,
    class_ids,
    class_name_lookup,
    draw_boxes,
    draw_labels,
    draw_conf,
    line_width,
    font_scale,
):
    canvas = image_bgr.copy()

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = np.round(box).astype(int)

        if draw_boxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), line_width)

        text_parts = []
        if draw_labels:
            text_parts.append(class_name_lookup.get(int(class_id), str(int(class_id))))
        if draw_conf:
            text_parts.append("{:.2f}".format(float(score)))

        if text_parts:
            text = " ".join(text_parts)
            cv2.putText(
                canvas,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                max(1, line_width),
                cv2.LINE_AA,
            )

    return canvas


class TensorRTEngine:
    def __init__(self, engine_path, device_id):
        cuda.init()
        self.cuda_context = cuda.Device(device_id).make_context()
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as engine_file:
            self.engine = self.runtime.deserialize_cuda_engine(engine_file.read())

        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialize TensorRT engine: {}".format(engine_path)
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self.stream = cuda.Stream()
        self.tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.input_names = [
            name
            for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name
            for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]

        if len(self.input_names) != 1:
            raise RuntimeError(
                "Expected exactly one input tensor, got {}".format(self.input_names)
            )

        self.input_name = self.input_names[0]
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))

        if any(dim == -1 for dim in self.input_shape):
            raise RuntimeError(
                "Dynamic-shape TensorRT engines are not supported by this node."
            )

        self.input_buffer = None
        self.output_buffers = {}
        self.tensor_addresses = {}
        self._allocate_buffers()
        self.cuda_context.pop()

    def _allocate_buffers(self):
        for name in self.tensor_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            buffer_info = {
                "host": host_mem,
                "device": device_mem,
                "shape": shape,
                "dtype": dtype,
            }
            self.tensor_addresses[name] = int(device_mem)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_buffer = buffer_info
            else:
                self.output_buffers[name] = buffer_info

    def infer(self, input_tensor):
        self.cuda_context.push()
        try:
            if tuple(input_tensor.shape) != tuple(self.input_shape):
                raise ValueError(
                    "Expected input shape {}, got {}".format(
                        self.input_shape,
                        input_tensor.shape,
                    )
                )

            np.copyto(self.input_buffer["host"], input_tensor.ravel())
            cuda.memcpy_htod_async(
                self.input_buffer["device"],
                self.input_buffer["host"],
                self.stream,
            )

            for name in self.tensor_names:
                self.context.set_tensor_address(name, self.tensor_addresses[name])

            self.context.execute_async_v3(stream_handle=self.stream.handle)

            outputs = {}
            for name, buffer_info in self.output_buffers.items():
                cuda.memcpy_dtoh_async(
                    buffer_info["host"],
                    buffer_info["device"],
                    self.stream,
                )

            self.stream.synchronize()

            for name, buffer_info in self.output_buffers.items():
                outputs[name] = buffer_info["host"].reshape(buffer_info["shape"]).copy()

            return outputs
        finally:
            self.cuda_context.pop()

    def close(self):
        try:
            try:
                self.cuda_context.push()
            except Exception:
                pass
            if hasattr(self, "context"):
                del self.context
            if hasattr(self, "engine"):
                del self.engine
            if hasattr(self, "runtime"):
                del self.runtime
        finally:
            try:
                self.cuda_context.pop()
            except Exception:
                pass
            try:
                self.cuda_context.detach()
            except Exception:
                pass


class RFDetrTRTNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_lock = threading.Lock()
        self.latest_image = None

        engine_path = rospy.get_param("~engine_path", "")
        if not engine_path:
            raise ValueError(
                "Parameter '~engine_path' must point to a TensorRT engine file."
            )

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(
                "Engine file not found: {}".format(self.engine_path)
            )

        self.publish_result_image = bool(rospy.get_param("~publish_result_image", True))
        self.conf_thres = float(rospy.get_param("~conf_thres", 0.25))
        self.max_det = int(rospy.get_param("~max_det", 300))
        self.allowed_classes = parse_int_set(rospy.get_param("~classes", []))
        self.class_id_map = parse_int_dict(rospy.get_param("~class_id_map", {}))
        self.class_name_lookup = parse_class_name_lookup(
            rospy.get_param("~class_names", [])
        )
        self.labels_output_name = rospy.get_param("~labels_output_name", "labels")
        self.boxes_output_name = rospy.get_param("~boxes_output_name", "dets")
        self.result_conf = bool(rospy.get_param("~result_conf", True))
        self.result_line_width = max(1, int(rospy.get_param("~result_line_width", 1)))
        self.result_font_size = float(rospy.get_param("~result_font_size", 1.0))
        self.result_labels = bool(rospy.get_param("~result_labels", True))
        self.result_boxes = bool(rospy.get_param("~result_boxes", True))
        self.device_id = parse_device_index(rospy.get_param("~device", "cuda:0"))

        self.result_pub = rospy.Publisher("result", YoloResult, queue_size=1)
        self.result_image_pub = None
        if self.publish_result_image:
            self.result_image_pub = rospy.Publisher(
                "result_image/compressed", CompressedImage, queue_size=1
            )
        self.image_sub = rospy.Subscriber(
            "input_image",
            Image,
            self._image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo(
            "[RFDETR-TRT] Loading engine '%s' on cuda:%d",
            str(self.engine_path),
            self.device_id,
        )
        self.engine = TensorRTEngine(str(self.engine_path), self.device_id)
        _, _, self.input_height, self.input_width = self.engine.input_shape

        rospy.loginfo(
            "[RFDETR-TRT] Ready. input=%s outputs=%s publish_result_image=%s",
            self.engine.input_shape,
            self.engine.output_names,
            self.publish_result_image,
        )

        rospy.on_shutdown(self.cleanup)

    def _image_callback(self, msg):
        with self.image_lock:
            self.latest_image = msg

    def _pop_latest_image(self):
        with self.image_lock:
            msg = self.latest_image
            self.latest_image = None
        return msg

    def _publish_result_image(self, header, cv_image, boxes, scores, class_ids):
        if not self.publish_result_image or self.result_image_pub is None:
            return

        vis_image = draw_detections(
            cv_image,
            boxes,
            scores,
            class_ids,
            self.class_name_lookup,
            self.result_boxes,
            self.result_labels,
            self.result_conf,
            self.result_line_width,
            self.result_font_size,
        )
        ok, encoded = cv2.imencode(".jpg", vis_image)
        if not ok:
            rospy.logwarn_throttle(5.0, "[RFDETR-TRT] Failed to encode debug image.")
            return

        image_msg = CompressedImage()
        image_msg.header = header
        image_msg.format = "jpeg"
        image_msg.data = encoded.tobytes()
        self.result_image_pub.publish(image_msg)

    def run(self):
        idle_rate = rospy.Rate(IDLE_RATE_HZ)

        while not rospy.is_shutdown():
            msg = self._pop_latest_image()
            if msg is None:
                idle_rate.sleep()
                continue

            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                input_tensor, orig_hw = preprocess_image(
                    cv_image,
                    (self.input_height, self.input_width),
                )
                outputs = self.engine.infer(input_tensor)
                boxes, scores, class_ids = postprocess_rfdetr(
                    outputs,
                    orig_hw,
                    self.labels_output_name,
                    self.boxes_output_name,
                    topk=self.max_det,
                    threshold=self.conf_thres,
                )
                class_ids = remap_class_ids(class_ids, self.class_id_map)
                boxes, scores, class_ids = filter_detections_by_classes(
                    boxes,
                    scores,
                    class_ids,
                    self.allowed_classes,
                )

                result_msg = make_yolo_result(msg.header, boxes, scores, class_ids)
                self.result_pub.publish(result_msg)
                self._publish_result_image(
                    msg.header, cv_image, boxes, scores, class_ids
                )
            except Exception as exc:
                rospy.logerr_throttle(5.0, "[RFDETR-TRT] Inference failed: %s", exc)

    def cleanup(self):
        try:
            if hasattr(self, "engine") and self.engine is not None:
                self.engine.close()
                self.engine = None
        except Exception as exc:
            rospy.logwarn("[RFDETR-TRT] Cleanup failed: %s", exc)


if __name__ == "__main__":
    rospy.init_node("rfdetr_trt_node")
    try:
        RFDetrTRTNode().run()
    except rospy.ROSInterruptException:
        pass
