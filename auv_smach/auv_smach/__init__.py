from rclpy.impl.rcutils_logger import RcutilsLogger

# smach_ros 3.0.3 calls logger.warn at import time; rclpy on Lyrical only has warning.
if not hasattr(RcutilsLogger, "warn"):
    RcutilsLogger.warn = RcutilsLogger.warning
