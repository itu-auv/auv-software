import smach
import rospy
from auv_smach.common import (
    SetDepthState,
)
import smach_ros
from std_srvs.srv import Trigger, TriggerRequest
from std_msgs.msg import Float64
from auv_msgs.srv import VisualServoing, VisualServoingRequest
from auv_smach.initialize import DelayState
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, UInt8
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse


class VisualServoingCenteringActivate(smach_ros.ServiceState):
    """
    Simple service state that just activates visual servoing centering
    without waiting for convergence. Used when you want to start centering
    and then use a DelayState to wait.
    """

    def __init__(self, target_prop: str):
        request = VisualServoingRequest()
        request.target_prop = target_prop
        super(VisualServoingCenteringActivate, self).__init__(
            "visual_servoing/start",
            VisualServoing,
            request=request,
            outcomes=["succeeded", "preempted", "aborted"],
        )


class VisualServoingCenteringWithFeedback(smach.State):
    """
    Enhanced visual servoing centering state that waits for actual convergence
    instead of using a fixed delay.
    """

    def __init__(
        self,
        target_prop: str,
        error_threshold: float = 0.1,
        convergence_time: float = 2.0,
        max_timeout: float = 30.0,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.target_prop = target_prop
        self.error_threshold = error_threshold  # radians
        self.convergence_time = convergence_time  # seconds to stay within threshold
        self.max_timeout = max_timeout

        # State tracking
        self.current_error = float("inf")
        self.convergence_start_time = None
        self.is_converged = False

        # ROS communication
        self.error_sub = None

    def error_callback(self, msg: Float64):
        """Callback for visual servoing error messages."""
        self.current_error = abs(msg.data)

        # Check if we're within the error threshold
        if self.current_error <= self.error_threshold:
            if self.convergence_start_time is None:
                self.convergence_start_time = rospy.Time.now()
                rospy.loginfo(
                    f"Centering: Error within threshold ({self.current_error:.3f} <= {self.error_threshold:.3f})"
                )
            else:
                # Check if we've been converged long enough
                convergence_duration = (
                    rospy.Time.now() - self.convergence_start_time
                ).to_sec()
                if convergence_duration >= self.convergence_time:
                    self.is_converged = True
                    rospy.loginfo(
                        f"Centering: Converged for {convergence_duration:.1f}s, centering complete!"
                    )
        else:
            # Reset convergence tracking if error goes above threshold
            if self.convergence_start_time is not None:
                rospy.loginfo(
                    f"Centering: Error increased ({self.current_error:.3f}), resetting convergence timer"
                )
            self.convergence_start_time = None
            self.is_converged = False

    def execute(self, userdata):
        # Reset state
        self.current_error = float("inf")
        self.convergence_start_time = None
        self.is_converged = False

        # Subscribe to error topic
        self.error_sub = rospy.Subscriber(
            "visual_servoing/error", Float64, self.error_callback, queue_size=1
        )

        try:
            # Start visual servoing
            rospy.loginfo(f"Starting visual servoing centering for: {self.target_prop}")
            rospy.wait_for_service("visual_servoing/start", timeout=5.0)
            start_service = rospy.ServiceProxy("visual_servoing/start", VisualServoing)

            request = VisualServoingRequest()
            request.target_prop = self.target_prop
            response = start_service(request)

            if not response.success:
                rospy.logerr(f"Failed to start visual servoing: {response.message}")
                return "aborted"

            # Wait for convergence
            start_time = rospy.Time.now()
            rate = rospy.Rate(10)  # 10 Hz

            while not rospy.is_shutdown() and not self.is_converged:
                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"

                # Check timeout
                if (rospy.Time.now() - start_time).to_sec() > self.max_timeout:
                    rospy.logwarn(
                        f"Centering timeout after {self.max_timeout}s. Current error: {self.current_error:.3f}"
                    )
                    return "aborted"

                # Log progress periodically
                if (
                    rospy.Time.now() - start_time
                ).to_sec() % 5.0 < 0.1:  # Every 5 seconds
                    if self.convergence_start_time is not None:
                        conv_time = (
                            rospy.Time.now() - self.convergence_start_time
                        ).to_sec()
                        rospy.loginfo(
                            f"Centering: Converging for {conv_time:.1f}s (need {self.convergence_time:.1f}s)"
                        )
                    else:
                        rospy.loginfo(
                            f"Centering: Current error: {self.current_error:.3f} (threshold: {self.error_threshold:.3f})"
                        )

                rate.sleep()

            if self.is_converged:
                rospy.loginfo("Visual servoing centering completed successfully!")
                return "succeeded"
            else:
                return "preempted"  # Shutdown case

        except rospy.ROSException as e:
            rospy.logerr(f"ROS exception during centering: {e}")
            return "aborted"
        finally:
            # Cleanup subscriber
            if self.error_sub is not None:
                self.error_sub.unregister()


class VisualServoingNavigationWithFeedback(smach.State):
    """
    Enhanced navigation state that monitors the visual servoing status
    and can detect when navigation is complete or failed.
    """

    def __init__(
        self, max_navigation_time: float = 60.0, prop_lost_timeout: float = 10.0
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.max_navigation_time = max_navigation_time
        self.prop_lost_timeout = prop_lost_timeout

        # State tracking
        self.current_error = float("inf")
        self.last_error_time = None
        self.error_sub = None

    def error_callback(self, msg: Float64):
        """Callback for visual servoing error messages."""
        self.current_error = abs(msg.data)
        self.last_error_time = rospy.Time.now()

    def execute(self, userdata):
        # Reset state
        self.current_error = float("inf")
        self.last_error_time = None

        # Subscribe to error topic
        self.error_sub = rospy.Subscriber(
            "visual_servoing/error", Float64, self.error_callback, queue_size=1
        )

        try:
            # Start navigation
            rospy.loginfo("Starting visual servoing navigation")
            rospy.wait_for_service("visual_servoing/navigate", timeout=5.0)
            nav_service = rospy.ServiceProxy("visual_servoing/navigate", Trigger)

            response = nav_service(TriggerRequest())
            if not response.success:
                rospy.logerr(f"Failed to start navigation: {response.message}")
                return "aborted"

            # Monitor navigation
            start_time = rospy.Time.now()
            rate = rospy.Rate(20)  # 20 Hz

            while not rospy.is_shutdown():
                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"

                current_time = rospy.Time.now()
                elapsed_time = (current_time - start_time).to_sec()

                # Check overall timeout
                if elapsed_time > self.max_navigation_time:
                    rospy.loginfo(
                        f"Navigation completed after {elapsed_time:.1f}s (max time reached)"
                    )
                    return "succeeded"

                # Check if prop is lost (no error updates)
                if self.last_error_time is not None:
                    time_since_error = (current_time - self.last_error_time).to_sec()
                    if time_since_error > self.prop_lost_timeout:
                        rospy.loginfo(
                            f"Navigation completed: prop lost for {time_since_error:.1f}s"
                        )
                        return "succeeded"

                # Log progress periodically
                if elapsed_time % 1.0 < 0.1:  # Every 10 seconds
                    if self.last_error_time is not None:
                        time_since_error = (
                            current_time - self.last_error_time
                        ).to_sec()
                        rospy.loginfo(
                            f"Navigation: t={elapsed_time:.1f}s, error={self.current_error:.3f}, "
                            f"time_since_prop={time_since_error:.1f}s"
                        )
                    else:
                        rospy.loginfo(
                            f"Navigation: t={elapsed_time:.1f}s, no prop detected yet"
                        )

                rate.sleep()

            return "preempted"  # Shutdown case

        except rospy.ROSException as e:
            rospy.logerr(f"ROS exception during navigation: {e}")
            return "aborted"
        finally:
            # Cleanup subscriber
            if self.error_sub is not None:
                self.error_sub.unregister()


class VisualServoingCancelNavigation(smach_ros.ServiceState):
    def __init__(self):
        super(VisualServoingCancelNavigation, self).__init__(
            "visual_servoing/cancel_navigation",
            Trigger,
            request=TriggerRequest(),
            outcomes=["succeeded", "preempted", "aborted"],
        )


class GateBackServiceEnableState(smach_ros.ServiceState):
    def __init__(self, req: bool):
        smach_ros.ServiceState.__init__(
            self,
            "enable_gate_back_detection",
            SetBool,
            request=SetBoolRequest(data=req),
        )


class VisualServoingCancel(smach_ros.ServiceState):
    def __init__(self):
        super(VisualServoingCancel, self).__init__(
            "visual_servoing/cancel",
            Trigger,
            request=TriggerRequest(),
            outcomes=["succeeded", "preempted", "aborted"],
        )


class PublishConstantCmdVelState(smach.State):
    """
    Publishes a constant cmd_vel and an enable flag for a fixed duration.
    Velocity is passed in as an argument when creating the state.
    """

    def __init__(self, duration: float, vel: Twist):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.duration = duration
        self.vel = vel
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=1)

    def execute(self, userdata):
        rospy.loginfo(
            f"Publishing constant cmd_vel for {self.duration:.1f}s "
            f"(lin=({self.vel.linear.x}, {self.vel.linear.y}, {self.vel.linear.z}), "
            f"ang=({self.vel.angular.x}, {self.vel.angular.y}, {self.vel.angular.z})), "
            f"and enabling control."
        )

        rate = rospy.Rate(10)  # 10 Hz
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.preempt_requested():
                rospy.logwarn("PublishConstantCmdVelState preempted")
                self.service_preempt()
                self.enable_pub.publish(False)  # disable before exit
                return "preempted"

            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > self.duration:
                break

            self.cmd_vel_pub.publish(self.vel)
            self.enable_pub.publish(True)
            rate.sleep()

        # Stop the robot and disable after publishing
        self.cmd_vel_pub.publish(Twist())
        self.enable_pub.publish(False)
        rospy.loginfo("Finished constant cmd_vel publishing, disabled control.")
        return "succeeded"


class WaitForAcoustics(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.enabled = False
        self.killswitch_subscriber = rospy.Subscriber("modem/rx", UInt8, self.callback)

    def callback(self, msg):
        if msg is not None:
            self.enabled = True

    def execute(self, userdata):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.enabled:
                return "succeeded"
            rate.sleep()
        return "aborted"


class NavigateThroughGateStateVS(smach.State):
    """
    Improved gate navigation state that uses feedback-based visual servoing
    instead of fixed delays.
    """

    def __init__(self, gate_depth: float, target_prop: str):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        # Initialize the state machine
        self.state_machine = smach.StateMachine(
            outcomes=["succeeded", "preempted", "aborted"]
        )

        with self.state_machine:
            smach.StateMachine.add(
                "WAIT_FOR_ACOUSTICS",
                WaitForAcoustics(),
                transitions={
                    "succeeded": "SET_GATE_DEPTH",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "SET_GATE_DEPTH",
                SetDepthState(depth=-1.0, sleep_duration=5.0),
                transitions={
                    "succeeded": "VISUAL_SERVOING_CENTERING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "VISUAL_SERVOING_CENTERING",
                VisualServoingCenteringWithFeedback(
                    target_prop="shark",
                    error_threshold=0.2,
                    convergence_time=10.0,
                    max_timeout=30.0,
                ),
                transitions={
                    "succeeded": "VISUAL_SERVOING_NAVIGATION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "VISUAL_SERVOING_NAVIGATION",
                VisualServoingNavigationWithFeedback(
                    max_navigation_time=60, prop_lost_timeout=3.0
                ),
                transitions={
                    "succeeded": "CANCEL_VISUAL_SERVOING",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_VISUAL_SERVOING",
                VisualServoingCancel(),
                transitions={
                    "succeeded": "ENABLE_GATE_BACK_DETECTION",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "ENABLE_GATE_BACK_DETECTION",
                GateBackServiceEnableState(req=True),
                transitions={
                    "succeeded": "VISUAL_SERVOING_CENTERING_RETURN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "VISUAL_SERVOING_CENTERING_RETURN",
                VisualServoingCenteringWithFeedback(
                    target_prop="gate_back",
                    error_threshold=0.2,
                    convergence_time=10.0,
                    max_timeout=30.0,
                ),
                transitions={
                    "succeeded": "VISUAL_SERVOING_NAVIGATION_RETURN",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "VISUAL_SERVOING_NAVIGATION_RETURN",
                VisualServoingNavigationWithFeedback(
                    max_navigation_time=60, prop_lost_timeout=3.0
                ),
                transitions={
                    "succeeded": "CANCEL_VISUAL_SERVOING_3",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )
            smach.StateMachine.add(
                "CANCEL_VISUAL_SERVOING_3",
                VisualServoingCancel(),
                transitions={
                    "succeeded": "succeeded",
                    "preempted": "preempted",
                    "aborted": "aborted",
                },
            )

    def execute(self, userdata):
        rospy.loginfo(
            "[ImprovedNavigateThroughGateStateVS] Starting improved gate navigation"
        )

        # Execute the state machine
        outcome = self.state_machine.execute()

        if outcome is None:  # ctrl + c
            return "preempted"
        return outcome
