#!/usr/bin/env python3

import asyncio
import threading

import rospy
from sensor_msgs.msg import BatteryState

try:
    from telethon import TelegramClient, events
except ImportError:
    TelegramClient = None
    events = None


class TelegramTelemetryNode:
    def __init__(self):
        self.namespace = rospy.get_param("namespace", rospy.get_namespace().strip("/"))
        self.battery_low_voltage = rospy.get_param("~battery_low_voltage", 14.5)
        self.battery_low_debounce_seconds = rospy.get_param(
            "~battery_low_debounce_seconds", 2.0
        )
        self.battery_low_repeat_interval = rospy.get_param(
            "~battery_low_repeat_interval", 60.0
        )
        self.power_message_timeout = rospy.get_param("~power_message_timeout", 5.0)
        self.session_name = rospy.get_param("~session_name", self.namespace)

        self.api_id = rospy.get_param("~telegram_api_id", "")
        self.api_hash = rospy.get_param("~telegram_api_hash", "")
        self.bot_token = rospy.get_param("~telegram_bot_token", "")
        self.chat_id = rospy.get_param("~telegram_chat_id", "")

        self.state_lock = threading.Lock()
        self.latest_voltage = None
        self.last_msg_time = None
        self.low_battery_active = False
        self.low_battery_since = None
        self.last_low_battery_notification_time = None

        self.client = None
        self.event_loop = None
        self.low_battery_timer = None
        self.telegram_chat = None

        rospy.Subscriber("power", BatteryState, self.power_callback)

    def is_configured(self):
        return all([self.api_id, self.api_hash, self.bot_token, self.chat_id])

    def power_callback(self, msg):
        with self.state_lock:
            self.latest_voltage = msg.voltage
            self.last_msg_time = rospy.Time.now()

            if msg.voltage < self.battery_low_voltage:
                self.low_battery_active = True
                if self.low_battery_since is None:
                    self.low_battery_since = self.last_msg_time
            else:
                self.low_battery_active = False
                self.low_battery_since = None
                self.last_low_battery_notification_time = None

    def build_status_message(self):
        with self.state_lock:
            voltage = self.latest_voltage
            last_msg_time = self.last_msg_time

        if voltage is None or last_msg_time is None:
            return f"{self.namespace} battery status is unavailable."

        age = (rospy.Time.now() - last_msg_time).to_sec()
        if age > self.power_message_timeout:
            return f"{self.namespace}: {voltage:.2f}V (stale for {age:.1f}s)"

        return f"{self.namespace}: {voltage:.2f}V"

    def evaluate_low_battery_alert(self, _event):
        if self.client is None or self.event_loop is None:
            return

        alert_text = None
        now = rospy.Time.now()

        with self.state_lock:
            if self.latest_voltage is None or self.last_msg_time is None:
                return

            if not self.low_battery_active or self.low_battery_since is None:
                return

            if (now - self.last_msg_time).to_sec() > self.power_message_timeout:
                return

            if (
                now - self.low_battery_since
            ).to_sec() < self.battery_low_debounce_seconds:
                return

            if self.last_low_battery_notification_time is not None:
                elapsed = (now - self.last_low_battery_notification_time).to_sec()
                if elapsed < self.battery_low_repeat_interval:
                    return

            self.last_low_battery_notification_time = now
            alert_text = (
                f"low battery alert for {self.namespace}: {self.latest_voltage:.2f}V"
            )

        self.schedule_send_message(alert_text)

    def schedule_send_message(self, text):
        if self.client is None or self.event_loop is None:
            return False

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.client.send_message(self.telegram_chat, text), self.event_loop
            )
        except RuntimeError:
            return False

        future.add_done_callback(self.log_future_exception)
        return True

    def log_future_exception(self, future):
        exc = future.exception()
        if exc is not None and not rospy.is_shutdown():
            rospy.logerr(f"Telegram operation failed: {exc}")

    def shutdown(self):
        if self.low_battery_timer is not None:
            self.low_battery_timer.shutdown()

        if self.client is None or self.event_loop is None:
            return

        try:
            self.event_loop.call_soon_threadsafe(
                lambda: self.event_loop.create_task(
                    self.client.disconnect()
                ).add_done_callback(self.log_future_exception)
            )
        except RuntimeError:
            return

    async def run(self):
        if TelegramClient is None:
            rospy.logerr(
                "Telethon is not installed. Install it in the runtime environment before using telegram_telemetry_node."
            )
            return

        if not self.is_configured():
            rospy.logwarn(
                "Telegram telemetry is disabled because credentials are missing."
            )
            return

        try:
            api_id = int(self.api_id)
        except (TypeError, ValueError):
            rospy.logerr("telegram_api_id must be an integer.")
            return

        try:
            self.telegram_chat = int(self.chat_id)
        except (TypeError, ValueError):
            self.telegram_chat = self.chat_id

        self.event_loop = asyncio.get_running_loop()
        self.client = TelegramClient(self.session_name, api_id, self.api_hash)
        rospy.on_shutdown(self.shutdown)
        self.client.add_event_handler(
            lambda event: event.reply(self.build_status_message()),
            events.NewMessage(
                chats=self.telegram_chat, pattern=r"(?i)^/status(?:@\w+)?$"
            ),
        )

        await self.client.start(bot_token=self.bot_token)
        self.low_battery_timer = rospy.Timer(
            rospy.Duration(0.5), self.evaluate_low_battery_alert
        )

        rospy.loginfo("telegram_telemetry_node connected to Telegram.")

        await self.client.run_until_disconnected()


if __name__ == "__main__":
    rospy.init_node("telegram_telemetry_node", anonymous=False)

    try:
        asyncio.run(TelegramTelemetryNode().run())
    except rospy.ROSInterruptException:
        pass
