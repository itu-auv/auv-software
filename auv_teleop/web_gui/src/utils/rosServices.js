// ROS Service Calls
export const callService = (ros, serviceName, serviceType, request = {}) => {
  if (!ros) {
    console.error('ROS not connected');
    return Promise.reject('ROS not connected');
  }

  return new Promise((resolve, reject) => {
    const service = new window.ROSLIB.Service({
      ros: ros,
      name: serviceName,
      serviceType: serviceType
    });

    const serviceRequest = new window.ROSLIB.ServiceRequest(request);

    service.callService(serviceRequest, 
      (result) => {
        console.log(`✓ Service ${serviceName} succeeded:`, result);
        resolve(result);
      }, 
      (error) => {
        console.error(`✗ Service ${serviceName} failed:`, error);
        reject(error);
      }
    );
  });
};

// Vehicle Control
export const publishCmdVel = (ros, linear, angular) => {
  if (!ros) return;

  const cmdVel = new window.ROSLIB.Topic({
    ros: ros,
    name: '/taluy/cmd_vel',
    messageType: 'geometry_msgs/Twist'
  });

  const twist = new window.ROSLIB.Message({
    linear: { x: linear.x || 0, y: linear.y || 0, z: linear.z || 0 },
    angular: { x: angular.x || 0, y: angular.y || 0, z: angular.z || 0 }
  });

  cmdVel.publish(twist);
};

// Control Enable/Disable
export const enableControl = (ros, setControlEnabled) => {
  if (!ros) return;
  
  setControlEnabled(true);
  const enableTopic = new window.ROSLIB.Topic({
    ros: ros,
    name: '/taluy/enable',
    messageType: 'std_msgs/Bool'
  });

  const publishInterval = setInterval(() => {
    const msg = new window.ROSLIB.Message({ data: true });
    enableTopic.publish(msg);
  }, 50); // 20Hz = 50ms

  window.enablePublishInterval = publishInterval;
};

export const disableControl = (ros, setControlEnabled) => {
  if (!ros) return;
  
  setControlEnabled(false);
  
  if (window.enablePublishInterval) {
    clearInterval(window.enablePublishInterval);
    window.enablePublishInterval = null;
  }

  // Stop vehicle movement first
  const cmdVel = new window.ROSLIB.Topic({
    ros: ros,
    name: '/taluy/cmd_vel',
    messageType: 'geometry_msgs/Twist'
  });
  const stopTwist = new window.ROSLIB.Message({
    linear: { x: 0, y: 0, z: 0 },
    angular: { x: 0, y: 0, z: 0 }
  });
  cmdVel.publish(stopTwist);

  // Then disable control
  const enableTopic = new window.ROSLIB.Topic({
    ros: ros,
    name: '/taluy/enable',
    messageType: 'std_msgs/Bool'
  });
  const msg = new window.ROSLIB.Message({ data: false });
  enableTopic.publish(msg);
};
