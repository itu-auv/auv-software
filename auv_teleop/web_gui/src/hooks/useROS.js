import { useState, useEffect } from 'react';

export function useROS() {
  const [ros, setRos] = useState(null);
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);

  // Power state
  const [voltage, setVoltage] = useState(0);
  const [current, setCurrent] = useState(0);
  const [power, setPower] = useState(0);
  const [powerHistory, setPowerHistory] = useState([]);
  const [powerTopicName, setPowerTopicName] = useState('');
  const [lastPowerUpdate, setLastPowerUpdate] = useState(null);
  const [powerSource, setPowerSource] = useState('unknown');

  const subscribeToPower = (rosInstance) => {
    console.log('🔍 Searching for power topic...');
    
    rosInstance.getTopics((result) => {
      console.log('📋 All topics:', result.topics);
      
      const possiblePowerTopics = [
        '/taluy/power',
        '/power',
        '/battery/status',
        '/auv/power'
      ];
      
      let foundTopic = null;
      for (const candidate of possiblePowerTopics) {
        if (result.topics.includes(candidate)) {
          foundTopic = candidate;
          break;
        }
      }
      
      if (!foundTopic) {
        foundTopic = result.topics.find(t => 
          t.toLowerCase().includes('power') || 
          t.toLowerCase().includes('battery')
        );
      }
      
      if (foundTopic) {
        console.log(`✅ Found power topic: ${foundTopic}`);
        setPowerTopicName(foundTopic);
        
        const hasGazeboTopics = result.topics.some(t => t.includes('gazebo'));
        setPowerSource(hasGazeboTopics ? 'simulation' : 'hardware');
        
        const powerListener = new window.ROSLIB.Topic({
          ros: rosInstance,
          name: foundTopic,
          messageType: 'auv_msgs/Power'
        });

        powerListener.subscribe((message) => {
          console.log('⚡ Power data:', message);
          
          if (message && typeof message.voltage === 'number') {
            setVoltage(message.voltage);
            setCurrent(message.current || 0);
            setPower(message.power || Math.abs(message.voltage * (message.current || 0)));
            setLastPowerUpdate(new Date());
            
            setPowerHistory(prev => {
              const newHistory = [...prev, {
                time: new Date(),
                power: message.power || Math.abs(message.voltage * (message.current || 0)),
                voltage: message.voltage,
                current: message.current || 0
              }];
              return newHistory.slice(-60);
            });
          }
        });
        
      } else {
        console.error('❌ No power topic found!');
        console.log('💡 Available topics:', result.topics);
        setPowerTopicName('NOT_FOUND');
      }
      
    }, (error) => {
      console.error('❌ Failed to get topics:', error);
      setPowerTopicName('ERROR');
    });
  };

  const connectToROS = () => {
    setConnecting(true);
    
    const rosInstance = new window.ROSLIB.Ros({
      url: 'ws://localhost:9090'
    });

    rosInstance.on('connection', () => {
      console.log('✓ Connected to ROS Bridge!');
      setConnected(true);
      setConnecting(false);
      subscribeToPower(rosInstance);
    });

    rosInstance.on('error', (error) => {
      console.error('✗ ROS connection error:', error);
      setConnected(false);
      setConnecting(false);
    });

    rosInstance.on('close', () => {
      console.log('Connection to ROS closed');
      setConnected(false);
      setConnecting(false);
      setPowerTopicName('');
      setVoltage(0);
      setCurrent(0);
      setPower(0);
      setLastPowerUpdate(null);
    });

    setRos(rosInstance);
  };

  useEffect(() => {
    connectToROS();
    
    return () => {
      if (ros) {
        ros.close();
      }
    };
  }, []);

  return {
    ros,
    connected,
    connecting,
    connectToROS,
    voltage,
    current,
    power,
    powerHistory,
    powerTopicName,
    lastPowerUpdate,
    powerSource
  };
}
