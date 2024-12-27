#!/usr/bin/env python3
import rospy
import yaml
import traceback
from auv_msgs.srv import ExecuteStateMachine

def test_state_service():
    rospy.init_node('test_state_service')
    
    # Load test states from configuration
    try:
        with open('/home/frk/catkin_ws/src/auv-software/auv_smach/config/state_params.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        rospy.loginfo(f"Loaded configuration: {config}")
    except Exception as e:
        rospy.logerr(f"Failed to load configuration: {e}")
        traceback.print_exc()
        return
    
    # Wait for the service to be available
    try:
        rospy.wait_for_service('/taluy/state_machine/execute', timeout=10)
    except rospy.ROSException:
        rospy.logerr("Service not available after 10 seconds")
        return
    
    try:
        # Create a service proxy
        execute_state = rospy.ServiceProxy('/taluy/state_machine/execute', ExecuteStateMachine)
        
        # Get states to run from configuration
        states_to_run = config.get('test_configuration', {}).get('run_states', [])
        rospy.loginfo(f"States to run: {states_to_run}")
        
        # Get state machine configurations
        state_machine_config = config.get('state_machine', {})
        
        for name in states_to_run:
            # Skip if state not found in configuration
            if name not in state_machine_config:
                rospy.logerr(f"State {name} not found in configuration")
                continue
            
            state_config = state_machine_config[name]
            rospy.loginfo(f"Testing state: {name}")
            
            # Prepare service call parameters
            service_params = {
                'name': name,
                'type': [state_config['type']]
            }
            
            # Add optional parameters if exist
            if 'parameters' in state_config:
                # Convert parameters to a list of strings for service call
                params_list = [f"{k}={v}" for k, v in state_config['parameters'].items()]
                service_params['type'].extend(params_list)
            
            rospy.loginfo(f"Calling service with params: {service_params}")
            
            try:
                response = execute_state(**service_params)
                
                if response.success:
                    rospy.loginfo(f"State {name} executed successfully")
                    rospy.loginfo(f"Message: {response.message}")
                else:
                    rospy.logerr(f"State {name} execution failed: {response.message}")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call for {name} failed: {e}")
    
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    test_state_service()