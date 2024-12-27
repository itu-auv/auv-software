#!/usr/bin/env python3
import rospy
import importlib
import traceback
import smach
from auv_msgs.srv import ExecuteStateMachine, ExecuteStateMachineResponse

def execute_state_callback(req):
    try:
        # Check if type is a list and take the first element
        state_type = req.type[0] if isinstance(req.type, list) else req.type

        # Dynamically import the state class
        module_name, class_name = state_type.rsplit('.', 1)
        
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            rospy.logerr(f"Could not import module: {module_name}")
            return ExecuteStateMachineResponse(
                success=False, 
                message=f"Module import failed: {module_name}"
            )

        try:
            StateClass = getattr(module, class_name)
        except AttributeError:
            rospy.logerr(f"Could not find class: {class_name} in module {module_name}")
            return ExecuteStateMachineResponse(
                success=False, 
                message=f"Class not found: {class_name}"
            )

        # Parse additional parameters
        extra_params = {}
        for param in req.type[1:]:
            if '=' in param:
                key, value = param.split('=')
                try:
                    # Convert to appropriate type
                    extra_params[key] = eval(value)
                except:
                    extra_params[key] = value

        # Determine which parameters to pass based on state class
        import inspect
        sig = inspect.signature(StateClass.__init__)
        valid_params = {
            k: v for k, v in extra_params.items() 
            if k in sig.parameters and k != 'self'
        }

        # Create state instance with valid parameters
        try:
            state_instance = StateClass(**valid_params)
        except Exception as inst:
            rospy.logerr(f"Failed to instantiate state: {str(inst)}")
            rospy.logerr(f"Attempted parameters: {valid_params}")
            return ExecuteStateMachineResponse(
                success=False, 
                message=f"State instantiation failed: {str(inst)}"
            )

        # Execute the state
        try:
            # Create an empty userdata if the state expects it
            if isinstance(state_instance, smach.State):
                # For SMACH states, create an empty UserData
                userdata = smach.UserData()
                outcome = state_instance.execute(userdata)
            else:
                # For states without userdata
                outcome = state_instance.execute()
        except TypeError:
            # If execute() doesn't accept arguments, try calling without
            outcome = state_instance.execute()
        except Exception as inst:
            rospy.logerr(f"State execution failed: {str(inst)}")
            rospy.logerr(traceback.format_exc())
            return ExecuteStateMachineResponse(
                success=False, 
                message=f"State execution failed: {str(inst)}"
            )

        # Determine success based on outcome
        if outcome == 'succeeded':
            return ExecuteStateMachineResponse(
                success=True, 
                message=f"State {req.name} executed successfully"
            )
        else:
            return ExecuteStateMachineResponse(
                success=False, 
                message=f"State {req.name} failed with outcome: {outcome}"
            )

    except Exception as e:
        rospy.logerr(f"Unexpected error executing state {req.name}: {str(e)}")
        rospy.logerr(traceback.format_exc())
        return ExecuteStateMachineResponse(
            success=False, 
            message=str(e)
        )

def state_service_handler():
    rospy.init_node('state_machine_service')
    rospy.Service('/taluy/state_machine/execute', 
                  ExecuteStateMachine, 
                  execute_state_callback)
    rospy.loginfo("State Machine Service Ready")
    rospy.spin()

if __name__ == '__main__':
    state_service_handler()