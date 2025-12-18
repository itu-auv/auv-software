#!/usr/bin/env python3

import rospy
import os
import json
import google.generativeai as genai
from auv_msgs.srv import SetDepth, AlignFrameController, VisualServoing
from auv_nlp.srv import Query, QueryResponse, Execute, ExecuteResponse

# Tool Definitions for valid AUV commands
AUV_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "set_depth_command",
                "description": "Sets the AUV target depth. Negative values mean underwater.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_depth": {
                            "type": "number",
                            "description": "Target depth in meters (e.g., -1.5)"
                        }
                    },
                    "required": ["target_depth"]
                }
            },
            {
                "name": "align_frame_command",
                "description": "Aligns the AUV to a specific target frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_frame": {
                            "type": "string",
                            "description": "Target frame name (e.g., gate, red_buoy, bin)"
                        }
                    },
                    "required": ["target_frame"]
                }
            },
            {
                "name": "vision_command",
                "description": "Activates visual servoing for a target object.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_object": {
                            "type": "string",
                            "description": "Target object name"
                        }
                    },
                    "required": ["target_object"]
                }
            },
            {
                "name": "surface_emergency",
                "description": "Emergency command to surface immediately.",
                "parameters": {
                    "type": "object",
                    "properties": {} # No params needed
                }
            }
        ]
    }
]

class NLPCommanderNode:
    def __init__(self):
        rospy.init_node('nlp_commander')
        
        # Get API Key
        self.api_key = rospy.get_param('~google_api_key', os.environ.get('GOOGLE_API_KEY'))
        if not self.api_key:
            rospy.logerr("Google API Key not found! Please set GOOGLE_API_KEY env var or param.")
            return

        genai.configure(api_key=self.api_key)
        
        # Init Model
        self.model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-exp', 
            tools=AUV_TOOLS,
            system_instruction="You are an intelligent assistant for an AUV (Autonomous Underwater Vehicle). "
                               "Map user instructions to the provided tools/functions. "
                               "Always prioritize safety. If a command is ambiguous, ask for clarification. "
                               "For depth, ensure values are typically negative (e.g., -0.5, -1.0)."
        )
        
        # Service Clients
        self.services = {
            'set_depth': rospy.ServiceProxy('/auv/set_depth', SetDepth),
            'align_frame': rospy.ServiceProxy('/auv/align_controller', AlignFrameController),
            'visual_servo': rospy.ServiceProxy('/auv/visual_servoing', VisualServoing)
        }
        
        # Service Servers
        rospy.Service('auv_nlp/query', Query, self.handle_query)
        rospy.Service('auv_nlp/execute', Execute, self.handle_execute)
        
        rospy.loginfo("NLP Commander Node Ready! Services: auv_nlp/query, auv_nlp/execute")

    def handle_query(self, req):
        """Processes prompt with Gemini and returns plan"""
        rospy.loginfo(f"Handling Query: {req.prompt}")
        try:
            response = self.model.generate_content(req.prompt)
            
            # Check for function calls
            if response.candidates and response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                if part.function_call:
                    fc = part.function_call
                    
                    # Convert RepeatedComposite to dict for serialization
                    args = dict(fc.args) 
                    
                    fc_data = {
                        "name": fc.name,
                        "args": args
                    }
                    
                    # Create a readable description
                    description = f"Call {fc.name} with {args}"
                    
                    return QueryResponse(
                        plan_description=description,
                        function_call_json=json.dumps(fc_data)
                    )
            
            return QueryResponse(
                plan_description="No action planned. Maybe just a chat response?",
                function_call_json="{}"
            )
            
        except Exception as e:
            rospy.logerr(f"Query failed: {e}")
            return QueryResponse(plan_description=f"Error: {str(e)}", function_call_json="{}")

    def handle_execute(self, req):
        """Executes the provided JSON function call"""
        try:
            fc_data = json.loads(req.function_call_json)
            if not fc_data or "name" not in fc_data:
                return ExecuteResponse(success=False, message="Empty or invalid plan")
            
            fname = fc_data["name"]
            args = fc_data["args"]
            
            rospy.loginfo(f"Executing: {fname} with {args}")
            
            if fname == 'set_depth_command':
                self.call_set_depth(args['target_depth'])
            elif fname == 'align_frame_command':
                self.call_align_frame(args['target_frame'])
            elif fname == 'surface_emergency':
                self.call_set_depth(0.0)
            elif fname == 'vision_command':
                self.call_vision_command(args['target_object'])
            else:
                return ExecuteResponse(success=False, message=f"Unknown function: {fname}")
            
            return ExecuteResponse(success=True, message="Command executed")
            
        except Exception as e:
            rospy.logerr(f"Execution failed: {e}")
            return ExecuteResponse(success=False, message=str(e))

    def call_set_depth(self, depth):
        # SetDepth srv: float32 target_depth, string frame_id
        self.services['set_depth'](target_depth=depth, frame_id="base_link")

    def call_align_frame(self, frame):
        self.services['align_frame'](
            source_frame="base_link",
            target_frame=frame,
            angle_offset=0.0,
            keep_orientation=False,
            max_linear_velocity=0.5,
            max_angular_velocity=0.5
        )

    def call_vision_command(self, target):
        self.services['visual_servo'](target_prop=target)

if __name__ == '__main__':
    try:
        node = NLPCommanderNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
