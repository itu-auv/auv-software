#!/usr/bin/env python3

import rospy
import sys
import argparse
from auv_nlp.srv import Query, Execute

def gen_cli():
    rospy.init_node('gen_cli', anonymous=True)
    
    # Parse args
    parser = argparse.ArgumentParser(description='AUV NLP Generator CLI')
    parser.add_argument('prompt', nargs='+', help='Natural language prompt')
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    
    full_prompt = " ".join(args.prompt)
    
    print(f"\n🔮 Generating plan for: '{full_prompt}'...")
    
    # 1. Query the plan
    rospy.wait_for_service('auv_nlp/query')
    try:
        query_service = rospy.ServiceProxy('auv_nlp/query', Query)
        response = query_service(full_prompt)
        
        if response.function_call_json == "{}":
            print("❌ No executable plan found.")
            print(f"Model says: {response.plan_description}")
            return

        print("\n" + "="*40)
        print("📋 PROPOSED ACTION:")
        print("="*40)
        print(f"{response.plan_description}")
        print("="*40)
        
        # 2. Ask for confirmation
        user_input = input("\n🚀 Execute this plan? [Enter] to confirm, [Ctrl+C] to cancel: ")
        
        # 3. Execute
        rospy.wait_for_service('auv_nlp/execute')
        execute_service = rospy.ServiceProxy('auv_nlp/execute', Execute)
        exec_res = execute_service(response.function_call_json)
        
        if exec_res.success:
            print(f"\n✅ SUCCESS: {exec_res.message}")
        else:
            print(f"\n❌ FAILED: {exec_res.message}")
            
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
    except KeyboardInterrupt:
        print("\n🚫 Cancelled by user.")

if __name__ == "__main__":
    gen_cli()
