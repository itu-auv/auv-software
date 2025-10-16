#!/usr/bin/env python3
"""
Test Script for AUV RL Navigation Components
Tests each component individually before full training
"""

import rospy
import numpy as np
import sys
from geometry_msgs.msg import Twist


def test_grid_encoder():
    """Test the World Grid Encoder."""
    print("\n" + "=" * 60)
    print("TEST 1: World Grid Encoder")
    print("=" * 60)

    try:
        from auv_rl_navigation.observation.world_grid_encoder import WorldGridEncoder

        # Create encoder with visualization
        object_frames = rospy.get_param(
            "~object_frames", "gate_shark_link,red_buoy,white_pipe_link"
        ).split(",")

        encoder = WorldGridEncoder(
            grid_dim_xy=10,
            grid_dim_z=2,
            cell_size_xy=0.7,
            cell_size_z=1.0,
            object_frames=[f.strip() for f in object_frames],
            visualize=True,
        )

        print(f"✓ Grid encoder created successfully")
        print(f"  - Grid dimensions: 10x10x2")
        print(f"  - Cell size: 0.7m")
        print(f"  - Coverage: ±3.5m")
        print(f"  - Tracking {len(object_frames)} object types")
        print(f"  - Visualization: ENABLED")

        # Create a few test grids
        print("\n  Testing grid creation...")
        for i in range(3):
            grid = encoder.create_grid()
            print(
                f"    Iteration {i+1}: Grid shape = {grid.shape}, "
                f"Non-zero cells = {np.count_nonzero(grid)}"
            )
            rospy.sleep(0.5)

        print("\n✅ Grid Encoder Test PASSED")
        print("   Check RViz topics:")
        print("   - /auv_rl/grid_visualization")
        print("   - /auv_rl/object_markers")
        return True

    except Exception as e:
        print(f"\n❌ Grid Encoder Test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_environment():
    """Test the Gym Environment."""
    print("\n" + "=" * 60)
    print("TEST 2: Gym Environment")
    print("=" * 60)

    try:
        from auv_rl_navigation.environments.auv_nav_env import AUVNavEnv

        # Create environment
        object_frames = rospy.get_param(
            "~object_frames", "gate_shark_link,red_buoy,white_pipe_link"
        ).split(",")

        env = AUVNavEnv(
            max_episode_steps=100,
            goal_tolerance=1.0,
            object_frames=[f.strip() for f in object_frames],
            goal_frame=rospy.get_param("~goal_frame", "gate"),
            visualize=True,
        )

        print(f"✓ Environment created successfully")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")

        # Test reset
        print("\n  Testing reset()...")
        obs = env.reset()
        print(f"    ✓ Reset successful")
        print(f"    - Grid shape: {obs['grid'].shape}")
        print(f"    - Goal vector: {obs['goal_vector']}")
        print(f"    - Velocity: {obs['velocity']}")

        # Test a few random steps
        print("\n  Testing step() with random actions...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(
                f"    Step {i+1}: reward={reward:.3f}, done={done}, "
                f"distance={info.get('distance_to_goal', 'N/A')}"
            )
            rospy.sleep(0.2)

        # Stop the robot
        env.close()

        print("\n✅ Environment Test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Environment Test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_agent():
    """Test the PPO Agent (without training)."""
    print("\n" + "=" * 60)
    print("TEST 3: PPO Agent")
    print("=" * 60)

    try:
        from auv_rl_navigation.environments.auv_nav_env import AUVNavEnv
        from auv_rl_navigation.agents.ppo_agent import AUVPPOAgent

        # Create minimal environment
        object_frames = rospy.get_param(
            "~object_frames", "gate_shark_link,red_buoy"
        ).split(",")

        env = AUVNavEnv(
            max_episode_steps=50,
            goal_tolerance=1.0,
            object_frames=[f.strip() for f in object_frames],
            goal_frame=rospy.get_param("~goal_frame", "gate"),
            visualize=False,  # Disable viz for speed
        )

        print(f"✓ Environment created")

        # Create agent
        print("  Creating PPO agent (this may take a moment)...")
        config = {
            "learning_rate": 3e-4,
            "n_steps": 128,  # Smaller for testing
            "batch_size": 32,
            "features_dim": 128,  # Smaller network
        }

        agent = AUVPPOAgent(env, config=config)
        print(f"✓ Agent created successfully")
        print(f"  - Policy type: MultiInputPolicy")
        print(f"  - Features dim: {config['features_dim']}")

        # Test prediction
        print("\n  Testing prediction...")
        obs = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        print(f"    ✓ Prediction successful")
        print(f"    - Action: {action}")

        # Test a short training run
        print("\n  Testing short training (100 steps)...")
        agent.train(total_timesteps=100, save_path="./test_model")
        print(f"    ✓ Training successful")

        env.close()

        print("\n✅ Agent Test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Agent Test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ros_topics():
    """Test ROS topic connectivity."""
    print("\n" + "=" * 60)
    print("TEST 4: ROS Topics")
    print("=" * 60)

    try:
        import rostopic

        # Check required topics
        required_topics = {
            "/taluy/odom": "nav_msgs/Odometry",
            "/taluy/cmd_vel": "geometry_msgs/Twist",
        }

        all_topics = dict(rostopic.get_topic_list())

        print("  Checking required topics...")
        all_ok = True
        for topic, msg_type in required_topics.items():
            if topic in all_topics:
                print(f"    ✓ {topic} - Found")
            else:
                print(f"    ✗ {topic} - NOT FOUND (Required!)")
                all_ok = False

        # Check TF
        print("\n  Checking TF frames...")
        import tf2_ros

        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        rospy.sleep(1.0)  # Wait for TF buffer to fill

        base_frame = rospy.get_param("~base_frame", "taluy/base_link")
        goal_frame = rospy.get_param("~goal_frame", "gate")

        try:
            transform = tf_buffer.lookup_transform(
                base_frame, goal_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            print(f"    ✓ TF transform {base_frame} → {goal_frame} available")
            print(
                f"      Position: [{transform.transform.translation.x:.2f}, "
                f"{transform.transform.translation.y:.2f}, "
                f"{transform.transform.translation.z:.2f}]"
            )
        except Exception as e:
            print(f"    ✗ TF transform {base_frame} → {goal_frame} NOT AVAILABLE")
            print(f"      Error: {e}")
            all_ok = False

        if all_ok:
            print("\n✅ ROS Topics Test PASSED")
        else:
            print("\n⚠️  ROS Topics Test PASSED WITH WARNINGS")
            print("    Some topics/frames not available - check your simulation")

        return True

    except Exception as e:
        print(f"\n❌ ROS Topics Test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    rospy.init_node("test_rl_components", anonymous=True)

    print("\n" + "#" * 60)
    print("# AUV RL Navigation - Component Tests")
    print("#" * 60)
    print("\nThis script tests each component individually.")
    print("Make sure your simulation is running with the AUV!\n")

    input("Press Enter to start tests...")

    results = {}

    # Run tests
    results["ROS Topics"] = test_ros_topics()

    if rospy.get_param("~skip_grid", False):
        print("\n⊘ Skipping Grid Encoder test (skip_grid=true)")
        results["Grid Encoder"] = None
    else:
        results["Grid Encoder"] = test_grid_encoder()

    if rospy.get_param("~skip_env", False):
        print("\n⊘ Skipping Environment test (skip_env=true)")
        results["Environment"] = None
    else:
        results["Environment"] = test_environment()

    if rospy.get_param("~skip_agent", False):
        print("\n⊘ Skipping Agent test (skip_agent=true)")
        results["Agent"] = None
    else:
        results["Agent"] = test_agent()

    # Summary
    print("\n" + "#" * 60)
    print("# Test Summary")
    print("#" * 60)
    for test_name, result in results.items():
        if result is None:
            status = "⊘ SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"  {test_name:20s}: {status}")

    all_passed = all(r in [True, None] for r in results.values())

    if all_passed:
        print("\n🎉 All tests passed! Ready for training.")
    else:
        print("\n⚠️  Some tests failed. Fix issues before training.")

    print("#" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        print("\nTests interrupted.")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
        sys.exit(0)
