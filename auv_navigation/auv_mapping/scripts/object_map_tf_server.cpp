#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <auv_msgs/SetObjectTransform.h>
#include <boost/thread/mutex.hpp>
#include <unordered_map>

class ObjectMapTFServer {
public:
    ObjectMapTFServer() : nh_("~"), rate_(10.0) {
        // Load parameters
        nh_.param<std::string>("static_frame", static_frame_, "odom");
        nh_.param<double>("rate", rate_, 10.0);

        // Initialize the service
        service_ = nh_.advertiseService("set_object_transform", &ObjectMapTFServer::handleSetTransform, this);

        // Initialize the transform listener and broadcaster
        tf_listener_ = std::make_shared<tf::TransformListener>();
        ROS_INFO("ObjectMapTFServer initialized. Static frame: %s", static_frame_.c_str());
    }

    bool handleSetTransform(auv_msgs::SetObjectTransform::Request &req,
                            auv_msgs::SetObjectTransform::Response &res) {
        std::string parent_frame = req.transform.header.frame_id;
        std::string target_frame = req.transform.child_frame_id;

        try {
            // Wait for the transform from static_frame to parent_frame
            tf_listener_->waitForTransform(static_frame_, parent_frame, ros::Time(0), ros::Duration(4.0));
            tf::StampedTransform tf_transform;
            tf_listener_->lookupTransform(static_frame_, parent_frame, ros::Time(0), tf_transform);

            // Convert the provided transform to the static frame
            geometry_msgs::TransformStamped static_transform;
            static_transform.header.stamp = ros::Time::now();
            static_transform.header.frame_id = static_frame_;
            static_transform.child_frame_id = target_frame;

            // Parent to target transformation matrix
            tf::Matrix3x3 parent_to_target(
                tf::Quaternion(req.transform.transform.rotation.x, 
                               req.transform.transform.rotation.y, 
                               req.transform.transform.rotation.z, 
                               req.transform.transform.rotation.w)
            );
            tf::Vector3 parent_to_target_translation(
                req.transform.transform.translation.x,
                req.transform.transform.translation.y,
                req.transform.transform.translation.z
            );

            // Static frame to parent transformation matrix
            tf::Matrix3x3 static_to_parent(tf_transform.getRotation());
            tf::Vector3 static_to_parent_translation = tf_transform.getOrigin();

            // Combined transformation: static frame to target
            tf::Matrix3x3 static_to_target = static_to_parent * parent_to_target;
            tf::Vector3 combined_translation = static_to_parent_translation + static_to_parent * parent_to_target_translation;

            // Extract translation and rotation
            static_transform.transform.translation.x = combined_translation.x();
            static_transform.transform.translation.y = combined_translation.y();
            static_transform.transform.translation.z = combined_translation.z();

            tf::Quaternion combined_rotation;
            static_to_target.getRotation(combined_rotation);
            static_transform.transform.rotation.x = combined_rotation.x();
            static_transform.transform.rotation.y = combined_rotation.y();
            static_transform.transform.rotation.z = combined_rotation.z();
            static_transform.transform.rotation.w = combined_rotation.w();

            // Store or update the transform in the map
            {
                boost::mutex::scoped_lock lock(mutex_);
                transforms_[target_frame] = static_transform;
            }

            ROS_INFO("Stored static transform for frame: %s", target_frame.c_str());
            res.success = true;
            res.message = "Stored transform for frame: " + target_frame;
            return true;
        }
        catch (tf::TransformException &ex) {
            ROS_ERROR("Error occurred while looking up transform: %s", ex.what());
            res.success = false;
            res.message = "Failed to capture transform: " + std::string(ex.what());
            return false;
        }
    }

    void publishTransforms() {
        ros::Rate rate(rate_);
        while (ros::ok()) {
            {
                boost::mutex::scoped_lock lock(mutex_);
                for (const auto &entry : transforms_) {
                    geometry_msgs::TransformStamped transform = entry.second;
                    transform.header.stamp = ros::Time::now();
                    tf_broadcaster_.sendTransform(transform);
                }
            }
            rate.sleep();
        }
    }

private:
    ros::NodeHandle nh_;
    ros::ServiceServer service_;
    std::shared_ptr<tf::TransformListener> tf_listener_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    boost::mutex mutex_;
    std::unordered_map<std::string, geometry_msgs::TransformStamped> transforms_;
    std::string static_frame_;
    double rate_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_map_tf_server");
    ObjectMapTFServer server;

    ros::AsyncSpinner spinner(2);
    spinner.start();

    server.publishTransforms();
    ros::waitForShutdown();

    return 0;
}
