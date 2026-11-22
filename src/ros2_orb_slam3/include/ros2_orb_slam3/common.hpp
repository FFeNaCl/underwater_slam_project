// Include file 
#ifndef COMMON_HPP  // Header guard to prevent multiple inclusions
#define COMMON_HPP

// C++ includes
#include <iostream> // The iostream library is an object-oriented library that provides input and output functionality using streams
#include <algorithm> // The header <algorithm> defines a collection of functions especially designed to be used on ranges of elements.
#include <fstream> // Input/output stream class to operate on files.
#include <chrono> // c++ timekeeper library
#include <vector> // vectors are sequence containers representing arrays that can change in size.
#include <queue>
#include <thread> // class to represent individual threads of execution.
#include <mutex> // A mutex is a lockable object that is designed to signal when critical sections of code need exclusive access, preventing other threads with the same protection from executing concurrently and access the same memory locations.
#include <cstdlib> // to find home directory

#include <cstring>
#include <sstream> // String stream processing functionalities
#include <memory> // std::unique_ptr, used to own the tf2_ros::TransformBroadcaster
#include <string> // std::string (explicit include, this header already stores std::string members)
#include <cstdint> // uint32_t, used by the hand written PointCloud2 layout
#include <cstddef> // std::size_t

//* ROS2 includes
//* std_msgs in ROS 2 https://docs.ros2.org/foxy/api/std_msgs/index-msg.html
#include "rclcpp/rclcpp.hpp"

// #include "your_custom_msg_interface/msg/custom_msg_field.hpp" // Example of adding in a custom message
#include <std_msgs/msg/header.hpp>
#include "std_msgs/msg/float64.hpp"
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include "sensor_msgs/msg/image.hpp"
//* Header.stamp 的字段类型（std_msgs/msg/header.hpp 已经间接包含它，这里显式写出来，
//* 避免将来依赖间接包含关系）
#include <builtin_interfaces/msg/time.hpp>

//* Message types required by the SLAM publishing interface added on top of the original node:
//* /slam/pose (PoseStamped), /slam/path (Path), /slam/map_points (PointCloud2) and the broadcast TF.
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp> // PointField::FLOAT32 constant, used to build the cloud fields
#include <tf2_ros/transform_broadcaster.h> // tf2_ros::TransformBroadcaster

using std::placeholders::_1; //* TODO why this is suggested in official tutorial

// Include Eigen
// Quick reference: https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
#include <Eigen/Dense> // Includes Core, Geometry, LU, Cholesky, SVD, QR, and Eigenvalues header file
#include <Eigen/Geometry> // Explicitly included: Eigen::Quaternionf is built from the 3x3 rotation matrix of Twc

// Include cv-bridge
#include <cv_bridge/cv_bridge.h>

// Include OpenCV computer vision library
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // Image processing tools
#include <opencv2/highgui/highgui.hpp> // GUI tools
#include <opencv2/core/eigen.hpp>
#include <image_transport/image_transport.h>

//* ORB SLAM 3 includes
#include "System.h" //* Also imports the ORB_SLAM3 namespace
//* /slam/map_points 需要对 MapPoint 调用 GetWorldPos()，因此必须拿到完整类型定义。
//* System.h 只是间接（System.h -> Tracking.h -> Atlas.h -> MapPoint.h）包含它，
//* 这里显式写出来，避免 ORB-SLAM3 内部调整 include 关系后编译失败。
#include "MapPoint.h"

//* Gobal defs
#define pass (void)0 // Python's equivalent of "pass" i.e. no operation


//* Node specific definitions
class MonocularMode : public rclcpp::Node
{   
    //* This slam node inherits from both rclcpp and ORB_SLAM3::System classes
    //* public keyword needs to come before the class constructor and anything else
    public:
    std::string experimentConfig = ""; // String to receive settings sent by the python driver
    double timeStep; // Timestep data received from the python node
    std::string receivedConfig = "";

    //* Class constructor
    MonocularMode(); // Constructor 

    ~MonocularMode(); // Destructor
        
    private:
        
        // Class internal variables
        std::string homeDir = "";
        std::string packagePath = "underwater_slam_project/src/ros2_orb_slam3/"; //! Change to match path to your workspace
        std::string OPENCV_WINDOW = ""; // Set during initialization
        std::string nodeName = ""; // Name of this node
        std::string vocFilePath = ""; // Path to ORB vocabulary provided by DBoW2 package
        std::string settingsFilePath = ""; // Path of the settings (.yaml) file actually handed to ORB_SLAM3::System
        //* 保存设置文件的「目录基准」路径（永远以 '/' 结尾）。
        //* 目的：initializeVSLAM() 每次都从该基准重新拼接 dir + name + ".yaml"，
        //* 而不是就地修改 settingsFilePath 成员 —— 后者在多次收到 experiment_settings 时会重复追加后缀。
        std::string settingsDirPath = "";
        bool bSettingsFromPython = false; // Flag set once when experiment setting from python node is received
        
        std::string subexperimentconfigName = ""; // Subscription topic name
        std::string pubconfigackName = ""; // Publisher topic name
        std::string subImgMsgName = ""; // Topic to subscribe to receive RGB images from a python node
        std::string subTimestepMsgName = ""; // Topic to subscribe to receive the timestep related to the 

        //* Definitions of publisher and subscribers
        rclcpp::Subscription<std_msgs::msg::String>::SharedPtr expConfig_subscription_;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr configAck_publisher_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subImgMsg_subscription_;
        rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr subTimestepMsg_subscription_;

        //* ORB_SLAM3 related variables
        //* 注意：pAgent 必须在构造时初始化为 nullptr。析构函数与 Img_callback() 都会先判空，
        //* 否则「还没完成握手就收到图像」或「没初始化就退出」都会解引用野指针而崩溃。
        ORB_SLAM3::System* pAgent = nullptr; // pointer to a ORB SLAM3 object
        ORB_SLAM3::System::eSensor sensorType;
        bool enablePangolinWindow = false; // Shows Pangolin window output
        bool enableOpenCVWindow = false; // Shows OpenCV window output

        //* ---------------------------------------------------------------------------------
        //* Parameters of the new SLAM publishing interface.
        //* All of them are declare_parameter()'d with a default value in the constructor, so the
        //* node stays usable without any launch-file / command-line parameter.
        //* ---------------------------------------------------------------------------------
        bool publishPose = true;    // -> /slam/pose (geometry_msgs/PoseStamped)
        bool publishPath = true;    // -> /slam/path (nav_msgs/Path)
        bool publishPoints = true;  // -> /slam/map_points (sensor_msgs/PointCloud2)
        bool publishTf = true;      // -> /tf, map_frame_id -> camera_frame_id
        bool publishStatus = true;  // -> /slam/status (std_msgs/String, JSON text)
        std::string mapFrameId = "map"; // Parent frame of the published pose/path/cloud and of the TF
        std::string cameraFrameId = "camera"; // Child frame of the broadcast TF
        int maxPathLength = 2000; // Max number of poses kept in /slam/path, <= 0 means unlimited
        int pointPublishPeriodMs = 1000; // Minimum period between two /slam/map_points messages in ms

        //* ---------------------------------------------------------------------------------
        //* Runtime bookkeeping for the publishers. Every one of these is touched only inside
        //* Img_callback() (single-threaded w.r.t. the SLAM object), never from another thread.
        //* ---------------------------------------------------------------------------------
        //* 累积的相机轨迹（Twc 形式）。nav_msgs::Path 自带 std_msgs/Header，可携带 map_frame_id 与时间戳。
        nav_msgs::msg::Path cameraPath;
        double lastImageTimestampSec = -1.0; // 上一帧图像的绝对时间戳（秒），用于估算 fps
        double currentFps = 0.0; // fps = 1 / (相邻图像时间戳之差)
        int lastTrackingStateCode = -1; // 上一次的 GetTrackingState() 返回值，仅用于日志

        //* 点云发布限频。使用 steady_clock（单调时钟）而不是 rclcpp::Time，
        //* 这样完全不涉及 ROS 时钟类型（RCL_ROS_TIME / RCL_SYSTEM_TIME）的运算，避免运行期抛异常。
        std::chrono::steady_clock::time_point lastPointPublishTime;
        bool bHasLastPointPublishTime = false; // false 表示还没发布过点云，下一帧立即发布
        //* 「SLAM 尚未初始化」警告的限频状态
        std::chrono::steady_clock::time_point lastNotInitializedWarnTime;
        bool bWarnedNotInitialized = false;

        //* Publishers of the new interface. They are always created so that the topics are
        //* visible in `ros2 topic list`; the publish_* parameters gate the actual publishing.
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mapPoints_publisher_;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;

        //* TF broadcaster (map_frame_id -> camera_frame_id). Owned by a std::unique_ptr and
        //* created in the constructor with `*this`, which is the documented tf2_ros usage.
        std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

        //* ROS callbacks
        void experimentSetting_callback(const std_msgs::msg::String& msg); // Callback to process settings sent over by Python node
        void Timestep_callback(const std_msgs::msg::Float64& time_msg); // Callback to process the timestep for this image
        void Img_callback(const sensor_msgs::msg::Image& msg); // Callback to process RGB image and semantic matrix sent by Python node
        
        //* Helper functions
        // ORB_SLAM3::eigenMatXf convertToEigenMat(const std_msgs::msg::Float32MultiArray& msg); // Helper method, converts semantic matrix eigenMatXf, a Eigen 4x4 float matrix
        void initializeVSLAM(std::string& configString); //* Method to bind an initialized VSLAM framework to this node


};

#endif