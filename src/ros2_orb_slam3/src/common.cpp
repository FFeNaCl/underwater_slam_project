/*

A bare-bones example node demonstrating the use of the Monocular mode in ORB-SLAM3

Author: Azmyin Md. Kamal
Date: 01/01/24

REQUIREMENTS
* Make sure to set path to your workspace in common.hpp file

*/

//* Includes
#include "ros2_orb_slam3/common.hpp"

//* ---------------------------------------------------------------------------------------------
//* File local helpers (internal linkage: `static`, so they cannot clash with other translation
//* units of this package). Both are deliberately kept dependency free.
//* ---------------------------------------------------------------------------------------------

/**
 * @brief 把 ORB-SLAM3 的 eTrackingState 数值映射成可读字符串，用于 /slam/status 的 JSON。
 *
 * 这里刻意只对「整数取值」做 switch，不直接引用 ORB_SLAM3::Tracking::eTrackingState 枚举：
 * 枚举定义在 orb_slam3/include/Tracking.h 的 Tracking 类内，直接引用会把本文件与
 * ORB-SLAM3 内部实现的可见性绑死。数值本身是 API 契约的一部分（-1..5），不会变。
 */
static std::string trackingStateToString(int state)
{
    switch (state)
    {
        case -1: return "SYSTEM_NOT_READY";
        case 0:  return "NO_IMAGES_YET";
        case 1:  return "NOT_INITIALIZED";
        case 2:  return "OK";
        case 3:  return "RECENTLY_LOST";
        case 4:  return "LOST";
        case 5:  return "OK_KLT";
        default: return "UNKNOWN";
    }
}

/**
 * @brief 把 Sophus::SE3f 转换为 geometry_msgs::msg::Pose。
 *
 * !!! 调用方必须传入 Twc（相机 -> 世界，即 Tcw.inverse()），不要把 TrackMonocular() 直接
 * !!! 返回的 Tcw 传进来，否则发布的位姿方向正好相反。
 *
 * 关键点：
 *  - Sophus::SE3f::translation() 返回 Eigen::Vector3f，直接展开到 position。
 *  - Sophus::SE3f::rotationMatrix() 返回 Eigen::Matrix3f，用 Eigen::Quaternionf 的
 *    「由旋转矩阵构造」构造函数得到四元数；Eigen 的访问器是 q.x()/q.y()/q.z()/q.w()，
 *    与 geometry_msgs 四元数的成员顺序 x, y, z, w 一一对应，可直接赋值（无需交换分量）。
 */
static geometry_msgs::msg::Pose se3ToPoseMsg(const Sophus::SE3f& T)
{
    geometry_msgs::msg::Pose pose;

    const Eigen::Vector3f translation = T.translation();
    pose.position.x = static_cast<double>(translation.x());
    pose.position.y = static_cast<double>(translation.y());
    pose.position.z = static_cast<double>(translation.z());

    const Eigen::Matrix3f rotation = T.rotationMatrix();
    Eigen::Quaternionf quaternion(rotation); // 旋转矩阵 -> 四元数
    quaternion.normalize();                  // 数值安全：确保单位四元数，RViz/TF 对未归一化的四元数很敏感

    pose.orientation.x = static_cast<double>(quaternion.x());
    pose.orientation.y = static_cast<double>(quaternion.y());
    pose.orientation.z = static_cast<double>(quaternion.z());
    pose.orientation.w = static_cast<double>(quaternion.w());

    return pose;
}

//* Constructor
MonocularMode::MonocularMode() :Node("mono_node_cpp")
{
    // Declare parameters to be passsed from command line
    // https://roboticsbackend.com/rclcpp-params-tutorial-get-set-ros2-params-with-cpp/
    
    //* Find path to home directory
    homeDir = getenv("HOME");
    // std::cout<<"Home: "<<homeDir<<std::endl;
    
    // std::cout<<"VLSAM NODE STARTED\n\n";
    RCLCPP_INFO(this->get_logger(), "\nORB-SLAM3-V1 NODE STARTED");

    this->declare_parameter("node_name_arg", "not_given"); // Name of this agent 
    this->declare_parameter("voc_file_arg", "file_not_set"); // Needs to be overriden with appropriate name  
    this->declare_parameter("settings_file_path_arg", "file_path_not_set"); // path to settings file  

    //* 新增参数：上面三个原有参数保持不变（不能改名，否则会破坏 Python driver 的握手）
    //* 重要：所有参数都必须在构造时带默认值声明，这样不带任何 launch/命令行参数也能跑起来。
    //* enable_pangolin / enable_opencv_window 默认 false —— 见 initializeVSLAM() 里的说明：
    //* 无显示器（纯 SSH / CI / Docker）时 Pangolin 会初始化失败，默认关闭才能保证节点能起来。
    this->declare_parameter("enable_pangolin", false);
    this->declare_parameter("enable_opencv_window", false);
    this->declare_parameter("publish_pose", true);
    this->declare_parameter("publish_path", true);
    this->declare_parameter("publish_points", true);
    this->declare_parameter("publish_tf", true);
    this->declare_parameter("publish_status", true);
    this->declare_parameter<std::string>("map_frame_id", "map");
    this->declare_parameter<std::string>("camera_frame_id", "camera");
    this->declare_parameter("max_path_length", 2000);
    this->declare_parameter("point_publish_period_ms", 1000);
    
    //* Watchdog, populate default values
    nodeName = "not_set";
    vocFilePath = "file_not_set";
    settingsFilePath = "file_not_set";

    //* Populate parameter values
    rclcpp::Parameter param1 = this->get_parameter("node_name_arg");
    nodeName = param1.as_string();
    
    rclcpp::Parameter param2 = this->get_parameter("voc_file_arg");
    vocFilePath = param2.as_string();

    rclcpp::Parameter param3 = this->get_parameter("settings_file_path_arg");
    settingsFilePath = param3.as_string();

    // rclcpp::Parameter param4 = this->get_parameter("settings_file_name_arg");

    //* 读取新增参数（全部在构造函数里给出默认值，因此这里一定取得到）
    rclcpp::Parameter param4 = this->get_parameter("enable_pangolin");
    enablePangolinWindow = param4.as_bool();

    rclcpp::Parameter param5 = this->get_parameter("enable_opencv_window");
    enableOpenCVWindow = param5.as_bool();

    rclcpp::Parameter param6 = this->get_parameter("publish_pose");
    publishPose = param6.as_bool();

    rclcpp::Parameter param7 = this->get_parameter("publish_path");
    publishPath = param7.as_bool();

    rclcpp::Parameter param8 = this->get_parameter("publish_points");
    publishPoints = param8.as_bool();

    rclcpp::Parameter param9 = this->get_parameter("publish_tf");
    publishTf = param9.as_bool();

    rclcpp::Parameter param10 = this->get_parameter("publish_status");
    publishStatus = param10.as_bool();

    rclcpp::Parameter param11 = this->get_parameter("map_frame_id");
    mapFrameId = param11.as_string();

    rclcpp::Parameter param12 = this->get_parameter("camera_frame_id");
    cameraFrameId = param12.as_string();

    rclcpp::Parameter param13 = this->get_parameter("max_path_length");
    maxPathLength = static_cast<int>(param13.as_int());

    rclcpp::Parameter param14 = this->get_parameter("point_publish_period_ms");
    pointPublishPeriodMs = static_cast<int>(param14.as_int());
    
  
    //* HARDCODED, set paths
    if (vocFilePath == "file_not_set" || settingsFilePath == "file_not_set")
    {
        pass;
        vocFilePath = homeDir + "/" + packagePath + "orb_slam3/Vocabulary/ORBvoc.txt.bin";
        settingsFilePath = homeDir + "/" + packagePath + "orb_slam3/config/Monocular/";
    }

    //* 保存 settings 文件的「目录基准」，供 initializeVSLAM() 幂等地拼接使用。
    //* settings_file_path_arg 的语义是「目录」（例如 .../orb_slam3/config/Monocular/）；
    //* 这里统一补一个结尾的 '/'，之后每次初始化都用 settingsDirPath + 文件名 + ".yaml" 重新构造，
    //* 这样重复收到 /mono_py_driver/experiment_settings 也不会像原实现那样重复追加后缀。
    settingsDirPath = settingsFilePath;
    if (!settingsDirPath.empty() && settingsDirPath.back() != '/')
    {
        settingsDirPath += "/";
    }

    // std::cout<<"vocFilePath: "<<vocFilePath<<std::endl;
    // std::cout<<"settingsFilePath: "<<settingsFilePath<<std::endl;
    
    
    //* DEBUG print
    RCLCPP_INFO(this->get_logger(), "nodeName %s", nodeName.c_str());
    RCLCPP_INFO(this->get_logger(), "voc_file %s", vocFilePath.c_str());
    // RCLCPP_INFO(this->get_logger(), "settings_file_path %s", settingsFilePath.c_str());
    RCLCPP_INFO(this->get_logger(), "settings dir %s", settingsDirPath.c_str());
    
    subexperimentconfigName = "/mono_py_driver/experiment_settings"; // topic that sends out some configuration parameters to the cpp ndoe
    pubconfigackName = "/mono_py_driver/exp_settings_ack"; // send an acknowledgement to the python node
    subImgMsgName = "/mono_py_driver/img_msg"; // topic to receive RGB image messages
    subTimestepMsgName = "/mono_py_driver/timestep_msg"; // topic to receive RGB image messages

    //* subscribe to python node to receive settings
    expConfig_subscription_ = this->create_subscription<std_msgs::msg::String>(subexperimentconfigName, 1, std::bind(&MonocularMode::experimentSetting_callback, this, _1));

    //* publisher to send out acknowledgement
    configAck_publisher_ = this->create_publisher<std_msgs::msg::String>(pubconfigackName, 10);

    //* subscrbite to the image messages coming from the Python driver node
    subImgMsg_subscription_= this->create_subscription<sensor_msgs::msg::Image>(subImgMsgName, 1, std::bind(&MonocularMode::Img_callback, this, _1));

    //* subscribe to receive the timestep
    subTimestepMsg_subscription_= this->create_subscription<std_msgs::msg::Float64>(subTimestepMsgName, 1, std::bind(&MonocularMode::Timestep_callback, this, _1));

    //* ---------------------------------------------------------------------------------
    //* 新增的 SLAM 发布接口
    //* 话题名是上层 Python 包的接口契约，必须一字不差：
    //*   /slam/pose        geometry_msgs/PoseStamped  相机在世界坐标系中的位姿 Twc
    //*   /slam/path        nav_msgs/Path              累积轨迹（同样是 Twc）
    //*   /slam/map_points  sensor_msgs/PointCloud2    地图点（x,y,z 均为 FLOAT32）
    //*   /slam/status      std_msgs/String            JSON 文本状态
    //*   /tf               map_frame_id -> camera_frame_id
    //* publisher 一律创建（这样话题在 ros2 topic list 里始终可见）；
    //* 是否真的发布由 publish_* 参数在 Img_callback() 里逐条控制。
    //* ---------------------------------------------------------------------------------
    pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/slam/pose", 10);
    path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/slam/path", 10);
    mapPoints_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/slam/map_points", 10);
    status_publisher_ = this->create_publisher<std_msgs::msg::String>("/slam/status", 10);

    //* TF 广播器，广播 map_frame_id -> camera_frame_id。
    //* 传 *this 是 tf2_ros 在 Humble 上的正式用法（构造函数接受节点引用），
    //* 用它比 auto broadcaster = ... 更安全：不需要 shared_from_this()，
    //* 因而不依赖「节点是否已经由 shared_ptr 持有」这一前提。
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    //* 累积轨迹的坐标系在初始化时就固定下来
    cameraPath.header.frame_id = mapFrameId;

    RCLCPP_INFO(this->get_logger(),
                "SLAM publishers: pose=%d path=%d points=%d tf=%d status=%d, frames %s -> %s",
                static_cast<int>(publishPose), static_cast<int>(publishPath),
                static_cast<int>(publishPoints), static_cast<int>(publishTf),
                static_cast<int>(publishStatus), mapFrameId.c_str(), cameraFrameId.c_str());

    RCLCPP_INFO(this->get_logger(), "Waiting to finish handshake ......");
    
}

//* Destructor
MonocularMode::~MonocularMode()
{   
    
    // Stop all threads
    // Call method to write the trajectory file
    // Release resources and cleanly shutdown
    //* 关键修复：只有 SLAM 真的被初始化过（即 pAgent != nullptr）才调用 Shutdown()。
    //* 原实现无条件调用 pAgent->Shutdown()，如果节点在收到 experiment_settings 之前就退出
    //* （Ctrl-C、driver 没起来、参数给错），pAgent 还是未初始化的野指针，会直接段错误。
    //* 另外这里刻意不 delete pAgent：ORB_SLAM3::System 没有自定义析构函数，
    //* 其内部线程的收尾工作已经全部由 Shutdown() 完成，delete 只会引入新的风险。
    if (pAgent != nullptr)
    {
        pAgent->Shutdown();
        pAgent = nullptr; // 防止重复 Shutdown
    }
    pass;

}

//* Callback which accepts experiment parameters from the Python node
void MonocularMode::experimentSetting_callback(const std_msgs::msg::String& msg){
    
    // std::cout<<"experimentSetting_callback"<<std::endl;
    bSettingsFromPython = true;
    experimentConfig = msg.data.c_str();
    // receivedConfig = experimentConfig; // Redundant
    
    RCLCPP_INFO(this->get_logger(), "Configuration YAML file name: %s", this->receivedConfig.c_str());

    //* Publish acknowledgement
    auto message = std_msgs::msg::String();
    message.data = "ACK";
    
    std::cout<<"Sent response: "<<message.data.c_str()<<std::endl;
    configAck_publisher_->publish(message);

    //* Wait to complete VSLAM initialization
    initializeVSLAM(experimentConfig);

}

//* Method to bind an initialized VSLAM framework to this node
void MonocularMode::initializeVSLAM(std::string& configString){
    
    // Watchdog, if the paths to vocabular and settings files are still not set
    if (vocFilePath == "file_not_set" || settingsFilePath == "file_not_set")
    {
        RCLCPP_ERROR(get_logger(), "Please provide valid voc_file and settings_file paths");       
        rclcpp::shutdown();
        return; // rclcpp::shutdown() 只是发出关闭请求，不会中断本函数；必须显式 return，
                // 否则接下来还会用无效路径去构造 ORB_SLAM3::System
    } 
    
    //* Build .yaml`s file path
    
    //* 幂等修复：每次都从保存好的目录基准 settingsDirPath 重新拼接「目录 + 文件名 + ".yaml"」。
    //* 绝对不要写 settingsFilePath.append(...) —— 那是就地修改成员，
    //* 第二次收到 experiment_settings 时会变成 dir/name.yamlname.yaml。
    //* 例：ros2_ws/src/orb_slam3_ros2/orb_slam3/config/Monocular/TUM2.yaml
    settingsFilePath = settingsDirPath + configString + ".yaml";

    RCLCPP_INFO(this->get_logger(), "Path to settings file: %s", settingsFilePath.c_str());
    
    // NOTE if you plan on passing other configuration parameters to ORB SLAM3 Systems class, do it here
    // NOTE you may also use a .yaml file here to set these values
    sensorType = ORB_SLAM3::System::MONOCULAR; 
    //* 关键决策：enablePangolinWindow / enableOpenCVWindow 不再硬编码为 true，
    //* 而是在构造函数里由 enable_pangolin / enable_opencv_window 参数填充（默认 false）。
    //* 原因：enablePangolinWindow=true 时 ORB_SLAM3 会创建 Viewer 线程并初始化 Pangolin/OpenGL 上下文，
    //* 在没有显示器、没有 X11/EGL（纯 SSH、CI、Docker、systemd 服务）的机器上会直接失败或崩溃。
    //* 需要可视化时显式覆盖：--ros-args -p enable_pangolin:=true
    RCLCPP_INFO(this->get_logger(), "enable_pangolin %d, enable_opencv_window %d",
                static_cast<int>(enablePangolinWindow), static_cast<int>(enableOpenCVWindow));

    //* 重复握手（第二次收到 experiment_settings）时，先安全关闭旧实例，避免线程泄漏。
    //* 正常流程下这里 pAgent 一定是 nullptr，不会走到。
    if (pAgent != nullptr)
    {
        RCLCPP_WARN(this->get_logger(), "SLAM already initialized, shutting down the previous instance first");
        pAgent->Shutdown();
        pAgent = nullptr;
    }

    //* 新的一次会话：清空上一轮累积的轨迹与统计量，避免新旧轨迹混在一起
    cameraPath.poses.clear();
    cameraPath.header.frame_id = mapFrameId;
    lastImageTimestampSec = -1.0;
    currentFps = 0.0;
    lastTrackingStateCode = -1;
    bHasLastPointPublishTime = false;
    bWarnedNotInitialized = false;

    pAgent = new ORB_SLAM3::System(vocFilePath, settingsFilePath, sensorType, enablePangolinWindow);
    std::cout << "MonocularMode node initialized" << std::endl; // TODO needs a better message
}

//* Callback that processes timestep sent over ROS
void MonocularMode::Timestep_callback(const std_msgs::msg::Float64& time_msg){
    // timeStep = 0; // Initialize
    timeStep = time_msg.data;
}

//* Callback to process image message and run SLAM node
void MonocularMode::Img_callback(const sensor_msgs::msg::Image& msg)
{
    //* ---------------------------------------------------------------------------------------------
    //* 关键设计：所有与 ORB-SLAM3 对象（pAgent）的交互**只允许发生在本回调内**。
    //* ORB-SLAM3 不是线程安全的：TrackMonocular / GetTrackingState / GetAllMapPoints 等都会访问
    //* 内部共享状态。因此这里绝不新开线程或定时器去读 SLAM，全部发布都在跟踪完成之后同步完成；
    //* 需要降频的话题（点云）用 steady_clock 做限频，而不是换线程。
    //* ---------------------------------------------------------------------------------------------
    //* 0) 尚未初始化就收到图像：pAgent 仍为 nullptr（Python driver 可能比握手先开始发图）。
    //*    必须安全跳过并限频打印警告（5 秒一次），绝不能解引用空指针。
    //* ---------------------------------------------------------------------------------------------
    if (pAgent == nullptr)
    {
        const std::chrono::steady_clock::time_point nowSteady = std::chrono::steady_clock::now();
        const int64_t warnPeriodMs = 5000; // 警告限频周期
        if (!bWarnedNotInitialized ||
            std::chrono::duration_cast<std::chrono::milliseconds>(nowSteady - lastNotInitializedWarnTime).count() >= warnPeriodMs)
        {
            RCLCPP_WARN(this->get_logger(),
                        "SLAM is not initialized yet, dropping the image. Waiting for a settings file name on %s",
                        subexperimentconfigName.c_str());
            lastNotInitializedWarnTime = nowSteady;
            bWarnedNotInitialized = true;
        }
        return;
    }

    // Initialize
    cv_bridge::CvImagePtr cv_ptr; //* Does not create a copy, memory efficient
    
    //* Convert ROS image to openCV image
    try
    {
        //cv::Mat im =  cv_bridge::toCvShare(msg.img, msg)->image;
        cv_ptr = cv_bridge::toCvCopy(msg); // Local scope
        
        // DEBUGGING, Show image
        // Update GUI Window
        // cv::imshow("test_window", cv_ptr->image);
        // cv::waitKey(3);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(),"Error reading image");
        return;
    }

    //* 额外防御：转换失败或空图不要送进 ORB-SLAM3
    if (cv_ptr == nullptr || cv_ptr->image.empty())
    {
        RCLCPP_WARN(this->get_logger(), "Received an empty image, skipping this frame");
        return;
    }
    
    //* 时间戳：优先使用图像自带的时间戳（msg.header.stamp），为 0 时退回节点时钟。
    //* 关键修复（2026-11-19）：SLAM 的时间基准必须和图像一起到达，不能依赖另一个
    //* /mono_py_driver/timestep_msg 话题。两个话题在 DDS 里没有跨话题顺序保证，
    //* 独立话题偶尔会让 TrackMonocular() 收到比上一帧更旧的时间戳，触发
    //* "Frame with a timestamp older than previous frame detected!" 并不断重置地图。
    builtin_interfaces::msg::Time stamp = msg.header.stamp;
    if (stamp.sec == 0 && stamp.nanosec == 0)
    {
        stamp = this->now(); // rclcpp::Time -> builtin_interfaces::msg::Time 的隐式转换是 rclcpp 的公开 API
    }
    const double stampSec = static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;

    //* Perform all ORB-SLAM3 operations in Monocular mode
    //! Tcw = 世界(地图)坐标系到相机坐标系的变换：X_cam = Tcw * X_world。
    //! ORB-SLAM3 的 TrackMonocular() 返回的就是 Tcw，不是相机位姿本身。
    Sophus::SE3f Tcw = pAgent->TrackMonocular(cv_ptr->image, stampSec);

    //* 原实现在这里把 Tcw 直接丢掉了，所以 ROS 侧永远看不到轨迹/点云。现在把它用起来：
    //* Twc = Tcw.inverse() —— 相机在世界/地图坐标系中的位姿（X_world = Twc * X_cam），
    //* /slam/pose、/slam/path 与 TF 发布的都是 Twc。Tcw 与 Twc 互为逆，方向弄反是最常见的错误。
    const Sophus::SE3f Twc = Tcw.inverse();

    //* fps：用相邻两帧图像时间戳之差估算。delta <= 0（时间戳相同/回退）时保持上一次的值。
    if (lastImageTimestampSec > 0.0)
    {
        const double delta = stampSec - lastImageTimestampSec;
        if (delta > 1e-6)
        {
            currentFps = 1.0 / delta;
        }
    }
    lastImageTimestampSec = stampSec;

    //* 跟踪状态与图像缩放系数（ORB_SLAM3::System 的公开方法）
    const int trackingStateCode = pAgent->GetTrackingState();
    const float imageScale = pAgent->GetImageScale();
    if (trackingStateCode != lastTrackingStateCode)
    {
        RCLCPP_INFO(this->get_logger(), "Tracking state changed to %s (%d)",
                    trackingStateToString(trackingStateCode).c_str(), trackingStateCode);
        lastTrackingStateCode = trackingStateCode;
    }

    //* 是否需要在本帧取地图点：
    //*  - /slam/status 需要 n_map_points，所以只要 publish_status 打开就每帧都要取；
    //*  - /slam/map_points 受 point_publish_period_ms 限频。
    //* GetAllMapPoints() 内部会复制整个地图点指针数组，开销不小，因此两者共用同一次查询结果。
    bool bTimeToPublishPoints = false;
    if (publishPoints)
    {
        if (!bHasLastPointPublishTime || pointPublishPeriodMs <= 0)
        {
            // 第一次发布，或者参数 <= 0（表示不做限频，每帧都发）
            bTimeToPublishPoints = true;
        }
        else
        {
            const std::chrono::steady_clock::time_point nowSteady = std::chrono::steady_clock::now();
            const int64_t elapsedMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(nowSteady - lastPointPublishTime).count();
            bTimeToPublishPoints = (elapsedMs >= static_cast<int64_t>(pointPublishPeriodMs));
        }
    }

    std::vector<ORB_SLAM3::MapPoint*> mapPoints;
    if (publishStatus || bTimeToPublishPoints)
    {
        //* GetAllMapPoints() 可能返回空 vector（刚初始化、还没建图，或者地图被清空），
        //* 此时退回 GetTrackedMapPoints()（当前帧被跟踪上的点）。
        mapPoints = pAgent->GetAllMapPoints();
        if (mapPoints.empty())
        {
            mapPoints = pAgent->GetTrackedMapPoints();
        }
    }

    //* Twc 对应的 ROS 位姿只计算一次，/slam/pose、/slam/path 和 TF 共用同一份数值，
    //* 保证三个出口给出的位姿严格一致。
    const geometry_msgs::msg::Pose poseTwc = se3ToPoseMsg(Twc);

    //* 1) /slam/pose —— 相机在世界/地图坐标系中的位姿 Twc
    //* 注意：即使跟踪状态是 RECENTLY_LOST / LOST，ORB-SLAM3 仍然会返回「最后一次成功的位姿」，
    //* 这里照常发布，由订阅方通过 /slam/status 判断该位姿是否可信。
    if (publishPose)
    {
        geometry_msgs::msg::PoseStamped poseMsg;
        poseMsg.header.stamp = stamp;
        poseMsg.header.frame_id = mapFrameId;
        poseMsg.pose = poseTwc;
        pose_publisher_->publish(poseMsg);
    }

    //* 2) /slam/path —— 累积的相机轨迹（同样是 Twc）
    if (publishPath)
    {
        geometry_msgs::msg::PoseStamped pathPose;
        pathPose.header.stamp = stamp;
        pathPose.header.frame_id = mapFrameId;
        pathPose.pose = poseTwc;

        cameraPath.header.stamp = stamp;
        cameraPath.header.frame_id = mapFrameId;
        cameraPath.poses.push_back(pathPose);

        //* max_path_length > 0：只保留最近 max_path_length 个位姿（滑动窗口）；
        //* <= 0：不限制长度。每帧最多新增一个位姿，所以这里的 while 正常只执行一次或零次。
        while (maxPathLength > 0 && cameraPath.poses.size() > static_cast<std::size_t>(maxPathLength))
        {
            cameraPath.poses.erase(cameraPath.poses.begin());
        }

        //* 说明：Path 每次都发布整条轨迹（RViz 需要完整消息才能画线），
        //* 轨迹长度由 max_path_length 控制，默认 2000 个位姿在 30Hz 下约 1MB/s 量级，可接受。
        path_publisher_->publish(cameraPath);
    }

    //* 3) /slam/map_points —— 地图点云（PointCloud2，字段 x,y,z 全部 FLOAT32）
    if (bTimeToPublishPoints)
    {
        sensor_msgs::msg::PointCloud2 cloudMsg;
        cloudMsg.header.stamp = stamp;
        cloudMsg.header.frame_id = mapFrameId;

        //* 手工设置 PointCloud2 的内存布局（不使用 PointCloud2Modifier，减少对 API 细节的依赖）：
        //*   height = 1（无序点云）、width = 点数 N、point_step = 3 * sizeof(float) = 12、
        //*   row_step = point_step * width、is_bigendian = false（小端，与 x86/ARM 一致）、
        //*   fields = x(offset 0) / y(offset 4) / z(offset 8)，datatype = FLOAT32，count = 1。
        cloudMsg.height = 1;
        cloudMsg.width = static_cast<uint32_t>(mapPoints.size());
        cloudMsg.is_bigendian = false;
        cloudMsg.point_step = 12;
        cloudMsg.row_step = cloudMsg.point_step * cloudMsg.width;
        cloudMsg.is_dense = true; // 无 NaN；下面遇到空指针时会置为 false

        cloudMsg.fields.resize(3);
        const char* const fieldNames[3] = {"x", "y", "z"};
        for (std::size_t i = 0; i < 3; ++i)
        {
            cloudMsg.fields[i].name = fieldNames[i];
            cloudMsg.fields[i].offset = static_cast<uint32_t>(i * 4); // 每个 float 占 4 字节
            cloudMsg.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
            cloudMsg.fields[i].count = 1;
        }

        //* data 的长度必须正好是 row_step（= 12 * N）。注意 width 为 0 时长度为 0 也是合法的，
        //* 此时下面的循环不会执行，发布的是空点云（这本身是有效消息）。
        cloudMsg.data.resize(static_cast<std::size_t>(cloudMsg.row_step));

        for (std::size_t i = 0; i < mapPoints.size(); ++i)
        {
            //* GetAllMapPoints() 返回的是裸指针，理论上可能为空，必须判空
            if (mapPoints[i] == nullptr)
            {
                const float zeros[3] = {0.0f, 0.0f, 0.0f};
                std::memcpy(&cloudMsg.data[i * 12], zeros, 12);
                cloudMsg.is_dense = false;
                continue;
            }

            const Eigen::Vector3f worldPos = mapPoints[i]->GetWorldPos(); // Eigen::Vector3f
            const float xyz[3] = {worldPos.x(), worldPos.y(), worldPos.z()};
            std::memcpy(&cloudMsg.data[i * 12], xyz, 12); // 3 个 float = 12 字节，与 point_step 一致
        }

        mapPoints_publisher_->publish(cloudMsg);

        lastPointPublishTime = std::chrono::steady_clock::now();
        bHasLastPointPublishTime = true;
    }

    //* 4) /slam/status —— JSON 文本（std_msgs/String）
    //* 用 std::ostringstream 手工拼接：所有值都是数字或固定的 ASCII 单词，不含需要 JSON 转义的字符。
    if (publishStatus)
    {
        std::ostringstream statusStream;
        statusStream << "{"
                     << "\"tracking_state\":\"" << trackingStateToString(trackingStateCode) << "\","
                     << "\"tracking_state_code\":" << trackingStateCode << ","
                     << "\"n_map_points\":" << static_cast<int>(mapPoints.size()) << ","
                     << "\"image_scale\":" << imageScale << ","
                     << "\"fps\":" << currentFps
                     << "}";

        std_msgs::msg::String statusMsg;
        statusMsg.data = statusStream.str();
        status_publisher_->publish(statusMsg);
    }

    //* 5) TF —— map_frame_id -> camera_frame_id，内容同样是 Twc
    //* 语义提醒：TransformStamped 表示「把点从 child_frame_id 变换到 header.frame_id」的变换，
    //* 也就是 ^map T_camera = Twc，与 /slam/pose 完全一致。千万不要填成 Tcw。
    if (publishTf && tf_broadcaster_ != nullptr)
    {
        geometry_msgs::msg::TransformStamped tfMsg;
        tfMsg.header.stamp = stamp;
        tfMsg.header.frame_id = mapFrameId;   // parent frame
        tfMsg.child_frame_id = cameraFrameId; // child frame
        tfMsg.transform.translation.x = poseTwc.position.x;
        tfMsg.transform.translation.y = poseTwc.position.y;
        tfMsg.transform.translation.z = poseTwc.position.z;
        tfMsg.transform.rotation.x = poseTwc.orientation.x;
        tfMsg.transform.rotation.y = poseTwc.orientation.y;
        tfMsg.transform.rotation.z = poseTwc.orientation.z;
        tfMsg.transform.rotation.w = poseTwc.orientation.w;
        tf_broadcaster_->sendTransform(tfMsg);
    }
}


