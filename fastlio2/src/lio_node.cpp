#include <mutex>
#include <vector>
#include <queue>
#include <memory>
#include <iostream>
#include <chrono>
// #include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

#include "utils.h"
#include "map_builder/commons.h"
#include "map_builder/map_builder.h"

#include <pcl_conversions/pcl_conversions.h>
#include "tf2_ros/transform_broadcaster.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "interface/srv/set_lio_pose.hpp"
#include <yaml-cpp/yaml.h>
#include <cmath>

using namespace std::chrono_literals;
struct NodeConfig
{
    std::string imu_topic = "/livox/imu";
    std::string lidar_topic = "/livox/lidar";
    std::string body_frame = "body";
    std::string world_frame = "lidar";
    bool print_time_cost = false;
    bool print_pose = false;
};
struct StateData
{
    bool lidar_pushed = false;
    std::mutex imu_mutex;
    std::mutex lidar_mutex;
    double last_lidar_time = -1.0;
    double last_imu_time = -1.0;
    std::deque<IMUData> imu_buffer;
    std::deque<std::pair<double, pcl::PointCloud<pcl::PointXYZINormal>::Ptr>> lidar_buffer;
    nav_msgs::msg::Path path;
};
struct OutputPose
{
    V3D position;
    Eigen::Quaterniond orientation;
};

class LIONode : public rclcpp::Node
{
public:
    LIONode() : Node("lio_node")
    {
        RCLCPP_INFO(this->get_logger(), "LIO Node Started");
        loadParameters();

        m_imu_sub = this->create_subscription<sensor_msgs::msg::Imu>(m_node_config.imu_topic, 10, std::bind(&LIONode::imuCB, this, std::placeholders::_1));
        m_lidar_sub = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(m_node_config.lidar_topic, 10, std::bind(&LIONode::lidarCB, this, std::placeholders::_1));

        m_body_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("body_cloud", 10000);
        m_world_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("world_cloud", 10000);
        m_path_pub = this->create_publisher<nav_msgs::msg::Path>("lio_path", 10000);
        m_odom_pub = this->create_publisher<nav_msgs::msg::Odometry>("lio_odom", 10000);
        m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

        m_state_data.path.poses.clear();
        m_state_data.path.header.frame_id = m_node_config.world_frame;

        m_kf = std::make_shared<IESKF>();
        m_builder = std::make_shared<MapBuilder>(m_builder_config, m_kf);
        m_set_pose_srv = this->create_service<interface::srv::SetLioPose>(
            "set_pose", std::bind(&LIONode::setPoseCB, this,
                                  std::placeholders::_1, std::placeholders::_2));
        m_timer = this->create_wall_timer(20ms, std::bind(&LIONode::timerCB, this));
    }

    void loadParameters()
    {
        this->declare_parameter("config_path", "");
        std::string config_path;
        this->get_parameter<std::string>("config_path", config_path);

        YAML::Node config = YAML::LoadFile(config_path);
        if (!config)
        {
            RCLCPP_WARN(this->get_logger(), "FAIL TO LOAD YAML FILE!");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "LOAD FROM YAML CONFIG PATH: %s", config_path.c_str());

        m_node_config.imu_topic = config["imu_topic"].as<std::string>();
        m_node_config.lidar_topic = config["lidar_topic"].as<std::string>();
        m_node_config.body_frame = config["body_frame"].as<std::string>();
        m_node_config.world_frame = config["world_frame"].as<std::string>();
        m_node_config.print_time_cost = config["print_time_cost"].as<bool>();
        if (config["print_pose"])
            m_node_config.print_pose = config["print_pose"].as<bool>();

        m_builder_config.lidar_filter_num = config["lidar_filter_num"].as<int>();
        m_builder_config.lidar_min_range = config["lidar_min_range"].as<double>();
        m_builder_config.lidar_max_range = config["lidar_max_range"].as<double>();
        m_builder_config.scan_resolution = config["scan_resolution"].as<double>();
        m_builder_config.map_resolution = config["map_resolution"].as<double>();
        m_builder_config.cube_len = config["cube_len"].as<double>();
        m_builder_config.det_range = config["det_range"].as<double>();
        m_builder_config.move_thresh = config["move_thresh"].as<double>();
        m_builder_config.na = config["na"].as<double>();
        m_builder_config.ng = config["ng"].as<double>();
        m_builder_config.nba = config["nba"].as<double>();
        m_builder_config.nbg = config["nbg"].as<double>();

        m_builder_config.imu_init_num = config["imu_init_num"].as<int>();
        m_builder_config.near_search_num = config["near_search_num"].as<int>();
        m_builder_config.ieskf_max_iter = config["ieskf_max_iter"].as<int>();
        m_builder_config.gravity_align = config["gravity_align"].as<bool>();
        m_builder_config.esti_il = config["esti_il"].as<bool>();
        std::vector<double> t_il_vec = config["t_il"].as<std::vector<double>>();
        std::vector<double> r_il_vec = config["r_il"].as<std::vector<double>>();
        m_builder_config.t_il << t_il_vec[0], t_il_vec[1], t_il_vec[2];
        m_builder_config.r_il << r_il_vec[0], r_il_vec[1], r_il_vec[2], r_il_vec[3], r_il_vec[4], r_il_vec[5], r_il_vec[6], r_il_vec[7], r_il_vec[8];
        m_builder_config.lidar_cov_inv = config["lidar_cov_inv"].as<double>();

        m_builder_config.scan_matching.point_to_plane.neighbor_count = m_builder_config.near_search_num;
        if (const YAML::Node scan_matching = config["scan_matching"])
        {
            if (scan_matching["method"])
                m_builder_config.scan_matching.method = scan_matching["method"].as<std::string>();
            if (scan_matching["min_effective_points"])
                m_builder_config.scan_matching.min_effective_points = scan_matching["min_effective_points"].as<int>();

            if (const YAML::Node cfg = scan_matching["point_to_plane"])
            {
                if (cfg["neighbor_count"])
                    m_builder_config.scan_matching.point_to_plane.neighbor_count = cfg["neighbor_count"].as<int>();
                if (cfg["max_neighbor_squared_distance"])
                    m_builder_config.scan_matching.point_to_plane.max_neighbor_squared_distance = cfg["max_neighbor_squared_distance"].as<double>();
                if (cfg["plane_fit_threshold"])
                    m_builder_config.scan_matching.point_to_plane.plane_fit_threshold = cfg["plane_fit_threshold"].as<double>();
                if (cfg["score_threshold"])
                    m_builder_config.scan_matching.point_to_plane.score_threshold = cfg["score_threshold"].as<double>();
            }
            if (const YAML::Node cfg = scan_matching["point_to_point"])
            {
                if (cfg["max_correspondence_distance"])
                    m_builder_config.scan_matching.point_to_point.max_correspondence_distance = cfg["max_correspondence_distance"].as<double>();
                if (cfg["min_effective_points"])
                    m_builder_config.scan_matching.point_to_point.min_effective_points = cfg["min_effective_points"].as<int>();
            }
            if (const YAML::Node cfg = scan_matching["ndt"])
            {
                if (cfg["resolution"])
                    m_builder_config.scan_matching.ndt.resolution = cfg["resolution"].as<double>();
                if (cfg["min_points_per_voxel"])
                    m_builder_config.scan_matching.ndt.min_points_per_voxel = cfg["min_points_per_voxel"].as<int>();
                if (cfg["covariance_regularization"])
                    m_builder_config.scan_matching.ndt.covariance_regularization = cfg["covariance_regularization"].as<double>();
                if (cfg["max_mahalanobis_distance"])
                    m_builder_config.scan_matching.ndt.max_mahalanobis_distance = cfg["max_mahalanobis_distance"].as<double>();
                if (cfg["min_effective_points"])
                    m_builder_config.scan_matching.ndt.min_effective_points = cfg["min_effective_points"].as<int>();
            }
            if (const YAML::Node cfg = scan_matching["loam"])
            {
                if (cfg["neighbor_count"])
                    m_builder_config.scan_matching.loam.neighbor_count = cfg["neighbor_count"].as<int>();
                if (cfg["max_neighbor_squared_distance"])
                    m_builder_config.scan_matching.loam.max_neighbor_squared_distance = cfg["max_neighbor_squared_distance"].as<double>();
                if (cfg["line_eigenvalue_ratio"])
                    m_builder_config.scan_matching.loam.line_eigenvalue_ratio = cfg["line_eigenvalue_ratio"].as<double>();
                if (cfg["plane_eigenvalue_ratio"])
                    m_builder_config.scan_matching.loam.plane_eigenvalue_ratio = cfg["plane_eigenvalue_ratio"].as<double>();
                if (cfg["min_effective_points"])
                    m_builder_config.scan_matching.loam.min_effective_points = cfg["min_effective_points"].as<int>();
            }
        }
    }

    void imuCB(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(m_state_data.imu_mutex);
        double timestamp = Utils::getSec(msg->header);
        if (timestamp < m_state_data.last_imu_time)
        {
            RCLCPP_WARN(this->get_logger(), "IMU Message is out of order");
            std::deque<IMUData>().swap(m_state_data.imu_buffer);
        }
        m_state_data.imu_buffer.emplace_back(V3D(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z) * 10.0,
                                             V3D(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                                             timestamp);
        m_state_data.last_imu_time = timestamp;
    }
    void lidarCB(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
    {
        CloudType::Ptr cloud = Utils::livox2PCL(msg, m_builder_config.lidar_filter_num, m_builder_config.lidar_min_range, m_builder_config.lidar_max_range);
        std::lock_guard<std::mutex> lock(m_state_data.lidar_mutex);
        double timestamp = Utils::getSec(msg->header);
        if (timestamp < m_state_data.last_lidar_time)
        {
            RCLCPP_WARN(this->get_logger(), "Lidar Message is out of order");
            std::deque<std::pair<double, pcl::PointCloud<pcl::PointXYZINormal>::Ptr>>().swap(m_state_data.lidar_buffer);
        }
        m_state_data.lidar_buffer.emplace_back(timestamp, cloud);
        m_state_data.last_lidar_time = timestamp;
    }

    bool syncPackage()
    {
        if (m_state_data.imu_buffer.empty() || m_state_data.lidar_buffer.empty())
            return false;
        if (!m_state_data.lidar_pushed)
        {
            m_package.cloud = m_state_data.lidar_buffer.front().second;
            std::sort(m_package.cloud->points.begin(), m_package.cloud->points.end(), [](PointType &p1, PointType &p2)
                      { return p1.curvature < p2.curvature; });
            m_package.cloud_start_time = m_state_data.lidar_buffer.front().first;
            m_package.cloud_end_time = m_package.cloud_start_time + m_package.cloud->points.back().curvature / 1000.0;
            m_state_data.lidar_pushed = true;
        }
        if (m_state_data.last_imu_time < m_package.cloud_end_time)
            return false;

        Vec<IMUData>().swap(m_package.imus);
        while (!m_state_data.imu_buffer.empty() && m_state_data.imu_buffer.front().time < m_package.cloud_end_time)
        {
            m_package.imus.emplace_back(m_state_data.imu_buffer.front());
            m_state_data.imu_buffer.pop_front();
        }
        m_state_data.lidar_buffer.pop_front();
        m_state_data.lidar_pushed = false;
        return true;
    }

    void publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub, CloudType::Ptr cloud, std::string frame_id, const double &time)
    {
        if (pub->get_subscription_count() <= 0)
            return;
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.frame_id = frame_id;
        cloud_msg.header.stamp = Utils::getTime(time);
        pub->publish(cloud_msg);
    }

    void setPoseCB(const std::shared_ptr<interface::srv::SetLioPose::Request> request,
                   std::shared_ptr<interface::srv::SetLioPose::Response> response)
    {
        if (!std::isfinite(request->x) || !std::isfinite(request->y) ||
            !std::isfinite(request->yaw_deg))
        {
            response->success = false;
            response->message = "x, y and yaw_deg must be finite";
            return;
        }
        std::lock_guard<std::mutex> lock(m_pose_mutex);
        if (!m_has_raw_pose || !m_last_raw_pose.position.allFinite() ||
            !m_last_raw_pose.orientation.coeffs().allFinite() ||
            m_last_raw_pose.orientation.squaredNorm() <= 0.0)
        {
            response->success = false;
            response->message = "No valid mapping pose available yet";
            return;
        }

        constexpr double kRadiansPerDegree = 3.14159265358979323846 / 180.0;
        const double target_yaw = std::remainder(request->yaw_deg, 360.0) * kRadiansPerDegree;
        const auto &q = m_last_raw_pose.orientation;
        const double raw_yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                                          1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
        const double yaw_delta = std::remainder(target_yaw - raw_yaw, 2.0 * 3.14159265358979323846);
        const Eigen::Quaterniond correction(Eigen::AngleAxisd(yaw_delta, V3D::UnitZ()));
        V3D target(request->x, request->y, m_last_raw_pose.position.z());
        V3D translation = target - correction * m_last_raw_pose.position;
        if (!translation.allFinite())
        {
            response->success = false;
            response->message = "Requested pose produces an invalid offset";
            return;
        }
        m_output_rotation = correction;
        m_output_translation = translation;
        // Earlier path poses belong to the old output coordinate convention.
        m_state_data.path.poses.clear();
        response->success = true;
        response->message = "Current pose set; next mapping update will publish corrected coordinates";
        RCLCPP_INFO(this->get_logger(), "Set current output pose to x=%.3f m, y=%.3f m, yaw=%.3f deg",
                    request->x, request->y, request->yaw_deg);
    }

    OutputPose correctedPose()
    {
        const OutputPose raw{m_kf->x().t_wi, Eigen::Quaterniond(m_kf->x().r_wi)};
        std::lock_guard<std::mutex> lock(m_pose_mutex);
        m_last_raw_pose = raw;
        m_has_raw_pose = true;
        return {m_output_rotation * raw.position + m_output_translation,
                m_output_rotation * raw.orientation};
    }

    void publishOdometry(rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub, std::string frame_id, std::string child_frame, const double &time, const OutputPose &output_pose)
    {
        if (odom_pub->get_subscription_count() <= 0)
            return;
        nav_msgs::msg::Odometry odom;
        odom.header.frame_id = frame_id;
        odom.header.stamp = Utils::getTime(time);
        odom.child_frame_id = child_frame;
        odom.pose.pose.position.x = output_pose.position.x();
        odom.pose.pose.position.y = output_pose.position.y();
        odom.pose.pose.position.z = output_pose.position.z();
        const auto &q = output_pose.orientation;
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();

        V3D vel = m_kf->x().r_wi.transpose() * m_kf->x().v;
        odom.twist.twist.linear.x = vel.x();
        odom.twist.twist.linear.y = vel.y();
        odom.twist.twist.linear.z = vel.z();
        odom_pub->publish(odom);
    }

    void publishPath(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub, std::string frame_id, const double &time, const OutputPose &output_pose)
    {
        if (path_pub->get_subscription_count() <= 0)
            return;
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = frame_id;
        pose.header.stamp = Utils::getTime(time);
        pose.pose.position.x = output_pose.position.x();
        pose.pose.position.y = output_pose.position.y();
        pose.pose.position.z = output_pose.position.z();
        const auto &q = output_pose.orientation;
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        m_state_data.path.poses.push_back(pose);
        path_pub->publish(m_state_data.path);
    }

    void broadCastTF(std::shared_ptr<tf2_ros::TransformBroadcaster> broad_caster, std::string frame_id, std::string child_frame, const double &time, const OutputPose &output_pose)
    {
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.frame_id = frame_id;
        transformStamped.child_frame_id = child_frame;
        transformStamped.header.stamp = Utils::getTime(time);
        const auto &q = output_pose.orientation;
        const auto &t = output_pose.position;
        transformStamped.transform.translation.x = t.x();
        transformStamped.transform.translation.y = t.y();
        transformStamped.transform.translation.z = t.z();
        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        transformStamped.transform.rotation.w = q.w();
        broad_caster->sendTransform(transformStamped);
        if (m_node_config.print_pose)
        {
            const double yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                                          1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Pose %s->%s: x=%.3f m, y=%.3f m, yaw=%.3f rad",
                                 frame_id.c_str(), child_frame.c_str(), t.x(), t.y(), yaw);
        }
    }

    void timerCB()
    {
        if (!syncPackage())
            return;
        auto t1 = std::chrono::high_resolution_clock::now();
        m_builder->process(m_package);
        auto t2 = std::chrono::high_resolution_clock::now();

        if (m_node_config.print_time_cost)
        {
            auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
            RCLCPP_WARN(this->get_logger(), "Time cost: %.2f ms", time_used);
        }

        if (m_builder->status() != BuilderStatus::MAPPING)
            return;

        const OutputPose output_pose = correctedPose();
        broadCastTF(m_tf_broadcaster, m_node_config.world_frame, m_node_config.body_frame, m_package.cloud_end_time, output_pose);

        publishOdometry(m_odom_pub, m_node_config.world_frame, m_node_config.body_frame, m_package.cloud_end_time, output_pose);

        CloudType::Ptr body_cloud = m_builder->lidar_processor()->transformCloud(m_package.cloud, m_kf->x().r_il, m_kf->x().t_il);

        publishCloud(m_body_cloud_pub, body_cloud, m_node_config.body_frame, m_package.cloud_end_time);

        CloudType::Ptr world_cloud = m_builder->lidar_processor()->transformCloud(m_package.cloud, m_builder->lidar_processor()->r_wl(), m_builder->lidar_processor()->t_wl());

        publishCloud(m_world_cloud_pub, world_cloud, m_node_config.world_frame, m_package.cloud_end_time);

        publishPath(m_path_pub, m_node_config.world_frame, m_package.cloud_end_time, output_pose);
    }

private:
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr m_lidar_sub;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr m_imu_sub;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_body_cloud_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_world_cloud_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr m_path_pub;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr m_odom_pub;

    rclcpp::TimerBase::SharedPtr m_timer;
    StateData m_state_data;
    SyncPackage m_package;
    NodeConfig m_node_config;
    Config m_builder_config;
    std::shared_ptr<IESKF> m_kf;
    std::shared_ptr<MapBuilder> m_builder;
    std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;
    rclcpp::Service<interface::srv::SetLioPose>::SharedPtr m_set_pose_srv;
    std::mutex m_pose_mutex;
    bool m_has_raw_pose = false;
    OutputPose m_last_raw_pose{V3D::Zero(), Eigen::Quaterniond::Identity()};
    Eigen::Quaterniond m_output_rotation = Eigen::Quaterniond::Identity();
    V3D m_output_translation = V3D::Zero();
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LIONode>());
    rclcpp::shutdown();
    return 0;
}
