#pragma once
#include "commons.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <limits>
#include <string>

struct KeyPoseWithCloud
{
    M3D r_local;
    V3D t_local;
    M3D r_global;
    V3D t_global;
    double time;
    CloudType::Ptr body_cloud;
};
struct LoopPair
{
    size_t source_id;
    size_t target_id;
    M3D r_offset;
    V3D t_offset;
    double score;
};

struct Config
{
    double key_pose_delta_deg = 10;
    double key_pose_delta_trans = 1.0;
    double loop_search_radius = 1.0;
    double loop_time_tresh = 60.0;
    double loop_score_tresh = 0.15;
    int loop_submap_half_range = 5;
    double submap_resolution = 0.1;
    double min_loop_detect_duration = 10.0;
    std::string loop_detector = "proximity";
    std::string loop_registration = "icp";

    struct ScanContextConfig
    {
        int rings = 20;
        int sectors = 60;
        double max_radius = 80.0;
        int candidate_count = 10;
        double distance_threshold = 0.15;
        int exclude_recent_keyframes = 30;
    } scan_context;

    struct RegistrationConfig
    {
        int max_iterations = 50;
        double max_correspondence_distance = 10.0;
        double transformation_epsilon = 1e-6;
        double euclidean_fitness_epsilon = 1e-6;
        double fitness_score_threshold = 0.15;
        double ndt_resolution = 1.0;
        double ndt_step_size = 0.1;
    } registration;
};

struct LoopCandidate
{
    int target_id = -1;
    M4F initial_guess = M4F::Identity();
    double descriptor_distance = 0.0;
};

struct RegistrationResult
{
    bool converged = false;
    M4F transformation = M4F::Identity();
    double score = std::numeric_limits<double>::infinity();
};

class LoopCandidateDetector
{
public:
    virtual ~LoopCandidateDetector() = default;
    virtual bool detect(const std::vector<KeyPoseWithCloud> &key_poses,
                        const Config &config,
                        LoopCandidate &candidate) const = 0;
};

class LoopRegistration
{
public:
    virtual ~LoopRegistration() = default;
    virtual RegistrationResult align(const CloudType::Ptr &source,
                                     const CloudType::Ptr &target,
                                     const M4F &initial_guess) = 0;
};

std::shared_ptr<LoopCandidateDetector> createLoopCandidateDetector(const Config &config);
std::shared_ptr<LoopRegistration> createLoopRegistration(const Config &config);

class SimplePGO
{
public:
    SimplePGO(const Config &config);

    bool isKeyPose(const PoseWithTime &pose);

    bool addKeyPose(const CloudWithPose &cloud_with_pose);

    bool hasLoop(){return m_cache_pairs.size() > 0;}

    void searchForLoopPairs();

    void smoothAndUpdate();

    CloudType::Ptr getSubMap(int idx, int half_range, double resolution);
    std::vector<std::pair<size_t, size_t>> &historyPairs() { return m_history_pairs; }
    std::vector<KeyPoseWithCloud> &keyPoses() { return m_key_poses; }

    M3D offsetR() { return m_r_offset; }
    V3D offsetT() { return m_t_offset; }

private:
    Config m_config;
    std::vector<KeyPoseWithCloud> m_key_poses;
    std::vector<std::pair<size_t, size_t>> m_history_pairs;
    std::vector<LoopPair> m_cache_pairs;
    M3D m_r_offset;
    V3D m_t_offset;
    std::shared_ptr<gtsam::ISAM2> m_isam2;
    gtsam::Values m_initial_values;
    gtsam::NonlinearFactorGraph m_graph;
    std::shared_ptr<LoopCandidateDetector> m_candidate_detector;
    std::shared_ptr<LoopRegistration> m_registration;
};
