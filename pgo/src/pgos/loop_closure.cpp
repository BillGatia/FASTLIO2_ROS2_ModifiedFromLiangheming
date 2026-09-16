#include "simple_pgo.h"

#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{
void requirePositive(double value, const char *name)
{
    if (!(value > 0.0) || !std::isfinite(value))
        throw std::invalid_argument(std::string(name) + " must be finite and > 0");
}

void requirePositive(int value, const char *name)
{
    if (value <= 0)
        throw std::invalid_argument(std::string(name) + " must be > 0");
}

M4F poseMatrix(const KeyPoseWithCloud &pose)
{
    M4F result = M4F::Identity();
    result.block<3, 3>(0, 0) = pose.r_global.cast<float>();
    result.block<3, 1>(0, 3) = pose.t_global.cast<float>();
    return result;
}

class ProximityDetector final : public LoopCandidateDetector
{
public:
    bool detect(const std::vector<KeyPoseWithCloud> &key_poses,
                const Config &config, LoopCandidate &candidate) const override
    {
        if (key_poses.size() < 10)
            return false;
        const auto &current = key_poses.back();
        pcl::PointCloud<pcl::PointXYZ>::Ptr pose_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pose_cloud->reserve(key_poses.size() - 1);
        for (size_t i = 0; i + 1 < key_poses.size(); ++i)
            pose_cloud->push_back(pcl::PointXYZ(static_cast<float>(key_poses[i].t_global.x()),
                                                static_cast<float>(key_poses[i].t_global.y()),
                                                static_cast<float>(key_poses[i].t_global.z())));

        pcl::KdTreeFLANN<pcl::PointXYZ> tree;
        tree.setInputCloud(pose_cloud);
        std::vector<int> indices;
        std::vector<float> squared_distances;
        const pcl::PointXYZ query(static_cast<float>(current.t_global.x()),
                                  static_cast<float>(current.t_global.y()),
                                  static_cast<float>(current.t_global.z()));
        if (tree.radiusSearch(query, config.loop_search_radius, indices, squared_distances) == 0)
            return false;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            const int index = indices[i];
            if (std::abs(current.time - key_poses[index].time) > config.loop_time_tresh)
            {
                candidate.target_id = index;
                candidate.initial_guess = M4F::Identity();
                candidate.descriptor_distance = std::sqrt(squared_distances[i]);
                return true;
            }
        }
        return false;
    }
};

using Descriptor = Eigen::MatrixXf;

Descriptor makeScanContext(const CloudType &cloud, const Config::ScanContextConfig &config)
{
    Descriptor descriptor = Descriptor::Zero(config.rings, config.sectors);
    constexpr double kTwoPi = 6.28318530717958647692;
    for (const auto &point : cloud.points)
    {
        const double radius = std::hypot(point.x, point.y);
        if (radius <= 0.0 || radius > config.max_radius)
            continue;
        double angle = std::atan2(point.y, point.x);
        if (angle < 0.0)
            angle += kTwoPi;
        const int ring = std::min(config.rings - 1,
                                  static_cast<int>(radius / config.max_radius * config.rings));
        const int sector = std::min(config.sectors - 1,
                                    static_cast<int>(angle / kTwoPi * config.sectors));
        descriptor(ring, sector) = std::max(descriptor(ring, sector), point.z + 2.0f);
    }
    return descriptor;
}

double ringKeyDistance(const Descriptor &lhs, const Descriptor &rhs)
{
    return (lhs.rowwise().mean() - rhs.rowwise().mean()).norm();
}

std::pair<double, int> scanContextDistance(const Descriptor &target, const Descriptor &source)
{
    double best_distance = std::numeric_limits<double>::infinity();
    int best_shift = 0;
    for (int shift = 0; shift < source.cols(); ++shift)
    {
        double cosine_sum = 0.0;
        int valid_columns = 0;
        for (int column = 0; column < source.cols(); ++column)
        {
            const int shifted = (column + shift) % source.cols();
            const Eigen::VectorXf target_column = target.col(column);
            const Eigen::VectorXf source_column = source.col(shifted);
            const double denominator = target_column.norm() * source_column.norm();
            if (denominator <= 1e-9)
                continue;
            cosine_sum += target_column.dot(source_column) / denominator;
            ++valid_columns;
        }
        if (valid_columns == 0)
            continue;
        const double distance = 1.0 - cosine_sum / static_cast<double>(valid_columns);
        if (distance < best_distance)
        {
            best_distance = distance;
            best_shift = shift;
        }
    }
    return {best_distance, best_shift};
}

class ScanContextDetector final : public LoopCandidateDetector
{
public:
    explicit ScanContextDetector(const Config &config)
    {
        const auto &cfg = config.scan_context;
        requirePositive(cfg.rings, "scan_context.rings");
        requirePositive(cfg.sectors, "scan_context.sectors");
        requirePositive(cfg.max_radius, "scan_context.max_radius");
        requirePositive(cfg.candidate_count, "scan_context.candidate_count");
        requirePositive(cfg.exclude_recent_keyframes, "scan_context.exclude_recent_keyframes");
        if (cfg.distance_threshold < 0.0 || !std::isfinite(cfg.distance_threshold))
            throw std::invalid_argument("scan_context.distance_threshold must be finite and >= 0");
    }

    bool detect(const std::vector<KeyPoseWithCloud> &key_poses,
                const Config &config, LoopCandidate &candidate) const override
    {
        const int current_index = static_cast<int>(key_poses.size()) - 1;
        const int max_candidate_index = current_index - config.scan_context.exclude_recent_keyframes;
        if (max_candidate_index < 0)
            return false;

        const Descriptor current = makeScanContext(*key_poses.back().body_cloud, config.scan_context);
        std::vector<std::pair<double, int>> shortlist;
        for (int index = 0; index <= max_candidate_index; ++index)
        {
            if (std::abs(key_poses.back().time - key_poses[index].time) <= config.loop_time_tresh)
                continue;
            const Descriptor descriptor = makeScanContext(*key_poses[index].body_cloud, config.scan_context);
            shortlist.emplace_back(ringKeyDistance(descriptor, current), index);
        }
        if (shortlist.empty())
            return false;

        const size_t count = std::min(shortlist.size(), static_cast<size_t>(config.scan_context.candidate_count));
        std::partial_sort(shortlist.begin(), shortlist.begin() + count, shortlist.end());
        double best_distance = std::numeric_limits<double>::infinity();
        int best_index = -1;
        int best_shift = 0;
        for (size_t i = 0; i < count; ++i)
        {
            const int index = shortlist[i].second;
            const Descriptor target = makeScanContext(*key_poses[index].body_cloud, config.scan_context);
            const auto [distance, shift] = scanContextDistance(target, current);
            if (distance < best_distance)
            {
                best_distance = distance;
                best_index = index;
                best_shift = shift;
            }
        }
        if (best_index < 0 || best_distance > config.scan_context.distance_threshold)
            return false;

        constexpr double kTwoPi = 6.28318530717958647692;
        const double yaw = -kTwoPi * static_cast<double>(best_shift) /
                           static_cast<double>(config.scan_context.sectors);
        M4F relative = M4F::Identity();
        relative.block<3, 3>(0, 0) = Eigen::AngleAxisf(static_cast<float>(yaw), V3F::UnitZ()).toRotationMatrix();
        candidate.target_id = best_index;
        candidate.initial_guess = poseMatrix(key_poses[best_index]) * relative * poseMatrix(key_poses.back()).inverse();
        candidate.descriptor_distance = best_distance;
        return true;
    }
};

template <typename RegistrationT>
class PCLRegistration : public LoopRegistration
{
public:
    explicit PCLRegistration(const Config &config) : m_config(config)
    {
        const auto &cfg = m_config.registration;
        requirePositive(cfg.max_iterations, "registration.max_iterations");
        requirePositive(cfg.max_correspondence_distance, "registration.max_correspondence_distance");
        requirePositive(cfg.transformation_epsilon, "registration.transformation_epsilon");
        requirePositive(cfg.euclidean_fitness_epsilon, "registration.euclidean_fitness_epsilon");
        requirePositive(cfg.fitness_score_threshold, "registration.fitness_score_threshold");
        m_registration.setMaximumIterations(cfg.max_iterations);
        m_registration.setMaxCorrespondenceDistance(cfg.max_correspondence_distance);
        m_registration.setTransformationEpsilon(cfg.transformation_epsilon);
        m_registration.setEuclideanFitnessEpsilon(cfg.euclidean_fitness_epsilon);
    }

    RegistrationResult align(const CloudType::Ptr &source, const CloudType::Ptr &target,
                             const M4F &initial_guess) override
    {
        RegistrationResult result;
        CloudType aligned;
        m_registration.setInputSource(source);
        m_registration.setInputTarget(target);
        m_registration.align(aligned, initial_guess);
        result.converged = m_registration.hasConverged();
        result.score = m_registration.getFitnessScore();
        result.transformation = m_registration.getFinalTransformation();
        result.converged = result.converged && std::isfinite(result.score) &&
                           result.score <= m_config.registration.fitness_score_threshold &&
                           result.transformation.allFinite();
        return result;
    }

protected:
    Config m_config;
    RegistrationT m_registration;
};

class NDTRegistration final : public PCLRegistration<pcl::NormalDistributionsTransform<PointType, PointType>>
{
public:
    explicit NDTRegistration(const Config &config)
        : PCLRegistration<pcl::NormalDistributionsTransform<PointType, PointType>>(config)
    {
        requirePositive(config.registration.ndt_resolution, "registration.ndt_resolution");
        requirePositive(config.registration.ndt_step_size, "registration.ndt_step_size");
        this->m_registration.setResolution(config.registration.ndt_resolution);
        this->m_registration.setStepSize(config.registration.ndt_step_size);
    }
};
} // namespace

std::shared_ptr<LoopCandidateDetector> createLoopCandidateDetector(const Config &config)
{
    if (config.loop_detector == "proximity")
        return std::make_shared<ProximityDetector>();
    if (config.loop_detector == "scan_context")
        return std::make_shared<ScanContextDetector>(config);
    throw std::invalid_argument("Unknown loop_detector '" + config.loop_detector +
                                "'. Expected proximity or scan_context");
}

std::shared_ptr<LoopRegistration> createLoopRegistration(const Config &config)
{
    if (config.loop_registration == "icp")
        return std::make_shared<PCLRegistration<pcl::IterativeClosestPoint<PointType, PointType>>>(config);
    if (config.loop_registration == "gicp")
        return std::make_shared<PCLRegistration<pcl::GeneralizedIterativeClosestPoint<PointType, PointType>>>(config);
    if (config.loop_registration == "ndt")
        return std::make_shared<NDTRegistration>(config);
    throw std::invalid_argument("Unknown loop_registration '" + config.loop_registration +
                                "'. Expected icp, gicp, or ndt");
}
