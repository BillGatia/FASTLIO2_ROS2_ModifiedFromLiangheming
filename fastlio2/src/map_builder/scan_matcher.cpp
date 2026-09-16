#include "scan_matcher.h"

#include <Eigen/Eigenvalues>
#include <sophus/so3.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

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

class PointToPlaneMatcher final : public ScanMatcher
{
public:
    explicit PointToPlaneMatcher(const Config &config) : ScanMatcher(config)
    {
        requirePositive(m_config.scan_matching.point_to_plane.neighbor_count, "point_to_plane.neighbor_count");
        requirePositive(m_config.scan_matching.point_to_plane.max_neighbor_squared_distance, "point_to_plane.max_neighbor_squared_distance");
        requirePositive(m_config.scan_matching.point_to_plane.plane_fit_threshold, "point_to_plane.plane_fit_threshold");
        requirePositive(m_config.scan_matching.min_effective_points, "scan_matching.min_effective_points");
    }

    int neighborCount() const override { return m_config.scan_matching.point_to_plane.neighbor_count; }
    int minEffectivePoints() const override { return m_config.scan_matching.min_effective_points; }

    bool linearize(const PointType &point_lidar, const V3D &point_world,
                   const PointVec &neighbors, const std::vector<float> &distances,
                   const State &state, LinearizedResidual &result) const override
    {
        const auto &cfg = m_config.scan_matching.point_to_plane;
        if (neighbors.size() < static_cast<size_t>(cfg.neighbor_count) ||
            distances.size() < static_cast<size_t>(cfg.neighbor_count) ||
            distances[cfg.neighbor_count - 1] > cfg.max_neighbor_squared_distance)
            return false;

        PointVec plane_points(neighbors.begin(), neighbors.begin() + cfg.neighbor_count);
        V4D plane;
        if (!esti_plane(plane_points, cfg.plane_fit_threshold, plane))
            return false;

        const V3D point_body(point_lidar.x, point_lidar.y, point_lidar.z);
        const double residual = plane.head<3>().dot(point_world) + plane(3);
        const double range = std::max(point_body.norm(), 1e-6);
        const double score = 1.0 - 0.9 * std::fabs(residual) / std::sqrt(range);
        if (score <= cfg.score_threshold)
            return false;

        result.dimension = 1;
        result.jacobian.row(0) = plane.head<3>().transpose() * pointJacobian(point_lidar, state);
        result.residual(0) = residual;
        result.information(0, 0) = m_config.lidar_cov_inv;
        return true;
    }
};

class PointToPointMatcher final : public ScanMatcher
{
public:
    explicit PointToPointMatcher(const Config &config) : ScanMatcher(config)
    {
        requirePositive(m_config.scan_matching.point_to_point.max_correspondence_distance, "point_to_point.max_correspondence_distance");
        requirePositive(m_config.scan_matching.point_to_point.min_effective_points, "point_to_point.min_effective_points");
    }

    int neighborCount() const override { return 1; }
    int minEffectivePoints() const override { return m_config.scan_matching.point_to_point.min_effective_points; }

    bool linearize(const PointType &point_lidar, const V3D &point_world,
                   const PointVec &neighbors, const std::vector<float> &distances,
                   const State &state, LinearizedResidual &result) const override
    {
        const double max_distance = m_config.scan_matching.point_to_point.max_correspondence_distance;
        if (neighbors.empty() || distances.empty() || distances.front() > max_distance * max_distance)
            return false;
        const PointType &target = neighbors.front();
        result.dimension = 3;
        result.jacobian = pointJacobian(point_lidar, state);
        result.residual = point_world - V3D(target.x, target.y, target.z);
        result.information = M3D::Identity() * m_config.lidar_cov_inv;
        return true;
    }
};

struct VoxelKey
{
    int x;
    int y;
    int z;
    bool operator==(const VoxelKey &other) const { return x == other.x && y == other.y && z == other.z; }
};

struct VoxelKeyHash
{
    size_t operator()(const VoxelKey &key) const
    {
        size_t seed = std::hash<int>{}(key.x);
        seed ^= std::hash<int>{}(key.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(key.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct VoxelAccumulator
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int count = 0;
    V3D sum = V3D::Zero();
    M3D sum_outer = M3D::Zero();
};

struct VoxelDistribution
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    V3D mean = V3D::Zero();
    M3D information = M3D::Identity();
};

class NDTMatcher final : public ScanMatcher
{
public:
    explicit NDTMatcher(const Config &config) : ScanMatcher(config)
    {
        const auto &cfg = m_config.scan_matching.ndt;
        requirePositive(cfg.resolution, "ndt.resolution");
        requirePositive(cfg.min_points_per_voxel, "ndt.min_points_per_voxel");
        requirePositive(cfg.covariance_regularization, "ndt.covariance_regularization");
        requirePositive(cfg.max_mahalanobis_distance, "ndt.max_mahalanobis_distance");
        requirePositive(cfg.min_effective_points, "ndt.min_effective_points");
    }

    int neighborCount() const override { return 0; }
    int minEffectivePoints() const override { return m_config.scan_matching.ndt.min_effective_points; }
    bool requiresMapPoints() const override { return true; }

    void prepare(const PointVec &map_points) override
    {
        m_voxels.clear();
        std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> accumulators;
        for (const auto &point : map_points)
        {
            const V3D value(point.x, point.y, point.z);
            auto &acc = accumulators[key(value)];
            ++acc.count;
            acc.sum += value;
            acc.sum_outer += value * value.transpose();
        }

        const auto &cfg = m_config.scan_matching.ndt;
        for (const auto &entry : accumulators)
        {
            const VoxelAccumulator &acc = entry.second;
            if (acc.count < cfg.min_points_per_voxel)
                continue;
            VoxelDistribution distribution;
            distribution.mean = acc.sum / static_cast<double>(acc.count);
            M3D covariance = acc.sum_outer / static_cast<double>(acc.count) -
                             distribution.mean * distribution.mean.transpose();
            covariance += M3D::Identity() * cfg.covariance_regularization;
            Eigen::LDLT<M3D> solver(covariance);
            if (solver.info() != Eigen::Success)
                continue;
            distribution.information = solver.solve(M3D::Identity());
            if (distribution.information.allFinite())
                m_voxels.emplace(entry.first, distribution);
        }
    }

    bool linearize(const PointType &point_lidar, const V3D &point_world,
                   const PointVec &, const std::vector<float> &,
                   const State &state, LinearizedResidual &result) const override
    {
        const auto it = m_voxels.find(key(point_world));
        if (it == m_voxels.end())
            return false;
        const V3D residual = point_world - it->second.mean;
        const double mahalanobis = residual.dot(it->second.information * residual);
        if (!std::isfinite(mahalanobis) || mahalanobis > m_config.scan_matching.ndt.max_mahalanobis_distance)
            return false;
        result.dimension = 3;
        result.jacobian = pointJacobian(point_lidar, state);
        result.residual = residual;
        result.information = it->second.information * m_config.lidar_cov_inv;
        return true;
    }

private:
    VoxelKey key(const V3D &point) const
    {
        const double resolution = m_config.scan_matching.ndt.resolution;
        return {static_cast<int>(std::floor(point.x() / resolution)),
                static_cast<int>(std::floor(point.y() / resolution)),
                static_cast<int>(std::floor(point.z() / resolution))};
    }

    std::unordered_map<VoxelKey, VoxelDistribution, VoxelKeyHash> m_voxels;
};

class LOAMMatcher final : public ScanMatcher
{
public:
    explicit LOAMMatcher(const Config &config) : ScanMatcher(config)
    {
        const auto &cfg = m_config.scan_matching.loam;
        requirePositive(cfg.neighbor_count, "loam.neighbor_count");
        requirePositive(cfg.max_neighbor_squared_distance, "loam.max_neighbor_squared_distance");
        requirePositive(cfg.line_eigenvalue_ratio, "loam.line_eigenvalue_ratio");
        requirePositive(cfg.plane_eigenvalue_ratio, "loam.plane_eigenvalue_ratio");
        requirePositive(cfg.min_effective_points, "loam.min_effective_points");
    }

    int neighborCount() const override { return m_config.scan_matching.loam.neighbor_count; }
    int minEffectivePoints() const override { return m_config.scan_matching.loam.min_effective_points; }

    bool linearize(const PointType &point_lidar, const V3D &point_world,
                   const PointVec &neighbors, const std::vector<float> &distances,
                   const State &state, LinearizedResidual &result) const override
    {
        const auto &cfg = m_config.scan_matching.loam;
        if (neighbors.size() < static_cast<size_t>(cfg.neighbor_count) ||
            distances.size() < static_cast<size_t>(cfg.neighbor_count) ||
            distances[cfg.neighbor_count - 1] > cfg.max_neighbor_squared_distance)
            return false;

        V3D centroid = V3D::Zero();
        for (int i = 0; i < cfg.neighbor_count; ++i)
            centroid += V3D(neighbors[i].x, neighbors[i].y, neighbors[i].z);
        centroid /= static_cast<double>(cfg.neighbor_count);

        M3D covariance = M3D::Zero();
        for (int i = 0; i < cfg.neighbor_count; ++i)
        {
            const V3D delta = V3D(neighbors[i].x, neighbors[i].y, neighbors[i].z) - centroid;
            covariance += delta * delta.transpose();
        }
        covariance /= static_cast<double>(cfg.neighbor_count);

        Eigen::SelfAdjointEigenSolver<M3D> solver(covariance);
        if (solver.info() != Eigen::Success)
            return false;
        const V3D eigenvalues = solver.eigenvalues();
        const auto point_jacobian = pointJacobian(point_lidar, state);
        const double sum = std::max(eigenvalues.sum(), 1e-12);

        if (eigenvalues(2) / std::max(eigenvalues(1), 1e-12) >= cfg.line_eigenvalue_ratio)
        {
            const V3D direction = solver.eigenvectors().col(2);
            const M3D projector = M3D::Identity() - direction * direction.transpose();
            result.dimension = 3;
            result.jacobian = projector * point_jacobian;
            result.residual = projector * (point_world - centroid);
            result.information = M3D::Identity() * m_config.lidar_cov_inv;
            return true;
        }

        if (eigenvalues(0) / sum <= cfg.plane_eigenvalue_ratio)
        {
            const V3D normal = solver.eigenvectors().col(0);
            result.dimension = 1;
            result.jacobian.row(0) = normal.transpose() * point_jacobian;
            result.residual(0) = normal.dot(point_world - centroid);
            result.information(0, 0) = m_config.lidar_cov_inv;
            return true;
        }
        return false;
    }
};
} // namespace

Eigen::Matrix<double, 3, 12> ScanMatcher::pointJacobian(const PointType &point_lidar,
                                                        const State &state) const
{
    Eigen::Matrix<double, 3, 12> jacobian = Eigen::Matrix<double, 3, 12>::Zero();
    const V3D point(point_lidar.x, point_lidar.y, point_lidar.z);
    jacobian.block<3, 3>(0, 0) = -state.r_wi * Sophus::SO3d::hat(state.r_il * point + state.t_il);
    jacobian.block<3, 3>(0, 3) = M3D::Identity();
    if (m_config.esti_il)
    {
        jacobian.block<3, 3>(0, 6) = -state.r_wi * state.r_il * Sophus::SO3d::hat(point);
        jacobian.block<3, 3>(0, 9) = state.r_wi;
    }
    return jacobian;
}

std::shared_ptr<ScanMatcher> createScanMatcher(const Config &config)
{
    const std::string &method = config.scan_matching.method;
    if (method == "point_to_plane")
        return std::make_shared<PointToPlaneMatcher>(config);
    if (method == "point_to_point")
        return std::make_shared<PointToPointMatcher>(config);
    if (method == "ndt")
        return std::make_shared<NDTMatcher>(config);
    if (method == "loam")
        return std::make_shared<LOAMMatcher>(config);
    throw std::invalid_argument("Unknown scan_matching.method '" + method +
                                "'. Expected point_to_plane, point_to_point, ndt, or loam");
}
