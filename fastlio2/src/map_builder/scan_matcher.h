#pragma once

#include "commons.h"
#include "ieskf.h"

#include <memory>
#include <vector>

struct LinearizedResidual
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int dimension = 0;
    Eigen::Matrix<double, 3, 12> jacobian = Eigen::Matrix<double, 3, 12>::Zero();
    V3D residual = V3D::Zero();
    M3D information = M3D::Zero();
};

class ScanMatcher
{
public:
    explicit ScanMatcher(const Config &config) : m_config(config) {}
    virtual ~ScanMatcher() = default;

    virtual int neighborCount() const = 0;
    virtual void prepare(const PointVec &map_points) { (void)map_points; }
    virtual bool requiresMapPoints() const { return false; }
    virtual bool linearize(const PointType &point_lidar,
                           const V3D &point_world,
                           const PointVec &neighbors,
                           const std::vector<float> &squared_distances,
                           const State &state,
                           LinearizedResidual &result) const = 0;
    virtual int minEffectivePoints() const = 0;

protected:
    Eigen::Matrix<double, 3, 12> pointJacobian(const PointType &point_lidar,
                                               const State &state) const;
    Config m_config;
};

std::shared_ptr<ScanMatcher> createScanMatcher(const Config &config);
