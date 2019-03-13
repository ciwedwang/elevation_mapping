/*
 * SensorProcessorBase.cpp
 *
 *  Created on: Jun 6, 2014
 *      Author: Péter Fankhauser, Hannes Keller
 *   Institute: ETH Zurich, ANYbotics
 */

#include <elevation_mapping/sensor_processors/SensorProcessorBase.hpp>

//PCL
#include <pcl/pcl_base.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
//TF
#include <tf_conversions/tf_eigen.h>

// STL
#include <limits>
#include <math.h>
#include <vector>

namespace elevation_mapping {

SensorProcessorBase::SensorProcessorBase(ros::NodeHandle& nodeHandle, tf::TransformListener& transformListener)
    : nodeHandle_(nodeHandle),
      transformListener_(transformListener),
      ignorePointsUpperThreshold_(std::numeric_limits<double>::infinity()),
      ignorePointsLowerThreshold_(-std::numeric_limits<double>::infinity())
{
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
	transformationSensorToMap_.setIdentity();
	transformListenerTimeout_.fromSec(1.0);
}

SensorProcessorBase::~SensorProcessorBase() {}

bool SensorProcessorBase::readParameters()
{
  nodeHandle_.param("sensor_frame_id", sensorFrameId_, std::string("/sensor")); // TODO Fail if parameters are not found.
  nodeHandle_.param("robot_base_frame_id", robotBaseFrameId_, std::string("/robot"));
  nodeHandle_.param("map_frame_id", mapFrameId_, std::string("/map"));

  double minUpdateRate;
  nodeHandle_.param("min_update_rate", minUpdateRate, 2.0);
  transformListenerTimeout_.fromSec(1.0 / minUpdateRate);
  ROS_ASSERT(!transformListenerTimeout_.isZero());

  nodeHandle_.param("sensor_processor/ignore_points_above", ignorePointsUpperThreshold_, std::numeric_limits<double>::infinity());
  nodeHandle_.param("sensor_processor/ignore_points_below", ignorePointsLowerThreshold_, std::numeric_limits<double>::infinity());  
  nodeHandle_.param("length_in_x", ignorePointsOutsideXThreshold_, std::numeric_limits<double>::infinity());
  nodeHandle_.param("length_in_y", ignorePointsOutsideYThreshold_, std::numeric_limits<double>::infinity());
  nodeHandle_.param("sensor_processor/ignore_points_inside_x_y", ignorePointsInsideXYThreshold_, std::numeric_limits<double>::infinity());
  
  ROS_INFO("Read parameter %s : %f", "ignorePointsOutsideXThreshold_", ignorePointsOutsideXThreshold_);
  ROS_INFO("Read parameter %s : %f", "ignorePointsOutsideYThreshold_", ignorePointsOutsideYThreshold_);
  ROS_INFO("Read parameter %s : %f", "ignorePointsInsideXYThreshold_", ignorePointsInsideXYThreshold_);

  return true;
}

bool SensorProcessorBase::process(
		const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr pointCloudInput,
		const Eigen::Matrix<double, 6, 6>& robotPoseCovariance,
		const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloudMapFrame,
		Eigen::VectorXf& variances, 
    bool calculate_variance)
{
  ros::Time timeStamp;
  timeStamp.fromNSec(1000 * pointCloudInput->header.stamp);
  if (!updateTransformations(timeStamp)) return false;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloudSensorFrame(new pcl::PointCloud<pcl::PointXYZRGB>);
	transformPointCloud(pointCloudInput, pointCloudSensorFrame, sensorFrameId_);
	cleanPointCloud(pointCloudSensorFrame);

	if (!transformPointCloud(pointCloudSensorFrame, pointCloudMapFrame, mapFrameId_)) return false;
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds({pointCloudMapFrame, pointCloudSensorFrame});
  // removePointsOutsideLimits(pointCloudMapFrame, pointClouds);
  removePointsOutsideLimitsByCropBox(pointCloudMapFrame, pointClouds);
  variances.resize(pointCloudMapFrame->size());
	if (calculate_variance && !computeVariances(pointCloudSensorFrame, robotPoseCovariance, variances)) return false;
  
	return true;
}

bool SensorProcessorBase::updateTransformations(const ros::Time& timeStamp)
{
  try {
    transformListener_.waitForTransform(sensorFrameId_, mapFrameId_, timeStamp, ros::Duration(1.0));

    tf::StampedTransform transformTf;
    transformListener_.lookupTransform(mapFrameId_, sensorFrameId_, timeStamp, transformTf);
    poseTFToEigen(transformTf, transformationSensorToMap_);

    transformListener_.lookupTransform(robotBaseFrameId_, sensorFrameId_, timeStamp, transformTf);  // TODO Why wrong direction?
    Eigen::Affine3d transform;
    poseTFToEigen(transformTf, transform);
    rotationBaseToSensor_.setMatrix(transform.rotation().matrix());
    translationBaseToSensorInBaseFrame_.toImplementation() = transform.translation();

    transformListener_.lookupTransform(mapFrameId_, robotBaseFrameId_, timeStamp, transformTf);  // TODO Why wrong direction?
    poseTFToEigen(transformTf, transform);
    rotationMapToBase_.setMatrix(transform.rotation().matrix());
    translationMapToBaseInMapFrame_.toImplementation() = transform.translation();

    return true;
  } catch (tf::TransformException &ex) {
    ROS_ERROR("%s", ex.what());
    return false;
  }
}

bool SensorProcessorBase::transformPointCloud(
		pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr pointCloud,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloudTransformed,
		const std::string& targetFrame)
{
  ros::Time timeStamp;
  timeStamp.fromNSec(1000 * pointCloud->header.stamp);
  const std::string inputFrameId(pointCloud->header.frame_id);

  tf::StampedTransform transformTf;
  try {
    transformListener_.waitForTransform(targetFrame, inputFrameId, timeStamp, ros::Duration(1.0));
    transformListener_.lookupTransform(targetFrame, inputFrameId, timeStamp, transformTf);
  } catch (tf::TransformException &ex) {
    ROS_ERROR("%s", ex.what());
    return false;
  }

  Eigen::Affine3d transform;
  poseTFToEigen(transformTf, transform);
  pcl::transformPointCloud(*pointCloud, *pointCloudTransformed, transform.cast<float>());
  pointCloudTransformed->header.frame_id = targetFrame;

	ROS_DEBUG("Point cloud transformed to frame %s for time stamp %f.", targetFrame.c_str(),
			ros::Time(pointCloudTransformed->header.stamp).toSec());
	return true;
}

void SensorProcessorBase::removePointsOutsideLimits(
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr reference, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& pointClouds)
{
  if (!std::isfinite(ignorePointsLowerThreshold_) && !std::isfinite(ignorePointsUpperThreshold_)) return;
  ROS_DEBUG("Limiting point cloud to the height interval of [%f, %f] relative to the robot base.", ignorePointsLowerThreshold_, ignorePointsUpperThreshold_);

  pcl::PassThrough<pcl::PointXYZRGB> passThroughFilter(true);
  passThroughFilter.setInputCloud(reference);
  passThroughFilter.setFilterFieldName("z"); // TODO: Should this be configurable?
  double relativeLowerThreshold = translationMapToBaseInMapFrame_.z() + ignorePointsLowerThreshold_;
  double relativeUpperThreshold = translationMapToBaseInMapFrame_.z() + ignorePointsUpperThreshold_;
  passThroughFilter.setFilterLimits(relativeLowerThreshold, relativeUpperThreshold);
  pcl::IndicesPtr insideIndeces(new std::vector<int>);
  passThroughFilter.filter(*insideIndeces);

  for (auto& pointCloud : pointClouds) {
    pcl::ExtractIndices<pcl::PointXYZRGB> extractIndicesFilter;
    extractIndicesFilter.setInputCloud(pointCloud);
    extractIndicesFilter.setIndices(insideIndeces);
    pcl::PointCloud<pcl::PointXYZRGB> tempPointCloud;
    extractIndicesFilter.filter(tempPointCloud);
    pointCloud->swap(tempPointCloud);
  }

  ROS_DEBUG("removePointsOutsideLimits() reduced point cloud to %i points.", (int) pointClouds[0]->size());
}

void SensorProcessorBase::removePointsOutsideLimitsByCropBox(
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr reference, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& pointClouds)
{
  if (   !std::isfinite(ignorePointsLowerThreshold_)      
      && !std::isfinite(ignorePointsUpperThreshold_)      
      && !std::isfinite(ignorePointsOutsideXThreshold_)
      && !std::isfinite(ignorePointsOutsideYThreshold_)
      && !std::isfinite(ignorePointsInsideXYThreshold_)) 
      return;
  ROS_DEBUG("Limiting point cloud to the height interval of [%f, %f] relative to the robot base.", ignorePointsLowerThreshold_, ignorePointsUpperThreshold_);

  pcl::CropBox<pcl::PointXYZRGB> crop_box_filter(true);
  crop_box_filter.setInputCloud(reference);
  Eigen::Vector4f min_point;
  Eigen::Vector4f max_point;

  //1. filter too far
  min_point = Eigen::Vector4f(translationMapToBaseInMapFrame_.x() - ignorePointsOutsideXThreshold_,
                              translationMapToBaseInMapFrame_.y() - ignorePointsOutsideYThreshold_,
                              translationMapToBaseInMapFrame_.z() + ignorePointsLowerThreshold_,
                              1.0);
  max_point = Eigen::Vector4f(translationMapToBaseInMapFrame_.x() + ignorePointsOutsideXThreshold_,
                              translationMapToBaseInMapFrame_.y() + ignorePointsOutsideYThreshold_,
                              translationMapToBaseInMapFrame_.z() + ignorePointsUpperThreshold_,
                              1.0);
  crop_box_filter.setMin(min_point);
  crop_box_filter.setMax(max_point);
  crop_box_filter.setNegative(false);
  pcl::IndicesPtr insideIndeces_far_rem(new std::vector<int>);
  crop_box_filter.filter(*insideIndeces_far_rem);

  //2. filter too near
  crop_box_filter.setIndices(insideIndeces_far_rem);
  min_point = Eigen::Vector4f(translationMapToBaseInMapFrame_.x() - ignorePointsInsideXYThreshold_,
                              translationMapToBaseInMapFrame_.y() - ignorePointsInsideXYThreshold_,
                              translationMapToBaseInMapFrame_.z() + ignorePointsLowerThreshold_,
                              1.0);
  max_point = Eigen::Vector4f(translationMapToBaseInMapFrame_.x() + ignorePointsInsideXYThreshold_,
                              translationMapToBaseInMapFrame_.y() + ignorePointsInsideXYThreshold_,
                              translationMapToBaseInMapFrame_.z() + ignorePointsUpperThreshold_,
                              1.0);
  crop_box_filter.setMin(min_point);
  crop_box_filter.setMax(max_point);
  crop_box_filter.setNegative(true);
  pcl::IndicesPtr insideIndeces_near_rem(new std::vector<int>);
  crop_box_filter.filter(*insideIndeces_near_rem);  

  for (auto& pointCloud : pointClouds) {
    pcl::ExtractIndices<pcl::PointXYZRGB> extractIndicesFilter;
    extractIndicesFilter.setInputCloud(pointCloud);
    extractIndicesFilter.setIndices(insideIndeces_near_rem);
    pcl::PointCloud<pcl::PointXYZRGB> tempPointCloud;
    extractIndicesFilter.filter(tempPointCloud);
    pointCloud->swap(tempPointCloud);
  }

  ROS_DEBUG("removePointsOutsideLimits() reduced point cloud to %i points.", (int) pointClouds[0]->size());
}

} /* namespace elevation_mapping */

