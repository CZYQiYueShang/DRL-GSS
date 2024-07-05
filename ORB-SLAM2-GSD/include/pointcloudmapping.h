/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <condition_variable>

using namespace ORB_SLAM2;

class PointCloudMapping
{
public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloudMapping(double resolution_, float minMaskValue_, float maxMaskValue_, int pixelRange_, float depthGapRangeXY_, float depthGapRangeGlass_, float maxDepthWeight_);

    // void insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    void insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& mask, cv::Mat& depthGS);

    // Plus
    float ComputeGlassPointsDepth(const cv::Mat &imDepth, const cv::Mat &imMask, const cv::Mat &imDepthGS, const int &x, const int &y, const float &maxDepth);
    float GetPointsDepth(const cv::Mat &imDepth, const cv::Mat &imMask, const cv::Mat &imDepthGS, const int &u, const int &v, const float &maxDepth);


    void shutdown();
    void viewer();

protected:
    // PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& mask, cv::Mat& depthGS);

    PointCloud::Ptr globalMap;
    shared_ptr<thread>  viewerThread;

    bool    shutDownFlag    =false;
    mutex   shutDownMutex;

    condition_variable  keyFrameUpdated;
    mutex               keyFrameUpdateMutex;

    // data to generate point clouds
    vector<KeyFrame*>       keyframes;
    vector<cv::Mat>         colorImgs;
    vector<cv::Mat>         depthImgs;

    vector<cv::Mat>         maskImgs;
    vector<cv::Mat>         depthGSImgs;

    mutex                   keyframeMutex;
    uint16_t                lastKeyframeSize =0;

    double resolution = 0.04;
    float minMaskValue;
    float maxMaskValue;
    int pixelRange;
    float depthGapRangeXY;
    float depthGapRangeGlass;
    float maxDepthWeight;

    pcl::VoxelGrid<PointT>  voxel;
};

#endif // POINTCLOUDMAPPING_H
