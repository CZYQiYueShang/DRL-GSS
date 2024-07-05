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

#include "pointcloudmping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>
#include "Converter.h"

#include <boost/make_shared.hpp>

#include <pcl/io/pcd_io.h>

PointCloudMapping::PointCloudMapping(double resolution_, float minMaskValue_, float maxMaskValue_, int pixelRange_, float depthGapRangeXY_, float depthGapRangeGlass_, float maxDepthWeight_)
{
    this->resolution = resolution_;
    this->minMaskValue = minMaskValue_;
    this->maxMaskValue = maxMaskValue_;
    this->pixelRange = pixelRange_;
    this->depthGapRangeXY = depthGapRangeXY_;
    this->depthGapRangeGlass = depthGapRangeGlass_;
    this->maxDepthWeight = maxDepthWeight_;

    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared<PointCloud>();

    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

// void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
// {
//     cout << "receive a keyframe, id = " << kf->mnId << endl;
//     unique_lock<mutex> lck(keyframeMutex);
//     keyframes.push_back(kf);
//     colorImgs.push_back(color.clone());
//     depthImgs.push_back(depth.clone());

//     keyFrameUpdated.notify_one();
// }

// Plus
void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& mask, cv::Mat& depthGS)
{
    cout << "receive a keyframe, id = " << kf->mnId << endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back(kf);
    colorImgs.push_back(color.clone());
    depthImgs.push_back(depth.clone());
    maskImgs.push_back(mask.clone());
    depthGSImgs.push_back(depthGS.clone());

    keyFrameUpdated.notify_one();
}

// pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
// {
//     PointCloud::Ptr tmp( new PointCloud() );
//     // point cloud is null ptr
//     for ( int m=0; m<depth.rows; m+=3 )
//     {
//         for ( int n=0; n<depth.cols; n+=3 )
//         {
//             float d = depth.ptr<float>(m)[n];

//             if (d < 0.01 || d>10)
//                 continue;
//             PointT p;
//             p.z = d;
//             p.x = ( n - kf->cx) * p.z / kf->fx;
//             p.y = ( m - kf->cy) * p.z / kf->fy;

//             p.b = color.ptr<uchar>(m)[n*3];
//             p.g = color.ptr<uchar>(m)[n*3+1];
//             p.r = color.ptr<uchar>(m)[n*3+2];

//             tmp->points.push_back(p);
//         }
//     }

//     Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
//     PointCloud::Ptr cloud(new PointCloud);
//     pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
//     cloud->is_dense = false;

//     cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
//     return cloud;
// }

// Plus
pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& mask, cv::Mat& depthGS)
{
    PointCloud::Ptr tmp( new PointCloud() );
    
    float maxDepth;
    if (!mask.empty())
        maxDepth = kf->mMaxDepth * maxDepthWeight;

    // std::cout << maxDepth << endl;

    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            float d;
            if (!mask.empty())
            {
                if (mask.ptr<uchar>(m)[n] >= minMaskValue)
                {
                    d = ComputeGlassPointsDepth(depth, mask, depthGS, n, m, maxDepth);
                    // d = depthGS.ptr<float>(m)[n];
                    // d = 0.0;
                    // std::cout << d << endl;
                }
                else
                    d = depth.ptr<float>(m)[n];
                if(d >= maxDepth)
                    d = 0.0;
            }
            else
                d = depth.ptr<float>(m)[n];

            if (d < 0.01 || d > 10)
                continue;
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;

            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

            tmp->points.push_back(p);
        }
    }

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;

    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}

float PointCloudMapping::ComputeGlassPointsDepth(const cv::Mat &imDepth, const cv::Mat &imMask, const cv::Mat &imDepthGS, const int &u, const int &v, const float &maxDepth)
{
    int width = imDepth.cols;
    int height = imDepth.rows;
    int x1, x2, y1, y2 = -1;
    int countNotNan = 0;
    float x1Depth, x2Depth, y1Depth, y2Depth;
    float GPDepth;

    for (int x = u - 1; x >= 0; x--)
        if (imMask.at<uchar>(v, x) < maxMaskValue)
        {
            x1Depth = GetPointsDepth(imDepth, imMask, imDepthGS, x, v, maxDepth);
            x1 = x;
            countNotNan++;
            break;
        }
    if (x1 == -1)
    {
        x1 == 0;
        x1Depth = imDepthGS.at<float>(v, x1);
    }
    for (int x = u + 1; x <= width; x++)
        if (imMask.at<uchar>(v, x) < maxMaskValue)
        {
            x2Depth = GetPointsDepth(imDepth, imMask, imDepthGS, x, v, maxDepth);
            x2 = x;
            countNotNan++;
            break;
        }
    if (x2 == -1)
    {
        x2 == width;
        x2Depth = imDepthGS.at<float>(v, x2);
    }
    for (int y = v - 1; y >= 0; y--)
        if (imMask.at<uchar>(y, u) < maxMaskValue)
        {
            y1Depth = GetPointsDepth(imDepth, imMask, imDepthGS, u, y, maxDepth);
            y1 = y;
            countNotNan++;
            break;
        }
    if (y1 == -1)
    {
        y1 == 0;
        y1Depth = imDepthGS.at<float>(y1, u);
    }
    for (int y = v + 1; y <= height; y++)
        if (imMask.at<uchar>(y, u) < maxMaskValue)
        {
            y2Depth = GetPointsDepth(imDepth, imMask, imDepthGS, u, y, maxDepth);
            y2 = y;
            countNotNan++;
            break;
        }
    if (y2 == -1)
    {
        y2 == height;
        y2Depth = imDepthGS.at<float>(y2, u);
    }

    if (countNotNan < 3)
        GPDepth = 0.0;
    else
    {
        int x2_weight = u - x1;
        int x1_weight = x2 - u;
        int y2_weight = v - y1;
        int y1_weight = y2 - v;
        
        if (x2_weight >= 1 && x1_weight >= 1 && y2_weight >= 1 && y1_weight >= 1)
        {
            float xDepth = x1Depth * x1_weight / (x1_weight + x2_weight) + x2Depth * x2_weight / (x1_weight + x2_weight);
            float yDepth = y1Depth * y1_weight / (y1_weight + y2_weight) + y2Depth * y2_weight / (y1_weight + y2_weight);
            float xyDepthGap = xDepth / yDepth;
            // std::cout << xDepth << ' ' << yDepth << endl;
            if (xyDepthGap > (1 - depthGapRangeXY) && xyDepthGap < (1 + depthGapRangeXY))
                GPDepth = (xDepth + yDepth) / 2;
            else
                GPDepth = 0.0;
        }    
        else
            GPDepth = 0.0;
    }

    // std::cout << depth << endl;
    return GPDepth;
}

float PointCloudMapping::GetPointsDepth(const cv::Mat &imDepth, const cv::Mat &imMask, const cv::Mat &imDepthGS, const int &x, const int &y, const float &maxDepth)
{
    int width = imDepth.cols;
    int height = imDepth.rows;

    float pDepth;
    float totalDepth = 0.0;
    int count = 0;

    float finalDepth;

    for (int xx = x + pixelRange; xx >= x - pixelRange; xx--)
        for (int yy = y + pixelRange; yy >= y - pixelRange; yy--)
            if (xx >= 0 && xx <= width && yy >= 0 && yy <= height)
            {   
                float depth = imDepth.at<float>(yy, xx);
                float depthGS = imDepthGS.at<float>(yy, xx);
                float depthGap = depth / depthGS;
                if (imMask.at<uchar>(yy, xx) < maxMaskValue)
                    pDepth = depth;
                else if (imMask.at<uchar>(yy, xx) < minMaskValue)
                    if (depthGap > (1 - depthGapRangeGlass) && depthGap < (1 + depthGapRangeGlass))
                        pDepth = depth;
                    else
                        pDepth = depthGS;
                else
                    pDepth = depthGS;
                if (pDepth < maxDepth && pDepth > 0.01)
                {
                    totalDepth += pDepth;
                    count++;
                }
            }
    if (count == 0)
        finalDepth = imDepthGS.at<float>(y, x);
    else
        finalDepth = totalDepth / count;

    return finalDepth;
}

void PointCloudMapping::viewer()
{
    pcl::visualization::CloudViewer viewer("Point Cloud Viewer");
    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }

        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }

        for ( size_t i=lastKeyframeSize; i<N ; i++ )
        {
            PointCloud::Ptr p;
            p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i], maskImgs[i], depthGSImgs[i]);
            *globalMap += *p;
        }
        pcl::io::savePCDFileBinary("result.pcd", *globalMap);
        PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud( globalMap );
        voxel.filter( *tmp );
        globalMap->swap( *tmp );
        viewer.showCloud( globalMap );
        cout << "show global map, size=" << globalMap->points.size() << endl;
        lastKeyframeSize = N;
    }
}

