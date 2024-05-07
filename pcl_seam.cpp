#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/octree/octree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/fast_bilateral_omp.h> 
#include <pcl/filters/bilateral.h>  
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/boundary.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>
#include <stdbool.h>
#include <chrono>
#include <math.h>
#include <sys/io.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <string>
#include <stdio.h>
#include <time.h>
#include <numeric>
#include <vector>

template <typename T>
T SumVector(std::vector<T> &vec)
{
    T res = 0;
    for (size_t i = 0; i < vec.size(); i++)
    {
        res += vec[i];
    }
    return res;
}

struct Clouds
{
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_tf;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bound;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bound_tf;
    pcl::ModelCoefficients::Ptr coefficients;
    pcl::ModelCoefficients::Ptr coefficients_tf;
    std::vector<double> mnormal;
    bool isbase;
};

void VoxelGridPointCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr output, float leaf_size)
{
    if (cloud->size() > 0)
    {
        pcl::VoxelGrid<pcl::PointXYZRGBNormal> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(leaf_size, leaf_size, leaf_size);
        sor.filter(*output);

        // Uniform sampling object
        // pcl::UniformSampling<pcl::PointXYZRGBNormal> filter;		// 创建均匀采样对象
        // filter.setInputCloud(cloud);					// 设置待采样点云
        // filter.setRadiusSearch(leaf_size);					// 设置采样半径
        // filter.filter(*output);					// 执行均匀采样，结果保存在cloud_filtered中
    }
}

void PassthroughPointCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                           pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr output,
                           float min,
                           float max)
{
    if (cloud->size() > 0)
    {
        pcl::PassThrough<pcl::PointXYZRGBNormal> pass;
        pass.setInputCloud(cloud);           // 在入点云
        pass.setFilterFieldName("z");        // 滤波的字段，即滤波的方向。可以是XYZ也可以是BGR
        pass.setFilterLimits(min, max);      // 滤除在z轴方向上不在0.0-0.1范围内的所有点
        pass.setFilterLimitsNegative(false); // 设置保留范围内部还是范围外部  默认false为内部
        pass.filter(*output);                // 开始滤波，结果输出至cloud_filtered中
    }
}

void ComputeNormals(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr incloud, pcl::PointCloud<pcl::Normal>::Ptr outnormals)
{
    if (!incloud->empty())
    {
        auto startnor = std::chrono::system_clock::now();
        std::cout << "compute normal cloud size: " << incloud->size() << std::endl;
        // 创建法线估计估计向量
        // pcl::NormalEstimation<pcl::PointXYZRGBNormal, pcl::Normal> ne;
        pcl::NormalEstimationOMP<pcl::PointXYZRGBNormal, pcl::Normal> ne;
        ne.setInputCloud(incloud);
        // 创建一个空的KdTree对象，并把它传递给法线估计向量
        // 基于给出的输入数据集，KdTree将被建立
        pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>());
        ne.setSearchMethod(tree);
        // 使用半径在查询点周围3厘米范围内的所有临近元素
        ne.setRadiusSearch(1);
        // 计算特征值
        ne.compute(*outnormals);
        auto endnor = std::chrono::system_clock::now();
        // std::chrono::duration<double> diff = end - start;
        // 得到程序运行总时间
        auto durationnor = std::chrono::duration_cast<std::chrono::milliseconds>(endnor - startnor);
        std::cout << "compute normal time "
                  << " ints : " << durationnor.count() << " ms\n";
    }
}
// 分割平面
void SegPlane(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr incloud,
              pcl::PointCloud<pcl::Normal>::Ptr innormals,
              pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr outcloud_plane,
              pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr outcloud_other,
              pcl::ModelCoefficients::Ptr coefficients)
{
    if (!incloud->empty())
    {
        pcl::PointIndices::Ptr indices(new pcl::PointIndices());
        pcl::SACSegmentationFromNormals<pcl::PointXYZRGBNormal, pcl::Normal> seg; // 依据法线　分割对象
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_NORMAL_PLANE); // 平面模型
        seg.setNormalDistanceWeight(0.1);             // 法线信息权重
        seg.setMethodType(pcl::SAC_RANSAC);           // 随机采样一致性算法
        seg.setMaxIterations(1000);                   // 最大迭代次数
        seg.setDistanceThreshold(1.0);                // 设置内点到模型的距离允许最大值
        seg.setInputCloud(incloud);                   // 输入点云
        seg.setInputNormals(innormals);               // 输入法线特征
        seg.segment(*indices, *coefficients);         // 分割　得到内点索引　和模型系数

        if (indices->indices.size() == 0)
        {
            std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        }
        else
        {
            pcl::ExtractIndices<pcl::PointXYZRGBNormal> my_extract_indices;
            my_extract_indices.setInputCloud(incloud);
            my_extract_indices.setIndices(indices);
            my_extract_indices.setNegative(false);
            my_extract_indices.filter(*outcloud_plane);

            my_extract_indices.setNegative(true);
            my_extract_indices.filter(*outcloud_other);
        }
    }
}

void SegLine(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr incloud,
             pcl::PointCloud<pcl::Normal>::Ptr innormals,
             pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr outcloud,
             pcl::ModelCoefficients::Ptr coefficients,
             bool negative)
{
    pcl::PointIndices::Ptr indices(new pcl::PointIndices());
    pcl::SACSegmentationFromNormals<pcl::PointXYZRGBNormal, pcl::Normal> seg; // 依据法线　分割对象
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PARALLEL_LINE); // 平面模型
    seg.setNormalDistanceWeight(0.1);              // 法线信息权重
    seg.setMethodType(pcl::SAC_RANSAC);            // 随机采样一致性算法
    seg.setMaxIterations(1000);                    // 最大迭代次数
    seg.setDistanceThreshold(0.8);                 // 设置内点到模型的距离允许最大值
    seg.setInputCloud(incloud);                    // 输入点云
    seg.setInputNormals(innormals);                // 输入法线特征
    // seg.setEpsAngle(M_PI / 120);
    seg.segment(*indices, *coefficients); // 分割　得到内点索引　和模型系数

    if (indices->indices.size() == 0)
    {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
    }
    pcl::ExtractIndices<pcl::PointXYZRGBNormal> my_extract_indices;
    my_extract_indices.setInputCloud(incloud);
    my_extract_indices.setIndices(indices);
    my_extract_indices.setNegative(negative);
    my_extract_indices.filter(*outcloud);
}

void BilateralCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr incloud,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr outcloud)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree1(new  pcl::search::KdTree<pcl::PointXYZI>);
    pcl::copyPointCloud(*incloud, *cloud);
    tree1->setInputCloud(cloud);
    // 创建双边滤波器对象
    pcl::BilateralFilter<pcl::PointXYZI> filter;
    filter.setInputCloud(cloud);
    filter.setSearchMethod(tree1); 
    // filter.setSigmaS(1); 
    // filter.setSigmaR(0.01); 
    // 执行双边滤波
    filter.setHalfSize(10);
    filter.setStdDev(0.1);
    filter.filter(*cloud1);
    pcl::copyPointCloud(*cloud1, *outcloud);
}

// 欧式聚类
void EuclideanClusterCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr incloud,
                           pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr outcloud,
                           int mincloudsize,
                           int eulists)
{
    auto starteu = std::chrono::system_clock::now();
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    tree->setInputCloud(incloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGBNormal> ec;
    if (eulists == 2)
    {
        ec.setClusterTolerance(4); // 4mm
    }
    else if (eulists == 1)
    {
        ec.setClusterTolerance(1.5); // 1.5mm
    }
    else if (eulists == 0)
    {
        ec.setClusterTolerance(1.0); // 1.0mm
    }

    ec.setMinClusterSize(mincloudsize);
    ec.setMaxClusterSize(1000000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(incloud);
    // 聚类抽取结果保存在一个数组中，数组中每个元素代表抽取的一个组件点云的下标
    ec.extract(cluster_indices);

    std::vector<pcl::PointIndices> cluster_indices1;
    std::vector<pcl::PointIndices> cluster_indices2;
    int j = 0;
    for (auto i : cluster_indices)
    {
        if (eulists == 0)
        {
            if (j < 2)
            {
                std::cout << "0" << std::endl;
                std::cout << i.indices.size() << std::endl;
                cluster_indices1.push_back(i);
                j = j + 1;
            }
            else if (j == 2 || j == 3)
            {
                std::cout << "0" << std::endl;
                std::cout << i.indices.size() << std::endl;
                if (i.indices.size() > 10000)
                    cluster_indices1.push_back(i);
                j = j + 1;
                std::cout << "j: " << j << std::endl;
            }
            else
            {
                break;
            }
        }
        else if (eulists == 1 || eulists == 2)
        {
            if (j < 1)
            {
                std::cout << "1" << std::endl;
                std::cout << i.indices.size() << std::endl;
                cluster_indices1.push_back(i);
                j = j + 1;
            }
            else
            {
                break;
            }
        }
    }
    pcl::copyPointCloud(*incloud, cluster_indices1, *outcloud); // 将对应索引的点存储
    auto endeu = std::chrono::system_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // 得到程序运行总时间
    auto durationeu = std::chrono::duration_cast<std::chrono::milliseconds>(endeu - starteu);
    std::cout << "eu cluster time "
              << " ints : " << durationeu.count() << " ms\n";
}

// 区域生长分割
void RegionGrowingrCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr incloud,
                         pcl::PointCloud<pcl::Normal>::Ptr innormals,
                         pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr outcloud,
                         int mincloudsize)
{
    auto starteu = std::chrono::system_clock::now();
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    tree->setInputCloud(incloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::RegionGrowing<pcl::PointXYZRGBNormal, pcl::Normal> reg; // 创建区域生长分割对象
    reg.setMinClusterSize(50);                                   // 设置一个聚类需要的最小点数
    reg.setMaxClusterSize(1000000);                              // 设置一个聚类需要的最大点数
    reg.setSearchMethod(tree);                                   // 设置搜索方法
    reg.setNumberOfNeighbours(30);                               // 设置搜索的临近点数目
    reg.setInputCloud(incloud);                                  // 设置输入点云
    // if(Bool_Cuting)reg.setIndices (indices);//通过输入参数设置，确定是否输入点云索引
    reg.setInputNormals(innormals);                // 设置输入点云的法向量
    reg.setSmoothnessThreshold(30 / 180.0 * M_PI); // 设置平滑阈值
    reg.setCurvatureThreshold(0.05);               // 设置曲率阈值
    reg.extract(cluster_indices);

    std::vector<pcl::PointIndices> cluster_indices1;
    int j = 0;
    for (auto i : cluster_indices)
    {
        if (j < 2)
        {
            std::cout << "0" << std::endl;
            std::cout << i.indices.size() << std::endl;
            cluster_indices1.push_back(i);
            j = j + 1;
        }
    }
    pcl::copyPointCloud(*incloud, cluster_indices1, *outcloud); // 将对应索引的点存储
    auto endeu = std::chrono::system_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // 得到程序运行总时间
    auto durationeu = std::chrono::duration_cast<std::chrono::milliseconds>(endeu - starteu);
    std::cout << "eu cluster time "
              << " ints : " << durationeu.count() << " ms\n";
}

// 条件滤波
void ConditionPointCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr incloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr outcloud)
{
    pcl::ConditionAnd<pcl::PointXYZRGBNormal>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZRGBNormal>());
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGBNormal>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGBNormal>("x", pcl::ComparisonOps::GT, -140)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGBNormal>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGBNormal>("x", pcl::ComparisonOps::LT, 140)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGBNormal>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGBNormal>("y", pcl::ComparisonOps::GT, -140)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGBNormal>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGBNormal>("y", pcl::ComparisonOps::LT, 100)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGBNormal>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGBNormal>("z", pcl::ComparisonOps::GT, 0)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZRGBNormal>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGBNormal>("z", pcl::ComparisonOps::LT, 700)));
    // 创建条件滤波器，设置滤除后依旧保持点云的有组织性（保持顺序）
    pcl::ConditionalRemoval<pcl::PointXYZRGBNormal> condrem;
    condrem.setCondition(range_cond);
    condrem.setInputCloud(incloud);
    // condrem.setKeepOrganized(true);
    // 开始滤波
    condrem.filter(*outcloud);
}

// 点云获取边界
void BoundaryCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr incloud,
                   pcl::PointCloud<pcl::Normal>::Ptr innormals,
                   pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr outcloud)
{
    if (!incloud->empty())
    {
        pcl::PointCloud<pcl::Boundary>::Ptr boundaries(new pcl::PointCloud<pcl::Boundary>);              // 声明一个boundary类指针，作为返回值
        boundaries->resize(incloud->size());                                                             // 初始化大小
        pcl::BoundaryEstimation<pcl::PointXYZRGBNormal, pcl::Normal, pcl::Boundary> boundary_estimation; // 声明一个BoundaryEstimation类
        boundary_estimation.setInputCloud(incloud);                                                      // 设置输入点云
        boundary_estimation.setInputNormals(innormals);                                                  // 设置输入法线
        pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr kdtree_ptr(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
        boundary_estimation.setSearchMethod(kdtree_ptr);   // 设置搜寻k近邻的方式
        boundary_estimation.setKSearch(30);                // 设置k近邻数量
        boundary_estimation.setAngleThreshold(M_PI * 0.6); // 设置角度阈值，大于阈值为边界
        boundary_estimation.compute(*boundaries);          // 计算点云边界，结果保存在boundaries中

        for (size_t i = 0; i < incloud->size(); i++)
        {
            if (boundaries->points[i].boundary_point != 0)
            {
                incloud->points[i].r = 255;
                incloud->points[i].g = 0;
                incloud->points[i].b = 0;
                outcloud->push_back(incloud->points[i]);
            }
        }
    }
}

// 求两平面的交线
void calcLine(pcl::ModelCoefficients::Ptr coefsOfPlane1,
              pcl::ModelCoefficients::Ptr coefsOfPlane2,
              pcl::ModelCoefficients::Ptr coefsOfLine)
{
    // 方向向量n=n1×n2=(b1*c2-c1*b2,c1*a2-a1*c2,a1*b2-b1*a2)
    pcl::ModelCoefficients temcoefs;
    double a1, b1, c1, d1, a2, b2, c2, d2;
    double tempy, tempz;
    a1 = coefsOfPlane1->values[0];
    b1 = coefsOfPlane1->values[1];
    c1 = coefsOfPlane1->values[2];
    d1 = coefsOfPlane1->values[3];
    a2 = coefsOfPlane2->values[0];
    b2 = coefsOfPlane2->values[1];
    c2 = coefsOfPlane2->values[2];
    d2 = coefsOfPlane2->values[3];
    tempz = -(d1 / b1 - d2 / b2) / (c1 / b1 - c2 / b2);
    tempy = (-c1 / b1) * tempz - d1 / b1;
    coefsOfLine->values.push_back(0.0);
    coefsOfLine->values.push_back(tempy);
    coefsOfLine->values.push_back(tempz);
    coefsOfLine->values.push_back(b1 * c2 - c1 * b2);
    coefsOfLine->values.push_back(c1 * a2 - a1 * c2);
    coefsOfLine->values.push_back(a1 * b2 - b1 * a2);
}

void ProjCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr incloud,
               pcl::ModelCoefficients::Ptr coefficients,
               pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr outcloud)
{
    pcl::ProjectInliers<pcl::PointXYZRGBNormal> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(incloud);
    proj.setModelCoefficients(coefficients);
    proj.filter(*outcloud);
}

void find_WeldingSeam(Clouds inPlanes1, Clouds inPlanes2,
                      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr outcloud,
                      double cloudmaxz, double cloudminz)
{
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_line(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_line(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_line1(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::ModelCoefficients::Ptr coefficients_line(new pcl::ModelCoefficients());

    if (!inPlanes1.bound->empty() && !inPlanes1.isbase && !inPlanes2.isbase)
    {
        double m1 = inPlanes2.coefficients->values[0] * inPlanes1.bound->points[0].x +
                    inPlanes2.coefficients->values[1] * inPlanes1.bound->points[0].y +
                    inPlanes2.coefficients->values[2] * inPlanes1.bound->points[0].z +
                    inPlanes2.coefficients->values[3];
        // std::cout << "m1====================: " << m1 << std::endl;
        int result1 = (m1 > 0) ? 1 : -1;
        for (int j = 0; j < inPlanes1.bound->points.size(); j++)
        {
            double m11 = inPlanes2.coefficients->values[0] * inPlanes1.bound->points[j].x +
                         inPlanes2.coefficients->values[1] * inPlanes1.bound->points[j].y +
                         inPlanes2.coefficients->values[2] * inPlanes1.bound->points[j].z +
                         inPlanes2.coefficients->values[3];
            // std::cout << "m11: " << m11 << std::endl;
            if (result1 == 1)
            {
                if (m11 > 0)
                {
                    continue;
                }
                else
                {
                    inPlanes1.isbase = true;
                    // std::cout << "m11---------------------------: " << m11 << std::endl;
                }
            }
            else
            {
                if (m11 < 0)
                {
                    continue;
                }
                else
                {
                    inPlanes1.isbase = true;
                    // std::cout << "m11---------------------------: " << m11 << std::endl;
                }
            }
        }
    }
    if (!inPlanes2.bound->empty() && !inPlanes1.isbase && !inPlanes2.isbase)
    {
        double m2 = inPlanes1.coefficients->values[0] * inPlanes2.bound->points[0].x +
                    inPlanes1.coefficients->values[1] * inPlanes2.bound->points[0].y +
                    inPlanes1.coefficients->values[2] * inPlanes2.bound->points[0].z +
                    inPlanes1.coefficients->values[3];
        // std::cout << "m2====================: " << m2 << std::endl;
        int result2 = (m2 > 0) ? 1 : -1;
        for (int j = 0; j < inPlanes2.bound->points.size(); j++)
        {
            double m22 = inPlanes1.coefficients->values[0] * inPlanes2.bound->points[j].x +
                         inPlanes1.coefficients->values[1] * inPlanes2.bound->points[j].y +
                         inPlanes1.coefficients->values[2] * inPlanes2.bound->points[j].z +
                         inPlanes1.coefficients->values[3];
            // std::cout << "m22: " << m22 << std::endl;
            if (result2 == 1)
            {
                if (m22 > 0)
                {
                    continue;
                }
                else
                {
                    inPlanes2.isbase = true;
                    // std::cout << "m22---------------------: " << m22 << std::endl;
                }
            }
            else
            {
                if (m22 < 0)
                {
                    continue;
                }
                else
                {
                    inPlanes2.isbase = true;
                    // std::cout << "m22---------------------: " << m22 << std::endl;
                }
            }
        }
    }
    std::cout << "inPlanes1.isbase : " << inPlanes1.isbase << std::endl;
    std::cout << "inPlanes2.isbase : " << inPlanes2.isbase << std::endl;
    if (!inPlanes1.isbase && !inPlanes2.isbase)
    {
        inPlanes1.isbase = true;
    }

    pcl::PointXYZRGBNormal minbound1_tf; // 用于存放三个轴的最小值
    pcl::PointXYZRGBNormal maxbound1_tf; // 用于存放三个轴的最大值
    pcl::PointXYZRGBNormal minbound2_tf; // 用于存放三个轴的最小值
    pcl::PointXYZRGBNormal maxbound2_tf; // 用于存放三个轴的最大值
    // 判断两个平面哪一个作为焊缝底平面,并得出非焊缝线
    if (inPlanes1.isbase)
    {
        for (int j = 0; j < inPlanes2.bound_tf->points.size(); j++)
        {
            double dis = pcl::pointToPlaneDistance(inPlanes2.bound_tf->points[j],
                                                   inPlanes1.coefficients_tf->values[0], inPlanes1.coefficients_tf->values[1],
                                                   inPlanes1.coefficients_tf->values[2], inPlanes1.coefficients_tf->values[3]);
            // std::cout << "dis: " << dis << std::endl;
            if (dis < 10)
                cloud_line->push_back(inPlanes2.bound_tf->points[j]);
        }
        pcl::getMinMax3D(*inPlanes2.bound_tf, minbound1_tf, maxbound1_tf);
        if (inPlanes1.bound_tf->size() <= 0)
        {
            pcl::getMinMax3D(*inPlanes1.cloud_tf, minbound2_tf, maxbound2_tf);
        }
        else
        {
            pcl::getMinMax3D(*inPlanes1.bound_tf, minbound2_tf, maxbound2_tf);
        }
        std::cout << "result1: " << inPlanes1.isbase << std::endl;
    }
    if (inPlanes2.isbase)
    {
        for (int j = 0; j < inPlanes1.bound_tf->points.size(); j++)
        {
            double dis = pcl::pointToPlaneDistance(inPlanes1.bound_tf->points[j],
                                                   inPlanes2.coefficients_tf->values[0], inPlanes2.coefficients_tf->values[1],
                                                   inPlanes2.coefficients_tf->values[2], inPlanes2.coefficients_tf->values[3]);
            // std::cout << "dis1: " << dis1 << std::endl;
            if (dis < 10)
                cloud_line->push_back(inPlanes1.bound_tf->points[j]);
        }
        pcl::getMinMax3D(*inPlanes1.bound_tf, minbound1_tf, maxbound1_tf);
        if (inPlanes2.bound_tf->size() <= 0)
        {
            pcl::getMinMax3D(*inPlanes2.cloud_tf, minbound2_tf, maxbound2_tf);
        }
        else
        {
            pcl::getMinMax3D(*inPlanes2.bound_tf, minbound2_tf, maxbound2_tf);
        }
        std::cout << "result: " << inPlanes2.isbase << std::endl;
    }
    bool isten = false;
    bool isten1 = false;
    int istens1 = 0;
    for (int j = 0; j < inPlanes1.cloud_tf->points.size(); j++)
    {
        for (int k = 0; k < inPlanes2.cloud_tf->points.size(); k++)
        {
            if (pcl::euclideanDistance(inPlanes1.cloud_tf->points[j], inPlanes2.cloud_tf->points[k]) < 50)
            {
                std::cout << "pcl::euclideanDistance(inPlanes1.cloud_tf->points[j], inPlanes2.cloud_tf->points[k]): "
                          << pcl::euclideanDistance(inPlanes1.cloud_tf->points[j], inPlanes2.cloud_tf->points[k]) << std::endl;
                istens1 += 1;
                break;
            }
        }
        if (istens1 >= 5)
        {
            isten1 = true;
            break;
        }
    }

    bool isten2 = false;
    int istens2 = 0;
    for (int j = 0; j < inPlanes2.cloud_tf->points.size(); j++)
    {
        for (int k = 0; k < inPlanes1.cloud_tf->points.size(); k++)
        {
            if (pcl::euclideanDistance(inPlanes2.cloud_tf->points[j], inPlanes1.cloud_tf->points[k]) < 50)
            {
                std::cout << "pcl::euclideanDistance(inPlanes2.cloud_tf->points[j], inPlanes1.cloud_tf->points[k]): "
                          << pcl::euclideanDistance(inPlanes2.cloud_tf->points[j], inPlanes1.cloud_tf->points[k]) << std::endl;
                istens2 += 1;
                break;
            }
        }
        if (istens2 >= 5)
        {
            isten2 = true;
            break;
        }
    }
    if (isten1 && isten2)
    {
        isten = true;
    }
    std::cout << "isten : " << isten << std::endl;
    std::cout << "cloud_line size : " << cloud_line->size() << std::endl;

    Eigen::Vector3f offset1(0.0, 0.0, 0.0);
    Eigen::Vector3f eulerAngle1(0.675, 2.965, -0.51);

    Eigen::AngleAxisf rollAngle1(Eigen::AngleAxisf(eulerAngle1(2), Eigen::Vector3f::UnitX()));
    Eigen::AngleAxisf pitchAngle1(Eigen::AngleAxisf(eulerAngle1(1), Eigen::Vector3f::UnitY()));
    Eigen::AngleAxisf yawAngle1(Eigen::AngleAxisf(eulerAngle1(0), Eigen::Vector3f::UnitZ()));

    Eigen::Quaternionf quaternion1;
    quaternion1 = yawAngle1 * pitchAngle1 * rollAngle1;

    pcl::ModelCoefficients::Ptr coefficients_projline_tf(new pcl::ModelCoefficients());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_outline(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_outline_tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    if (cloud_line->size() > 5 && isten)
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_line0(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_line0(new pcl::PointCloud<pcl::Normal>);
        pcl::ModelCoefficients::Ptr coefficients_line0(new pcl::ModelCoefficients());
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_line_tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_line0_tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_line_tf(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_line1_tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::ModelCoefficients::Ptr coefficients_line1_tf(new pcl::ModelCoefficients());

        ComputeNormals(cloud_line, cloud_normals_line0);
        SegLine(cloud_line, cloud_normals_line0, cloud_line1, coefficients_line1_tf, false);

        // pcl::transformPointCloud(*cloud_line0, *cloud_line_tf, offset1, quaternion1); // 得到世界坐标系下的点云

        pcl::PointXYZRGBNormal minline_tf; // 用于存放三个轴的最小值
        pcl::PointXYZRGBNormal maxline_tf; // 用于存放三个轴的最大值
        pcl::getMinMax3D(*cloud_line_tf, minline_tf, maxline_tf);

        double minboundz = (abs(minbound1_tf.z) > abs(minbound2_tf.z)) ? minbound2_tf.z : minbound1_tf.z;
        double maxboundz = (abs(maxbound1_tf.z) > abs(maxbound2_tf.z)) ? maxbound1_tf.z : maxbound2_tf.z;
        if (abs(maxboundz - cloudmaxz) > 10 && abs(maxbound2_tf.z - maxbound1_tf.z) > 5)
        {
            maxboundz = cloudmaxz;
        }
        std::cout << "bound1_tf max min : " << maxbound1_tf.z << minbound1_tf.z << std::endl;
        std::cout << "bound2_tf max min : " << maxbound2_tf.z << minbound2_tf.z << std::endl;
        std::cout << "bound2_tf max min : " << maxboundz << minboundz << std::endl;
        // double boundtf_dz = maxbound1_tf.z - minbound1_tf.z;
        // double linetf_dz = maxline_tf.z - minline_tf.z;
        // std::cout << "line_tf max min : " << maxline_tf.z << minline_tf.z << std::endl;

        // if (boundtf_dz - linetf_dz > 3)
        // {
        //     std::cout << "boundtf_dz - linetf_dz 的差大于3 , 补齐直线: " << boundtf_dz - linetf_dz << std::endl;
        //     ComputeNormals(cloud_line_tf, cloud_normals_line_tf);
        //     SegLine(cloud_line_tf, cloud_normals_line_tf, cloud_line1, coefficients_line1_tf, false);
        //     std::cout << "coefficients_line1_tf z:" << coefficients_line1_tf->values[5] << std::endl;
        //     if (abs(coefficients_line1_tf->values[5]) > 0.5)
        //     {
        //         for (double ik = minboundz + 1; ik < maxboundz; ++ik)
        //         {
        //             // std::cout << "ik: " << ik << std::endl;
        //             pcl::PointXYZRGBNormal pt;
        //             pt.z = ik;
        //             double dd = double(ik - coefficients_line1_tf->values[2]) / coefficients_line1_tf->values[5];
        //             // std::cout << "dd: " << dd << std::endl;
        //             pt.x = dd * coefficients_line1_tf->values[3] + coefficients_line1_tf->values[0];
        //             pt.y = dd * coefficients_line1_tf->values[4] + coefficients_line1_tf->values[1];
        //             cloud_line1->push_back(pt);
        //         }
        //         // pcl::transformPointCloud(*cloud_line1_tf, *cloud_line1, offset1, quaternion1.inverse());
        //     }
        // }
        // else if (linetf_dz - boundtf_dz > 3)
        // {
        //     std::cout << "linetf_dz - boundtf_dz的差大于3 , 补齐直线: " << linetf_dz - boundtf_dz << std::endl;
        //     ComputeNormals(cloud_line_tf, cloud_normals_line_tf);
        //     SegLine(cloud_line_tf, cloud_normals_line_tf, cloud_line0_tf, coefficients_line1_tf, false);
        //     std::cout << "coefficients_line1_tf z:" << coefficients_line1_tf->values[5] << std::endl;
        //     if (abs(coefficients_line1_tf->values[5]) > 0.5)
        //     {
        //         for (double ik = minboundz + 1; ik < maxboundz; ++ik)
        //         {
        //             // std::cout << "ik: " << ik << std::endl;
        //             pcl::PointXYZRGBNormal pt;
        //             pt.z = ik;
        //             double dd = double(ik - coefficients_line1_tf->values[2]) / coefficients_line1_tf->values[5];
        //             // std::cout << "dd: " << dd << std::endl;
        //             pt.x = dd * coefficients_line1_tf->values[3] + coefficients_line1_tf->values[0];
        //             pt.y = dd * coefficients_line1_tf->values[4] + coefficients_line1_tf->values[1];
        //             cloud_line1->push_back(pt);
        //         }
        //         // pcl::transformPointCloud(*cloud_line1_tf, *cloud_line1, offset1, quaternion1.inverse());
        //     }
        // }
        // else
        // {
        //     std::cout << "boundtf_dz - linetf_dz 的差小于3 , 不补齐直线；" << std::endl;
        //     // 非底平面焊缝边界拟合直线
        //     pcl::copyPointCloud(*cloud_line_tf, *cloud_line1);
        //     coefficients_line1_tf = coefficients_line0;
        // }
        std::cout << "cloud_line1 size : " << cloud_line1->size() << std::endl;

        std::vector<double> hfdx;
        if (inPlanes1.isbase)
        {
            for (int j = 0; j < cloud_line1->points.size(); j++)
            {
                double dis_hf = pcl::pointToPlaneDistance(cloud_line1->points[j],
                                                          inPlanes1.coefficients_tf->values[0], inPlanes1.coefficients_tf->values[1],
                                                          inPlanes1.coefficients_tf->values[2], inPlanes1.coefficients_tf->values[3]);
                hfdx.push_back(dis_hf);
            }
        }
        if (inPlanes2.isbase)
        {
            for (int j = 0; j < cloud_line1->points.size(); j++)
            {
                double dis_hf = pcl::pointToPlaneDistance(cloud_line1->points[j],
                                                          inPlanes2.coefficients_tf->values[0], inPlanes2.coefficients_tf->values[1],
                                                          inPlanes2.coefficients_tf->values[2], inPlanes2.coefficients_tf->values[3]);
                hfdx.push_back(dis_hf);
            }
        }

        double sumhfdx = std::accumulate(std::begin(hfdx), std::end(hfdx), 0.0);
        double meanhfdx = sumhfdx / hfdx.size(); // 均值

        if (meanhfdx > 0.75)
        {
            for (int i = 0; i < hfdx.size(); i++)
            {
                hfdx[i] -= 0.75;
            }
        }

        sumhfdx = std::accumulate(std::begin(hfdx), std::end(hfdx), 0.0);
        meanhfdx = sumhfdx / hfdx.size();
        double hfk = meanhfdx;
        std::cout << "焊缝平均距离纠正后： " << meanhfdx << std::endl;

        calcLine(inPlanes1.coefficients_tf, inPlanes2.coefficients_tf, coefficients_projline_tf);
        std::cout << "两平面交线模型系数: " << std::endl;
        for (auto i : coefficients_projline_tf->values)
        {
            std::cout << i << std::endl;
        }
        std::cout << "贯穿面边界线模型系数: " << std::endl;
        for (auto i : coefficients_line1_tf->values)
        {
            std::cout << i << std::endl;
        }
        if (abs(coefficients_projline_tf->values[5]) > 0.4)
        {
            for (double ik = minboundz; ik < maxboundz; ++ik)
            {
                // std::cout << "ik: " << ik << std::endl;
                pcl::PointXYZRGBNormal pt0;
                pt0.z = ik;
                double dd0 = double(ik - coefficients_projline_tf->values[2]) / coefficients_projline_tf->values[5];
                // std::cout << "dd: " << dd << std::endl;
                pt0.x = dd0 * coefficients_projline_tf->values[3] + coefficients_projline_tf->values[0];
                pt0.y = dd0 * coefficients_projline_tf->values[4] + coefficients_projline_tf->values[1];
                cloud_outline->push_back(pt0);
            }
            std::cout << "cloud_outline size: " << cloud_outline->size() << std::endl;

            if (cloud_outline->size() > 5)
            {
                pcl::PointXYZRGBNormal minoutline;
                pcl::PointXYZRGBNormal maxoutline;
                pcl::getMinMax3D(*cloud_outline, minoutline, maxoutline);

                for (int j = 0; j < cloud_outline->size(); j++)
                {
                    if (cloud_outline->points[j].z == maxoutline.z || cloud_outline->points[j].z == minoutline.z)
                    {
                        cloud_outline->points[j].r = 0;
                        cloud_outline->points[j].b = 255;
                        cloud_outline->points[j].g = 0;
                        cloud_outline->points.at(j).normal_x = (inPlanes1.mnormal[0] + inPlanes2.mnormal[0]) / 2;
                        cloud_outline->points.at(j).normal_y = (inPlanes1.mnormal[1] + inPlanes2.mnormal[1]) / 2;
                        cloud_outline->points.at(j).normal_z = (inPlanes1.mnormal[2] + inPlanes2.mnormal[2]) / 2;
                        cloud_outline_tf->push_back(cloud_outline->points[j]);
                    }
                    if (cloud_outline_tf->size() > 2)
                    {
                        break;
                    }
                }
                int len = maxoutline.z - minoutline.z;
                int size0;
                if (len >= 0 && len < 40)
                {
                    size0 = 0;
                }
                else
                {
                    size0 = (int(len - 40) / 20) + 1;
                }
                int size = cloud_outline->points.size() / (size0 + 1);
                for (int j = 1; j < cloud_outline->size(); j++)
                {
                    if (cloud_outline_tf->size() < size0 + 2)
                    {
                        if (j % size == 0)
                        {
                            cloud_outline->points[j].r = 0;
                            cloud_outline->points[j].b = 0;
                            cloud_outline->points[j].g = 255;
                            cloud_outline_tf->push_back(cloud_outline->points[j]);
                        }
                    }
                }
                for (auto i : cloud_outline_tf->points)
                {
                    std::cout << "x: " << i.x << " y: " << i.y << " z: " << i.z << std::endl;
                }
                pcl::transformPointCloud(*cloud_outline_tf, *outcloud, offset1, quaternion1.inverse()); // 得到世界坐标系下的点云

                std::cout << "outcloud line size: " << outcloud->size() << std::endl;
            }
        }
        else
        {
            std::cout << "两平面交线的线方向小于0.4 " << std::endl;
            if (maxbound2_tf.z > maxbound1_tf.z)
            {
                if (abs(maxbound1_tf.y - minbound1_tf.y) > abs(maxbound1_tf.x - minbound1_tf.x))
                {
                    std::cout << "maxbound1_tf.x - minbound1_tf.x: " << maxbound1_tf.x << " - " << minbound1_tf.x << " = " << maxbound1_tf.x - minbound1_tf.x << std::endl;
                    for (double ik = minbound1_tf.x; ik < maxbound1_tf.x; ++ik)
                    {
                        // std::cout << "ik: " << ik << std::endl;
                        pcl::PointXYZRGBNormal pt0;
                        pt0.x = ik;
                        double dd0 = double(ik - coefficients_projline_tf->values[0]) / coefficients_projline_tf->values[3];
                        // std::cout << "dd: " << dd << std::endl;
                        pt0.z = dd0 * coefficients_projline_tf->values[5] + coefficients_projline_tf->values[2];
                        pt0.y = dd0 * coefficients_projline_tf->values[4] + coefficients_projline_tf->values[1];
                        cloud_outline->push_back(pt0);
                    }
                    std::cout << "cloud_outline size: " << cloud_outline->size() << std::endl;

                    if (cloud_outline->size() > 5)
                    {
                        pcl::PointXYZRGBNormal minoutline;
                        pcl::PointXYZRGBNormal maxoutline;
                        pcl::getMinMax3D(*cloud_outline, minoutline, maxoutline);

                        for (int j = 0; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline->points[j].x == maxoutline.x || cloud_outline->points[j].x == minoutline.x)
                            {
                                cloud_outline->points[j].r = 0;
                                cloud_outline->points[j].b = 255;
                                cloud_outline->points[j].g = 0;
                                cloud_outline->points.at(j).normal_x = (inPlanes1.mnormal[0] + inPlanes2.mnormal[0]) / 2;
                                cloud_outline->points.at(j).normal_y = (inPlanes1.mnormal[1] + inPlanes2.mnormal[1]) / 2;
                                cloud_outline->points.at(j).normal_z = (inPlanes1.mnormal[2] + inPlanes2.mnormal[2]) / 2;
                                cloud_outline_tf->push_back(cloud_outline->points[j]);
                            }
                            if (cloud_outline_tf->size() > 2)
                            {
                                break;
                            }
                        }
                        int len = maxoutline.x - minoutline.x;
                        int size0;
                        if (len >= 0 && len < 40)
                        {
                            size0 = 0;
                        }
                        else
                        {
                            size0 = (int(len - 40) / 20) + 1;
                        }
                        int size = cloud_outline->points.size() / (size0 + 1);
                        for (int j = 1; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline_tf->size() < size0 + 2)
                            {
                                if (j % size == 0)
                                {
                                    cloud_outline->points[j].r = 0;
                                    cloud_outline->points[j].b = 0;
                                    cloud_outline->points[j].g = 255;
                                    cloud_outline_tf->push_back(cloud_outline->points[j]);
                                }
                            }
                        }
                        for (auto i : cloud_outline_tf->points)
                        {
                            std::cout << "x: " << i.x << " y: " << i.y << " z: " << i.z << std::endl;
                        }
                        pcl::transformPointCloud(*cloud_outline_tf, *outcloud, offset1, quaternion1.inverse()); // 得到世界坐标系下的点云

                        std::cout << "outcloud line size: " << outcloud->size() << std::endl;
                    }
                }
                else
                {
                    std::cout << "maxbound1_tf.y - minbound1_tf.y: " << maxbound1_tf.y << " - " << minbound1_tf.y << " = " << maxbound1_tf.y - minbound1_tf.y << std::endl;
                    for (double ik = minbound1_tf.y; ik < maxbound1_tf.y; ++ik)
                    {
                        // std::cout << "ik: " << ik << std::endl;
                        pcl::PointXYZRGBNormal pt0;
                        pt0.y = ik;
                        double dd0 = double(ik - coefficients_projline_tf->values[1]) / coefficients_projline_tf->values[4];
                        // std::cout << "dd: " << dd << std::endl;
                        pt0.z = dd0 * coefficients_projline_tf->values[5] + coefficients_projline_tf->values[2];
                        pt0.x = dd0 * coefficients_projline_tf->values[3] + coefficients_projline_tf->values[0];
                        cloud_outline->push_back(pt0);
                    }
                    std::cout << "cloud_outline size: " << cloud_outline->size() << std::endl;

                    if (cloud_outline->size() > 5)
                    {
                        pcl::PointXYZRGBNormal minoutline;
                        pcl::PointXYZRGBNormal maxoutline;
                        pcl::getMinMax3D(*cloud_outline, minoutline, maxoutline);

                        for (int j = 0; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline->points[j].y == maxoutline.y || cloud_outline->points[j].y == minoutline.y)
                            {
                                cloud_outline->points[j].r = 0;
                                cloud_outline->points[j].b = 255;
                                cloud_outline->points[j].g = 0;
                                cloud_outline->points.at(j).normal_x = (inPlanes1.mnormal[0] + inPlanes2.mnormal[0]) / 2;
                                cloud_outline->points.at(j).normal_y = (inPlanes1.mnormal[1] + inPlanes2.mnormal[1]) / 2;
                                cloud_outline->points.at(j).normal_z = (inPlanes1.mnormal[2] + inPlanes2.mnormal[2]) / 2;
                                cloud_outline_tf->push_back(cloud_outline->points[j]);
                            }
                            if (cloud_outline_tf->size() > 2)
                            {
                                break;
                            }
                        }
                        int len = maxoutline.y - minoutline.y;
                        int size0;
                        if (len >= 0 && len < 40)
                        {
                            size0 = 0;
                        }
                        else
                        {
                            size0 = (int(len - 40) / 20) + 1;
                        }
                        int size = cloud_outline->points.size() / (size0 + 1);
                        for (int j = 1; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline_tf->size() < size0 + 2)
                            {
                                if (j % size == 0)
                                {
                                    cloud_outline->points[j].r = 0;
                                    cloud_outline->points[j].b = 0;
                                    cloud_outline->points[j].g = 255;
                                    cloud_outline_tf->push_back(cloud_outline->points[j]);
                                }
                            }
                        }
                        for (auto i : cloud_outline_tf->points)
                        {
                            std::cout << "x: " << i.x << " y: " << i.y << " z: " << i.z << std::endl;
                        }
                        pcl::transformPointCloud(*cloud_outline_tf, *outcloud, offset1, quaternion1.inverse()); // 得到世界坐标系下的点云

                        std::cout << "outcloud line size: " << outcloud->size() << std::endl;
                    }
                }
            }
            else
            {
                if (abs(maxbound2_tf.y - minbound2_tf.y) > abs(maxbound2_tf.x - minbound2_tf.x))
                {
                    std::cout << "maxbound2_tf.x - minbound2_tf.x: " << maxbound2_tf.x << " - " << minbound2_tf.x << " = " << maxbound2_tf.x - minbound2_tf.x << std::endl;
                    for (double ik = minbound2_tf.x; ik < maxbound2_tf.x; ++ik)
                    {
                        // std::cout << "ik: " << ik << std::endl;
                        pcl::PointXYZRGBNormal pt0;
                        pt0.x = ik;
                        double dd0 = double(ik - coefficients_projline_tf->values[0]) / coefficients_projline_tf->values[3];
                        // std::cout << "dd: " << dd << std::endl;
                        pt0.z = dd0 * coefficients_projline_tf->values[5] + coefficients_projline_tf->values[2];
                        pt0.y = dd0 * coefficients_projline_tf->values[4] + coefficients_projline_tf->values[1];
                        cloud_outline->push_back(pt0);
                    }
                    std::cout << "cloud_outline size: " << cloud_outline->size() << std::endl;

                    if (cloud_outline->size() > 5)
                    {
                        pcl::PointXYZRGBNormal minoutline;
                        pcl::PointXYZRGBNormal maxoutline;
                        pcl::getMinMax3D(*cloud_outline, minoutline, maxoutline);

                        for (int j = 0; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline->points[j].x == maxoutline.x || cloud_outline->points[j].x == minoutline.x)
                            {
                                cloud_outline->points[j].r = 0;
                                cloud_outline->points[j].b = 255;
                                cloud_outline->points[j].g = 0;
                                cloud_outline->points.at(j).normal_x = (inPlanes1.mnormal[0] + inPlanes2.mnormal[0]) / 2;
                                cloud_outline->points.at(j).normal_y = (inPlanes1.mnormal[1] + inPlanes2.mnormal[1]) / 2;
                                cloud_outline->points.at(j).normal_z = (inPlanes1.mnormal[2] + inPlanes2.mnormal[2]) / 2;
                                cloud_outline_tf->push_back(cloud_outline->points[j]);
                            }
                            if (cloud_outline_tf->size() > 2)
                            {
                                break;
                            }
                        }
                        int len = maxoutline.x - minoutline.x;
                        int size0;
                        if (len >= 0 && len < 40)
                        {
                            size0 = 0;
                        }
                        else
                        {
                            size0 = (int(len - 40) / 20) + 1;
                        }
                        int size = cloud_outline->points.size() / (size0 + 1);
                        for (int j = 1; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline_tf->size() < size0 + 2)
                            {
                                if (j % size == 0)
                                {
                                    cloud_outline->points[j].r = 0;
                                    cloud_outline->points[j].b = 0;
                                    cloud_outline->points[j].g = 255;
                                    cloud_outline_tf->push_back(cloud_outline->points[j]);
                                }
                            }
                        }
                        for (auto i : cloud_outline_tf->points)
                        {
                            std::cout << "x: " << i.x << " y: " << i.y << " z: " << i.z << std::endl;
                        }
                        pcl::transformPointCloud(*cloud_outline_tf, *outcloud, offset1, quaternion1.inverse()); // 得到世界坐标系下的点云

                        std::cout << "outcloud line size: " << outcloud->size() << std::endl;
                    }
                }
                else
                {
                    std::cout << "maxbound2_tf.y - minbound2_tf.y: " << maxbound2_tf.y << " - " << minbound2_tf.y << " = " << maxbound2_tf.y - minbound2_tf.y << std::endl;
                    for (double ik = minbound2_tf.y; ik < maxbound2_tf.y; ++ik)
                    {
                        // std::cout << "ik: " << ik << std::endl;
                        pcl::PointXYZRGBNormal pt0;
                        pt0.y = ik;
                        double dd0 = double(ik - coefficients_projline_tf->values[1]) / coefficients_projline_tf->values[4];
                        // std::cout << "dd: " << dd << std::endl;
                        pt0.z = dd0 * coefficients_projline_tf->values[5] + coefficients_projline_tf->values[2];
                        pt0.x = dd0 * coefficients_projline_tf->values[3] + coefficients_projline_tf->values[0];
                        cloud_outline->push_back(pt0);
                    }
                    std::cout << "cloud_outline size: " << cloud_outline->size() << std::endl;

                    if (cloud_outline->size() > 5)
                    {
                        pcl::PointXYZRGBNormal minoutline;
                        pcl::PointXYZRGBNormal maxoutline;
                        pcl::getMinMax3D(*cloud_outline, minoutline, maxoutline);

                        for (int j = 0; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline->points[j].y == maxoutline.y || cloud_outline->points[j].y == minoutline.y)
                            {
                                cloud_outline->points[j].r = 0;
                                cloud_outline->points[j].b = 255;
                                cloud_outline->points[j].g = 0;
                                cloud_outline->points.at(j).normal_x = (inPlanes1.mnormal[0] + inPlanes2.mnormal[0]) / 2;
                                cloud_outline->points.at(j).normal_y = (inPlanes1.mnormal[1] + inPlanes2.mnormal[1]) / 2;
                                cloud_outline->points.at(j).normal_z = (inPlanes1.mnormal[2] + inPlanes2.mnormal[2]) / 2;
                                cloud_outline_tf->push_back(cloud_outline->points[j]);
                            }
                            if (cloud_outline_tf->size() > 2)
                            {
                                break;
                            }
                        }
                        int len = maxoutline.y - minoutline.y;
                        int size0;
                        if (len >= 0 && len < 40)
                        {
                            size0 = 0;
                        }
                        else
                        {
                            size0 = (int(len - 40) / 20) + 1;
                        }
                        int size = cloud_outline->points.size() / (size0 + 1);
                        for (int j = 1; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline_tf->size() < size0 + 2)
                            {
                                if (j % size == 0)
                                {
                                    cloud_outline->points[j].r = 0;
                                    cloud_outline->points[j].b = 0;
                                    cloud_outline->points[j].g = 255;
                                    cloud_outline_tf->push_back(cloud_outline->points[j]);
                                }
                            }
                        }
                        for (auto i : cloud_outline_tf->points)
                        {
                            std::cout << "x: " << i.x << " y: " << i.y << " z: " << i.z << std::endl;
                        }
                        pcl::transformPointCloud(*cloud_outline_tf, *outcloud, offset1, quaternion1.inverse()); // 得到世界坐标系下的点云

                        std::cout << "outcloud line size: " << outcloud->size() << std::endl;
                    }
                }
            }
        }
        std::cout << "cloud_outline size: " << cloud_outline->size() << std::endl;
    }
    else if (isten)
    {
        std::cout << "进入直接求两面交线得出焊缝!!!" << std::endl;
        pcl::PointXYZRGBNormal minbound1_tf; // 用于存放三个轴的最小值
        pcl::PointXYZRGBNormal maxbound1_tf; // 用于存放三个轴的最大值
        pcl::getMinMax3D(*inPlanes1.cloud_tf, minbound1_tf, maxbound1_tf);
        pcl::PointXYZRGBNormal minbound2_tf; // 用于存放三个轴的最小值
        pcl::PointXYZRGBNormal maxbound2_tf; // 用于存放三个轴的最大值
        pcl::getMinMax3D(*inPlanes2.cloud_tf, minbound2_tf, maxbound2_tf);
        double minboundz = (abs(minbound1_tf.z) > abs(minbound2_tf.z)) ? minbound1_tf.z : minbound2_tf.z;
        double maxboundz = (abs(maxbound1_tf.z) > abs(maxbound2_tf.z)) ? maxbound1_tf.z : maxbound2_tf.z;
        std::cout << "bound_tf max min : " << maxboundz << minboundz << std::endl;
        if (abs(maxboundz - cloudmaxz) > 10 && abs(maxbound2_tf.z - maxbound1_tf.z) > 5)
        {
            maxboundz = cloudmaxz;
        }
        if (abs(minboundz - cloudminz) > 3 && abs(minbound2_tf.z - minbound1_tf.z) > 5)
        {
            if (abs(minboundz - cloudminz) < 10)
            {
                minboundz = cloudminz;
            }
            else
            {
                minboundz = (abs(minbound1_tf.z) > abs(minbound2_tf.z)) ? minbound2_tf.z : minbound1_tf.z;
            }
        }
        std::cout << "bound1_tf max min : " << maxbound1_tf.z << minbound1_tf.z << std::endl;
        std::cout << "bound2_tf max min : " << maxbound2_tf.z << minbound2_tf.z << std::endl;
        std::cout << "bound_tf max min : " << maxboundz << minboundz << std::endl;

        calcLine(inPlanes1.coefficients_tf, inPlanes2.coefficients_tf, coefficients_projline_tf);
        std::cout << "两平面交线模型系数: " << std::endl;
        for (auto i : coefficients_projline_tf->values)
        {
            std::cout << i << std::endl;
        }
        if (abs(coefficients_projline_tf->values[5]) > 0.4)
        {
            pcl::PointXYZRGBNormal pt01;
            pt01.z = minboundz;
            double dd01 = double(minboundz - coefficients_projline_tf->values[2]) / coefficients_projline_tf->values[5];
            pt01.x = dd01 * coefficients_projline_tf->values[3] + coefficients_projline_tf->values[0];
            pt01.y = dd01 * coefficients_projline_tf->values[4] + coefficients_projline_tf->values[1];

            pcl::PointXYZRGBNormal pt02;
            pt02.z = maxboundz;
            double dd02 = double(maxboundz - coefficients_projline_tf->values[2]) / coefficients_projline_tf->values[5];
            pt02.x = dd02 * coefficients_projline_tf->values[3] + coefficients_projline_tf->values[0];
            pt02.y = dd02 * coefficients_projline_tf->values[4] + coefficients_projline_tf->values[1];

            std::cout << "pt01.x - pt02.x: " << abs(pt01.x - pt02.x) << std::endl;
            std::cout << "pt01.y - pt02.y: " << abs(pt01.y - pt02.y) << std::endl;
            if (abs(pt01.x - pt02.x) <= 2 && abs(pt01.y - pt02.y) <= 2)
            {
                for (double ik = minboundz; ik < maxboundz; ++ik)
                {
                    // std::cout << "ik: " << ik << std::endl;
                    pcl::PointXYZRGBNormal pt0;
                    pt0.z = ik;
                    double dd0 = double(ik - coefficients_projline_tf->values[2]) / coefficients_projline_tf->values[5];
                    // std::cout << "dd0: " << dd0 << std::endl;
                    pt0.x = dd0 * coefficients_projline_tf->values[3] + coefficients_projline_tf->values[0];
                    pt0.y = dd0 * coefficients_projline_tf->values[4] + coefficients_projline_tf->values[1];
                    cloud_outline->push_back(pt0);
                }
            }
            else
            {
                for (double ik = minboundz; ik < maxboundz; ++ik)
                {
                    pcl::PointXYZRGBNormal pt0;
                    pt0.z = ik;
                    pt0.x = pt01.x;
                    pt0.y = pt01.y;
                    cloud_outline->push_back(pt0);
                }
            }
            std::cout << "cloud_outline size: " << cloud_outline->size() << std::endl;

            if (cloud_outline->size() > 5)
            {
                pcl::PointXYZRGBNormal minoutline;
                pcl::PointXYZRGBNormal maxoutline;
                pcl::getMinMax3D(*cloud_outline, minoutline, maxoutline);

                for (int j = 0; j < cloud_outline->size(); j++)
                {
                    if (cloud_outline->points[j].z == maxoutline.z || cloud_outline->points[j].z == minoutline.z)
                    {
                        cloud_outline->points[j].r = 0;
                        cloud_outline->points[j].b = 255;
                        cloud_outline->points[j].g = 0;
                        cloud_outline->points.at(j).normal_x = (inPlanes1.mnormal[0] + inPlanes2.mnormal[0]) / 2;
                        cloud_outline->points.at(j).normal_y = (inPlanes1.mnormal[1] + inPlanes2.mnormal[1]) / 2;
                        cloud_outline->points.at(j).normal_z = (inPlanes1.mnormal[2] + inPlanes2.mnormal[2]) / 2;
                        cloud_outline_tf->push_back(cloud_outline->points[j]);
                    }
                    if (cloud_outline_tf->size() > 2)
                    {
                        break;
                    }
                }
                int len = maxoutline.z - minoutline.z;
                int size0;
                if (len >= 0 && len < 40)
                {
                    size0 = 0;
                }
                else
                {
                    size0 = (int(len - 40) / 20) + 1;
                }
                int size = cloud_outline->points.size() / (size0 + 1);
                for (int j = 1; j < cloud_outline->size(); j++)
                {
                    if (cloud_outline_tf->size() < size0 + 2)
                    {
                        if (j % size == 0)
                        {
                            cloud_outline->points[j].r = 0;
                            cloud_outline->points[j].b = 0;
                            cloud_outline->points[j].g = 255;
                            cloud_outline_tf->push_back(cloud_outline->points[j]);
                        }
                    }
                }
                for (auto i : cloud_outline_tf->points)
                {
                    std::cout << "x: " << i.x << " y: " << i.y << " z: " << i.z << std::endl;
                }
                pcl::transformPointCloud(*cloud_outline_tf, *outcloud, offset1, quaternion1.inverse()); // 得到世界坐标系下的点云

                std::cout << "outcloud line size: " << outcloud->size() << std::endl;
            }
        }
        else
        {
            std::cout << "两平面交线的线方向小于0.4 " << std::endl;
            if (maxbound2_tf.z > maxbound1_tf.z)
            {
                if (abs(maxbound1_tf.y - minbound1_tf.y) > abs(maxbound1_tf.x - minbound1_tf.x))
                {
                    std::cout << "maxbound1_tf.x - minbound1_tf.x: " << maxbound1_tf.x << " - " << minbound1_tf.x << " = " << maxbound1_tf.x - minbound1_tf.x << std::endl;
                    for (double ik = minbound1_tf.x; ik < maxbound1_tf.x; ++ik)
                    {
                        // std::cout << "ik: " << ik << std::endl;
                        pcl::PointXYZRGBNormal pt0;
                        pt0.x = ik;
                        double dd0 = double(ik - coefficients_projline_tf->values[0]) / coefficients_projline_tf->values[3];
                        // std::cout << "dd: " << dd << std::endl;
                        pt0.z = dd0 * coefficients_projline_tf->values[5] + coefficients_projline_tf->values[2];
                        pt0.y = dd0 * coefficients_projline_tf->values[4] + coefficients_projline_tf->values[1];
                        cloud_outline->push_back(pt0);
                    }
                    std::cout << "cloud_outline size: " << cloud_outline->size() << std::endl;

                    if (cloud_outline->size() > 5)
                    {
                        pcl::PointXYZRGBNormal minoutline;
                        pcl::PointXYZRGBNormal maxoutline;
                        pcl::getMinMax3D(*cloud_outline, minoutline, maxoutline);

                        for (int j = 0; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline->points[j].x == maxoutline.x || cloud_outline->points[j].x == minoutline.x)
                            {
                                cloud_outline->points[j].r = 0;
                                cloud_outline->points[j].b = 255;
                                cloud_outline->points[j].g = 0;
                                cloud_outline->points.at(j).normal_x = (inPlanes1.mnormal[0] + inPlanes2.mnormal[0]) / 2;
                                cloud_outline->points.at(j).normal_y = (inPlanes1.mnormal[1] + inPlanes2.mnormal[1]) / 2;
                                cloud_outline->points.at(j).normal_z = (inPlanes1.mnormal[2] + inPlanes2.mnormal[2]) / 2;
                                cloud_outline_tf->push_back(cloud_outline->points[j]);
                            }
                            if (cloud_outline_tf->size() > 2)
                            {
                                break;
                            }
                        }
                        int len = maxoutline.x - minoutline.x;
                        int size0;
                        if (len >= 0 && len < 40)
                        {
                            size0 = 0;
                        }
                        else
                        {
                            size0 = (int(len - 40) / 20) + 1;
                        }
                        int size = cloud_outline->points.size() / (size0 + 1);
                        for (int j = 1; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline_tf->size() < size0 + 2)
                            {
                                if (j % size == 0)
                                {
                                    cloud_outline->points[j].r = 0;
                                    cloud_outline->points[j].b = 0;
                                    cloud_outline->points[j].g = 255;
                                    cloud_outline_tf->push_back(cloud_outline->points[j]);
                                }
                            }
                        }
                        for (auto i : cloud_outline_tf->points)
                        {
                            std::cout << "x: " << i.x << " y: " << i.y << " z: " << i.z << std::endl;
                        }
                        pcl::transformPointCloud(*cloud_outline_tf, *outcloud, offset1, quaternion1.inverse()); // 得到世界坐标系下的点云

                        std::cout << "outcloud line size: " << outcloud->size() << std::endl;
                    }
                }
                else
                {
                    std::cout << "maxbound1_tf.y - minbound1_tf.y: " << maxbound1_tf.y << " - " << minbound1_tf.y << " = " << maxbound1_tf.y - minbound1_tf.y << std::endl;
                    for (double ik = minbound1_tf.y; ik < maxbound1_tf.y; ++ik)
                    {
                        // std::cout << "ik: " << ik << std::endl;
                        pcl::PointXYZRGBNormal pt0;
                        pt0.y = ik;
                        double dd0 = double(ik - coefficients_projline_tf->values[1]) / coefficients_projline_tf->values[4];
                        // std::cout << "dd: " << dd << std::endl;
                        pt0.z = dd0 * coefficients_projline_tf->values[5] + coefficients_projline_tf->values[2];
                        pt0.x = dd0 * coefficients_projline_tf->values[3] + coefficients_projline_tf->values[0];
                        cloud_outline->push_back(pt0);
                    }
                    std::cout << "cloud_outline size: " << cloud_outline->size() << std::endl;

                    if (cloud_outline->size() > 5)
                    {
                        pcl::PointXYZRGBNormal minoutline;
                        pcl::PointXYZRGBNormal maxoutline;
                        pcl::getMinMax3D(*cloud_outline, minoutline, maxoutline);

                        for (int j = 0; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline->points[j].y == maxoutline.y || cloud_outline->points[j].y == minoutline.y)
                            {
                                cloud_outline->points[j].r = 0;
                                cloud_outline->points[j].b = 255;
                                cloud_outline->points[j].g = 0;
                                cloud_outline->points.at(j).normal_x = (inPlanes1.mnormal[0] + inPlanes2.mnormal[0]) / 2;
                                cloud_outline->points.at(j).normal_y = (inPlanes1.mnormal[1] + inPlanes2.mnormal[1]) / 2;
                                cloud_outline->points.at(j).normal_z = (inPlanes1.mnormal[2] + inPlanes2.mnormal[2]) / 2;
                                cloud_outline_tf->push_back(cloud_outline->points[j]);
                            }
                            if (cloud_outline_tf->size() > 2)
                            {
                                break;
                            }
                        }
                        int len = maxoutline.y - minoutline.y;
                        int size0;
                        if (len >= 0 && len < 40)
                        {
                            size0 = 0;
                        }
                        else
                        {
                            size0 = (int(len - 40) / 20) + 1;
                        }
                        int size = cloud_outline->points.size() / (size0 + 1);
                        for (int j = 1; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline_tf->size() < size0 + 2)
                            {
                                if (j % size == 0)
                                {
                                    cloud_outline->points[j].r = 0;
                                    cloud_outline->points[j].b = 0;
                                    cloud_outline->points[j].g = 255;
                                    cloud_outline_tf->push_back(cloud_outline->points[j]);
                                }
                            }
                        }
                        for (auto i : cloud_outline_tf->points)
                        {
                            std::cout << "x: " << i.x << " y: " << i.y << " z: " << i.z << std::endl;
                        }
                        pcl::transformPointCloud(*cloud_outline_tf, *outcloud, offset1, quaternion1.inverse()); // 得到世界坐标系下的点云

                        std::cout << "outcloud line size: " << outcloud->size() << std::endl;
                    }
                }
            }
            else
            {
                if (abs(maxbound2_tf.y - minbound2_tf.y) > abs(maxbound2_tf.x - minbound2_tf.x))
                {
                    std::cout << "maxbound2_tf.x - minbound2_tf.x: " << maxbound2_tf.x << " - " << minbound2_tf.x << " = " << maxbound2_tf.x - minbound2_tf.x << std::endl;
                    for (double ik = minbound2_tf.x; ik < maxbound2_tf.x; ++ik)
                    {
                        // std::cout << "ik: " << ik << std::endl;
                        pcl::PointXYZRGBNormal pt0;
                        pt0.x = ik;
                        double dd0 = double(ik - coefficients_projline_tf->values[0]) / coefficients_projline_tf->values[3];
                        // std::cout << "dd: " << dd << std::endl;
                        pt0.z = dd0 * coefficients_projline_tf->values[5] + coefficients_projline_tf->values[2];
                        pt0.y = dd0 * coefficients_projline_tf->values[4] + coefficients_projline_tf->values[1];
                        cloud_outline->push_back(pt0);
                    }
                    std::cout << "cloud_outline size: " << cloud_outline->size() << std::endl;

                    if (cloud_outline->size() > 5)
                    {
                        pcl::PointXYZRGBNormal minoutline;
                        pcl::PointXYZRGBNormal maxoutline;
                        pcl::getMinMax3D(*cloud_outline, minoutline, maxoutline);

                        for (int j = 0; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline->points[j].x == maxoutline.x || cloud_outline->points[j].x == minoutline.x)
                            {
                                cloud_outline->points[j].r = 0;
                                cloud_outline->points[j].b = 255;
                                cloud_outline->points[j].g = 0;
                                cloud_outline->points.at(j).normal_x = (inPlanes1.mnormal[0] + inPlanes2.mnormal[0]) / 2;
                                cloud_outline->points.at(j).normal_y = (inPlanes1.mnormal[1] + inPlanes2.mnormal[1]) / 2;
                                cloud_outline->points.at(j).normal_z = (inPlanes1.mnormal[2] + inPlanes2.mnormal[2]) / 2;
                                cloud_outline_tf->push_back(cloud_outline->points[j]);
                            }
                            if (cloud_outline_tf->size() > 2)
                            {
                                break;
                            }
                        }
                        int len = maxoutline.x - minoutline.x;
                        int size0;
                        if (len >= 0 && len < 40)
                        {
                            size0 = 0;
                        }
                        else
                        {
                            size0 = (int(len - 40) / 20) + 1;
                        }
                        int size = cloud_outline->points.size() / (size0 + 1);
                        for (int j = 1; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline_tf->size() < size0 + 2)
                            {
                                if (j % size == 0)
                                {
                                    cloud_outline->points[j].r = 0;
                                    cloud_outline->points[j].b = 0;
                                    cloud_outline->points[j].g = 255;
                                    cloud_outline_tf->push_back(cloud_outline->points[j]);
                                }
                            }
                        }
                        for (auto i : cloud_outline_tf->points)
                        {
                            std::cout << "x: " << i.x << " y: " << i.y << " z: " << i.z << std::endl;
                        }
                        pcl::transformPointCloud(*cloud_outline_tf, *outcloud, offset1, quaternion1.inverse()); // 得到世界坐标系下的点云

                        std::cout << "outcloud line size: " << outcloud->size() << std::endl;
                    }
                }
                else
                {
                    std::cout << "maxbound2_tf.y - minbound2_tf.y: " << maxbound2_tf.y << " - " << minbound2_tf.y << " = " << maxbound2_tf.y - minbound2_tf.y << std::endl;
                    for (double ik = minbound2_tf.y; ik < maxbound2_tf.y; ++ik)
                    {
                        // std::cout << "ik: " << ik << std::endl;
                        pcl::PointXYZRGBNormal pt0;
                        pt0.y = ik;
                        double dd0 = double(ik - coefficients_projline_tf->values[1]) / coefficients_projline_tf->values[4];
                        // std::cout << "dd: " << dd << std::endl;
                        pt0.z = dd0 * coefficients_projline_tf->values[5] + coefficients_projline_tf->values[2];
                        pt0.x = dd0 * coefficients_projline_tf->values[3] + coefficients_projline_tf->values[0];
                        cloud_outline->push_back(pt0);
                    }
                    std::cout << "cloud_outline size: " << cloud_outline->size() << std::endl;

                    if (cloud_outline->size() > 5)
                    {
                        pcl::PointXYZRGBNormal minoutline;
                        pcl::PointXYZRGBNormal maxoutline;
                        pcl::getMinMax3D(*cloud_outline, minoutline, maxoutline);

                        for (int j = 0; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline->points[j].y == maxoutline.y || cloud_outline->points[j].y == minoutline.y)
                            {
                                cloud_outline->points[j].r = 0;
                                cloud_outline->points[j].b = 255;
                                cloud_outline->points[j].g = 0;
                                cloud_outline->points.at(j).normal_x = (inPlanes1.mnormal[0] + inPlanes2.mnormal[0]) / 2;
                                cloud_outline->points.at(j).normal_y = (inPlanes1.mnormal[1] + inPlanes2.mnormal[1]) / 2;
                                cloud_outline->points.at(j).normal_z = (inPlanes1.mnormal[2] + inPlanes2.mnormal[2]) / 2;
                                cloud_outline_tf->push_back(cloud_outline->points[j]);
                            }
                            if (cloud_outline_tf->size() > 2)
                            {
                                break;
                            }
                        }
                        int len = maxoutline.z - minoutline.z;
                        int size0;
                        if (len >= 0 && len < 40)
                        {
                            size0 = 0;
                        }
                        else
                        {
                            size0 = (int(len - 40) / 20) + 1;
                        }
                        int size = cloud_outline->points.size() / (size0 + 1);
                        for (int j = 1; j < cloud_outline->size(); j++)
                        {
                            if (cloud_outline_tf->size() < size0 + 2)
                            {
                                if (j % size == 0)
                                {
                                    cloud_outline->points[j].r = 0;
                                    cloud_outline->points[j].b = 0;
                                    cloud_outline->points[j].g = 255;
                                    cloud_outline_tf->push_back(cloud_outline->points[j]);
                                }
                            }
                        }
                        for (auto i : cloud_outline_tf->points)
                        {
                            std::cout << "x: " << i.x << " y: " << i.y << " z: " << i.z << std::endl;
                        }
                        pcl::transformPointCloud(*cloud_outline_tf, *outcloud, offset1, quaternion1.inverse()); // 得到世界坐标系下的点云

                        std::cout << "outcloud line size: " << outcloud->size() << std::endl;
                    }
                }
            }
        }
    }
}

std::vector<double> MeanNormal(pcl::PointCloud<pcl::Normal>::Ptr innormals)
{
    std::cout << "innormals size: " << innormals->size() << std::endl;
    std::vector<double> nx;
    std::vector<double> ny;
    std::vector<double> nz;
    std::vector<double> out_meannormal;
    if (innormals->size() > 0)
    {
        for (int i = 0; i < innormals->points.size(); i++)
        {
            if (i < 3000)
            {
                if (!isnan(innormals->points[i].normal_x) &&
                    !isnan(innormals->points[i].normal_y) &&
                    !isnan(innormals->points[i].normal_z))
                {
                    nx.push_back(double(innormals->points[i].normal_x));
                    ny.push_back(double(innormals->points[i].normal_y));
                    nz.push_back(double(innormals->points[i].normal_z));
                }
                else
                {
                    continue;
                }
            }
        }

        // double sumnx = std::accumulate(nx.begin(), nx.end(), 0.0);
        double sumnx = SumVector(nx);
        std::cout << "sumnx: " << sumnx << std::endl;
        double meannx = sumnx / nx.size(); // 均值
        std::cout << "meannx: " << meannx << std::endl;
        out_meannormal.push_back(meannx);

        double sumny = std::accumulate(ny.begin(), ny.end(), double(0.0));

        std::cout << "sumny: " << sumny << std::endl;

        double meanny = sumny / ny.size(); // 均值
        std::cout << "meanny: " << meanny << std::endl;
        out_meannormal.push_back(meanny);

        double sumnz = std::accumulate(nz.begin(), nz.end(), double(0.0));
        std::cout << "sumnz: " << sumnz << std::endl;
        double meannz = sumnz / nz.size(); // 均值
        std::cout << "meannz: " << meannz << std::endl;
        out_meannormal.push_back(meannz);
    }

    return out_meannormal;
}

double getAngleTwoPoint(Eigen::Vector3d &v1, Eigen::Vector3d &v2)
{
    double radian_angle = atan2(v1.cross(v2).norm(), v1.transpose() * v2);
    return radian_angle;
}

bool isLevel(Clouds inPlanes1, Clouds inPlanes2)
{
    if (abs(inPlanes1.mnormal[2]) > 0.6 && abs(inPlanes2.mnormal[2]) < 0.6)
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1(new pcl::PointCloud<pcl::Normal>);
        ComputeNormals(inPlanes1.cloud_tf, cloud_normals1);
        std::vector<double> plane_normal1;
        plane_normal1 = MeanNormal(cloud_normals1);
        Eigen::Vector3d pv11(plane_normal1[0], plane_normal1[1], plane_normal1[2]);
        for (int i = 0; i < int(inPlanes1.cloud_tf->size()); ++i)
        {
            Eigen::Vector3d pv12(cloud_normals1->points[i].normal_x,
                                 cloud_normals1->points[i].normal_y,
                                 cloud_normals1->points[i].normal_z);
            double angle = getAngleTwoPoint(pv11, pv12);
            if (abs(angle) < 0.4)
            {
                cloud1->push_back(inPlanes1.cloud_tf->points[i]);
            }
        }

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
        ComputeNormals(inPlanes2.cloud_tf, cloud_normals2);
        std::vector<double> plane_normal2;
        plane_normal2 = MeanNormal(cloud_normals2);
        Eigen::Vector3d pv21(plane_normal2[0], plane_normal2[1], plane_normal2[2]);
        for (int i = 0; i < int(inPlanes2.cloud_tf->size()); ++i)
        {
            Eigen::Vector3d pv22(cloud_normals2->points[i].normal_x,
                                 cloud_normals2->points[i].normal_y,
                                 cloud_normals2->points[i].normal_z);
            double angle = getAngleTwoPoint(pv21, pv22);
            if (abs(angle) < 0.4)
            {
                cloud2->push_back(inPlanes2.cloud_tf->points[i]);
            }
        }

        pcl::PointXYZRGBNormal min1; // 用于存放三个轴的最小值
        pcl::PointXYZRGBNormal max1; // 用于存放三个轴的最大值
        pcl::getMinMax3D(*cloud1, min1, max1);
        pcl::PointXYZRGBNormal min2; // 用于存放三个轴的最小值
        pcl::PointXYZRGBNormal max2; // 用于存放三个轴的最大值
        pcl::getMinMax3D(*cloud2, min2, max2);
        if (min1.z <= min2.z)
        {
            std::cout << "水平判断" << std::endl;
            std::cout << "1" << std::endl;
            std::cout << "max1.z: " << max1.z << std::endl;
            std::cout << "min1.z: " << min1.z << std::endl;
            std::cout << "max2.z: " << max2.z << std::endl;
            std::cout << "min2.z: " << min2.z << std::endl;
            if ((min2.z - min1.z) >= 0 && (min2.z - min1.z) < 10)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }

    else if (abs(inPlanes1.mnormal[2]) < 0.6 && abs(inPlanes2.mnormal[2]) > 0.6)
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1(new pcl::PointCloud<pcl::Normal>);
        ComputeNormals(inPlanes1.cloud_tf, cloud_normals1);
        std::vector<double> plane_normal1;
        plane_normal1 = MeanNormal(cloud_normals1);
        Eigen::Vector3d pv11(plane_normal1[0], plane_normal1[1], plane_normal1[2]);
        for (int i = 0; i < int(inPlanes1.cloud_tf->size()); ++i)
        {
            Eigen::Vector3d pv12(cloud_normals1->points[i].normal_x,
                                 cloud_normals1->points[i].normal_y,
                                 cloud_normals1->points[i].normal_z);
            double angle = getAngleTwoPoint(pv11, pv12);
            if (abs(angle) < 0.4)
            {
                cloud1->push_back(inPlanes1.cloud_tf->points[i]);
            }
        }

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
        ComputeNormals(inPlanes2.cloud_tf, cloud_normals2);
        std::vector<double> plane_normal2;
        plane_normal2 = MeanNormal(cloud_normals2);
        Eigen::Vector3d pv21(plane_normal2[0], plane_normal2[1], plane_normal2[2]);
        for (int i = 0; i < int(inPlanes2.cloud_tf->size()); ++i)
        {
            Eigen::Vector3d pv22(cloud_normals2->points[i].normal_x,
                                 cloud_normals2->points[i].normal_y,
                                 cloud_normals2->points[i].normal_z);
            double angle = getAngleTwoPoint(pv21, pv22);
            if (abs(angle) < 0.4)
            {
                cloud2->push_back(inPlanes2.cloud_tf->points[i]);
            }
        }

        pcl::PointXYZRGBNormal min1; // 用于存放三个轴的最小值
        pcl::PointXYZRGBNormal max1; // 用于存放三个轴的最大值
        pcl::getMinMax3D(*cloud1, min1, max1);
        pcl::PointXYZRGBNormal min2; // 用于存放三个轴的最小值
        pcl::PointXYZRGBNormal max2; // 用于存放三个轴的最大值
        pcl::getMinMax3D(*cloud2, min2, max2);
        if (min1.z >= min2.z)
        {
            std::cout << "水平判断" << std::endl;
            std::cout << "2" << std::endl;
            std::cout << "max1.z: " << max1.z << std::endl;
            std::cout << "min1.z: " << min1.z << std::endl;
            std::cout << "max2.z: " << max2.z << std::endl;
            std::cout << "min2.z: " << min2.z << std::endl;
            if ((min1.z - min2.z) >= 0 && (min1.z - min2.z) < 10)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
    return false;
}

bool isVertical(Clouds inPlanes1, Clouds inPlanes2)
{
    if (abs(inPlanes1.mnormal[2]) < 0.6 && abs(inPlanes2.mnormal[2]) < 0.6)
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1(new pcl::PointCloud<pcl::Normal>);
        ComputeNormals(inPlanes1.cloud_tf, cloud_normals1);
        std::vector<double> plane_normal1;
        plane_normal1 = MeanNormal(cloud_normals1);
        Eigen::Vector3d pv11(plane_normal1[0], plane_normal1[1], plane_normal1[2]);
        for (int i = 0; i < int(inPlanes1.cloud_tf->size()); ++i)
        {
            Eigen::Vector3d pv12(cloud_normals1->points[i].normal_x,
                                 cloud_normals1->points[i].normal_y,
                                 cloud_normals1->points[i].normal_z);
            double angle = getAngleTwoPoint(pv11, pv12);
            if (abs(angle) < 0.4)
            {
                cloud1->push_back(inPlanes1.cloud_tf->points[i]);
            }
        }

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
        ComputeNormals(inPlanes2.cloud_tf, cloud_normals2);
        std::vector<double> plane_normal2;
        plane_normal2 = MeanNormal(cloud_normals2);
        Eigen::Vector3d pv21(plane_normal2[0], plane_normal2[1], plane_normal2[2]);
        for (int i = 0; i < int(inPlanes2.cloud_tf->size()); ++i)
        {
            Eigen::Vector3d pv22(cloud_normals2->points[i].normal_x,
                                 cloud_normals2->points[i].normal_y,
                                 cloud_normals2->points[i].normal_z);
            double angle = getAngleTwoPoint(pv21, pv22);
            if (abs(angle) < 0.5)
            {
                cloud2->push_back(inPlanes2.cloud_tf->points[i]);
            }
        }

        pcl::PointXYZRGBNormal min1; // 用于存放三个轴的最小值
        pcl::PointXYZRGBNormal max1; // 用于存放三个轴的最大值
        pcl::getMinMax3D(*cloud1, min1, max1);
        pcl::PointXYZRGBNormal min2; // 用于存放三个轴的最小值
        pcl::PointXYZRGBNormal max2; // 用于存放三个轴的最大值
        pcl::getMinMax3D(*cloud2, min2, max2);

        std::cout << "竖直判断" << std::endl;
        std::cout << "max1.z: " << max1.z << "max1.x: " << max1.x << "max1.y: " << max1.y << std::endl;
        std::cout << "min1.z: " << min1.z << "min1.x: " << min1.x << "min1.y: " << min1.y << std::endl;
        std::cout << "max2.z: " << max2.z << "max2.x: " << max2.x << "max2.y: " << max2.y << std::endl;
        std::cout << "min2.z: " << min2.z << "min2.x: " << min2.x << "min2.y: " << min2.y << std::endl;
        if (min2.z >= min1.z)
        {
            if (min2.z - min1.z >= 0 && min2.z - min1.z <= 25)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else if (min1.z > min2.z)
        {
            if (min1.z - min2.z >= 0 && min1.z - min2.z <= 25)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }

    return false;
}

bool isCollision(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                 pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                 pcl::PointXYZRGBNormal point)
{
    auto startisco = std::chrono::system_clock::now();

    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree0(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    tree0->setInputCloud(cloud1);

    std::vector<pcl::PointIndices> cluster_indices0;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGBNormal> ec0;
    ec0.setClusterTolerance(4); // 4mm
    ec0.setMinClusterSize(1000);
    ec0.setMaxClusterSize(1000000);
    ec0.setSearchMethod(tree0);
    ec0.setInputCloud(cloud1);
    // 聚类抽取结果保存在一个数组中，数组中每个元素代表抽取的一个组件点云的下标
    ec0.extract(cluster_indices0);
    int size1 = int(cluster_indices0.size());

    double hqdx, hqdy, hqdz;
    hqdx = point.x - 0.29939;
    hqdy = point.y - (-8.2333) - 60;
    hqdz = 555.0 - point.z;
    std::cout << "hqdx: " << hqdx << "  hqdy: " << hqdy << "  hqdz: " << hqdz << std::endl;

    // for (int i = 0; i < int(cloud2->size()); ++i)
    // {
    //     cloud2->points[i].x = double(cloud2->points[i].x) + hqdx;
    //     cloud2->points[i].y = double(cloud2->points[i].y) + hqdy;
    //     cloud2->points[i].z = double(cloud2->points[i].z) + hqdz;
    // }
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud3(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    for (int i = 0; i < int(cloud2->size()); ++i)
    {
        pcl::PointXYZRGBNormal pt;
        pt.x = double(cloud2->points[i].x) + hqdx;
        pt.y = double(cloud2->points[i].y) + hqdy;
        pt.z = double(cloud2->points[i].z) + hqdz;
        cloud3->push_back(pt);
    }
    std::cout << "cloud3 size: " << cloud3->size() << std::endl;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_a(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    *cloud_a = *cloud1 + *cloud3;

    // pcl::io::savePCDFileASCII ("/home/liukun/cameraS/er92_a.pcd", *cloud_a);

    std::cout << "cloud_a size: " << cloud_a->size() << std::endl;

    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    tree->setInputCloud(cloud_a);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGBNormal> ec;
    ec.setClusterTolerance(4); // 4mm
    ec.setMinClusterSize(1000);
    ec.setMaxClusterSize(1000000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_a);
    // 聚类抽取结果保存在一个数组中，数组中每个元素代表抽取的一个组件点云的下标
    ec.extract(cluster_indices);

    std::cout << "焊枪碰撞检测cluster_indices0 size : " << size1 << std::endl;
    std::cout << "焊枪碰撞检测cluster_indices size : " << cluster_indices.size() << std::endl;
    auto endisco = std::chrono::system_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // 得到程序运行总时间
    auto durationeu = std::chrono::duration_cast<std::chrono::milliseconds>(endisco - startisco);
    std::cout << "isCollision time "
              << " ints : " << durationeu.count() << " ms\n";

    if (cluster_indices.size() == size1)
    {
        return false;
    }
    else if (cluster_indices.size() == size1 + 1)
    {
        return true;
    }

    return false;
}

bool cmp2(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
          pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2)
{
    return abs(cloud1->points[0].x) + abs(cloud1->points[0].y) < abs(cloud2->points[0].x) + abs(cloud2->points[0].y);
}

void Welding_line(std::vector<Clouds> Planes,
                  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_Voxel1,
                  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_hq,
                  std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &wending_line,
                  double cloudmaxz, double cloudminz)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> WeldingLines_V;

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> WeldingLines_L;
    auto start1 = std::chrono::system_clock::now();
    for (int i = 0; i < Planes.size(); i++)
    {
        for (int j = i + 1; j < Planes.size(); j++)
        {
            Eigen::Vector3f bound1coe;
            bound1coe << Planes[i].mnormal[0], Planes[i].mnormal[1], Planes[i].mnormal[2];
            Eigen::Vector3f bound2coe;
            bound2coe << Planes[j].mnormal[0], Planes[j].mnormal[1], Planes[j].mnormal[2];
            double angle = pcl::getAngle3D(bound1coe, bound2coe, true);

            std::cout << "angle: " << angle << std::endl;
            if (abs(int(angle)) > 5)
            {
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_isresult(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                // 查找竖直焊缝
                if (isVertical(Planes[i], Planes[j]))
                {
                    std::cout << "竖直焊缝边界夹角为: " << angle << std::endl;

                    find_WeldingSeam(Planes[i], Planes[j], cloud_isresult, cloudmaxz, cloudminz);
                    if (!cloud_isresult->empty())
                    {
                        if (isCollision(cloud_Voxel1, cloud_hq, cloud_isresult->points[0]))
                        {
                            WeldingLines_V.push_back(cloud_isresult);
                        }
                        else
                        {
                            std::cout << "焊枪不能到达焊缝！！！！！" << std::endl;
                        }
                    }
                }
            }
            else
            {
                std::cout << "两边界夹角为0 两平面平行" << std::endl;
            }
        }
    }
    std::cout << "找到" << WeldingLines_V.size() << "条竖直条焊缝" << std::endl;

    for (int i = 0; i < Planes.size(); i++)
    {
        for (int j = i + 1; j < Planes.size(); j++)
        {
            Eigen::Vector3f bound1coe;
            bound1coe << Planes[i].mnormal[0], Planes[i].mnormal[1], Planes[i].mnormal[2];
            Eigen::Vector3f bound2coe;
            bound2coe << Planes[j].mnormal[0], Planes[j].mnormal[1], Planes[j].mnormal[2];
            double angle = pcl::getAngle3D(bound1coe, bound2coe, true);

            if (abs(int(angle)) > 80)
            {
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_isresult(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                // 查找水平焊缝
                if (isLevel(Planes[i], Planes[j]))
                {
                    std::cout << "水平焊缝边界夹角为: " << angle << std::endl;
                    find_WeldingSeam(Planes[i], Planes[j], cloud_isresult, cloudmaxz, cloudminz);
                    if (!cloud_isresult->empty())
                    {
                        WeldingLines_L.push_back(cloud_isresult);
                        cloud_isresult.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
                    }
                }
            }
            else
            {
                std::cout << "两边界夹角为0 两平面平行" << std::endl;
            }
        }
    }
    std::cout << "找到" << WeldingLines_L.size() << "条水平条焊缝" << std::endl;

    auto end1 = std::chrono::system_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // 得到程序运行总时间
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "Find shuzhi and shuiping tatol time "
              << " ints : " << duration1.count() << " ms\n";

    sort(WeldingLines_V.begin(), WeldingLines_V.end(), cmp2);

    if (WeldingLines_V.size() > 0)
    {
        for (int i = 0; i < WeldingLines_V.size(); i++)
        {
            bool issave_V = true;
            for (int j = 0; j < WeldingLines_V[i]->size(); ++j)
            {
                std::cout << "WeldingLines_V[i]->points[j]: "
                          << "  x: " << WeldingLines_V[i]->points[j].x << "  y:" << WeldingLines_V[i]->points[j].y << std::endl;
                if (WeldingLines_V[i]->points[j].x >= -60 && WeldingLines_V[i]->points[j].x <= 60 &&
                    WeldingLines_V[i]->points[j].y >= -100 && WeldingLines_V[i]->points[j].y <= 100)
                {
                    if (wending_line.size() > 0)
                    {
                        for (int k = 0; k < wending_line.size(); ++k)
                        {
                            for (int ki = 0; ki < wending_line[k]->size(); ++ki)
                            {
                                if (pcl::euclideanDistance(WeldingLines_V[i]->points[j], wending_line[k]->points[ki]) < 20)
                                {
                                    issave_V = false;
                                }
                            }
                        }
                    }
                }
                else
                {
                    issave_V = false;
                }
            }
            std::cout << "issave_V: " << issave_V << std::endl;
            if (issave_V)
            {
                wending_line.push_back(WeldingLines_V[i]);
                std::cout << "issave_V: " << issave_V << std::endl;
            }
        }
    }

    if (WeldingLines_L.size() > 0)
    {
        for (int i = 0; i < WeldingLines_L.size(); ++i)
        {
            bool issave_L = true;
            for (int j = 0; j < WeldingLines_L[i]->size(); ++j)
            {
                if (wending_line.size() > 0)
                {
                    for (int k = 0; k < wending_line.size(); ++k)
                    {
                        for (int ki = 0; ki < wending_line[k]->size(); ++ki)
                        {
                            if (pcl::euclideanDistance(WeldingLines_L[i]->points[j], wending_line[k]->points[ki]) < 20)
                            {
                                issave_L = false;
                            }
                        }
                    }
                }
            }
            if (issave_L)
            {
                wending_line.push_back(WeldingLines_L[i]);
            }
        }
    }
}

bool cmp(Clouds pt1, Clouds pt2)
{
    return abs(pt1.mnormal[2]) < abs(pt2.mnormal[2]);
}

bool cmp3(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pt1,
          pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pt2)
{
    return abs(pt1->points[0].z) < abs(pt2->points[0].z);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "hf");
    ros::NodeHandle nh;

    ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("pcl_output", 1);
    ros::Publisher pcl_pub_hq = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_hq", 1);
    ros::Publisher pcl_pub_condition = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_condition", 1);
    ros::Publisher pcl_pub_bilateral = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_bilateral", 1);
    ros::Publisher pcl_pub_pass_tf = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_pass", 1);
    ros::Publisher pcl_pub_eucluste = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_eucluste", 1);

    ros::Publisher pcl_pub_bound = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_bound", 1);
    ros::Publisher pcl_pub_bound0 = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_bound0", 1);
    ros::Publisher pcl_pub_bound01 = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_bound01", 1);
    ros::Publisher pcl_pub_bound02 = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_bound02", 1);
    ros::Publisher pcl_pub_bound03 = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_bound03", 1);
    ros::Publisher pcl_pub_bound1 = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_bound1", 1);
    ros::Publisher pcl_pub_bound2 = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_bound2", 1);
    ros::Publisher pcl_pub_bound3 = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_bound3", 1);
    ros::Publisher pcl_pub_bound4 = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_bound4", 1);
    ros::Publisher pcl_pub_bound_tf = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_bound_tf", 1);
    ros::Publisher pcl_pub_bound1_tf = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_bound1_tf", 1);
    ros::Publisher pcl_pub_resultline = nh.advertise<sensor_msgs::PointCloud2>("pcl_output_resultline", 1);

    sensor_msgs::PointCloud2 output;
    sensor_msgs::PointCloud2 output_hq;
    sensor_msgs::PointCloud2 output_condition;
    sensor_msgs::PointCloud2 output_bilateral;
    sensor_msgs::PointCloud2 output_pass;
    sensor_msgs::PointCloud2 output_eucluster;

    sensor_msgs::PointCloud2 output_bound;
    sensor_msgs::PointCloud2 output_bound0;
    sensor_msgs::PointCloud2 output_bound01;
    sensor_msgs::PointCloud2 output_bound02;
    sensor_msgs::PointCloud2 output_bound03;
    sensor_msgs::PointCloud2 output_bound1;
    sensor_msgs::PointCloud2 output_bound2;
    sensor_msgs::PointCloud2 output_bound3;
    sensor_msgs::PointCloud2 output_bound4;
    sensor_msgs::PointCloud2 output_bound_tf;
    sensor_msgs::PointCloud2 output_bound1_tf;
    sensor_msgs::PointCloud2 output_resultline;

    std::vector<Clouds> Planes;

    // Eigen::Vector3f offset(150.0, -250.0, 550.0);
    Eigen::Vector3f offset(0.0, 0.0, 0.0);
    // Eigen::Vector3f eulerAngle(0.905, 2.965, -0.51);
    Eigen::Vector3f eulerAngle(0.675, 2.965, -0.51);

    Eigen::AngleAxisf rollAngle(Eigen::AngleAxisf(eulerAngle(2), Eigen::Vector3f::UnitX()));
    Eigen::AngleAxisf pitchAngle(Eigen::AngleAxisf(eulerAngle(1), Eigen::Vector3f::UnitY()));
    Eigen::AngleAxisf yawAngle(Eigen::AngleAxisf(eulerAngle(0), Eigen::Vector3f::UnitZ()));

    Eigen::Quaternionf quaternion;
    quaternion = yawAngle * pitchAngle * rollAngle;

    // 为了不报错,读取文件用绝对路径(pcd)
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_hq0(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_hq(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    std::string pcdfilename;
    ros::param::param<std::string>("~pcdfilename", pcdfilename, "/home/liukun/cameraS/er122.pcd");
    std::cout << "pcdfile name: " << pcdfilename << std::endl;
    pcl::io::loadPCDFile(pcdfilename, *cloud);
    // 读取ply文件，也可以用 pcl::io::loadPLYFile("/home/liukun/cameraS/1.ply", cloud)
    // pcl::PLYReader reader;
    // reader.read<pcl::PointXYZRGBNormal>(pcdfilename, *cloud);
    std::cout << "cloud points size = " << cloud->size() << std::endl;
    // 读取焊枪点云并转移到工件进枪点
    pcl::io::loadPCDFile("/home/liukun/cameraS/welding_31.pcd", *cloud_hq0);

    Eigen::Vector3f offset0(-60.5, -85.0, 190.0);
    // Eigen::Vector3f offset0(-60.5 - 0.548459, -85.0 - 31.9178, 190.0 - 21.1873);
    Eigen::Vector3f eulerAngle0(3.15, -0.17, 0);

    Eigen::AngleAxisf rollAngle0(Eigen::AngleAxisf(eulerAngle0(2), Eigen::Vector3f::UnitX()));
    Eigen::AngleAxisf pitchAngle0(Eigen::AngleAxisf(eulerAngle0(1), Eigen::Vector3f::UnitY()));
    Eigen::AngleAxisf yawAngle0(Eigen::AngleAxisf(eulerAngle0(0), Eigen::Vector3f::UnitZ()));

    Eigen::Quaternionf quaternion0;
    quaternion0 = yawAngle0 * pitchAngle0 * rollAngle0;

    pcl::transformPointCloud(*cloud_hq0, *cloud_hq, offset0, quaternion0); // 得到世界坐标系下的点云

    auto start = std::chrono::system_clock::now();

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_Voxel(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_Voxel1(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_Condition(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_Condition_tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_all0(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_all1(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_passgrough_tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_Voxel(new pcl::PointCloud<pcl::Normal>);
    VoxelGridPointCloud(cloud, cloud_Voxel, 1);
    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    std::cout << "cloud points size = " << cloud->size() << std::endl;
    std::cout << "cloud_Voxel points size = " << cloud_Voxel->size() << std::endl;
    pcl::toROSMsg(*cloud_Voxel, output);
    output.header.frame_id = "map";

    ConditionPointCloud(cloud_Voxel, cloud_Condition);
    cloud_Voxel.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    std::cout << "cloud_Voxel points size = " << cloud_Voxel->size() << std::endl;
    std::cout << "cloud_Condition points size = " << cloud_Condition->size() << std::endl;
    pcl::toROSMsg(*cloud_Condition, output_condition);
    output_condition.header.frame_id = "map";

    // pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_Bilateral(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    // BilateralCloud(cloud_Condition, cloud_Bilateral);
    // std::cout << "cloud_Bilateral points size = " << cloud_Bilateral->size() << std::endl;
    // pcl::toROSMsg(*cloud_Bilateral, output_bilateral);
    // output_bilateral.header.frame_id = "map";

    // 得到大的聚类后的点云
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_eucluster(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    EuclideanClusterCloud(cloud_Condition, cloud_eucluster, 10000, 0);
    pcl::toROSMsg(*cloud_eucluster, output_eucluster);
    output_eucluster.header.frame_id = "map";

    // 去掉底座平面
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_0(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_1_pass(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_0tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_1tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_segplane0(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_segplane0_other(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::ModelCoefficients::Ptr coefficients_plane0(new pcl::ModelCoefficients());
    ComputeNormals(cloud_eucluster, cloud_normals_Voxel);
    SegPlane(cloud_eucluster, cloud_normals_Voxel, cloud_segplane0, cloud_segplane0_other, coefficients_plane0);
    // VoxelGridPointCloud(cloud_all1, cloud_Voxel1, 3);

    double basez = -(double(coefficients_plane0->values[3]) / coefficients_plane0->values[2]);
    std::cout << "basez: " << basez << std::endl;
    pcl::PointXYZRGBNormal ptbase;
    ptbase.x = 0;
    ptbase.y = 0;
    ptbase.z = basez;
    cloud_0->push_back(ptbase);
    pcl::transformPointCloud(*cloud_0, *cloud_0tf, offset, quaternion);
    double basez_tf = cloud_0tf->points[0].z;
    std::cout << "basez_tf: " << basez_tf << std::endl;
    pcl::transformPointCloud(*cloud_eucluster, *cloud_all0, offset, quaternion);
    PassthroughPointCloud(cloud_all0, cloud_passgrough_tf, int(basez_tf) + 3, -300);
    double cloudminz = int(basez_tf) + 3;
    std::cout << "cloudminz: " << cloudminz << std::endl;
    pcl::transformPointCloud(*cloud_Condition, *cloud_1tf, offset, quaternion);
    PassthroughPointCloud(cloud_1tf, cloud_1_pass, int(basez_tf) + 3, -300);

    pcl::transformPointCloud(*cloud_1_pass, *cloud_1, offset, quaternion.inverse());
    pcl::toROSMsg(*cloud_1, output_pass);
    output_pass.header.frame_id = "map";
    VoxelGridPointCloud(cloud_1, cloud_Voxel1, 3);

    pcl::transformPointCloud(*cloud_passgrough_tf, *cloud_all1, offset, quaternion.inverse());

    int SumCloud_size = cloud_all1->size();
    std::cout << "cloud_all1 points size = " << SumCloud_size << std::endl;
    int MinCloud_size = SumCloud_size;

    int kk = 0;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_segplane_other(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    // int MinMaxCloud_size = SumCloud_size * 0.1; // 最小平面阈值
    int MinMaxCloud_size = 3200; // 最小平面阈值

    cv::Mat a = cv::Mat::zeros(1, 4, CV_64F);

    a.at<double>(0, 2) += 1.0;

    std::cout << "a: " << a << std::endl;

    while (MinCloud_size > MinMaxCloud_size)
    {
        Clouds plane;
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_all(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_all(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_segplane(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients());
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_plane(new pcl::PointCloud<pcl::Normal>);
        std::vector<double> plane_normal;
        if (kk == 0)
        {
            cloud_all = cloud_all1;
            ++kk;
        }
        else
        {
            cloud_all = cloud_segplane_other;
            std::cout << "cloud_all points size = " << cloud_all->size() << std::endl;
            cloud_segplane_other.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            std::cout << "cloud_segplane_other points size = " << cloud_segplane_other->size() << std::endl;
        }
        ComputeNormals(cloud_all, cloud_normals_all);
        SegPlane(cloud_all, cloud_normals_all, cloud_segplane, cloud_segplane_other, coefficients_plane);
        std::cout << "cloud_segplane_other points size = " << cloud_segplane_other->size() << std::endl;
        ComputeNormals(cloud_segplane, cloud_normals_plane);
        plane_normal = MeanNormal(cloud_normals_plane);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcloud1(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_p1(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_p(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bound(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud0_tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud0_tf_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::ModelCoefficients::Ptr coefficients_plane_tf(new pcl::ModelCoefficients());
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_segplane_other1(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bound_tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        EuclideanClusterCloud(cloud_segplane, pcloud, 3000, 1);
        ComputeNormals(pcloud, cloud_normals_p);
        BoundaryCloud(pcloud, cloud_normals_p, bound);
        std::cout << "bound size: " << bound->size() << std::endl;

        pcl::transformPointCloud(*cloud_segplane, *cloud0_tf, offset, quaternion); // 得到世界坐标系下的点云
        ComputeNormals(cloud0_tf, cloud0_tf_normals);
        SegPlane(cloud0_tf, cloud0_tf_normals, cloud_tf, cloud_segplane_other1, coefficients_plane_tf);
        EuclideanClusterCloud(cloud_tf, pcloud1, 3000, 2);
        ComputeNormals(pcloud1, cloud_normals_p1);
        BoundaryCloud(pcloud1, cloud_normals_p1, bound_tf);

        MinCloud_size = cloud_segplane_other->size();
        plane.cloud = cloud_segplane;
        plane.cloud_tf = cloud0_tf;
        plane.bound = bound;
        plane.bound_tf = bound_tf;
        plane.coefficients = coefficients_plane;
        plane.coefficients_tf = coefficients_plane_tf;
        plane.mnormal = plane_normal;
        plane.isbase = false;
        Planes.push_back(plane);
    }

    std::cout << "cloud_segplane_other points size = " << cloud_segplane_other->size() << std::endl;
    std::cout << "Planes: " << Planes.size() << std::endl;

    for (auto i : Planes)
    {
        std::cout << "---------------------" << std::endl;
        std::cout << "---------------------" << std::endl;
        std::cout << "平面模型系数为："
                  << i.coefficients->values[0] << "      "
                  << i.coefficients->values[1] << "      "
                  << i.coefficients->values[2] << std::endl;
        std::cout << i.mnormal.size() << std::endl;
        std::cout << "平面平均法向量为："
                  << i.mnormal[0] << "      "
                  << i.mnormal[1] << "      "
                  << i.mnormal[2] << std::endl;
        std::cout << "+++++++++++++++++++++" << std::endl;
        std::cout << "+++++++++++++++++++++" << std::endl;
    }

    sort(Planes.begin(), Planes.end(), cmp);
    std::cout << "排序后： " << std::endl;
    for (auto i : Planes)
    {
        std::cout << "---------------------" << std::endl;
        std::cout << "---------------------" << std::endl;
        std::cout << "平面模型系数为："
                  << i.coefficients->values[0] << "      "
                  << i.coefficients->values[1] << "      "
                  << i.coefficients->values[2] << std::endl;
        std::cout << i.mnormal.size() << std::endl;
        std::cout << "平面平均法向量为："
                  << i.mnormal[0] << "      "
                  << i.mnormal[1] << "      "
                  << i.mnormal[2] << std::endl;
        std::cout << "+++++++++++++++++++++" << std::endl;
        std::cout << "+++++++++++++++++++++" << std::endl;
    }

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> minmaxz;
    double cloudmaxz;
    for (int i = 0; i < Planes.size(); i++)
    {
        if (abs(Planes[i].mnormal[2]) > 0.6)
        {
            minmaxz.push_back(Planes[i].cloud_tf);
        }
    }
    sort(minmaxz.begin(), minmaxz.end(), cmp3);
    std::cout << "minmaxz size: " << minmaxz.size() << std::endl;

    pcl::PointXYZRGBNormal minzz;  // 用于存放三个轴的最小值
    pcl::PointXYZRGBNormal maxzz;  // 用于存放三个轴的最大值
    pcl::PointXYZRGBNormal minzz1; // 用于存放三个轴的最小值
    pcl::PointXYZRGBNormal maxzz1; // 用于存放三个轴的最大值
    if (minmaxz.size() == 1)
    {
        pcl::getMinMax3D(*minmaxz[0], minzz, maxzz);
        cloudmaxz = double(minzz.z + maxzz.z) / 2;
    }
    else if (minmaxz.size() == 2)
    {
        pcl::getMinMax3D(*minmaxz[0], minzz, maxzz);
        pcl::getMinMax3D(*minmaxz[1], minzz1, maxzz1);
        cloudmaxz = (abs(maxzz.z) > abs(maxzz1.z)) ? (maxzz.z) : (maxzz1.z);
    }
    else
    {
        pcl::getMinMax3D(*minmaxz[0], minzz, maxzz);
        pcl::getMinMax3D(*minmaxz[int(minmaxz.size() - 1)], minzz1, maxzz1);
        cloudmaxz = (abs(maxzz.z) > abs(maxzz1.z)) ? (maxzz.z) : (maxzz1.z);
    }
    std::cout << "cloudmaxz: " << cloudmaxz << std::endl;

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> wending_line;

    Welding_line(Planes, cloud_Voxel1, cloud_hq, wending_line, cloudmaxz, cloudminz);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud100(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    std::cout << "wending_line size: " << wending_line.size() << std::endl;
    for (auto i : wending_line)
    {
        std::cout << "wending_line cloud size: " << i->size() << std::endl;
        *cloud100 = *cloud100 + *i;
        for (auto j : i->points)
        {
            std::cout << "x: " << j.x << " y: " << j.y << " z: " << j.z << std::endl;
        }
    }

    if (wending_line.size() == 0 && cloud_segplane_other->size() > 1000 && Planes.size() < 4)
    {
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals0_other(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_other(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_all0(new pcl::PointCloud<pcl::Normal>);
        std::vector<double> plane_normal0;
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_all(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_all_other(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::ModelCoefficients::Ptr coefficients_all(new pcl::ModelCoefficients());
        ComputeNormals(cloud_Condition, cloud_normals0_other);
        ComputeNormals(cloud_segplane_other, cloud_normals_other);
        SegPlane(cloud_segplane_other, cloud_normals_other, cloud_all, cloud_all_other, coefficients_all);
        ComputeNormals(cloud_all, cloud_normals_all0);
        plane_normal0 = MeanNormal(cloud_normals_all0);
        Eigen::Vector3d pv1(plane_normal0[0], plane_normal0[1], plane_normal0[2]);
        for (int i = 0; i < cloud_Condition->size(); ++i)
        {
            if (!isnan(cloud_normals0_other->points[i].normal_x) &&
                !isnan(cloud_normals0_other->points[i].normal_y) &&
                !isnan(cloud_normals0_other->points[i].normal_z))
            {
                double aa = pcl::pointToPlaneDistance(cloud_Condition->points[i],
                                                      coefficients_all->values[0], coefficients_all->values[1],
                                                      coefficients_all->values[2], coefficients_all->values[3]);
                Eigen::Vector3d pv2(cloud_normals0_other->points[i].normal_x,
                                    cloud_normals0_other->points[i].normal_y,
                                    cloud_normals0_other->points[i].normal_z);
                double angle = getAngleTwoPoint(pv1, pv2);
                if (aa >= -0.5 && aa <= 0.5 && abs(angle) < 0.1)
                {
                    // std::cout << "angle: " << angle << std::endl;
                    cloud_all->push_back(cloud_Condition->points[i]);
                }
            }
        }
        Clouds plane;

        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_all(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_segplane(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_segplane_0(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients());
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_plane(new pcl::PointCloud<pcl::Normal>);
        std::vector<double> plane_normal;
        ComputeNormals(cloud_all, cloud_normals_all);
        SegPlane(cloud_all, cloud_normals_all, cloud_segplane, cloud_segplane_0, coefficients_plane);
        std::cout << "222cloud_segplane points size = " << cloud_segplane->size() << std::endl;
        std::cout << "222cloud_segplane_other points size = " << cloud_segplane_0->size() << std::endl;
        ComputeNormals(cloud_segplane, cloud_normals_plane);
        plane_normal = MeanNormal(cloud_normals_plane);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bound(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud0_tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud0_tf_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_tf_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::ModelCoefficients::Ptr coefficients_plane_tf(new pcl::ModelCoefficients());
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_segplane_other1(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bound_tf(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        BoundaryCloud(cloud_segplane, cloud_normals_plane, bound);
        std::cout << "222bound size: " << bound->size() << std::endl;

        pcl::transformPointCloud(*cloud_segplane, *cloud0_tf, offset, quaternion); // 得到世界坐标系下的点云
        ComputeNormals(cloud0_tf, cloud0_tf_normals);
        SegPlane(cloud0_tf, cloud0_tf_normals, cloud_tf, cloud_segplane_other1, coefficients_plane_tf);
        ComputeNormals(cloud_tf, cloud_tf_normals);
        BoundaryCloud(cloud_tf, cloud_tf_normals, bound_tf);
        std::cout << "222bound_tf size: " << bound_tf->size() << std::endl;

        MinCloud_size = cloud_segplane_other->size();
        plane.cloud = cloud_segplane;
        plane.cloud_tf = cloud0_tf;
        plane.bound = bound;
        plane.bound_tf = bound_tf;
        plane.coefficients = coefficients_plane;
        plane.coefficients_tf = coefficients_plane_tf;
        plane.mnormal = plane_normal;
        plane.isbase = false;
        Planes.push_back(plane);

        std::cout << "Planes: " << Planes.size() << std::endl;

        sort(Planes.begin(), Planes.end(), cmp);

        Welding_line(Planes, cloud_Voxel1, cloud_hq, wending_line, cloudmaxz, cloudminz);
        std::cout << "第二次寻找 wending_line size: " << wending_line.size() << std::endl;
        for (auto i : wending_line)
        {
            std::cout << "wending_line cloud size: " << i->size() << std::endl;
            *cloud100 = *cloud100 + *i;
            for (auto j : i->points)
            {
                std::cout << "x: " << j.x << " y: " << j.y << " z: " << j.z << std::endl;
            }
        }
    }

    pcl::toROSMsg(*cloud_hq, output_hq);
    output_hq.header.frame_id = "map";

    pcl::toROSMsg(*cloud100, output_resultline);
    output_resultline.header.frame_id = "map";

    pcl::toROSMsg(*Planes[0].cloud, output_bound);
    output_bound.header.frame_id = "map";
    pcl::toROSMsg(*Planes[0].bound, output_bound0);
    output_bound0.header.frame_id = "map";

    pcl::toROSMsg(*Planes[1].cloud, output_bound1);
    output_bound1.header.frame_id = "map";
    pcl::toROSMsg(*Planes[1].bound, output_bound01);
    output_bound01.header.frame_id = "map";

    pcl::toROSMsg(*Planes[0].bound_tf, output_bound_tf);
    output_bound_tf.header.frame_id = "map";
    pcl::toROSMsg(*Planes[1].bound_tf, output_bound1_tf);
    output_bound1_tf.header.frame_id = "map";
    if (Planes.size() > 2)
    {
        if (Planes[2].cloud->size() > 0)
        {
            pcl::toROSMsg(*Planes[2].cloud, output_bound2);
            output_bound2.header.frame_id = "map";
            pcl::toROSMsg(*Planes[2].bound, output_bound02);
            output_bound02.header.frame_id = "map";
        }
    }

    if (Planes.size() > 3)
    {
        if (Planes[3].cloud->size() > 0)
        {
            pcl::toROSMsg(*Planes[3].cloud, output_bound3);
            output_bound3.header.frame_id = "map";
            pcl::toROSMsg(*Planes[3].bound, output_bound03);
            output_bound03.header.frame_id = "map";
        }
    }

    if (Planes.size() > 4)
    {
        if (Planes[4].cloud->size() > 0)
        {
            pcl::toROSMsg(*Planes[4].cloud, output_bound4);
            output_bound4.header.frame_id = "map";
        }
    }

    auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // 得到程序运行总时间
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time to fill and iterate a vector of "
              << " ints : " << duration.count() << " ms\n";

    ros::Rate loop_rate(1);
    while (ros::ok())
    {
        pcl_pub.publish(output);
        pcl_pub_hq.publish(output_hq);
        pcl_pub_condition.publish(output_condition);
        pcl_pub_bilateral.publish(output_bilateral);
        pcl_pub_pass_tf.publish(output_pass);
        pcl_pub_eucluste.publish(output_eucluster);
        pcl_pub_bound.publish(output_bound);
        pcl_pub_bound0.publish(output_bound0);
        pcl_pub_bound01.publish(output_bound01);
        pcl_pub_bound02.publish(output_bound02);
        pcl_pub_bound03.publish(output_bound03);
        pcl_pub_bound1.publish(output_bound1);
        pcl_pub_bound2.publish(output_bound2);
        pcl_pub_bound3.publish(output_bound3);
        pcl_pub_bound4.publish(output_bound4);
        pcl_pub_bound_tf.publish(output_bound_tf);
        pcl_pub_bound1_tf.publish(output_bound1_tf);
        pcl_pub_resultline.publish(output_resultline);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
