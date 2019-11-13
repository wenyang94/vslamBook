#include<iostream>
#include<fstream>
#include<stdio.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include<Eigen/Geometry>
#include<Eigen/Core>
#include<Eigen/SVD>
#include<pcl/point_types.h>
#include<pcl/io/pcd_io.h>
#include<pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h> 
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <octomap/octomap.h>  
#include <octomap/ColorOcTree.h>  
#include<ceres/ceres.h>
#include<ceres/rotation.h>
#include <boost/concept_check.hpp>
 
using namespace std;
using namespace cv;
 
struct cost_function_define
{
  //构造函数初始化并传入参数
  cost_function_define(Point3d p1,Point2d p2):_p1(p1),_p2(p2){}
  template<typename T>
  bool operator()(const T* const cere_r,T* residual)const
  {
    //将空间点转为类型T，可以调用旋转函数转化
    T p_1[3];
    T p_2[3];
    p_1[0]=T(_p1.x);
    p_1[1]=T(_p1.y);
    p_1[2]=T(_p1.z);
    //调用选择矩阵，将一图的空间点映射到第二图
    ceres::AngleAxisRotatePoint(cere_r,p_1,p_2);
    //将空间点变为像素坐标
    p_2[0]=p_2[0]+cere_r[3];
    p_2[1]=p_2[1]+cere_r[4];
    p_2[2]=p_2[2]+cere_r[5];
    const T x=p_2[0]/p_2[2];
    const T y=p_2[1]/p_2[2];
    const T u=x*518.0+325.5;
    const T v=y*519.0+253.5;
    //求出第二图的像素坐标
    const T u1=T(_p2.x);
    const T v1=T(_p2.y);
    //定义残差
    residual[0]=u-u1;
    residual[1]=v-v1;
    return true;
  }
   Point3d _p1;
   Point2d _p2;
};
 
int main()
{
  Point2d transform(const Point2d& p,const Mat& K);
  void writefile(const Mat& mat,const char* filename);
  Mat K=(Mat_<double>(3,3)<<518.0,0,325.5,0,519.0,253.5,0,0,1);
  //存放照片编号
  vector<int>number;
  //存放照片顺序
  vector<int>queue;
  ifstream fin("./number.txt");
  for(int i=0;i<32;i++)
  {
    int a=0;
    fin>>a;
    number.push_back(a);
    queue.push_back(i);
  }
  ifstream fin_1("./data.txt");
  vector<Mat>poses;
  for(int i=0;i<32;i++)
  {
    double data[12]={0};
    fin_1>>data[0]>>data[1]>>data[2]>>data[3]>>data[4]>>data[5]>>data[6]>>data[7]>>data[8]>>data[9]>>data[10]>>data[11];
    Mat currentpose=(Mat_<double>(3,4)<<data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11]);
    poses.push_back(currentpose);
  }
  //分别存放三维坐标和像素坐标和照片对应顺序
  vector<Point3d> loaclopt;
  vector<Point2d> localnum;
  vector<int> localqueue;
  for(int i=0;i<31;i++)
    for(int j=i+1;j<32;j++)
    {
      char last_c[30];
      char color[30];
      char last_d[30];
      char depth[30];
      sprintf(last_c,"%s%d%s","./rgb_png/",number[i],".png");
      sprintf(color,"%s%d%s","./rgb_png/",number[j],".png");
      sprintf(last_d,"%s%d%s","./depth_png/",number[i],".png");
      sprintf(depth,"%s%d%s","./depth_png/",number[j],".png");
      Mat last_co=imread(last_c);
      Mat colornow=imread(color);
      Mat last_dep=imread(last_d,-1);
      vector<KeyPoint> keypoints1,keypoints2;
      Ptr<FeatureDetector> detector=ORB::create();
      detector->detect(last_co,keypoints1);
      detector->detect(colornow,keypoints2);
      Mat descriptor1,descriptor2;
      Ptr<DescriptorExtractor> descriptor=ORB::create();
      descriptor->compute(last_co,keypoints1,descriptor1);
      descriptor->compute(colornow,keypoints2,descriptor2);
      vector<DMatch> matches;
      Ptr<DescriptorMatcher> matcher=DescriptorMatcher::create("BruteForce-Hamming");
      matcher->match(descriptor1,descriptor2,matches);
      double mindis=1000;
      for(int k=0;k<matches.size();k++)
      {
	if(matches[k].distance<mindis)
	{
	  mindis=matches[k].distance;
	}
      }
      vector<DMatch> goodmatch;
      for(int k=0;k<matches.size();k++)
      {
	if(matches[k].distance<=min(4*mindis,20.0))
	{
	  goodmatch.push_back(matches[k]);
	}
      }
      for(int k=0;k<goodmatch.size();k++)
      {
	Point2d pixel1=keypoints1[goodmatch[k].queryIdx].pt;
	Point2d pixel2=keypoints2[goodmatch[k].trainIdx].pt;
	Point2d pixel_cam1=transform(pixel1,K);
	ushort d=last_dep.ptr<unsigned short>(int(pixel1.y))[int(pixel1.x)];
	if(d==0)
	{
	  continue;
	}
        localnum.push_back(pixel2);
	localqueue.push_back(queue[j]);
	Point3d camera_3d=Point3d(pixel_cam1.x*double(d)/1000.0,pixel_cam1.y*double(d)/1000.0,double(d)/1000.0);
	Mat worldpoint=(Mat_<double>(3,3)<<poses[i].at<double>(0,0),poses[i].at<double>(0,1),poses[i].at<double>(0,2),
	                                                                  poses[i].at<double>(1,0),poses[i].at<double>(1,1),poses[i].at<double>(1,2),
                                                                          poses[i].at<double>(2,0),poses[i].at<double>(2,1),poses[i].at<double>(2,2)
	)*(Mat_<double>(3,1)<<camera_3d.x,camera_3d.y,camera_3d.z)+(Mat_<double>(3,1)<<poses[i].at<double>(0,3),poses[i].at<double>(1,3),poses[i].at<double>(2,3));
	Point3d world_3d=Point3d(worldpoint.at<double>(0,0),worldpoint.at<double>(1,0),worldpoint.at<double>(2,0));
	loaclopt.push_back(world_3d);
      }
    }
  cout<<"共有观测点"<<loaclopt.size()<<endl;
  vector<double> lastdata;
  for(int i=1;i<32;i++)
  {
    Mat reserve=(Mat_<double>(3,3)<<poses[i].at<double>(0,0),poses[i].at<double>(1,0),poses[i].at<double>(2,0),
	                                                                  poses[i].at<double>(0,1),poses[i].at<double>(1,1),poses[i].at<double>(2,1),
                                                                          poses[i].at<double>(0,2),poses[i].at<double>(1,2),poses[i].at<double>(2,2));
    Mat reserve_t=-reserve*(Mat_<double>(3,1)<<poses[i].at<double>(0,3),poses[i].at<double>(1,3),poses[i].at<double>(2,3));
    Mat res;
    Rodrigues(reserve,res);
    for(int j=0;j<3;j++)
    {
      lastdata.push_back(res.at<double>(j,0));
    }
    for(int j=0;j<3;j++)
    {
      lastdata.push_back(reserve_t.at<double>(j,0));
    }
  }
  double dataest[186]={0};
  for(int i=0;i<186;i++)
  {
    dataest[i]=lastdata[i];
  }
  double* camera=dataest;
  //定义求解问题
  ceres::Problem problem;
  for(int i=0;i<localnum.size();i++)
  {
    //残差的维度，变量的维度,使用自动求导，定义一个求解残差的构造函数
    ceres::CostFunction* costfunction=new ceres::AutoDiffCostFunction<cost_function_define,2,6>(new cost_function_define(loaclopt[i],localnum[i]));
    ceres::LossFunction* lossfunction=new ceres::HuberLoss(1.0);
    //添加每一次观测的残差
    double* cere_r=camera+6*(localqueue[i]-1);
    problem.AddResidualBlock(costfunction,lossfunction,cere_r);
  }
  //配置求解器
  ceres::Solver::Options option;
  //选择迭代方式
  option.linear_solver_type=ceres::SPARSE_SCHUR;
  //输出迭代信息到屏幕
  option.minimizer_progress_to_stdout=true;
  //显示优化信息
  ceres::Solver::Summary summary;
  //开始求解
  ceres::Solve(option,&problem,&summary);
  //显示优化信息
  cout<<summary.BriefReport()<<endl;
  
  Mat worldaxis=(Mat_<double>(3,4)<<1,0,0,0,0,1,0,0,0,0,1,0);
  writefile(worldaxis,"./local.txt");
  for(int i=0;i<31;i++)
  {
    //将转角和位移变成变化矩阵并存入
    Mat initial_r=(Mat_<double>(3,1)<<camera[6*i],camera[6*i+1],camera[6*i+2]);
    Mat initial_R;
    Rodrigues(initial_r,initial_R);
    Mat initial_t=(Mat_<double>(3,1)<<camera[6*i+3],camera[6*i+4],camera[6*i+5]);
    Mat final_R=initial_R.t();
    Mat final_t=-final_R*initial_t;
    Mat combine=(Mat_<double>(3,4)<<final_R.at<double>(0,0),final_R.at<double>(0,1),final_R.at<double>(0,2),final_t.at<double>(0,0),
	                                         final_R.at<double>(1,0),final_R.at<double>(1,1),final_R.at<double>(1,2),final_t.at<double>(1,0),
	                                         final_R.at<double>(2,0),final_R.at<double>(2,1),final_R.at<double>(2,2),final_t.at<double>(2,0));
    writefile(combine,"./local.txt");
  }
  //写数据到点云
  ifstream fin_2("./local.txt");
  vector<Mat>pose;
  for(int i=0;i<32;i++)
  {
    double data[12]={0};
    fin_2>>data[0]>>data[1]>>data[2]>>data[3]>>data[4]>>data[5]>>data[6]>>data[7]>>data[8]>>data[9]>>data[10]>>data[11];
    Mat currentpose=(Mat_<double>(3,4)<<data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11]);
    pose.push_back(currentpose);
  }
  cout<<"正在存放点云"<<endl;
  //定义点云指针存放关键帧的点云
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  for(int i=0;i<32;i++)
  {
    //创建点云，用于统计滤波
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr currentpoint(new pcl::PointCloud<pcl::PointXYZRGB>);
    char colorname[30];
    char depthname[30];
    sprintf(colorname,"%s%d%s","./rgb_png/",number[i],".png");
    sprintf(depthname,"%s%d%s","./depth_png/",number[i],".png");
    //存放关键帧
    Mat keyframe,depthkeyframe;
    keyframe=imread(colorname);
    depthkeyframe=imread(depthname,-1);
    for(int v=0;v<keyframe.rows;v++)
      for(int u=0;u<keyframe.cols;u++)
      {
	ushort d=depthkeyframe.ptr<unsigned short>(v)[u];
	if(d==0||d>7000.0||d<400.0)
	{
	  continue;
	}
	//还原三维点
	Point3d camera_point;
	camera_point.z=double(d)/1000.0;
	camera_point.x=(u-K.at<double>(0,2))/K.at<double>(0,0)*camera_point.z;
	camera_point.y=(v-K.at<double>(1,2))/K.at<double>(1,1)*camera_point.z;
	Mat worldpoint=(Mat_<double>(3,3)<<pose[i].at<double>(0,0),pose[i].at<double>(0,1),pose[i].at<double>(0,2),
	                                                                  pose[i].at<double>(1,0),pose[i].at<double>(1,1),pose[i].at<double>(1,2),
                                                                          pose[i].at<double>(2,0),pose[i].at<double>(2,1),pose[i].at<double>(2,2)
	)*(Mat_<double>(3,1)<<camera_point.x,camera_point.y,camera_point.z)+(Mat_<double>(3,1)<<pose[i].at<double>(0,3),pose[i].at<double>(1,3),pose[i].at<double>(2,3));
	pcl::PointXYZRGB p_3d;
	p_3d.x=worldpoint.at<double>(0,0);
	p_3d.y=worldpoint.at<double>(1,0);
	p_3d.z=worldpoint.at<double>(2,0);
	p_3d.b=keyframe.data[v*keyframe.step+u*keyframe.channels()];
	p_3d.g=keyframe.data[v*keyframe.step+u*keyframe.channels()+1];
	p_3d.r=keyframe.data[v*keyframe.step+u*keyframe.channels()+2];
	pointcloud->points.push_back(p_3d);
      }
      
      //统计滤波
     // pcl::PointCloud<pcl::PointXYZRGB>::Ptr tem(new pcl::PointCloud<pcl::PointXYZRGB>);
     // pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> statistical_filter;
      //设置聚类，阀值
     // statistical_filter.setMeanK(50);
     // statistical_filter.setStddevMulThresh(1.0);
     // statistical_filter.setInputCloud(currentpoint);
     // statistical_filter.filter(*tem);
     // (*pointcloud)+=*tem;
  }
  cout<<"点云读取完成"<<endl;
  pointcloud->is_dense=false;
  cout<<pointcloud->size()<<endl;
  //体素滤波
  pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
  //设置最小方格
  voxel_filter.setLeafSize(0.01,0.01,0.01);
  //定义一个点云指针用于保存滤波后的点
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr tempoint(new pcl::PointCloud<pcl::PointXYZRGB>);
  voxel_filter.setInputCloud(pointcloud);
  //滤波
  voxel_filter.filter(*tempoint);
  tempoint->swap(*pointcloud);
  cout<<pointcloud->size()<<endl;
  //保存
  pcl::io::savePCDFileBinary("globalmap.pcd",*pointcloud);
  return 0;
}
Point2d transform(const Point2d& p,const Mat& K)
{
  return Point2d(
    (p.x-K.at<double>(0,2))/K.at<double>(0,0),
    (p.y-K.at<double>(1,2))/K.at<double>(1,1)
  );
}
 
    //将位置写入txt便于读取
void writefile(const Mat& mat,const char* filename)
{
  ofstream fout(filename,ios_base::app);
  for(int i=0;i<mat.rows;i++)
  {
    for(int k=0;k<mat.cols;k++)
    {
      fout<<mat.at<double>(i,k)<<" ";
    }
    fout<<endl;
  }
  fout.close();
}
