// General
#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

// Cloud Operations
#include <pcl/common/projection_matrix.h>

// I/O File Operation
#include <pcl/io/pcd_io.h>

// Filters
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>

// Features
#include <pcl/features/normal_3d.h>

// ICP Regristation
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

// Visualization
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

// Kinect IO
#include <pcl/io/openni_grabber.h>

// Name Shortners
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

// Useful Typedefs
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

// Visualizer
pcl::visualization::PCLVisualizer *p;
// Global Cloud
PointCloud::Ptr global(new PointCloud), scene(new PointCloud);
// Left / Right Viewports
int vp_1, vp_2;
// Keep updating scene?
bool updateScene = true;

// PCD File Object
struct PCD
{
	PointCloud::Ptr cloud;
	std::string f_name;

	PCD() : cloud (new PointCloud) {};
};

// TODO: Figure out what this is used for
struct PCDComparator
{
	bool operator () (const PCD& p1, const PCD& p2)
	{
		return (p1.f_name < p2.f_name);
	}
};

// TOOD: Rename and communt to be useful

// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
{
	using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;

public:
	MyPointRepresentation ()
	{
		// Define the number of dimensions
		nr_dimensions_ = 4;
	}

	// Override the copyToFloatArray method to define our feature vector
	virtual void copyToFloatArray (const PointNormalT &p, float * out) const
	{
		// < x, y, z, curvature >
		out[0] = p.x;
		out[1] = p.y;
		out[2] = p.z;
		out[3] = p.curvature;
	}
};

class KinectViewer
{
public:
	KinectViewer () : viewer ("PCL OpenNI Viewer") {}

	void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr temp (new pcl::PointCloud<pcl::PointXYZ>);

		if (cloud->points.size() > 0)
		{
			pcl::VoxelGrid<PointT> grid;
			grid.setLeafSize (0.05, 0.05, 0.05);
			grid.setInputCloud (cloud);
			grid.filter(*temp);

			pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
			sor.setInputCloud(temp);
			sor.setMeanK(50);
			sor.setStddevMulThresh(1.0);

			if (updateScene)
				sor.filter (*scene);

			if (global->points.size() == 0)
				sor.filter (*global);
			
			if (!viewer.wasStopped())
			{
				viewer.showCloud (temp);
			}
		}
	}

	void run ()
	{
		pcl::Grabber* interface = new pcl::OpenNIGrabber();

		boost::function<void (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr&)> f =
			boost::bind (&KinectViewer::cloud_cb_, this, _1);

		interface->registerCallback (f);

		interface->start ();
	}

	pcl::visualization::CloudViewer viewer;
};

DWORD WINAPI showFinal(LPVOID lpParameter)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Global Output"));
	viewer->setBackgroundColor (0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ> (global, "Global Output");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Global Output");
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();

	while (!viewer->wasStopped ())
	{
		//PointCloud::Ptr global_up (&global);
		viewer->updatePointCloud(global, "Global Output");
		viewer->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::milliseconds (100));
	}
	return 0;
}

// Show left point cloud in viewport
void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
	p->removePointCloud ("vp1_target");
	p->removePointCloud ("vp1_source");

	PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 255, 0);
	PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 255, 0, 0);
	p->addPointCloud (cloud_target, tgt_h, "vp1_target", vp_1);
	p->addPointCloud (cloud_source, src_h, "vp1_source", vp_1);
}

// Show right point cloud in viewport
void showCloudsRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source)
{
	p->removePointCloud ("source");
	p->removePointCloud ("target");


	PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler (cloud_target, "curvature");
	if (!tgt_color_handler.isCapable ())
		PCL_WARN ("Cannot create curvature color handler!");

	PointCloudColorHandlerGenericField<PointNormalT> src_color_handler (cloud_source, "curvature");
	if (!src_color_handler.isCapable ())
		PCL_WARN ("Cannot create curvature color handler!");


	p->addPointCloud (cloud_target, tgt_color_handler, "target", vp_2);
	p->addPointCloud (cloud_source, src_color_handler, "source", vp_2);

	p->spinOnce();
}


void pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
	// Downsample
	PointCloud::Ptr src (new PointCloud);
	PointCloud::Ptr tgt (new PointCloud);

	printf("Filter Cloud\n");
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloud_tgt, *tgt, indices);
	pcl::removeNaNFromPointCloud(*cloud_src, *src, indices);

	pcl::VoxelGrid<PointT> grid;
	if (downsample)
	{
		grid.setLeafSize (0.075, 0.075, 0.075);
		grid.setInputCloud (src);
		grid.filter (*src);

		grid.setInputCloud (tgt);
		grid.filter (*tgt);
	}


	printf("Compute Normals\n");
	// Compute surface normals and curvature
	PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
	PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

	pcl::NormalEstimation<PointT, PointNormalT> norm_est;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
	norm_est.setSearchMethod (tree);
	norm_est.setKSearch (30);
	//norm_est.setRadiusSearch(0.2);

	printf("Compute source\n");
	norm_est.setInputCloud (src);
	norm_est.compute (*points_with_normals_src);
	pcl::copyPointCloud (*src, *points_with_normals_src);

	printf("Compute global\n");
	norm_est.setInputCloud (tgt);
	norm_est.compute (*points_with_normals_tgt);
	pcl::copyPointCloud (*tgt, *points_with_normals_tgt);

	printf("Curvature instantiation\n");
	// Instantiate our custom point representation (defined above) ...
	MyPointRepresentation point_representation;
	// ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
	float alpha[4] = {1.0, 1.0, 1.0, 1.0};
	point_representation.setRescaleValues (alpha);

	printf("Compute Align\n");
	// Align
	pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
	reg.setTransformationEpsilon (1e-6);
	// Set the maximum distance between two correspondences (src<->tgt) to 10cm
	// Note: adjust this based on the size of your datasets
	reg.setMaxCorrespondenceDistance (0.1);  
	// Set the point representation
	reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));

	reg.setInputSource (points_with_normals_src);
	reg.setInputTarget (points_with_normals_tgt);


	printf("Start Iterations\n");
	//
	// Run the same optimization in a loop and visualize the results
	Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
	PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
	reg.setMaximumIterations (2);
	//TODO: Can remove this after playing around with it to get the best output
	for (int i = 0; i < 30; ++i)
	{
		PCL_INFO ("Iteration Nr. %d.\n", i);

		// Save cloud for visualization purpose
		points_with_normals_src = reg_result;

		// Estimate
		reg.setInputSource (points_with_normals_src);
		reg.align (*reg_result);

		// Accumulate transformation between each Iteration
		Ti = reg.getFinalTransformation () * Ti;

		// If the difference between this transformation and the previous one
		// is smaller than the threshold, refine the process by reducing
		// the maximal correspondence distance
		if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
			reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);

		prev = reg.getLastIncrementalTransformation ();

		// Visualize current state
		showCloudsRight(points_with_normals_tgt, points_with_normals_src);
	}

	// Get the transformation from target to source
	printf("Get Transformation\n");
	targetToSource = Ti.inverse();

	// Transform target back in source frame
	printf("Apply transorm\n");
	pcl::transformPointCloud (*tgt, *output, targetToSource);

	printf("Remove cloud from viewer\n");
	p->removePointCloud ("source");
	p->removePointCloud ("target");

	printf("Show cloud\n");
	PointCloudColorHandlerCustom<PointT> cloud_tgt_h (output, 0, 255, 0);
	PointCloudColorHandlerCustom<PointT> cloud_src_h (cloud_src, 255, 0, 0);
	p->addPointCloud (output, cloud_tgt_h, "target", vp_2);
	p->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);

	PCL_INFO ("Press q to continue the registration.\n");
	p->spin ();

	p->removePointCloud ("source"); 
	p->removePointCloud ("target");

	printf("Add to output cloud\n");
	//add the source to the transformed target
	*output += *cloud_src;

	final_transform = targetToSource;
}



int main (int argc, char** argv)
{
	KinectViewer k_viewer;
	k_viewer.run();


	boost::this_thread::sleep (boost::posix_time::seconds (5));


	// Create a PCLVisualizer object
	p = new pcl::visualization::PCLVisualizer (argc, argv, "Pairwise Incremental Registration example");
	p->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
	p->createViewPort (0.5, 0, 1.0, 1.0, vp_2);

	PointCloud::Ptr result (new PointCloud);
	Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity (), pairTransform;

	CreateThread(NULL, 0, showFinal, NULL, 0, 0);

	while (true)
	{
		PCL_INFO ("Press q to continue\n");
		p->spin ();

		// Add visualization data
		showCloudsLeft(scene, global);

		printf("Start pairAllign using ICP\n");
		PointCloud::Ptr temp (new PointCloud);
		updateScene = false;
		//PCL_INFO ("Aligning %s (%d) with %s (%d).\n", data[i-1].f_name.c_str (), source->points.size (), data[i].f_name.c_str (), target->points.size ());
		pairAlign (global, scene, temp, pairTransform, true);
		updateScene = true;
		printf("Done with pairAllign\n");

		//transform current pair into the global transform
		pcl::transformPointCloud (*temp, *result, GlobalTransform);

		// Combine current into global cloud
		*global += *temp;

		printf("Shrinking global cloud\n");
		pcl::VoxelGrid<PointT> grid;
		grid.setLeafSize (0.05, 0.05, 0.05);
		grid.setInputCloud (global);
		grid.filter (*global);
		printf("Done shrinking\n");

		//update the global transform
		GlobalTransform = GlobalTransform * pairTransform;
	}
}