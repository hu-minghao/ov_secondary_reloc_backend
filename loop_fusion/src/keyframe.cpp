/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "keyframe.h"

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
	int j = 0;
	for (int i = 0; i < int(v.size()); i++)
		if (status[i])
			v[j++] = v[i];
	v.resize(j);
}

// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
				   vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
				   vector<double> &_point_id, int _sequence)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	computeWindowBRIEFPoint();
	computeBRIEFPoint();
	if(!DEBUG_IMAGE)
		image.release();
}

// load previous keyframe
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
				   cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1> &_loop_info,
				   vector<cv::Point2f> &_keypoints, vector<cv::Point2f> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors,
				   vector<cv::Point3f> _point_3d)
{
	time_stamp = _time_stamp;
	index = _index;
	//vio_T_w_i = _vio_T_w_i;
	//vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
	if (DEBUG_IMAGE)
	{
		image = _image.clone();
		cv::resize(image, thumbnail, cv::Size(80, 60));
	}
	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	sequence = 0;
	point_2d_uv = _keypoints;
	point_2d_norm = _keypoints_norm;
	window_brief_descriptors = _brief_descriptors;
	point_3d = _point_3d;
}

KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
				   int _sequence)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	computeInitialBRIEFPoint();
	if (!DEBUG_IMAGE)
		image.release();
}

void KeyFrame::computeWindowBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	for (int i = 0; i < (int)point_2d_uv.size(); i++)
	{
		cv::KeyPoint key;
		key.pt = point_2d_uv[i];
		window_keypoints.push_back(key);
	}
	extractor(image, window_keypoints, window_brief_descriptors);
}

void KeyFrame::computeBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int fast_th = 10; // corner detector response threshold
	if (1)
	{
		// cv::FAST(image, keypoints, fast_th, true);
		Grider_FAST::perform_griding(image, keypoints, 200, 1, 1, fast_th, true);
	}
	else
	{
		vector<cv::Point2f> tmp_pts;
		cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
		for (int i = 0; i < (int)tmp_pts.size(); i++)
		{
			cv::KeyPoint key;
			key.pt = tmp_pts[i];
			keypoints.push_back(key);
		}
	}

	// push back the uvs used in vio
	for (int i = 0; i < (int)point_2d_uv.size(); i++)
	{
		cv::KeyPoint key;
		key.pt = point_2d_uv[i];
		keypoints.push_back(key);
	}

	// extract and save
	extractor(image, keypoints, brief_descriptors);
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}

void KeyFrame::computeInitialBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int fast_th = 10; // corner detector response threshold
	Grider_FAST::perform_griding(image, keypoints, 400, 1, 1, fast_th, true);

	// push back the uvs used in vio
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		point_2d_uv.emplace_back(keypoints[i].pt);
	}

	// extract and save
	extractor(image, keypoints, brief_descriptors);
	for (int i = 0; i < (int)point_2d_uv.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(point_2d_uv[i].x, point_2d_uv[i].y), tmp_p);
		point_2d_norm.push_back(cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z()));
	}
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}

void BriefExtractor::operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
	m_brief.compute(im, keys, descriptors);
}

bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
							const std::vector<BRIEF::bitset> &descriptors_cur,
							const std::vector<cv::KeyPoint> &keypoints_normal_cur,
							const std::vector<cv::KeyPoint> &keypoints_cur,
							cv::Point2f &best_match_normal,
							cv::Point2f &best_match,
							std::vector<bool> &matched_flag)
{
	int bestDist = 128;
	int secondBestDist = 128;
	int bestIndex = -1;
	for (int i = 0; i < (int)descriptors_cur.size(); i++)
	{
		if (matched_flag[i])
			continue; // 已匹配，跳过

		int dis = HammingDis(window_descriptor, descriptors_cur[i]);

		if (dis < bestDist)
		{
			secondBestDist = bestDist;
			bestDist = dis;
			bestIndex = i;
		}
		else if (dis < secondBestDist)
		{
			secondBestDist = dis;
		}
	}

	// 比值约束 + 距离阈值
	if (bestIndex != -1 && bestDist < 80 && static_cast<float>(bestDist) / secondBestDist < 0.7f)
	{
		best_match_normal = keypoints_normal_cur[bestIndex].pt;
		best_match = keypoints_cur[bestIndex].pt;
		matched_flag[bestIndex] = true; // 标记该点已匹配
		return true;
	}
	return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_normal_cur,
								std::vector<cv::Point2f> &matched_2d_cur,
								std::vector<uchar> &status,
								const std::vector<BRIEF::bitset> &window_descriptors_old,
								const std::vector<cv::KeyPoint> &keypoints_2d_norm,
								const std::vector<cv::KeyPoint> &keypoints)
{
	std::vector<bool> matched_flag((int)brief_descriptors.size(), false);
	for (int i = 0; i < (int)window_descriptors_old.size(); i++)
	{
		cv::Point2f pt_normal(0.f, 0.f);
		cv::Point2f pt(0.f, 0.f);
		if (searchInAera(window_descriptors_old[i], brief_descriptors, keypoints_2d_norm, keypoints, pt_normal, pt, matched_flag))
			status.push_back(1);
		else
			status.push_back(0);
		matched_2d_normal_cur.emplace_back(pt_normal);
		matched_2d_cur.emplace_back(pt);
	}
}

void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
									  const std::vector<cv::Point2f> &matched_2d_old_norm,
									  vector<uchar> &status)
{
	int n = (int)matched_2d_cur_norm.size();
	for (int i = 0; i < n; i++)
		status.push_back(0);
	if (n >= 8)
	{
		vector<cv::Point2f> tmp_cur(n), tmp_old(n);
		for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
		{
			double FOCAL_LENGTH = 460.0;
			double tmp_x, tmp_y;
			tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
			tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
			tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

			tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
			tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
			tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
		}
		cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
	}
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
						 const std::vector<cv::Point3f> &matched_3d,
						 std::vector<uchar> &status,
						 Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
	cv::Mat r, rvec, t, D, tmp_r;
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
	Matrix3d R_inital;
	Vector3d P_inital;
	Matrix3d R_w_c = PnP_R_old * qic;
	Vector3d T_w_c = PnP_T_old + PnP_R_old * tic;

	R_inital = R_w_c.inverse();
	P_inital = -(R_inital * T_w_c);

	cv::eigen2cv(R_inital, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_inital, t);

	cv::Mat inliers;
	TicToc t_pnp_ransac;

	int flags = cv::SOLVEPNP_EPNP; // SOLVEPNP_EPNP, SOLVEPNP_ITERATIVE
	if (CV_MAJOR_VERSION < 3)
	{
		solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 200, PNP_INFLATION / max_focallength, 100, inliers, flags);
	}
	else
	{
		if (CV_MINOR_VERSION < 2)
			solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 200, sqrt(PNP_INFLATION / max_focallength), 0.99, inliers, flags);
		else
			solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 200, PNP_INFLATION / max_focallength, 0.99, inliers, flags);
	}

	for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
		status.push_back(0);

	for (int i = 0; i < inliers.rows; i++)
	{
		int n = inliers.at<int>(i);
		status[n] = 1;
	}

	cv::Rodrigues(rvec, r);
	Matrix3d R_pnp, R_w_c_old;
	cv::cv2eigen(r, R_pnp);
	R_w_c_old = R_pnp.transpose();
	Vector3d T_pnp, T_w_c_old;
	cv::cv2eigen(t, T_pnp);
	T_w_c_old = R_w_c_old * (-T_pnp);

	PnP_R_old = R_w_c_old * qic.transpose();
	PnP_T_old = T_w_c_old - PnP_R_old * tic;
}

bool KeyFrame::findConnection(const KeyFramePtr &old_kf)
{
	TicToc tmp_t;
	printf("[POSEGRAPH]: find Connection\n");
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d, matched_3d_old;
	vector<double> matched_id;
	vector<uchar> status;

	point_2d_norm.clear();
	// re-undistort with the latest intrinsic values
	for (int i = 0; i < (int)point_2d_uv.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(point_2d_uv[i].x, point_2d_uv[i].y), tmp_p);
		point_2d_norm.push_back(cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z()));
	}
	keypoints_norm.clear();
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}

	matched_3d = point_3d;
	matched_3d_old = old_kf->point_3d;
	matched_id = point_id;
	matched_2d_old = old_kf->point_2d_uv;

	TicToc t_match;
#if 0
		if (DEBUG_IMAGE)    
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
	searchByBRIEFDes(matched_2d_cur_norm, matched_2d_cur, status, old_kf->window_brief_descriptors, keypoints_norm, keypoints);
	if(matched_2d_cur.empty()){
		return false;
	}
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	// reduceVector(matched_2d_old_norm, status);
	// reduceVector(matched_3d, status);
	reduceVector(matched_3d_old, status);
	// reduceVector(matched_id, status);
	// for(const auto& pt:matched_3d){
	//	std::cout<<" , 3d curr "<<pt.x<<" "<<pt.y<<" "<<pt.z<<std::endl;
	// }

#if 0 
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path, path1, path2;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        /*
	        path1 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);	        
	        */
	        
	    }
#endif
	status.clear();
/*
FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
reduceVector(matched_2d_cur, status);
reduceVector(matched_2d_old, status);
reduceVector(matched_2d_cur_norm, status);
reduceVector(matched_2d_old_norm, status);
reduceVector(matched_3d, status);
reduceVector(matched_id, status);
*/
#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif

	Eigen::Vector3d origin_T_old;
	Eigen::Matrix3d origin_R_old;
	old_kf->getPose(origin_T_old, origin_R_old);
	Eigen::Vector3d PnP_T_old = origin_T_old;
	Eigen::Matrix3d PnP_R_old = origin_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;
	if ((int)matched_2d_cur_norm.size() > MIN_LOOP_NUM)
	{
		status.clear();
		// PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
		// PnP_T_old: PnP_T_cur, PnP_R_old: PnP_R_cur
		PnPRANSAC(matched_2d_cur_norm, matched_3d_old, status, PnP_T_old, PnP_R_old);
		reduceVector(matched_2d_cur, status);
		reduceVector(matched_2d_old, status);
		reduceVector(matched_2d_cur_norm, status);
		reduceVector(matched_2d_old_norm, status);
		reduceVector(matched_3d, status);
		reduceVector(matched_3d_old, status);
		reduceVector(matched_id, status);
#if 1
		if (DEBUG_IMAGE)
		{
			int gap = 10;
			cv::Mat gap_image(old_kf->image.rows, gap, CV_8UC1, cv::Scalar(255, 255, 255));
			cv::Mat gray_img, loop_match_img;
			cv::Mat old_img = old_kf->image;
			cv::hconcat(image, gap_image, gap_image);
			cv::hconcat(gap_image, old_img, gray_img);
			cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
			for (int i = 0; i < (int)matched_2d_cur.size(); i++)
			{
				cv::Point2f cur_pt = matched_2d_cur[i];
				cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
			}
			for (int i = 0; i < (int)matched_2d_old.size(); i++)
			{
				cv::Point2f old_pt = matched_2d_old[i];
				old_pt.x += (old_kf->image.cols + gap);
				cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
			}
			for (int i = 0; i < (int)matched_2d_cur.size(); i++)
			{
				cv::Point2f old_pt = matched_2d_old[i];
				old_pt.x += (old_kf->image.cols + gap);
				cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
			}
			cv::Mat notation(50, old_kf->image.cols + gap + old_kf->image.cols, CV_8UC3, cv::Scalar(255, 255, 255));
			putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

			putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + old_kf->image.cols + gap, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
			cv::vconcat(notation, loop_match_img, loop_match_img);

			/*
			ostringstream path;
			path <<  "/home/tony-ws1/raw_data/loop_image/"
					<< index << "-"
					<< old_kf->index << "-" << "3pnp_match.jpg";
			cv::imwrite( path.str().c_str(), loop_match_img);
			*/
			if ((int)matched_2d_cur_norm.size() > MIN_LOOP_NUM)
			{
				/*
				cv::imshow("loop connection",loop_match_img);
				cv::waitKey(10);
				*/
				cv::Mat thumbimage;
				cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
				sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
				msg->header.stamp = ros::Time(time_stamp);
				pub_match_img.publish(msg);
			}
		}
#endif
	}
	printf(" ,[POSEGRAPH]: loop final use num %d %lf--------------- \n", (int)matched_2d_cur_norm.size(), t_match.toc());
	if ((int)matched_2d_cur_norm.size() > MIN_LOOP_NUM)
	{
		// relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
		// relative_q = PnP_R_old.transpose() * origin_vio_R;
		// relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
		// T^old_cur
		relative_q = origin_R_old.transpose() * PnP_R_old;
		relative_t = origin_R_old.transpose() * (PnP_T_old - origin_T_old);
		relative_yaw = Utility::normalizeAngle(Utility::R2ypr(PnP_R_old).x() - Utility::R2ypr(origin_R_old).x());
		// printf("[POSEGRAPH]: PNP relative\n");
		// cout << "pnp relative_t " << relative_t.transpose() << endl;
		// cout << "pnp relative_yaw " << relative_yaw << endl;
		printf(" ,[POSEGRAPH]: abs(relative_yaw)  %f  MAX_THETA_DIFF %f relative_t.norm() %lf MAX_POS_DIFF %f  \n",
			   abs(relative_yaw), MAX_THETA_DIFF, relative_t.norm(), MAX_POS_DIFF);
		if (abs(relative_yaw) < MAX_THETA_DIFF && relative_t.norm() < MAX_POS_DIFF)
		{
			printf(" , add loop info \n");

			has_loop = true;
			loop_index = old_kf->index;
			loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
				relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
				relative_yaw;
			cout << "pnp relative_t " << relative_t.transpose() << endl;
			cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
			return true;
		}
	}
	// printf(" ,[POSEGRAPH]: loop final use num %d %lf--------------- end \n", (int)matched_2d_cur.size(), t_match.toc());
	return false;
}

int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
	BRIEF::bitset xor_of_bitset = a ^ b;
	int dis = xor_of_bitset.count();
	return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
	_T_w_i = vio_T_w_i;
	_R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
	_T_w_i = T_w_i;
	_R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
	return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
	return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
	return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		// printf("[POSEGRAPH]: update loop info\n");
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
	// The DVision::BRIEF extractor computes a random pattern by default when
	// the object is created.
	// We load the pattern that we used to build the vocabulary, to make
	// the descriptors compatible with the predefined vocabulary

	// loads the pattern
	cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
	if (!fs.isOpened())
		throw string("Could not open file ") + pattern_file;

	vector<int> x1, y1, x2, y2;
	fs["x1"] >> x1;
	fs["x2"] >> x2;
	fs["y1"] >> y1;
	fs["y2"] >> y2;

	m_brief.importPairs(x1, y1, x2, y2);
}
