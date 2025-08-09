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

#include "pose_graph.h"
#include <unordered_set>
#include <limits>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>  // pcl::toROSMsg
#include <sensor_msgs/PointCloud2.h>

PoseGraph::PoseGraph()
{
    posegraph_visualization = new CameraPoseVisualization(1.0, 0.0, 1.0, 1.0);
    posegraph_visualization->setScale(0.1);
    posegraph_visualization->setLineWidth(0.01);
    earliest_loop_index = -1;
    t_drift = Eigen::Vector3d(0, 0, 0);
    yaw_drift = 0;
    r_drift = Eigen::Matrix3d::Identity();
    w_t_vio = Eigen::Vector3d(0, 0, 0);
    w_r_vio = Eigen::Matrix3d::Identity();
    global_index = 0;
    sequence_cnt = 0;
    sequence_loop.push_back(0);
    base_sequence = 1;
    use_imu = 0;
    pre_key_pose_.time_ = -1;
}

PoseGraph::~PoseGraph()
{
    t_optimization.detach();
    thread_run = false;
    if (tf_thread.joinable())
    {
        tf_thread.join();
    }
    else
    {
        tf_thread.detach();
    }
}

void PoseGraph::registerPub(ros::NodeHandle &n)
{
    pub_pg_path = n.advertise<nav_msgs::Path>("pose_graph_path", 1000);
    pub_base_path = n.advertise<nav_msgs::Path>("base_path", 1000);
    pub_pose_graph = n.advertise<visualization_msgs::MarkerArray>("pose_graph", 1000);
    pub_map_points =  n.advertise<sensor_msgs::PointCloud2>("map_points", 1, true);
    for (int i = 1; i < 10; i++)
        pub_path[i] = n.advertise<nav_msgs::Path>("path_" + to_string(i), 1000);
}

void PoseGraph::setIMUFlag(bool _use_imu)
{
    use_imu = _use_imu;
    if (use_imu)
    {
        printf("[POSEGRAPH]: VIO input, perfrom 4 DoF (x, y, z, yaw) pose graph optimization\n");
        t_optimization = std::thread(&PoseGraph::optimize4DoF, this);
    }
    else
    {
        printf("[POSEGRAPH]: VO input, perfrom 6 DoF pose graph optimization\n");
        t_optimization = std::thread(&PoseGraph::optimize6DoF, this);
    }
}

void PoseGraph::loadVocabulary(std::string voc_path)
{
    voc = new BriefVocabulary(voc_path);
    db.setVocabulary(*voc, false, 0);
}

void PoseGraph::addKeyFrame(const KeyFramePtr &cur_kf, bool flag_detect_loop)
{
    printf("addKeyFrame keyframelist size %zu  detect loop %d \n", keyframelist.size(), flag_detect_loop);
    // shift to base frame
    Vector3d vio_P_cur;
    Matrix3d vio_R_cur;
    // initial
    if (sequence_cnt != cur_kf->sequence)
    {
        sequence_cnt++;
        sequence_loop.push_back(0);
        // 该轨迹段的坐标系相对于世界坐标系的初始变换
        w_t_vio = Eigen::Vector3d(0, 0, 0);
        w_r_vio = Eigen::Matrix3d::Identity();
        // 当前轨迹段相对于全局地图的漂移量
        m_drift.lock();
        t_drift = Eigen::Vector3d(0, 0, 0);
        r_drift = Eigen::Matrix3d::Identity();
        m_drift.unlock();
    }

    cur_kf->getVioPose(vio_P_cur, vio_R_cur);
    // 漂移补偿作用
    vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
    vio_R_cur = w_r_vio * vio_R_cur;
    cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
    cur_kf->index = global_index;
    global_index++;
    int loop_index = -1;
    if (flag_detect_loop)
    {
        TicToc tmp_t;
        loop_index = detectLoop(cur_kf, cur_kf->index);
    }
    else
    {
        addKeyFrameIntoVoc(cur_kf);
    }
    if (loop_index != -1)
    {
        printf("[POSEGRAPH]:  %d detect loop with %d \n", cur_kf->index, loop_index);
        KeyFramePtr old_kf = getKeyFrame(loop_index);

        if ((cur_kf->has_loop && loop_index == old_kf->index) || cur_kf->findConnection(old_kf))
        {
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
                earliest_loop_index = loop_index;

            Vector3d w_P_old, w_P_cur, vio_P_cur;
            Matrix3d w_R_old, w_R_cur, vio_R_cur;
            old_kf->getVioPose(w_P_old, w_R_old);
            cur_kf->getVioPose(vio_P_cur, vio_R_cur);

            Vector3d relative_t;
            Quaterniond relative_q;
            relative_t = cur_kf->getLoopRelativeT();
            relative_q = (cur_kf->getLoopRelativeQ()).toRotationMatrix();
            w_P_cur = w_R_old * relative_t + w_P_old;
            w_R_cur = w_R_old * relative_q;

            double shift_yaw;
            Matrix3d shift_r;
            Vector3d shift_t;
            if (use_imu)
            {
                shift_yaw = Utility::R2ypr(w_R_cur).x() - Utility::R2ypr(vio_R_cur).x();
                shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
            }
            else
                shift_r = w_R_cur * vio_R_cur.transpose();
            shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur;
            // shift vio pose of whole sequence to the world frame
            if (old_kf->sequence != cur_kf->sequence && sequence_loop[cur_kf->sequence] == 0)
            {
                if(!isMappingMode()){
                    relocation_ = true;
                    printf("relocation success\n");
                }
                w_r_vio = shift_r;
                w_t_vio = shift_t;
                vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
                vio_R_cur = w_r_vio * vio_R_cur;
                cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
                list<KeyFramePtr >::iterator it = keyframelist.begin();
                for (; it != keyframelist.end(); it++)
                {
                    if ((*it)->sequence == cur_kf->sequence)
                    {
                        Vector3d vio_P_cur;
                        Matrix3d vio_R_cur;
                        (*it)->getVioPose(vio_P_cur, vio_R_cur);
                        vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
                        vio_R_cur = w_r_vio * vio_R_cur;
                        (*it)->updateVioPose(vio_P_cur, vio_R_cur);
                    }
                }
                sequence_loop[cur_kf->sequence] = 1;
            }
            m_optimize_buf.lock();
            optimize_buf.push(cur_kf->index);
            m_optimize_buf.unlock();
        }
    }
    m_keyframelist.lock();
    Vector3d P;
    Matrix3d R;
    cur_kf->getVioPose(P, R);
    P = r_drift * P + t_drift;
    R = r_drift * R;
    cur_kf->updatePose(P, R);
    Quaterniond Q{R};
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(cur_kf->time_stamp);
    pose_stamped.header.frame_id = "global";
    pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
    pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
    pose_stamped.pose.position.z = P.z();
    pose_stamped.pose.orientation.x = Q.x();
    pose_stamped.pose.orientation.y = Q.y();
    pose_stamped.pose.orientation.z = Q.z();
    pose_stamped.pose.orientation.w = Q.w();
    path[sequence_cnt].poses.push_back(pose_stamped);
    path[sequence_cnt].header = pose_stamped.header;

    if (SAVE_LOOP_PATH)
    {
        ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
        loop_path_file.setf(ios::fixed, ios::floatfield);
        loop_path_file.precision(0);
        loop_path_file << cur_kf->time_stamp * 1e9 << ",";
        loop_path_file.precision(5);
        loop_path_file << P.x() << ","
                       << P.y() << ","
                       << P.z() << ","
                       << Q.w() << ","
                       << Q.x() << ","
                       << Q.y() << ","
                       << Q.z() << ","
                       << endl;
        loop_path_file.close();
    }
    // draw local connection
    if (SHOW_S_EDGE)
    {
        list<KeyFramePtr >::reverse_iterator rit = keyframelist.rbegin();
        for (int i = 0; i < 4; i++)
        {
            if (rit == keyframelist.rend())
                break;
            Vector3d conncected_P;
            Matrix3d connected_R;
            if ((*rit)->sequence == cur_kf->sequence)
            {
                (*rit)->getPose(conncected_P, connected_R);
                posegraph_visualization->add_edge(P, conncected_P);
            }
            rit++;
        }
    }
    if (SHOW_L_EDGE)
    {
        if (cur_kf->has_loop)
        {
            // printf("[POSEGRAPH]: has loop \n");
            KeyFramePtr connected_KF = getKeyFrame(cur_kf->loop_index);
            Vector3d connected_P, P0;
            Matrix3d connected_R, R0;
            connected_KF->getPose(connected_P, connected_R);
            // cur_kf->getVioPose(P0, R0);
            cur_kf->getPose(P0, R0);
            if (cur_kf->sequence > 0)
            {
                // printf("[POSEGRAPH]: add loop into visual \n");
                posegraph_visualization->add_loopedge(P0, connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
            }
        }
    }
    // posegraph_visualization->add_pose(P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0), Q);
    if (SysMode::kMAPPING == mode)
    {
        keyframelist.push_back(cur_kf);
    }
    publish();
    m_keyframelist.unlock();
    printf("system mode %d, relocation_ %d  loop_index %d \n", (int)mode, relocation_, loop_index);
}

void PoseGraph::loadKeyFrame(const KeyFramePtr &cur_kf, bool flag_detect_loop)
{
    cur_kf->index = global_index;
    global_index++;
    int loop_index = -1;
    if (flag_detect_loop)
        loop_index = detectLoop(cur_kf, cur_kf->index);
    else
    {
        addKeyFrameIntoVoc(cur_kf);
    }
    if (loop_index != -1)
    {
        printf("[POSEGRAPH]:  %d detect loop with %d \n", cur_kf->index, loop_index);
        KeyFramePtr old_kf = getKeyFrame(loop_index);
        if (cur_kf->findConnection(old_kf))
        {
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
                earliest_loop_index = loop_index;
            m_optimize_buf.lock();
            optimize_buf.push(cur_kf->index);
            m_optimize_buf.unlock();
        }
    }
    m_keyframelist.lock();
    Vector3d P;
    Matrix3d R;
    cur_kf->getPose(P, R);
    Quaterniond Q{R};
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(cur_kf->time_stamp);
    pose_stamped.header.frame_id = "global";
    pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
    pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
    pose_stamped.pose.position.z = P.z();
    pose_stamped.pose.orientation.x = Q.x();
    pose_stamped.pose.orientation.y = Q.y();
    pose_stamped.pose.orientation.z = Q.z();
    pose_stamped.pose.orientation.w = Q.w();
    base_path.poses.push_back(pose_stamped);
    base_path.header = pose_stamped.header;

    // draw local connection
    if (SHOW_S_EDGE)
    {
        list<KeyFramePtr >::reverse_iterator rit = keyframelist.rbegin();
        for (int i = 0; i < 1; i++)
        {
            if (rit == keyframelist.rend())
                break;
            Vector3d conncected_P;
            Matrix3d connected_R;
            if ((*rit)->sequence == cur_kf->sequence)
            {
                (*rit)->getPose(conncected_P, connected_R);
                posegraph_visualization->add_edge(P, conncected_P);
            }
            rit++;
        }
    }
    /*
    if (cur_kf->has_loop)
    {
        KeyFrame* connected_KF = getKeyFrame(cur_kf->loop_index);
        Vector3d connected_P;
        Matrix3d connected_R;
        connected_KF->getPose(connected_P,  connected_R);
        posegraph_visualization->add_loopedge(P, connected_P, SHIFT);
    }
    */

    keyframelist.push_back(cur_kf);
    // publish();
    m_keyframelist.unlock();
}

KeyFramePtr PoseGraph::getKeyFrame(int index)
{
    //    unique_lock<mutex> lock(m_keyframelist);
    list<KeyFramePtr>::iterator it = keyframelist.begin();
    for (; it != keyframelist.end(); it++)
    {
        if ((*it)->index == index)
            break;
    }
    if (it != keyframelist.end())
        return *it;
    else
        return nullptr;
}

int PoseGraph::detectLoop(const KeyFramePtr& keyframe, int frame_index)
{
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE)
    {
        int feature_num = keyframe->keypoints.size();
        cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[frame_index] = compressed_image;
    }
    TicToc tmp_t;
    // first query; then add this frame into database!
    QueryResults ret;
    TicToc t_query;
    unsigned int max_frame_id_allowed = std::max(0, frame_index - RECALL_IGNORE_RECENT_COUNT);
    db.query(keyframe->brief_descriptors, ret, 3, max_frame_id_allowed);
    printf("[POSEGRAPH]: query time: %f \n", t_query.toc());
    cout << "Searching for Image " << frame_index << ". " << ret << endl;

    TicToc t_add;
    if (isMappingMode())
    {
        db.add(keyframe->brief_descriptors);
    }
    // printf("[POSEGRAPH]: add feature time: %f", t_add.toc());
    //  ret[0] is the nearest neighbour's score. threshold change with neighour score
    // bool find_loop = false;
    cv::Mat loop_result;
    if (DEBUG_IMAGE)
    {
        loop_result = compressed_image.clone();
        if (ret.size() > 0)
            putText(loop_result, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
    }
    // visual loop result
    if (DEBUG_IMAGE)
    {
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            int tmp_index = ret[i].Id;
            auto it = image_pool.find(tmp_index);
            cv::Mat tmp_image = (it->second).clone();
            putText(tmp_image, "index:  " + to_string(tmp_index) + "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
            cv::hconcat(loop_result, tmp_image, loop_result);
        }
    }

    for (unsigned int i = 0; i < ret.size(); i++)
        cout << "  " << i << " - " << ret[i].Score << endl;

    // a good match with its nerghbour
    // if (ret.size() >= 1 && ret[0].Score > MIN_SCORE)
    {
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            // if (ret[i].Score > ret[0].Score * 0.3)
            if (ret[i].Score > MIN_SCORE && ret[i].Id < max_frame_id_allowed)
            {
                // find_loop = true;
                int tmp_index = ret[i].Id;
                if (DEBUG_IMAGE && 0)
                {
                    auto it = image_pool.find(tmp_index);
                    cv::Mat tmp_image = (it->second).clone();
                    putText(tmp_image, "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
                    cv::hconcat(loop_result, tmp_image, loop_result);
                }
            }
        }
    }
    /*
        if (DEBUG_IMAGE)
        {
            cv::imshow("loop_result", loop_result);
            cv::waitKey(20);
        }
    */
    // if (find_loop && frame_index > 50)
    if (frame_index < 50 && isMappingMode())
        return -1;

    // Loop through all, and see if we have one that is a good match!
    std::vector<int> done_ids;
    while (done_ids.size() < ret.size())
    {

        // First find the oldest that we have not tried yet
        unsigned int min_index = std::numeric_limits<unsigned int>::max();
        bool has_min = false;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (ret[i].Id < min_index && ret[i].Id < max_frame_id_allowed && ret[i].Score > MIN_SCORE && std::find(done_ids.begin(), done_ids.end(), ret[i].Id) == done_ids.end())
            {
                min_index = ret[i].Id;
                has_min = true;
            }
        }

        // Break out if we have not found a min
        if (!has_min)
            return -1;

        // Then try to see if we can loop close with it
        KeyFramePtr old_kf = getKeyFrame(min_index);
        if (keyframe->findConnection(old_kf))
            return min_index;
        else
            done_ids.push_back(min_index);
    }

    // failure
    return -1;
}

void PoseGraph::addKeyFrameIntoVoc(const KeyFramePtr& keyframe)
{
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE)
    {
        int feature_num = keyframe->keypoints.size();
        cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[keyframe->index] = compressed_image;
    }

    db.add(keyframe->window_brief_descriptors);
}

void PoseGraph::optimize4DoF()
{
    while (!stop_.load())
    {
        int cur_index = -1;
        int first_looped_index = -1;
        m_optimize_buf.lock();
        while (!optimize_buf.empty())
        {
            cur_index = optimize_buf.front();
            first_looped_index = earliest_loop_index;
            optimize_buf.pop();
        }
        m_optimize_buf.unlock();
        if (cur_index != -1)
        {
            TicToc tmp_t1;
            m_keyframelist.lock();
            KeyFramePtr cur_kf = getKeyFrame(cur_index);
            if (NULL == cur_kf)
            {
                m_keyframelist.unlock();
                continue;
            }
            int max_length = cur_index + 1;

            // w^t_i   w^q_i
            double t_array[max_length][3];
            Quaterniond q_array[max_length];
            double euler_array[max_length][3];
            double sequence_array[max_length];

            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            // options.minimizer_progress_to_stdout = true;
            options.max_solver_time_in_seconds = 5;
            options.max_num_iterations = 20;
            options.num_threads = 1;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(0.1);
            // loss_function = new ceres::CauchyLoss(1.0);
            ceres::LocalParameterization *angle_local_parameterization = AngleLocalParameterization::Create();

            list<KeyFramePtr >::iterator it;

            int i = 0;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                (*it)->local_index = i;
                Quaterniond tmp_q;
                Matrix3d tmp_r;
                Vector3d tmp_t;
                (*it)->getVioPose(tmp_t, tmp_r);
                tmp_q = tmp_r;
                t_array[i][0] = tmp_t(0);
                t_array[i][1] = tmp_t(1);
                t_array[i][2] = tmp_t(2);
                q_array[i] = tmp_q;

                Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
                euler_array[i][0] = euler_angle.x();
                euler_array[i][1] = euler_angle.y();
                euler_array[i][2] = euler_angle.z();

                sequence_array[i] = (*it)->sequence;

                problem.AddParameterBlock(euler_array[i], 1, angle_local_parameterization);
                problem.AddParameterBlock(t_array[i], 3);

                if ((*it)->index == first_looped_index || (*it)->sequence == 0)
                {
                    problem.SetParameterBlockConstant(euler_array[i]);
                    problem.SetParameterBlockConstant(t_array[i]);
                }

                // add edge
                for (int j = 1; j < 5; j++)
                {
                    if (i - j >= 0 && sequence_array[i] == sequence_array[i - j])
                    {
                        Vector3d euler_conncected = Utility::R2ypr(q_array[i - j].toRotationMatrix());
                        Vector3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1], t_array[i][2] - t_array[i - j][2]);
                        relative_t = q_array[i - j].inverse() * relative_t;
                        double relative_yaw = euler_array[i][0] - euler_array[i - j][0];
                        ceres::CostFunction *cost_function = FourDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                  relative_yaw, euler_conncected.y(), euler_conncected.z());
                        problem.AddResidualBlock(cost_function, NULL, euler_array[i - j],
                                                 t_array[i - j],
                                                 euler_array[i],
                                                 t_array[i]);
                    }
                }

                // add loop edge
                if ((*it)->has_loop)
                {
                    assert((*it)->loop_index >= first_looped_index);
                    int connected_index = getKeyFrame((*it)->loop_index)->local_index;
                    Vector3d euler_conncected = Utility::R2ypr(q_array[connected_index].toRotationMatrix());
                    Vector3d relative_t;
                    relative_t = (*it)->getLoopRelativeT();
                    double relative_yaw = (*it)->getLoopRelativeYaw();
                    ceres::CostFunction *cost_function = FourDOFWeightError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                    relative_yaw, euler_conncected.y(), euler_conncected.z());
                    problem.AddResidualBlock(cost_function, loss_function, euler_array[connected_index],
                                             t_array[connected_index],
                                             euler_array[i],
                                             t_array[i]);
                }

                if ((*it)->index == cur_index)
                    break;
                i++;
            }
            m_keyframelist.unlock();
            double t_create = tmp_t1.toc();
            TicToc tmp_t2;
            ceres::Solve(options, &problem, &summary);
            double t_opt = tmp_t2.toc();
            std::cout << summary.BriefReport() << "\n";

            printf("[POSEGRAPH]: pose optimization time: %f \n", tmp_t2.toc());
            /*
            for (int j = 0 ; j < i; j++)
            {
                printf("[POSEGRAPH]: optimize i: %d p: %f, %f, %f\n", j, t_array[j][0], t_array[j][1], t_array[j][2] );
            }
            */
            TicToc tmp_t3;
            m_keyframelist.lock();
            i = 0;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                Quaterniond tmp_q;
                tmp_q = Utility::ypr2R(Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
                Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
                Matrix3d tmp_r = tmp_q.toRotationMatrix();
                (*it)->updatePose(tmp_t, tmp_r);

                if ((*it)->index == cur_index)
                    break;
                i++;
            }

            Vector3d cur_t, vio_t;
            Matrix3d cur_r, vio_r;
            cur_kf->getPose(cur_t, cur_r);
            cur_kf->getVioPose(vio_t, vio_r);
            m_drift.lock();
            yaw_drift = Utility::R2ypr(cur_r).x() - Utility::R2ypr(vio_r).x();
            r_drift = Utility::ypr2R(Vector3d(yaw_drift, 0, 0));
            t_drift = cur_t - r_drift * vio_t;
            m_drift.unlock();
            // cout << "t_drift " << t_drift.transpose() << endl;
            // cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;
            // cout << "yaw drift " << yaw_drift << endl;

            it++;
            for (; it != keyframelist.end(); it++)
            {
                Vector3d P;
                Matrix3d R;
                (*it)->getVioPose(P, R);
                P = r_drift * P + t_drift;
                R = r_drift * R;
                (*it)->updatePose(P, R);
            }
            m_keyframelist.unlock();
            updatePath();
            double t_update = tmp_t3.toc();

            // Nice debug print
            printf(" ,[POSEGRAPH]: creation %.3f ms | optimization %.3f ms | update %.3f ms | %.3f dyaw, %.3f dpos\n", t_create, t_opt, t_update, yaw_drift, t_drift.norm());
        }

        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }
    return;
}

void PoseGraph::optimize6DoF()
{
    while (!stop_.load())
    {
        int cur_index = -1;
        int first_looped_index = -1;
        m_optimize_buf.lock();
        while (!optimize_buf.empty())
        {
            cur_index = optimize_buf.front();
            first_looped_index = earliest_loop_index;
            optimize_buf.pop();
        }
        m_optimize_buf.unlock();
        if (cur_index != -1)
        {
            // printf("[POSEGRAPH]: optimize pose graph \n");
            TicToc tmp_t;
            m_keyframelist.lock();
            KeyFramePtr cur_kf = getKeyFrame(cur_index);

            int max_length = cur_index + 1;

            // w^t_i   w^q_i
            double t_array[max_length][3];
            double q_array[max_length][4];
            double sequence_array[max_length];

            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            // ptions.minimizer_progress_to_stdout = true;
            options.max_solver_time_in_seconds = 5;
            options.max_num_iterations = 20;
            options.num_threads = 1;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(0.1);
            // loss_function = new ceres::CauchyLoss(1.0);
            ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();

            list<KeyFramePtr >::iterator it;

            int i = 0;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                (*it)->local_index = i;
                Quaterniond tmp_q;
                Matrix3d tmp_r;
                Vector3d tmp_t;
                (*it)->getVioPose(tmp_t, tmp_r);
                tmp_q = tmp_r;
                t_array[i][0] = tmp_t(0);
                t_array[i][1] = tmp_t(1);
                t_array[i][2] = tmp_t(2);
                q_array[i][0] = tmp_q.w();
                q_array[i][1] = tmp_q.x();
                q_array[i][2] = tmp_q.y();
                q_array[i][3] = tmp_q.z();

                sequence_array[i] = (*it)->sequence;

                problem.AddParameterBlock(q_array[i], 4, local_parameterization);
                problem.AddParameterBlock(t_array[i], 3);

                if ((*it)->index == first_looped_index || (*it)->sequence == 0)
                {
                    problem.SetParameterBlockConstant(q_array[i]);
                    problem.SetParameterBlockConstant(t_array[i]);
                }

                // add edge
                for (int j = 1; j < 5; j++)
                {
                    if (i - j >= 0 && sequence_array[i] == sequence_array[i - j])
                    {
                        Vector3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1], t_array[i][2] - t_array[i - j][2]);
                        Quaterniond q_i_j = Quaterniond(q_array[i - j][0], q_array[i - j][1], q_array[i - j][2], q_array[i - j][3]);
                        Quaterniond q_i = Quaterniond(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
                        relative_t = q_i_j.inverse() * relative_t;
                        Quaterniond relative_q = q_i_j.inverse() * q_i;
                        ceres::CostFunction *vo_function = RelativeRTError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                   relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
                                                                                   0.1, 0.01);
                        problem.AddResidualBlock(vo_function, NULL, q_array[i - j], t_array[i - j], q_array[i], t_array[i]);
                    }
                }

                // add loop edge

                if ((*it)->has_loop)
                {
                    assert((*it)->loop_index >= first_looped_index);
                    int connected_index = getKeyFrame((*it)->loop_index)->local_index;
                    Vector3d relative_t;
                    relative_t = (*it)->getLoopRelativeT();
                    Quaterniond relative_q;
                    relative_q = (*it)->getLoopRelativeQ();
                    ceres::CostFunction *loop_function = RelativeRTError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                 relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
                                                                                 0.1, 0.01);
                    problem.AddResidualBlock(loop_function, loss_function, q_array[connected_index], t_array[connected_index], q_array[i], t_array[i]);
                }

                if ((*it)->index == cur_index)
                    break;
                i++;
            }
            m_keyframelist.unlock();

            ceres::Solve(options, &problem, &summary);
            // std::cout << summary.BriefReport() << "\n";

            // printf("[POSEGRAPH]: pose optimization time: %f \n", tmp_t.toc());
            /*
            for (int j = 0 ; j < i; j++)
            {
                printf("[POSEGRAPH]: optimize i: %d p: %f, %f, %f\n", j, t_array[j][0], t_array[j][1], t_array[j][2] );
            }
            */
            m_keyframelist.lock();
            i = 0;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                Quaterniond tmp_q(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
                Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
                Matrix3d tmp_r = tmp_q.toRotationMatrix();
                (*it)->updatePose(tmp_t, tmp_r);

                if ((*it)->index == cur_index)
                    break;
                i++;
            }

            Vector3d cur_t, vio_t;
            Matrix3d cur_r, vio_r;
            cur_kf->getPose(cur_t, cur_r);
            cur_kf->getVioPose(vio_t, vio_r);
            m_drift.lock();
            r_drift = cur_r * vio_r.transpose();
            t_drift = cur_t - r_drift * vio_t;
            m_drift.unlock();
            // cout << "t_drift " << t_drift.transpose() << endl;
            // cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;

            it++;
            for (; it != keyframelist.end(); it++)
            {
                Vector3d P;
                Matrix3d R;
                (*it)->getVioPose(P, R);
                P = r_drift * P + t_drift;
                R = r_drift * R;
                (*it)->updatePose(P, R);
            }
            m_keyframelist.unlock();
            updatePath();

            // Nice debug print
            printf("[POSEGRAPH]: pose optimization in %.3f seconds | %.3f dori, %.3f dpos\n", tmp_t.toc(), Utility::R2ypr(r_drift).norm(), t_drift.norm());
        }

        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }
    return;
}

void PoseGraph::updatePath()
{
    m_keyframelist.lock();
    list<KeyFramePtr >::iterator it;
    for (int i = 1; i <= sequence_cnt; i++)
    {
        path[i].poses.clear();
    }
    base_path.poses.clear();
    posegraph_visualization->reset();

    if (SAVE_LOOP_PATH)
    {
        ofstream loop_path_file_tmp(VINS_RESULT_PATH, ios::out);
        loop_path_file_tmp.close();
    }

    for (it = keyframelist.begin(); it != keyframelist.end(); it++)
    {
        Vector3d P;
        Matrix3d R;
        (*it)->getPose(P, R);
        Quaterniond Q;
        Q = R;
        //        printf("[POSEGRAPH]: path p: %f, %f, %f\n",  P.x(),  P.z(),  P.y() );

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time((*it)->time_stamp);
        pose_stamped.header.frame_id = "global";
        pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
        pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
        pose_stamped.pose.position.z = P.z();
        pose_stamped.pose.orientation.x = Q.x();
        pose_stamped.pose.orientation.y = Q.y();
        pose_stamped.pose.orientation.z = Q.z();
        pose_stamped.pose.orientation.w = Q.w();
        if ((*it)->sequence == 0)
        {
            base_path.poses.push_back(pose_stamped);
            base_path.header = pose_stamped.header;
        }
        else
        {
            path[(*it)->sequence].poses.push_back(pose_stamped);
            path[(*it)->sequence].header = pose_stamped.header;
        }

        if (SAVE_LOOP_PATH)
        {
            ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
            loop_path_file.setf(ios::fixed, ios::floatfield);
            loop_path_file.precision(0);
            loop_path_file << (*it)->time_stamp * 1e9 << ",";
            loop_path_file.precision(5);
            loop_path_file << P.x() << ","
                           << P.y() << ","
                           << P.z() << ","
                           << Q.w() << ","
                           << Q.x() << ","
                           << Q.y() << ","
                           << Q.z() << ","
                           << endl;
            loop_path_file.close();
        }
        // draw local connection
        if (SHOW_S_EDGE)
        {
            list<KeyFramePtr >::reverse_iterator rit = keyframelist.rbegin();
            list<KeyFramePtr >::reverse_iterator lrit;
            for (; rit != keyframelist.rend(); rit++)
            {
                if ((*rit)->index == (*it)->index)
                {
                    lrit = rit;
                    lrit++;
                    for (int i = 0; i < 4; i++)
                    {
                        if (lrit == keyframelist.rend())
                            break;
                        if ((*lrit)->sequence == (*it)->sequence)
                        {
                            Vector3d conncected_P;
                            Matrix3d connected_R;
                            (*lrit)->getPose(conncected_P, connected_R);
                            posegraph_visualization->add_edge(P, conncected_P);
                        }
                        lrit++;
                    }
                    break;
                }
            }
        }
        if (SHOW_L_EDGE)
        {
            if ((*it)->has_loop && (*it)->sequence == sequence_cnt)
            {

                KeyFramePtr connected_KF = getKeyFrame((*it)->loop_index);
                Vector3d connected_P;
                Matrix3d connected_R;
                connected_KF->getPose(connected_P, connected_R);
                //(*it)->getVioPose(P, R);
                (*it)->getPose(P, R);
                if ((*it)->sequence > 0)
                {
                    posegraph_visualization->add_loopedge(P, connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
                }
            }
        }
    }
    publish();
    m_keyframelist.unlock();
}

void PoseGraph::savePoseGraph()
{
    m_keyframelist.lock();
    TicToc tmp_t;
    FILE *pFile;
    printf("[POSEGRAPH]: pose graph path: %s\n", POSE_GRAPH_SAVE_PATH.c_str());
    printf("[POSEGRAPH]: pose graph saving... \n");
    string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
    pFile = fopen(file_path.c_str(), "w");
    // fprintf(pFile, "index time_stamp Tx Ty Tz Qw Qx Qy Qz loop_index loop_info\n");
    // 已保存过的点 ID
    std::unordered_set<double> saved_ids;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    list<KeyFramePtr >::iterator it;
    for (it = keyframelist.begin(); it != keyframelist.end(); it++)
    {
        std::string image_path, descriptor_path, brief_path, keypoints_path;
        if (DEBUG_IMAGE)
        {
            image_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_image.png";
            imwrite(image_path.c_str(), (*it)->image);
        }
        Quaterniond VIO_tmp_Q{(*it)->vio_R_w_i};
        Quaterniond PG_tmp_Q{(*it)->R_w_i};
        Vector3d VIO_tmp_T = (*it)->vio_T_w_i;
        Vector3d PG_tmp_T = (*it)->T_w_i;

        if((*it)->window_brief_descriptors.empty()){
            continue;
        }

        fprintf(pFile, " %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f %f %d\n", (*it)->index, (*it)->time_stamp,
                VIO_tmp_T.x(), VIO_tmp_T.y(), VIO_tmp_T.z(),
                PG_tmp_T.x(), PG_tmp_T.y(), PG_tmp_T.z(),
                VIO_tmp_Q.w(), VIO_tmp_Q.x(), VIO_tmp_Q.y(), VIO_tmp_Q.z(),
                PG_tmp_Q.w(), PG_tmp_Q.x(), PG_tmp_Q.y(), PG_tmp_Q.z(),
                (*it)->loop_index,
                (*it)->loop_info(0), (*it)->loop_info(1), (*it)->loop_info(2), (*it)->loop_info(3),
                (*it)->loop_info(4), (*it)->loop_info(5), (*it)->loop_info(6), (*it)->loop_info(7),
                (int)(*it)->point_2d_uv.size());

        // write keypoints, brief_descriptors   vector<cv::KeyPoint> keypoints vector<BRIEF::bitset> brief_descriptors;
        // printf("2d uv size %zu ,window_brief_descriptors size %zu \n",(*it)->point_2d_uv.size(),(*it)->window_brief_descriptors.size());
        assert((*it)->point_2d_uv.size() == (*it)->window_brief_descriptors.size());
        brief_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_briefdes.dat";
        std::ofstream brief_file(brief_path, std::ios::binary);
        keypoints_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_keypoints.txt";
        FILE *keypoints_file;
        keypoints_file = fopen(keypoints_path.c_str(), "w");
        // int uv_size = static_cast<int>((*it)->point_2d_uv.size());
        for (int i = 0; i < (int)(*it)->point_2d_uv.size(); i++)
        {
            brief_file << (*it)->window_brief_descriptors[i] << endl;
            // float pt_id = i > uv_size? -1.0 : (*it)->point_id[i];
            fprintf(keypoints_file, "%f %f %f %f %f\n", (*it)->point_2d_uv[i].x, (*it)->point_2d_uv[i].y,
                    (*it)->point_2d_norm[i].x, (*it)->point_2d_norm[i].y,(*it)->point_id[i]);
        }
        brief_file.close();
        fclose(keypoints_file);

        const auto& pts = (*it)->point_3d;
        const auto& ids = (*it)->point_id;
        if (pts.size() != ids.size()) {
            std::cerr << "KeyFrame has mismatched point_3d and point_id sizes!" << std::endl;
            continue;
        }
        for (size_t i = 0; i < pts.size(); ++i) {
            double id = ids[i];
            if (saved_ids.find(id) != saved_ids.end()) {
                continue;
            }

            const cv::Point3f& pt = pts[i];
            pcl::PointXYZI pcl_point;
            pcl_point.x = pt.x;
            pcl_point.y = pt.y;
            pcl_point.z = pt.z;
            pcl_point.intensity = static_cast<float>(id);  // 使用 intensity 存储 id
            cloud->points.push_back(pcl_point);

            saved_ids.insert(id);
        }

    }
    fclose(pFile);
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    string pcd_file_name = VINS_RESULT_PATH + "/keyframes_point3d.pcd";
    std::cout<<"pcd point cloud save path "<<pcd_file_name<<std::endl;
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // 创建颜色处理器：根据 intensity 映射颜色
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(cloud, "intensity");
    if (!color_handler.isCapable()) {
        std::cerr << "Color handler not capable. Showing in white." << std::endl;
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> white(cloud, 255, 255, 255);
        viewer->addPointCloud<pcl::PointXYZI>(cloud, white, "cloud");
    } else {
        viewer->addPointCloud<pcl::PointXYZI>(cloud, color_handler, "cloud");
    }

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // 可视化主循环
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (pcl::io::savePCDFileBinary(pcd_file_name, *cloud) == 0) {
        std::cout << "Saved " << cloud->points.size() << " unique points to " << pcd_file_name << std::endl;
    } else {
        std::cerr << "Failed to save PCD file: " << pcd_file_name << std::endl;
    }

    printf("[POSEGRAPH]: save pose graph time: %f s\n", tmp_t.toc() / 1000);
    m_keyframelist.unlock();
}
void PoseGraph::loadPoseGraph()
{
    id_point_map.clear();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    string pcd_file_name = VINS_RESULT_PATH + "keyframes_point3d.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file_name, *cloud) == -1) {
        std::cerr << "Failed to load PCD file: " << pcd_file_name << std::endl;
    }


    for (const auto& pt : cloud->points) {
        int id = static_cast<int>(pt.intensity);  // 将 intensity 作为 ID
        cv::Point3f position(pt.x, pt.y, pt.z);
        // std::cout<<"id "<<id<<" pt: "<<pt.x<<" "<<pt.y<<" "<<pt.z<<std::endl;
        id_point_map[id] = position;  // 如果已有相同 ID，会被覆盖
    }
    std::cout << "Loaded " << cloud->size() << " points from " << pcd_file_name << std::endl;

    
    TicToc tmp_t;
    FILE *pFile;
    string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
    printf("[POSEGRAPH]: lode pose graph from: %s \n", file_path.c_str());
    printf("[POSEGRAPH]: pose graph loading...\n");
    pFile = fopen(file_path.c_str(), "r");
    if (pFile == NULL)
    {
        printf("[POSEGRAPH]: lode previous pose graph error: wrong previous pose graph path or no previous pose graph \n the system will start with new pose graph \n");
        return;
    }
    int index;
    double time_stamp;
    double VIO_Tx, VIO_Ty, VIO_Tz;
    double PG_Tx, PG_Ty, PG_Tz;
    double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
    double PG_Qw, PG_Qx, PG_Qy, PG_Qz;
    double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
    double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
    int loop_index;
    int keypoints_num;
    Eigen::Matrix<double, 8, 1> loop_info;
    int cnt = 0;
    while (fscanf(pFile, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d", &index, &time_stamp,
                  &VIO_Tx, &VIO_Ty, &VIO_Tz,
                  &PG_Tx, &PG_Ty, &PG_Tz,
                  &VIO_Qw, &VIO_Qx, &VIO_Qy, &VIO_Qz,
                  &PG_Qw, &PG_Qx, &PG_Qy, &PG_Qz,
                  &loop_index,
                  &loop_info_0, &loop_info_1, &loop_info_2, &loop_info_3,
                  &loop_info_4, &loop_info_5, &loop_info_6, &loop_info_7,
                  &keypoints_num) != EOF)
    {
        /*
        printf("[POSEGRAPH]: I read: %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d\n", index, time_stamp,
                                    VIO_Tx, VIO_Ty, VIO_Tz,
                                    PG_Tx, PG_Ty, PG_Tz,
                                    VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz,
                                    PG_Qw, PG_Qx, PG_Qy, PG_Qz,
                                    loop_index,
                                    loop_info_0, loop_info_1, loop_info_2, loop_info_3,
                                    loop_info_4, loop_info_5, loop_info_6, loop_info_7,
                                    keypoints_num);
        */
        cv::Mat image;
        std::string image_path, descriptor_path;
        if (DEBUG_IMAGE)
        {
            image_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_image.png";
            image = cv::imread(image_path.c_str(), 0);
        }

        Vector3d VIO_T(VIO_Tx, VIO_Ty, VIO_Tz);
        Vector3d PG_T(PG_Tx, PG_Ty, PG_Tz);
        Quaterniond VIO_Q;
        VIO_Q.w() = VIO_Qw;
        VIO_Q.x() = VIO_Qx;
        VIO_Q.y() = VIO_Qy;
        VIO_Q.z() = VIO_Qz;
        Quaterniond PG_Q;
        PG_Q.w() = PG_Qw;
        PG_Q.x() = PG_Qx;
        PG_Q.y() = PG_Qy;
        PG_Q.z() = PG_Qz;
        Matrix3d VIO_R, PG_R;
        VIO_R = VIO_Q.toRotationMatrix();
        PG_R = PG_Q.toRotationMatrix();
        Eigen::Matrix<double, 8, 1> loop_info;
        loop_info << loop_info_0, loop_info_1, loop_info_2, loop_info_3, loop_info_4, loop_info_5, loop_info_6, loop_info_7;

        if (loop_index != -1)
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
            {
                earliest_loop_index = loop_index;
            }

        // load keypoints, brief_descriptors
        string brief_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_briefdes.dat";
        std::ifstream file(brief_path);
        if (!file.good()) {
          // 文件不存在或无法打开
           continue;
        }
        std::ifstream brief_file(brief_path, std::ios::binary);
        string keypoints_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_keypoints.txt";
        FILE *keypoints_file;
        keypoints_file = fopen(keypoints_path.c_str(), "r");
        vector<cv::Point2f> keypoints;
        vector<cv::Point2f> keypoints_norm;
        vector<cv::Point3f> point_3d;
        vector<BRIEF::bitset> brief_descriptors;
        for (int i = 0; i < keypoints_num; i++)
        {
            BRIEF::bitset tmp_des;
            brief_file >> tmp_des;
            brief_descriptors.push_back(tmp_des);
            double p_x, p_y, p_x_norm, p_y_norm, point_id;
            if (!fscanf(keypoints_file, "%lf %lf %lf %lf %lf", &p_x, &p_y, &p_x_norm, &p_y_norm, &point_id))
                printf("[POSEGRAPH]:  fail to load pose graph \n");
            cv::Point2f tmp_keypoint;
            cv::Point2f tmp_keypoint_norm;
            tmp_keypoint.x = p_x;
            tmp_keypoint.y = p_y;
            tmp_keypoint_norm.x = p_x_norm;
            tmp_keypoint_norm.y = p_y_norm;
            keypoints.push_back(tmp_keypoint);
            keypoints_norm.push_back(tmp_keypoint_norm);
            auto it = id_point_map.find(static_cast<int>(point_id));
            if(it != id_point_map.end()){
                point_3d.emplace_back(it->second);
            }
        }
        if(point_3d.size() != static_cast<long unsigned int>(keypoints_num)){
            point_3d.clear();
        }
        brief_file.close();
        fclose(keypoints_file);

        KeyFramePtr keyframe = std::make_shared<KeyFrame>(time_stamp, index, VIO_T, VIO_R, PG_T, PG_R, image, 
                                loop_index, loop_info, keypoints, keypoints_norm, brief_descriptors, point_3d);
        if(!keyframe->point_3d.empty()){
            loadKeyFrame(keyframe, 0);
        }
        if (cnt % 20 == 0)
        {
            publish();
        }
        cnt++;
    }
    fclose(pFile);
    printf("[POSEGRAPH]: load pose graph time: %f s\n", tmp_t.toc() / 1000);
    base_sequence = 0;

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "map";  // 指定坐标系
    cloud_msg.header.stamp = ros::Time::now();

    // 3. 发布
    pub_map_points.publish(cloud_msg);
    ROS_INFO("publish map points end \n");
}

void PoseGraph::publish()
{
    for (int i = 1; i <= sequence_cnt; i++)
    {
        // if (sequence_loop[i] == true || i == base_sequence)
        if (1 || i == base_sequence)
        {
            pub_pg_path.publish(path[i]);
            pub_path[i].publish(path[i]);
            posegraph_visualization->publish_by(pub_pose_graph, path[sequence_cnt].header);
        }
    }
    pub_base_path.publish(base_path);
    // posegraph_visualization->publish_by(pub_pose_graph, path[sequence_cnt].header);
}

bool PoseGraph::isKeyFrame(const TimedPose &pose)
{
    if ((pose.t_ - pre_key_pose_.t_).norm() > 0.04 ||
        std::fabs(Utility::GetYaw(pre_key_pose_.R_.inverse() * pose.R_)) > 5. * M_PI / 180.0 ||
        pose.time_ - pre_key_pose_.time_ > 30 || pre_key_pose_.time_ < -1)
    {
        pre_key_pose_ = pose;
        return true;
    }
    return false;
}

void PoseGraph::startTFThread()
{
    tf_thread = std::thread([this]()
                            { pubTFThread(); });
}

void PoseGraph::pubTFThread()
{
    static tf::TransformBroadcaster tf_broadcaster;

    const double PUB_RATE = 30.0; // 20Hz
    const double PUB_PERIOD = 1.0 / PUB_RATE;

    ros::Time last_pub_time = ros::Time::now();

    while (thread_run && !stop_.load())
    {
        ros::Time now = ros::Time::now();
        double dt = (now - last_pub_time).toSec();

        if (dt >= PUB_PERIOD)
        {
            // 获取 posegraph 中的漂移信息
            m_drift.lock();
            Eigen::Matrix3d R = r_drift * w_r_vio;
            Eigen::Vector3d t = r_drift * w_t_vio + t_drift;
            Eigen::Quaterniond q(R);
            m_drift.unlock();

            tf::StampedTransform trans;
            trans.frame_id_ = "map";
            trans.child_frame_id_ = "odom";
            trans.stamp_ = now;
            tf::Quaternion tf_q(q.x(), q.y(), q.z(), q.w());
            tf::Vector3 tf_t(t.x(), t.y(), t.z());
            trans.setRotation(tf_q);
            trans.setOrigin(tf_t);

            tf_broadcaster.sendTransform(trans);

            last_pub_time = now;
        }

        // 降低 CPU 占用（不要写成 sleep 太久，以免错过时间点）
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
}
