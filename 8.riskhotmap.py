# coding=utf-8

# 数据路径
import ast
import csv
import getopt
import logging
import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import coordinateconvert
# import seaborn as sns
csv.field_size_limit(500 * 1024 * 1024)

object_classification_class = ['未知目标', '汽车', '卡车', '摩托车', '行人', '自行车', '动物', '巴士', '其他', '购物车',
                               '柱子', '锥桶', '已被锁上的车位锁', '未被锁上的车位锁 ']

linestyle_tuple = [
    ('loosely dotted', (0, (1, 10))),
    ('dotted', (0, (1, 1))),
    ('densely dotted', (0, (1, 2))),
    ('loosely dashed', (0, (5, 10))),
    ('dashed', (0, (5, 5))),
    ('densely dashed', (0, (5, 1))),
    ('loosely dashdotted', (0, (3, 10, 1, 10))),
    ('dashdotted', (0, (3, 5, 1, 5))),
    ('densely dashdotted', (0, (3, 1, 1, 1))),
    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


def prepare_data(path,targetfile,is_save_generate_image=1, is_record_inf=1, no_image_file_format=1, is_generated_video=1):
    # 读取文件夹下所有文件的名字并把他们用列表存起来
    # path = "D:/DataContest/data/0802 (无图数据1)/0802/"
    is_fig = 0
    if is_save_generate_image == 1 or not no_image_file_format == 1 or is_generated_video==1:
        is_fig = 1
    # path='/home/lawrence/Documents/Bigcontest/data/0802(无图数据1)/0802/'
    # file_image = 'F:\\car\\object\\image4\\0805 (1)\\0805\\image'
    datanames = os.listdir(path)
    file_list = []
    risk_map = []
    global_complex_map = []
    for i in datanames:
        file_list.append(i)
    already_draw_one = 0
    for b in file_list:
        row_total = 0
        (filename, extension) = os.path.splitext(b)
        file = path + b
        if not os.access(file, os.X_OK):
            continue
        if not b == targetfile:
            continue
        try:
            row_total = len(open(file).readlines())
        except:
            continue
        # row_total = 1
        storepath = path + 'images/' + filename + '/image2/'
        if not os.path.exists(path + 'images/'):
            os.mkdir(path + 'images/')
        if not os.path.exists(path + 'images/' + filename):
            os.mkdir(path + 'images/' + filename)
        if not os.path.exists(storepath):
            os.mkdir(storepath)
        rec_header = ['index', 'forward_speed', 'location_x', 'location_y', 'rotation_yaw','complex_value']
        shotname, extension = os.path.splitext(extension)
        if is_record_inf == 1:
            log_file = open(storepath + 'carinfo.csv', 'a+', encoding='utf-8', newline='')
            log_csv_writer = csv.writer(log_file)
            log_csv_writer.writerow(rec_header)
        with open(file) as f:
            reader = csv.reader(f)
            header = next(reader)
            esp_yaw_rate_stp_motion_his = []
            esp_vehicle_speed_stp_motion_his = []
            esp_lat_accel_stp_motion_his = []
            esp_long_accel_stp_motion_his = []
            sas_steering_angle_stp_motion_his = []
            timestamp_of_location_his = []
            latitude_of_location_his = []
            longitude_of_location_his = []
            altitude_of_location_his = []

            timeindex = []  # np.arange(0, row_total, 1)
            static_complex_map = []
            dynamic_complex_map = []

            already_draw_one = 0
            for a in range(row_total):
                # fig = plt.figure()

                # if already_draw_one == 2:
                #     break;
                timeindex.append(already_draw_one)
                first0 = next(reader, None)
                if first0 is None:
                    break
                # 1.目标检测
                object_detection_list = first0[0]
                object_detection_list = ast.literal_eval(object_detection_list)

                # 2.自车运动
                esp_yaw_rate_stp_motion = first0[1]  # 偏航率(车横摆角速度)[degree/s]
                esp_yaw_rate_stp_motion = ast.literal_eval(esp_yaw_rate_stp_motion)
                esp_yaw_rate_stp_motion_his.append(esp_yaw_rate_stp_motion)

                esp_vehicle_speed_stp_motion = first0[2]  # 车速信号
                esp_vehicle_speed_stp_motion = ast.literal_eval(esp_vehicle_speed_stp_motion)
                esp_vehicle_speed_stp_motion_his.append(esp_vehicle_speed_stp_motion)

                esp_lat_accel_stp_motion = first0[3]  # 横向加速度
                esp_lat_accel_stp_motion = ast.literal_eval(esp_lat_accel_stp_motion)
                esp_lat_accel_stp_motion_his.append(esp_vehicle_speed_stp_motion)

                esp_long_accel_stp_motion = first0[4]  # 纵向加速度
                esp_long_accel_stp_motion = ast.literal_eval(esp_long_accel_stp_motion)
                esp_long_accel_stp_motion_his.append(esp_long_accel_stp_motion)

                sas_steering_angle_stp_motion = first0[5]  # 方向盘转向角度[degree]
                sas_steering_angle_stp_motion = ast.literal_eval(sas_steering_angle_stp_motion)
                sas_steering_angle_stp_motion_his.append(esp_long_accel_stp_motion)

                if is_fig == 1:
                    plt.subplot(235)
                    plt.title('Self Vec Motion')  # 自车运动
                    plt.plot(timeindex, esp_yaw_rate_stp_motion_his, linestyle='solid', color='g',
                             label='偏航率(车横摆角速度)[degree/s]')
                    plt.plot(timeindex, esp_vehicle_speed_stp_motion_his, 'ro:', linewidth=1.0, label='车速信号')
                    plt.plot(timeindex, esp_lat_accel_stp_motion_his, 'bo:', linewidth=1.0, label='横向加速度')
                    plt.plot(timeindex, esp_long_accel_stp_motion_his, 'yo:', linewidth=1.0, label='纵向加速度')
                    plt.plot(timeindex, sas_steering_angle_stp_motion_his, 'mo:', linewidth=1.0,
                             label='方向盘转向角度[degree]')
                # 3.车道线
                Lane_line_list = first0[6]
                Lane_line_list = ast.literal_eval(Lane_line_list)
                # 4.动态地图
                vehicle_pos_lng_hdmap = first0[7]  # 锚点经度坐标，请求地图定位点
                vehicle_pos_lat_hdmap = first0[8]  # 锚点维度坐标，请求地图定位点
                vehicle_pos_lng_hdmap = ast.literal_eval(vehicle_pos_lng_hdmap)
                vehicle_pos_lat_hdmap = ast.literal_eval(vehicle_pos_lat_hdmap)
                x0, y0 = coordinateconvert.millerToXY(float(vehicle_pos_lng_hdmap), float(vehicle_pos_lat_hdmap))
                vehicle_pos_current_link_id_hdmap = first0[9]  # 当前位置所在的link ID
                vehicle_pos_current_lane_num_hdmap = first0[10]  # 当前所在车道编号
                path_planning_routing_path_hdmap = first0[11]  # 导航路径上的link-id和可行车道
                lane_curvature_100m_hdmap = first0[12]  # 前方100m车道曲率[1/m]
                lane_curvature_200m_hdmap = first0[13]  # 前方200m车道曲率
                lane_curvature_300m_hdmap = first0[14]  # 前方300m车道曲率

                 # [bd_lng, bd_lat]=coordinateconvert.gcj02tobd09(float(vehicle_pos_lng_hdmap), float(vehicle_pos_lat_hdmap))
                # logging.debug('动态地图经纬度：['+str(bd_lng)+','+str(bd_lat)+']')

                # 5.静态地图
                link_list_hdmap = first0[15]
                link_list_hdmap = ast.literal_eval(link_list_hdmap)
                # 画静态地图
                if is_fig == 1:
                    plt.subplot(234)
                    plt.title('Statics Map')  # 静态地图
                    plt.xlim(-500, 50)
                    plt.ylim(-500, 500)
                statics_comple_value2 =Draw_link_list_hdmap_static_map(x0, y0, link_list_hdmap,is_fig)
                # static_complex_map.append(statics_comple_value2)
                # 6.图片数据
                if is_fig == 1:
                    plt.subplot(221)
                    # fig.add_subplot(221)
                    plt.title('Image Data')  # 图片数据
                BacauseOfThereIsImage = 0
                if not no_image_file_format == 1:
                    BacauseOfThereIsImage = 1
                    imageData_image = first0[16]
                    Image_Data(imageData_image, '')  # storepath+ '/scence' + str(a) + '.png'

                # 7.定位
                timestamp_of_location = first0[16 + BacauseOfThereIsImage]  # 时间戳
                heading_of_location = first0[17 + BacauseOfThereIsImage]  # 航向角:[deg]
                # plt.subplot(236)
                # plt.title('latitude_+longitude')  # 航向角+车辆经纬度
                # plt.plot(timestamp_of_location, heading_of_location, 'bo:')
                latitude_of_location = first0[18 + BacauseOfThereIsImage]  # 车辆定位纬度
                latitude_of_location = ast.literal_eval(latitude_of_location)

                longitude_of_location = first0[19 + BacauseOfThereIsImage]  # 车辆定位经度
                longitude_of_location = ast.literal_eval(longitude_of_location)

                altitude_of_location = first0[20 + BacauseOfThereIsImage]  # 车辆定位高度
                altitude_of_location = ast.literal_eval(altitude_of_location)

                timestamp_of_location_his.append(timestamp_of_location)
                latitude_of_location_his.append(latitude_of_location)
                longitude_of_location_his.append(longitude_of_location)
                altitude_of_location_his.append(altitude_of_location)

                # 如果车速超速1倍 超过曲率限制的风险为5
                if float(lane_curvature_100m_hdmap) > 0:
                    if esp_vehicle_speed_stp_motion > math.sqrt(0.8 * 9.8 * abs(1 / float(lane_curvature_100m_hdmap))):
                        risk_map = add_to_risk_map(5, vehicle_pos_lat_hdmap, vehicle_pos_lng_hdmap, risk_map,latitude_of_location,longitude_of_location)
                if float(lane_curvature_200m_hdmap) > 0:
                    if esp_vehicle_speed_stp_motion > math.sqrt(0.8 * 9.8 * abs(1 / float(lane_curvature_200m_hdmap))):
                        risk_map = add_to_risk_map(4, vehicle_pos_lat_hdmap, vehicle_pos_lng_hdmap, risk_map,latitude_of_location,longitude_of_location)
                if float(lane_curvature_300m_hdmap) > 0:
                    if esp_vehicle_speed_stp_motion > math.sqrt(0.8 * 9.8 * abs(1 / float(lane_curvature_300m_hdmap))):
                        risk_map = add_to_risk_map(3, vehicle_pos_lat_hdmap, vehicle_pos_lng_hdmap, risk_map,latitude_of_location,longitude_of_location)

                # plt.plot(timestamp_of_location_his, latitude_of_location_his, 'bo:', label='latitude_of_location')
                # plt.plot(timestamp_of_location_his, longitude_of_location_his, 'ro:', label='longitude_of_location')
                # plt.plot(timestamp_of_location_his, altitude_of_location_his, 'go:', label='altitude_of_location')

                # 8.驾驶员行为数据
                bcmlight = first0[21 + BacauseOfThereIsImage]  # 转向灯开关状态信号（未使用：0，左转：1，右转：2，未知：3）
                # 9.可行驶区域点集
                if is_fig == 1:
                    plt.subplot(232)
                    plt.title('Freespace')  # 可行驶区域点集
                Draw_Free_Space_Desc(first0, BacauseOfThereIsImage,is_fig)

                longitudinal = 0  # 纵向
                lateral = 0  # 横向
                length = 5
                width = 5
                txt = ['car']
                # 10.----开始画图------
                # 目标检测 objs/fus_objs
                # 遍历object数组
                if is_fig == 1:
                    plt.subplot(222)
                    # fig.add_subplot(222)
                    plt.title('POI')  # 目标数据
                    plt.xlim(-40, 40)
                    # 设置y轴的刻度范围
                    plt.ylim(-40, 40)
                risk_map,statics_complex_value,dynamic_complex_value = Draw_Dection_POI(is_fig,object_detection_list,risk_map,latitude_of_location,longitude_of_location,risk_map)
                static_complex_map.append(statics_complex_value+statics_comple_value2)
                dynamic_complex_map.append(dynamic_complex_value)
                # 把所有车道线搞出来
                Draw_Lane_Line(is_fig,Lane_line_list)

                global_complex_map.append(statics_complex_value+statics_comple_value2+dynamic_complex_value)

                # 画热点图
                if is_fig == 1:
                    # plt.subplot(222)
                    # plt.title('POI')  # 目标数据
                    # plt.xlim(-40, 40)
                    # # 设置y轴的刻度范围
                    # plt.ylim(-40, 40)
                    fig = plt.figure()
                    # fig.add_subplot(212)
                    # ax=plt.subplot(212)
                    plt.title('Env Complex Value')  # 目标数据
                    plt.plot(timeindex, static_complex_map, color='green', label='static_complex_map')
                    plt.plot(timeindex, dynamic_complex_map, color='red', label='dynamic_complex_map')

                # plt.plot(sub_axix, test_acys, color='red', label='testing accuracy')                #
                # X=[]
                # Y=[]
                # Z=[]
                # for i in range(0, risk_map.__len__()):
                #     X.append(risk_map[i].lateral)
                #     Y.append(risk_map[i].longitudinal)
                #     Z.append(risk_map[i].risk_value)
                # X1 = np.array(X)
                # X2 = np.vstack([X1, X1, X1])
                # Y1 = np.array(Y)
                # Y2 = np.vstack([Y1, Y1, Y1])
                # Z1 = np.array(Z)
                # matrix = np.diag(Z)
                # xmax = max(X)
                # xmin = min(X)
                # ymax = max(Y)
                # ymin = min(Y)
                # uniform_data = [[0] * (ymax-ymin)] *(xmax-xmin)
                # for i in range(0,X.__len__()):
                #     uniform_data[X[i]-xmin-1][Y[i]-ymin-1]=Z[i]*10
                # sns.heatmap(uniform_data, ax=ax, vmin=0, vmax=1, cmap='YlOrRd', annot=True, linewidths=0.1, cbar=True)
                # ax.set_ylabel('longitudinal')  # 设置纵轴标签
                # ax.set_xlabel('lateral')  # 设置横轴标签
                # ax.plot_surface(X1, Y1, matrix, rstride=8, cstride=8, alpha=0.3)
                # ax.plot_trisurf(X1, Y1, Z1, cmap="rainbow")
                if is_fig == 1:
                    plt.legend()
                # plt.show()
                # 保存到文件中去
                if is_record_inf == 1:
                    # rec_header = ['index','forward_speed','location_x', 'location_y', 'rotation_yaw','lat_accel_stp','long_accel_stp']
                    log_csv_writer.writerow([str(already_draw_one), esp_vehicle_speed_stp_motion, vehicle_pos_lng_hdmap,
                                             vehicle_pos_lat_hdmap, esp_lat_accel_stp_motion,
                                             esp_long_accel_stp_motion,statics_complex_value+statics_comple_value2+dynamic_complex_value])
                if is_save_generate_image == 1 and is_fig == 1:
                    print("path+'/image2/'+filename):" + path + '/image2/' + filename + '/complex' +
                          str(a) + '.jpg')
                    if not os.path.exists(path + '/image2/'):
                        os.mkdir(path + 'image2/')
                    if not os.path.exists(path + '/image2/' + filename):
                        os.mkdir(path + 'image2/' + filename)
                    try:
                        plt.savefig(path + '/image2/' + filename + '/complex' +
                                    str(a) + '.jpg', dpi=200, bbox_inches="tight")
                    except:
                        print('save figure error')
                if is_fig == 1:
                    plt.close()
                # plt.clf()
                # plt.xlim(-40, 40)
                # # 设置y轴的刻度范围
                # plt.ylim(-40, 40)
                # plot_circle((0, 0), r=1)
                # pass
                already_draw_one = already_draw_one + 1
        if is_record_inf == 1:
            log_file.close()
        if is_generated_video == 1:
            print(
                'ffmpeg  -y -framerate 24.0 -i "' + path + 'image2/' + filename + '/%d.png" -c:v libx264 -crf 30 -preset:v medium -pix_fmt yuv420p  -vf "scale=960:-2" "' + path + 'image2/' + filename + '/' + filename + '.mov"')
            os.system(
                'ffmpeg  -y -framerate 24.0 -i "' + path + 'image2/' + filename + '/%d.png" -c:v libx264 -crf 30 -preset:v medium -pix_fmt yuv420p  -vf "scale=960:-2" "' + path + 'image2/' + filename + '/' + filename + '.mov"')
    total_complex = 0
    for i in range(0, global_complex_map.__len__()):
        total_complex = total_complex+global_complex_map[i]
    print('\n total_complex='+str(total_complex)+'')
    return


def Draw_link_list_hdmap_static_map(x0, y0, link_list_hdmap,is_fig = 1):
    statics_comple_value = 0
    for i in range(0, link_list_hdmap.__len__()):
        link_id = link_list_hdmap['links_' + str(i)]['link_id']
        link_length = link_list_hdmap['links_' + str(i)]['link_length']  # 路段的长度:[m]
        link_type = link_list_hdmap['links_' + str(i)]['type']  # 道路类型:[/],(0,0,3),[/],(1,0),区分路段为高速、匝道、城区
        lane_attributelists = link_list_hdmap['links_' + str(i)]['lane_attributelists']  # 车道属性集合
        lane_lines_sets = link_list_hdmap['links_' + str(i)]['lines']  # 车道线集合
        if is_fig == 1:
            plot_circle((0, 0), r=5, colorstr='r')
        for j in range(0, lane_lines_sets.__len__()):
            lane_line_point_sets = lane_lines_sets['lines_' + str(j)]['line_points']
            line_x = []
            line_y = []
            for k in range(0, lane_line_point_sets.__len__()):
                lane_line_point_sets_cell = lane_line_point_sets['line_points_' + str(k)]
                x, y = coordinateconvert.millerToXY(lane_line_point_sets_cell['lng'], lane_line_point_sets_cell['lat'])
                line_x.append(x - x0)
                line_y.append(y - y0)
            if lane_lines_sets['lines_' + str(j)]['line_type'] == 0:  # 实线
                if is_fig == 1:
                    plt.plot(line_x, line_y, linestyle='solid', color='b', linewidth=1.0)
                logging.debug('实线')
            elif lane_lines_sets['lines_' + str(j)]['line_type'] == 1:  # 虚线
                if is_fig == 1:
                    plt.plot(line_x, line_y, linestyle='dotted', color='b', linewidth=1)
                logging.debug('虚线')
            elif lane_lines_sets['lines_' + str(j)]['line_type'] == 2:  # 双虚线
                if is_fig == 1:
                    plt.plot(line_x, line_y, color='b',
                         linestyle='dotted', linewidth=1.0)
                    line_x2 = [x + 0.1 for x in line_x]
                    line_y2 = [x + 0.1 for x in line_y]
                    plt.plot(line_x2, line_y2, color='b',
                             linestyle='dotted', linewidth=1.0)
                logging.debug('双虚线')
            elif lane_lines_sets['lines_' + str(j)]['line_type'] == 3:  # 虚实线
                if is_fig == 1:
                    plt.plot(line_x, line_y, color='b',
                             linestyle='dotted', linewidth=1.0)
                    line_x2 = [x + 0.1 for x in line_x]
                    line_y2 = [x + 0.1 for x in line_y]
                    plt.plot(line_x2, line_y2, color='b',
                             linestyle='solid', linewidth=1.0)
                logging.debug('虚实线')
            elif lane_lines_sets['lines_' + str(j)]['line_type'] == 4:  # 实虚线
                if is_fig == 1:
                    plt.plot(line_x, line_y, color='b',
                             linestyle='solid', linewidth=1.0)
                    line_x2 = [x + 0.1 for x in line_x]
                    line_y2 = [x + 0.1 for x in line_y]
                    plt.plot(line_x2, line_y2, color='b',
                             linestyle='dotted', linewidth=1.0)
                logging.debug('实虚线')
            elif lane_lines_sets['lines_' + str(j)]['line_type'] == 5:  # 双实线
                if is_fig == 1:
                    plt.plot(line_x, line_y, color='b',
                             linestyle='solid', linewidth=1.0)
                    line_x2 = [x + 0.1 for x in line_x]
                    line_y2 = [x + 0.1 for x in line_y]
                    plt.plot(line_x2, line_y2, color='b',
                             linestyle='solid', linewidth=1.0)
                logging.debug('双实线')
            # plt.plot(line_x,line_y, color='blue', label='test1')
            # plot_circle((lane_line_point_sets_cell['lng'],lane_line_point_sets_cell['lat']), r=0.5,colorstr='green')
            # print('('+str(lane_line_point_sets_cell['lng'])+','+str(lane_line_point_sets_cell['lat'])+')----('+str(x-x0)+','+str(y-y0)+')')
        lan_line_ground_markings = link_list_hdmap['links_' + str(i)]['ground_markings']  # 地面标识
        if not lan_line_ground_markings == 0:
            statics_comple_value = statics_comple_value + 0.1
        lan_line_ground_traffic_light = link_list_hdmap['links_' + str(i)]['traffic_light']  # 交通灯信息
        lan_line_traffic_info = link_list_hdmap['links_' + str(i)]['traffic_info']  # 交通标志牌信息
        lan_line_complex_intersection = link_list_hdmap['links_' + str(i)][
            'complex_intersection']  # 是否为路口link
        if lan_line_complex_intersection == 1:
            statics_comple_value = statics_comple_value + 0.1
        lan_line_successive_link_ids = link_list_hdmap['links_' + str(i)]['successive_link_ids']  # 后继link编号
        # print('lan_line_successive_link_ids='+lan_line_successive_link_ids+',current_link_id='+str(link_id))
        lan_line_is_routing_path = link_list_hdmap['links_' + str(i)]['is_routing_path']  # 当前link是否在导航路径上
        lan_line_split_merge_list = link_list_hdmap['links_' + str(i)]['split_merge_list']  # 分流汇流状态
        if not lan_line_split_merge_list == 0:
            statics_comple_value = statics_comple_value + 0.1
        lan_line_is_in_tunnel = link_list_hdmap['links_' + str(i)]['is_in_tunnel']  # link是否为隧道
        if lan_line_is_in_tunnel == 1:
            statics_comple_value = statics_comple_value + 0.1
        lan_line_is_in_toll_booth = link_list_hdmap['links_' + str(i)]['is_in_toll_booth']  # link是否为收费站
        if lan_line_is_in_toll_booth == 1:
            statics_comple_value = statics_comple_value + 0.1
        lan_line_is_in_certified_road = link_list_hdmap['links_' + str(i)][
            'is_in_certified_road']  # link是否为检查站
        if lan_line_is_in_certified_road == 1:
            statics_comple_value = statics_comple_value + 0.1
        lan_line_is_in_odd = link_list_hdmap['links_' + str(i)]['is_in_odd']  # link是否在ODD（运行设计域）范围内
    return statics_comple_value

def Image_Data(imagedata_image, img_filename=''):
    imagedata_image = ast.literal_eval(imagedata_image)
    if not imagedata_image == '':
        jpg_as_np = np.frombuffer(imagedata_image, np.uint8)
        img = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
        if not img_filename == '':
            print(img_filename)
            cv2.imwrite(img_filename, img)
        plt.imshow(img)


# 可行驶区域
def Draw_Free_Space_Desc(first0, BacauseOfThereIsImage, is_fig =1):
    plt.xlim(-40, 40)
    # 设置y轴的刻度范围
    plt.ylim(-40, 40)
    freespace_fc_list = first0[22 + BacauseOfThereIsImage]
    freespace_fc_list = ast.literal_eval(freespace_fc_list)
    plot_circle((0, 0), r=2, colorstr='darkred')
    try:
        if freespace_fc_list == 0:
            return
        for i in range(0, freespace_fc_list.__len__()):
            freespace_fc_unit = freespace_fc_list['points_' + str(i)]
            motion_state = freespace_fc_unit['motion_state']
            point_type = freespace_fc_unit['type']  # reespace点类型
            position_longitudinal_distance = freespace_fc_unit['position_longitudinal_distance']
            position_lateral_distance = freespace_fc_unit['position_lateral_distance']
            if point_type == 0:  # 忽略：0
                logging.debug('忽略')
                if is_fig == 1:
                    plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1, colorstr='dimgray')
            elif point_type == 1:  # 汽车 ：1
                logging.debug('汽车')
                if is_fig == 1:
                    plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1, colorstr='blue')
            elif point_type == 2:  # 路沿：2
                logging.debug('路边')
                if is_fig == 1:
                    plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1,
                            colorstr='brown')
            elif point_type == 3:  # 行人 ：3
                logging.debug('行人')
                if is_fig == 1:
                    plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1,
                            colorstr='red')
            elif point_type == 4:  # 锥形桶 ：4
                logging.debug('锥形桶')
                if is_fig == 1:
                    plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1,
                            colorstr='coral')
            elif point_type == 5:  # 静态目标：5
                logging.debug('静态目标')
                if is_fig == 1:
                    plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1,
                            colorstr='darkcyan')
            elif point_type == 6:  # 未知：6
                logging.debug('未知')
                if is_fig == 1:
                    plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1,
                            colorstr='aliceblue')
    except:
        pass
    return


def plot_img(img_bin):
    # figsize = 11, 9  # 设定图片大小，数字可以调整
    # figure, ax = plt.subplots(figsize=figsize)
    jpg_as_np = np.frombuffer(img_bin, np.uint8)
    img = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
    plt.imshow(img, interpolation='nearest')
    return


class risk:  # 结构体
    def __init__(self):
        self.longitudinal = 0  # 纵向
        self.lateral = 0  # 横向
        self.risk_value = 0  # 风险值


def Draw_Dection_POI(is_fig,object_detection_list, local_riskmap,latitude_of_location,longitude_of_location,local_complex):
    riskMapIndex = 0
    statics_object_classification_class = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    statics_dynamic_classification_class = [0,0,0,0,0,0,0,0,0,0,0,0,0]

    statics_theta = 0 #静态复杂度系数（信息熵）
    local_risk_value_list = []

    # ['未知目标', '汽车', '卡车', '摩托车', '行人', '自行车', '动物',
    #                                    '巴士', '其他', '购物车',
    #                                    '柱子', '锥桶', '已被锁上的车位锁', '未被锁上的车位锁 ']
    for i in range(0, object_detection_list.__len__()):
        local_risk_value = 0
        # 如果目标目标跟踪ID不为空
        track_id = object_detection_list['objs_' + str(i)]['track_id']
        # track_id = ast.literal_eval(track_id)
        if track_id == 0:
            continue
        # if already_draw_one == 2:
        #     break;
        # if track_id != 0:
        #local_riskmap.append(risk())
        # 获取目标类别
        objs_classification = object_detection_list['objs_' + str(i)]['classification']
        # 获取目标纵向和横向距离

        objs_longitudinal = object_detection_list['objs_' + str(i)]['longitudinal_distance']
        objs_lateral = object_detection_list['objs_' + str(i)]['lateral_distance']
        # objs_lateral=ast.literal_eval(objs_lateral)
        # objs_longitudinal = ast.literal_eval(objs_longitudinal)
        flag_There_is = 0

        # 获取trackID
        # track_id = object_detection_list['objs_' + str(i)]['track_id']
        # track_id_txt = str(track_id)
        # riskMap[riskMapIndex].track_id = track_id_txt
        # print(type(track_id_txt))
        objs_length = object_detection_list['objs_' + str(i)]['length']
        objs_width = object_detection_list['objs_' + str(i)]['width']
        heading_angle = object_detection_list['objs_' + str(i)]['heading_angle']
        # 体地协同转换矩阵
        Cbe = np.transpose(
            np.array([[np.cos(heading_angle), np.sin(heading_angle), 0],
                      [- np.sin(heading_angle), np.cos(heading_angle), 0], [0, 0, 1]]))

        # print( objs_lateral,objs_longitudinal,objs_length,objs_width,track_id)
        longitudinal = objs_longitudinal / 2  # 目标纵向距离
        lateral = objs_lateral / 2  # 目标纵向距离驾驶目标横向宽度
        # 基本分及权重
        distance_weight=1
        size_weight=1
        class_type_score = 0
        status_score = 0
        cutin_status_score = 0
        complex_class_type_score=0
        if longitudinal <30 and lateral <30:
            distance_weight=1.5
        elif longitudinal <100 and lateral <100:
            distance_weight = 1.2
        else:
            distance_weight = 1.
        length = objs_length / 2 #目标长度[m]
        width = objs_width / 2
        if length>5 and  width > 5:
            size_weight= 1.5
        elif length>3 and  width > 2:
            size_weight= 1.2
        else:
            size_weight = 1
        txt = str(objs_classification) + ':' + str(track_id) + ',(' + str(objs_longitudinal) + ',' + str(
            objs_lateral) + '),', str(objs_length), str(objs_width), \
        # txt = str(heading_angle)
        logging.debug(
            'objs_classification:track_id_txt,(objs_longitudinal,objs_lateral),objs_length,objs_width,heading_angle')
        logging.debug('目标是', txt)

        cut_status = object_detection_list['objs_' + str(i)]['cut_status']  # Cut-in状态
        status = object_detection_list['objs_' + str(i)]['status']  # 目标运动状态
        if status == 0 or status==2:
            status_score=0.5
        elif  status == 3:
            status_score = 0.8
        elif status == 1:
            status_score = 1.2

        if cut_status == 0: #未知：0
            cutin_status_score=1
        elif  cut_status == 2: #正在切入:2
            cutin_status_score = 1.5
        elif  cut_status == 3: #正在切出:3
            cutin_status_score = 0.8
        elif cut_status == 1: #正常:1
            cutin_status_score = 1.2

        longitudinal_relative_velocity = object_detection_list['objs_' + str(i)][
            'longitudinal_relative_velocity']  # 纵向相对速度
        lateral_relative_velocity = object_detection_list['objs_' + str(i)][
            'lateral_relative_velocity']  # 横向相对速度
        obj_refer_points = object_detection_list['objs_' + str(i)][
            'obj_refer_points']  # 目标测量参考点：下一级为纵向距离、横向距离

        # if abs(objs_longitudinal) <= 10:
        # object_classification_class = ['未知目标', '汽车', '卡车', '摩托车', '行人', '自行车', '动物',
        #                                '巴士', '其他', '购物车',
        #                                '柱子', '锥桶', '已被锁上的车位锁', '未被锁上的车位锁 ']

        if objs_classification == 0:
            if is_fig==1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='black')
            class_type_score = 2
            if status == 1 or status == 3 or status == 4 :
                statics_object_classification_class[0]= statics_object_classification_class[0]+1
            else:
                statics_dynamic_classification_class[0] = math.cos(heading_angle)*(longitudinal_relative_velocity*0.5+lateral_relative_velocity*0.5)/math.sqrt(objs_longitudinal**2+objs_lateral**2)
            complex_class_type_score=1
            logging.debug("unknow")
        elif objs_classification == 1:
            if is_fig==1:
                Draw_Rectangle_With_angle(objs_lateral, objs_longitudinal, objs_width, objs_length,
                                      heading_angle, 'yellow', 'car')
            class_type_score = 5
            if status == 1 or status == 3 or status == 4 :
                statics_object_classification_class[1] = statics_object_classification_class[1] + 1
            else:
                statics_dynamic_classification_class[1] = math.cos(heading_angle)*(longitudinal_relative_velocity*0.5+lateral_relative_velocity*0.5)/math.sqrt(objs_longitudinal**2+objs_lateral**2)
            logging.debug("汽车")
        elif objs_classification == 2:
            if is_fig == 1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='orange', labelstr='truck')
            class_type_score = 5
            if status == 1 or status == 3 or status == 4 :
                statics_object_classification_class[2] = statics_object_classification_class[2] + 1
            else:
                statics_dynamic_classification_class[2] = math.cos(heading_angle)*(longitudinal_relative_velocity*0.5+lateral_relative_velocity*0.5)/math.sqrt(objs_longitudinal**2+objs_lateral**2)
            logging.debug("卡车")
        elif objs_classification == 3:
            logging.debug("摩托车")
            class_type_score = 5
            if status == 1 or status == 3 or status == 4 :
                statics_object_classification_class[3] = statics_object_classification_class[3]+ 1
            else:
                statics_dynamic_classification_class[3] = math.cos(heading_angle)*(longitudinal_relative_velocity*0.5+lateral_relative_velocity*0.5)/math.sqrt(objs_longitudinal**2+objs_lateral**2)
            if is_fig==1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='palegreen', labelstr='motocycle')
        elif objs_classification == 4:
            logging.debug("行人")
            if status == 1 or status == 3 or status == 4 :
                statics_object_classification_class[4] = statics_object_classification_class[4]+ 1
            else:
                statics_dynamic_classification_class[4] = math.cos(heading_angle)*(longitudinal_relative_velocity*0.5+lateral_relative_velocity*0.5)/math.sqrt(objs_longitudinal**2+objs_lateral**2)
            class_type_score = 8
            if is_fig==1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='red', labelstr='pedestrian')
        elif objs_classification == 5:
            logging.debug("自行车")
            if status == 1 or status == 3 or status == 4 :
                statics_object_classification_class[5] = statics_object_classification_class[5]+ 1
            else:
                statics_dynamic_classification_class[5] = math.cos(heading_angle)*(longitudinal_relative_velocity*0.5+lateral_relative_velocity*0.5)/math.sqrt(objs_longitudinal**2+objs_lateral**2)
            class_type_score = 5
            if is_fig==1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='purple', labelstr='bicycle')
        elif objs_classification == 6:
            logging.debug("动物")
            class_type_score = 5
            if status == 1 or status == 3 or status == 4 :
                statics_object_classification_class[6] = statics_object_classification_class[6]+ 1
            else:
                statics_dynamic_classification_class[6] = math.cos(heading_angle)*(longitudinal_relative_velocity*0.5+lateral_relative_velocity*0.5)/math.sqrt(objs_longitudinal**2+objs_lateral**2)
            if is_fig==1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='darkgreen', labelstr='animal')
        elif objs_classification == 7:
            logging.debug("巴士")
            class_type_score = 5
            if status == 1 or status == 3 or status == 4 :
                statics_object_classification_class[7] = statics_object_classification_class[7]+ 1
            else:
                statics_dynamic_classification_class[7] = math.cos(heading_angle)*(longitudinal_relative_velocity*0.5+lateral_relative_velocity*0.5)/math.sqrt(objs_longitudinal**2+objs_lateral**2)
            if is_fig==1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='green', labelstr='bus')
        elif objs_classification == 8:
            logging.debug("其他")
            class_type_score = 5
            if status == 1 or status == 3 or status == 4 :
                statics_object_classification_class[8] = statics_object_classification_class[8]+ 1
            else:
                statics_dynamic_classification_class[8] = math.cos(heading_angle)*(longitudinal_relative_velocity*0.5+lateral_relative_velocity*0.5)/math.sqrt(objs_longitudinal**2+objs_lateral**2)
            if is_fig==1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='gray', labelstr='others')
        elif objs_classification == 9:
            logging.debug("购物车")
            class_type_score = 5
            if is_fig==1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='cyan', labelstr='Shopping Cart')
        elif objs_classification == 10:
            logging.debug("柱子")
            class_type_score = 4
            if is_fig==1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='crimson', labelstr='pillar')
        elif objs_classification == 11:
            class_type_score = 3
            if is_fig==1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='blanchedalmond', labelstr='cone')
            logging.debug("锥桶")
        elif objs_classification == 12:
            class_type_score = 2
            logging.debug("已被锁上的车位锁")
            if is_fig==1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='darkslategray',
                        labelstr='Locked parking space')
        else:
            logging.debug("未被锁上的车位锁")
            if is_fig==1:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='darkslategray',
                        labelstr='unLocked parking space')


        if longitudinal_relative_velocity > 100 or lateral_relative_velocity >100:
            distance_weight = distance_weight * 1.8
        elif  longitudinal_relative_velocity > 50 or lateral_relative_velocity >50:
            distance_weight = distance_weight * 1.5
        for j in range(0, obj_refer_points.__len__()):
            longitudinal_distance = obj_refer_points['obj_refer_points_' + str(j)]['longitudinal_distance']
            lateral_distance = obj_refer_points['obj_refer_points_' + str(j)]['lateral_distance']
        if abs(longitudinal_distance)<5 or abs(lateral_distance)<5:
            distance_weight  = distance_weight*1.5
        local_risk_value = distance_weight * size_weight * class_type_score * status_score *cutin_status_score
        local_risk_value_list.append(local_risk_value)
        logging.info(str(local_risk_value)+'='+str(distance_weight)+'*'+str(size_weight)+'*'+str(class_type_score)+'*'+str(status_score)+'*'+str(cutin_status_score))
        local_riskmap = add_to_risk_map(local_risk_value, objs_lateral, objs_longitudinal, local_riskmap,latitude_of_location,longitude_of_location)

        # plt.Rectangle(
        #     (objs_lateral, objs_longitudinal), objs_length, objs_width, fill=True, edgecolor='red',
        #     linewidth=1)
        # plot_circle((objs_lateral, objs_longitudinal), r=1)

        if is_fig == 1:
            ax = plt.gca()  # 获得坐标轴的句柄
            plt.text(lateral + objs_width / 2, longitudinal + objs_length / 2, '%s' %
                 object_classification_class[objs_classification], ha='center', va='bottom', fontsize=7,
                 transform=ax.transAxes)
        # ax.add_patch(plt.Rectangle(
        #     (objs_lateral, objs_longitudinal), objs_length, objs_width))
        # ax.add_patch(plt.Rectangle((0, 0), 1, 1))
        # ax = fig.add_subplot(111)
        riskMapIndex = riskMapIndex + 1
        already_draw_one = 1
    total_statics_object_classification_class_value = 0  # 静态复杂度系数（信息熵）
    total_dynamic_object_classification_class_value = 0  # 静态复杂度系数（信息熵）
    total_local_risk_value = 0
    for i in range(0, local_risk_value_list.__len__()):
        total_local_risk_value = total_local_risk_value + local_risk_value_list[i]
    for i in range(0, statics_dynamic_classification_class.__len__()):
        total_dynamic_object_classification_class_value = total_dynamic_object_classification_class_value + statics_dynamic_classification_class[i]
    total_dynamic_object_classification_class_value = abs(total_dynamic_object_classification_class_value) *4000
    for i in range(0, statics_object_classification_class.__len__()):
        statics_object_classification_class[i] = statics_object_classification_class[i]/object_detection_list.__len__()
        total_statics_object_classification_class_value = total_statics_object_classification_class_value + statics_object_classification_class[i]
    total_statics_object_complex = total_local_risk_value * total_statics_object_classification_class_value*10
    return local_riskmap,total_statics_object_complex,total_dynamic_object_classification_class_value


def add_to_risk_map(local_risk_value, objs_lateral, objs_longitudinal, riskMap,latitude_of_location,longitude_of_location):
    if len(riskMap) == 0:
        riskMap.append(risk())
        riskMap[riskMap.__len__()-1].lateral = round(objs_lateral-latitude_of_location)
        riskMap[riskMap.__len__()-1].longitudinal = round(objs_longitudinal-longitude_of_location)
        riskMap[riskMap.__len__()-1].risk_value = local_risk_value.__round__(2)
    else:
        flag_There_is = 0
        for ii in range(len(riskMap)):
            if riskMap[ii].longitudinal == objs_lateral.__round__(6) and riskMap[
                ii].longitudinal == objs_longitudinal.__round__(6):
                flag_There_is = 1
                riskMap[ii].risk_value = local_risk_value + riskMap[ii].risk_value.__round__(2)
                break
        if flag_There_is == 0:
            riskMap.append(risk())
            riskMap[riskMap.__len__() - 1].lateral = round(objs_lateral - latitude_of_location)
            riskMap[riskMap.__len__() - 1].longitudinal = round(objs_longitudinal - longitude_of_location)
            riskMap[riskMap.__len__()-1].risk_value = local_risk_value.__round__(2)
    return riskMap


def Draw_Rectangle_With_angle(x, y, width, height, angle, colorstr, label):
    # x,y 矩形中心点坐标
    anglePi = -angle * math.pi / 180.0
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)

    # 没有转之前的
    #  (x2,y2)  ---(x3,y3)
    #  (x1,y1） ---(x0,y0)
    x1 = x - 0.5 * width  # 没有转之前的
    y1 = y - 0.5 * height

    x0 = x + 0.5 * width
    y0 = y1

    x2 = x1
    y2 = y + 0.5 * height

    x3 = x0
    y3 = y2

    x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
    y0n = (x0 - x) * sinA + (y0 - y) * cosA + y

    x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
    y1n = (x1 - x) * sinA + (y1 - y) * cosA + y

    x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
    y2n = (x2 - x) * sinA + (y2 - y) * cosA + y

    x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
    y3n = (x3 - x) * sinA + (y3 - y) * cosA + y

    # plt.plot([(x0n, y0n), (x1n, y1n)],  "b", label="line_b")
    # plt.plot([(x1n, y1n), (x2n, y2n)],  "b", label="line_b")
    # plt.plot([(x2n, y2n), (x3n, y3n)],  "b", label="line_b")
    # plt.plot([(x0n, y0n), (x3n, y3n)], "b", label="line_b")
    xx = [x0n, x1n, x2n, x3n, x0n]
    yy = [y0n, y1n, y2n, y3n, y0n]
    plt.plot(xx, yy, "b", color=colorstr, label=label)


def Draw_Lane_Line(is_fig,Lane_line_list):
    for i in range(0, Lane_line_list.__len__()):
        # Lane_line_list['lines_' + str(i)]
        if Lane_line_list['lines_' + str(i)]['curve_parameter_a0'] != 0 or \
                Lane_line_list['lines_' + str(i)]['curve_parameter_a1'] != 0 or \
                Lane_line_list['lines_' + str(i)]['curve_parameter_a2'] != 0:
            curve_parameter_a0 = Lane_line_list['lines_' + str(i)]['curve_parameter_a0']
            curve_parameter_a1 = Lane_line_list['lines_' + str(i)]['curve_parameter_a1']
            curve_parameter_a2 = Lane_line_list['lines_' + str(i)]['curve_parameter_a2']
            curve_parameter_a3 = Lane_line_list['lines_' + str(i)]['curve_parameter_a3']

            color = Lane_line_list['lines_' + str(i)]['color']
            # logging.debug('车道线颜色：'+color)

            y = np.linspace(-50, 50, 50)

            x = curve_parameter_a3 * y ** 3 + curve_parameter_a2 * \
                y ** 2 + curve_parameter_a1 * y + curve_parameter_a0

            if Lane_line_list['lines_' + str(i)]['type'] == 0:  # 实线
                if is_fig == 1:
                    plt.plot(x, y, linestyle='solid', color='b', linewidth=1.0)
                logging.debug('实线')
            elif Lane_line_list['lines_' + str(i)]['type'] == 1:  # 虚线
                if is_fig == 1:
                    plt.plot(x, y, linestyle='dotted', color='b', linewidth=1)
                logging.debug('虚线')
            elif Lane_line_list['lines_' + str(i)]['type'] == 2:  # 双虚线
                if is_fig == 1:
                    plt.plot(x, y, color='b',
                         linestyle='dotted', linewidth=1.0)
                    plt.plot(x + 0.1, y + 0.1, color='b',
                         linestyle='dotted', linewidth=1.0)
                logging.debug('双虚线')
            elif Lane_line_list['lines_' + str(i)]['type'] == 3:  # 虚实线
                if is_fig == 1:
                    plt.plot(x, y, color='b',
                         linestyle='dotted', linewidth=1.0)
                    plt.plot(x + 0.1, y + 0.1, color='b',
                         linestyle='solid', linewidth=1.0)
                logging.debug('虚实线')
            elif Lane_line_list['lines_' + str(i)]['type'] == 4:  # 实虚线
                if is_fig == 1:
                    plt.plot(x, y, color='b',
                         linestyle='solid', linewidth=1.0)
                    plt.plot(x + 0.1, y + 0.1, color='b',
                         linestyle='dotted', linewidth=1.0)
                logging.debug('实虚线')
            elif Lane_line_list['lines_' + str(i)]['type'] == 5:  # 双实线
                if is_fig == 1:
                    plt.plot(x, y, color='b',
                         linestyle='solid', linewidth=1.0)
                    plt.plot(x + 0.1, y + 0.1, color='b',
                         linestyle='solid', linewidth=1.0)
                logging.debug('双实线')
    return


def dx(dis, k):
    return np.sqrt(dis / (k ** 2 + 1.))


def dy(dis, k):
    return k * dx(dis, k)


def parallel_curve(xs, ys, ks, dis=1.):
    """ 由于对称性, 会返回两条平行曲线上的点
    :param xs: ndarray, 原始曲线上点的x值   [shape:(N,)]
    :param xs: ndarray, 原始曲线上点的y值   [shape:(N,)]
    :param ks: ndarray, 原始曲线上点的斜率  [shape:(N,)]
    :param dis: float, 曲线间的距离
    :return: ndarray, pxs [shape:(2, N)], pys [shape:(2, N)]
    """
    g = np.sign(ks)
    g[g == 0] = 1
    ms = -1. / (ks + 1e-20)
    pxs = np.vstack((xs + dx(dis ** 2, ms) * g, xs - dx(dis ** 2, ms) * g))
    pys = np.vstack((ys + dy(dis ** 2, ms) * g, ys - dy(dis ** 2, ms) * g))
    return pxs, pys


def plot_circle(center=(3, 3), r=2, colorstr='k', labelstr=''):
    try:
        x = np.linspace(center[0] - r, center[0] + r, 5000)
        y1 = np.sqrt(r ** 2 - (x - center[0]) ** 2) + center[1]
        y2 = -np.sqrt(r ** 2 - (x - center[0]) ** 2) + center[1]
        plt.plot(x, y1, c=colorstr, label=labelstr)
        plt.plot(x, y2, c=colorstr, label=labelstr)
    except:
        pass


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# 绘制三角形函数
def draw_tri(points, color='red', alpha=0.5):
    plt.scatter(points[:, 0], points[:, 1], s=0.1, color=color, alpha=0.001)
    tri = plt.Polygon(points, color=color, alpha=alpha)
    plt.gca().add_patch(tri)


# brief 将VTDheading转换为53 - 2017标准
def heading_angle_Cali(vtdh):
    vtdh = 90 - (vtdh * 180.0 / math.pi)
    if (vtdh < 0):
        vtdh = vtdh + 360
    if (vtdh > 360):
        vtdh = vtdh - 360
    return vtdh * 80


def declare_a_global_variable():
    global plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["ifile=", "wdir=", ])
    for opt, arg in opts:
        if opt == '-h':
            print
            '8.riskhotmap.py -i <inputfile> -w <working directory>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            targetfile = arg
        elif opt in ("-w", "--wdir"):
            path = arg
            if not path.endswith('/'):
                path = path +'/'
    path = 'F:/DataContest/data/0805 (有图数据)/'
    targetfile = '1659666219.43_1659666263.64.csv'

    declare_a_global_variable()
    config = {
        "font.family": 'serif',  # sans-serif/serif/cursive/fantasy/monospace
        "font.size": 10,  # medium/large/small
        'font.style': 'normal',  # normal/italic/oblique
        'font.weight': 'normal',  # bold
        "mathtext.fontset": 'cm',  # 'cm' (Computer Modern)
        "font.serif": ['cmb10'],  # 'Simsun'宋体
        "axes.unicode_minus": False,  # 用来正常显示负号
    }

    plt.figure()  # 字符型linestyle使用方法
    plt.rcParams.update(config)
    # plt.xlim(-40, 40)
    # # 设置y轴的刻度范围
    # plt.ylim(-40, 40)

    # plt.rcParams['font.sans-serif'] = ["SimHei"]
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['figure.figsize'] = (8.0, 6.0)
    # plt.rcParams['figure.dpi'] = 50 #分辨率
    # plt.rcParams['savefig.dpi'] = 150  # 图片像素
    isSaveGenerateImage = 0
    isRecordInf = 1
    NoImageFileFormat = 1 # 没有图片
    is_generated_video = 0
    prepare_data(path,targetfile,isSaveGenerateImage, isRecordInf, NoImageFileFormat, is_generated_video)
    # plt.legend()

    plt.show()
    #

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
