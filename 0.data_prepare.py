# coding=utf-8

# 数据路径
import ast
import base64
import csv
import math
import os
import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
import coordinateconvert

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


def prepare_data(isGenerateImage=1, isRecordInf=1,NoImageFileFormat=1):
    '''
    读取文件夹下所有文件的名字并把他们用列表存起来
    '''
    path = "D:/DataContest/data/0802 (无图数据1)/0802/"
    # path='/home/lawrence/Documents/Bigcontest/data/0802(无图数据1)/0802/'
    # file_image = 'F:\\car\\object\\image4\\0805 (1)\\0805\\image'
    datanames = os.listdir(path)
    list = []
    for i in datanames:
        list.append(i)
    already_draw_one = 0
    for b in list:
        row_total=0
        (filename, extension) = os.path.splitext(b)
        file = path + b
        if not os.access(file, os.X_OK):
            continue;
        try:
            row_total = len(open(file).readlines())
        except:
            continue
        # row_total = 1

        rec_header = ['index', 'forward_speed', 'location_x', 'location_y', 'rotation_yaw']
        filepath, tmpfilename = os.path.split(b)
        shotname, extension = os.path.splitext(tmpfilename)
        if isRecordInf == 1:
            rec_filename = '/home/lawrence/Documents/Bigcontest/data/image2/' + shotname + '/'
            if not os.path.exists(rec_filename):
                os.mkdir(rec_filename)
            log_file = open(rec_filename+ '/' + 'carinfo.cvs', 'a+', encoding='utf-8', newline='')
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

            timeindex = []  # np.arange(0, row_total, 1)
            already_draw_one = 0
            for a in range(row_total):
                # if already_draw_one == 2:
                #     break;
                timeindex.append(already_draw_one)
                first0 = next(reader, None)
                if first0 is None:
                    break;
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

                plt.subplot(235)
                plt.title('Self Vec Motion') #自车运动
                X1N = np.array(timeindex)
                X2N = np.array(esp_yaw_rate_stp_motion_his)
                plt.plot(timeindex, esp_yaw_rate_stp_motion_his, linestyle='solid', color='g',
                         label='偏航率(车横摆角速度)[degree/s]')
                plt.plot(timeindex, esp_vehicle_speed_stp_motion_his, 'ro:', linewidth=1.0, label='车速信号')
                plt.plot(timeindex, esp_lat_accel_stp_motion_his, 'bo:', linewidth=1.0, label='横向加速度')
                plt.plot(timeindex, esp_long_accel_stp_motion_his, 'yo:', linewidth=1.0, label='纵向加速度')
                plt.plot(timeindex, sas_steering_angle_stp_motion_his, 'mo:', linewidth=1.0,
                         label='方向盘转向角度[degree]')
                #plt.legend()
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
                # 静态地图
                plt.subplot(234)
                plt.title('Statics Map')#静态地图
                plt.xlim(-500, 50)
                # 设置y轴的刻度范围
                plt.ylim(-500, 500)
                Draw_link_list_hdmap_static_map(x0, y0, link_list_hdmap)

                # 6.图片数据
                plt.subplot(231)
                plt.title('Image Data')#图片数据
                BacauseOfThereIsImage=0
                if not NoImageFileFormat==1:
                    BacauseOfThereIsImage
                    imageData_image = first0[16]
                    Image_Data(imageData_image ,'') #'D:/DataContest/data/image2/' + filename + '/scence' + str(a) + '.png'

                # 7.定位
                timestamp_of_location = first0[16+BacauseOfThereIsImage]  # 时间戳
                heading_of_location = first0[17+BacauseOfThereIsImage]  # 航向角:[deg]
                plt.subplot(236)
                plt.title('latitude_+longitude')#航向角+车辆经纬度
                plt.plot(timestamp_of_location, heading_of_location, 'bo:')
                latitude_of_location = first0[18+BacauseOfThereIsImage]  # 车辆定位纬度
                longitude_of_location = first0[19+BacauseOfThereIsImage]  # 车辆定位经度
                altitude_of_location = first0[20+BacauseOfThereIsImage]  # 车辆定位高度
                plt.plot(timestamp_of_location, latitude_of_location, 'bo:',label='latitude_of_location')
                plt.plot(timestamp_of_location, longitude_of_location, 'ro:',label='longitude_of_location')
                plt.plot(timestamp_of_location, altitude_of_location, 'go:',label='altitude_of_location')

                # 8.驾驶员行为数据
                bcmlight = first0[21+BacauseOfThereIsImage]  # 转向灯开关状态信号（未使用：0，左转：1，右转：2，未知：3）
                # 9.可行驶区域点集
                plt.subplot(232)
                plt.title('Freespace')#可行驶区域点集
                Free_Space_Desc(first0,BacauseOfThereIsImage)

                longitudinal = 0  # 纵向
                lateral = 0  # 横向
                length = 5
                width = 5
                txt = ['car']
                # 10.----开始画图------
                # 目标检测 objs/fus_objs
                # 遍历object数组
                plt.subplot(233)
                plt.title('POI')#目标数据
                plt.xlim(-40, 40)
                # 设置y轴的刻度范围
                plt.ylim(-40, 40)
                Draw_Dection_POI(object_detection_list)
                # 把所有车道线搞出来
                Draw_Lane_Line(Lane_line_list)


                # 保存到文件中去
                if isRecordInf == 1:
                    # rec_header = ['index','forward_speed','location_x', 'location_y', 'rotation_yaw','lat_accel_stp','long_accel_stp']
                    log_csv_writer.writerow([str(already_draw_one), esp_vehicle_speed_stp_motion, vehicle_pos_lng_hdmap,
                                             vehicle_pos_lat_hdmap, esp_lat_accel_stp_motion,
                                             esp_long_accel_stp_motion])

                if isGenerateImage == 1:
                    print("path+'/image2/'+filename):" + path + '/image2/' + filename + '\\' +
                          str(a) + '.png')
                    if not os.path.exists(path + '/image2/'):
                        os.mkdir(path + 'image2/')
                    if not os.path.exists(path + '/image2/' + filename):
                        os.mkdir(path + 'image2/' + filename)
                    plt.savefig(path + '/image2/' + filename + '\\' +
                                str(a) + '.png', bbox_inches="tight")
                plt.clf()
                plt.xlim(-40, 40)
                # 设置y轴的刻度范围
                plt.ylim(-40, 40)
                plot_circle((0, 0), r=1)
                # pass
                already_draw_one = already_draw_one + 1
        if isRecordInf == 1:
            log_file.close()
        print(
            'fmpeg  -y -framerate 24.0 -i "' + path + 'image2/' + filename + '/%d.png" -c:v libx264 -crf 30 -preset:v medium -pix_fmt yuv420p  -vf "scale=960:-2" "' + path + 'image2/' + filename + '/' + filename + '.mov"')
        os.system(
            'ffmpeg  -y -framerate 24.0 -i "' + path + 'image2/' + filename + '/%d.png" -c:v libx264 -crf 30 -preset:v medium -pix_fmt yuv420p  -vf "scale=960:-2" "' + path + 'image2/' + filename + '/' + filename + '.mov"')
    return


def Draw_link_list_hdmap_static_map(x0, y0, link_list_hdmap):
    for i in range(0, link_list_hdmap.__len__()):
        link_id = link_list_hdmap['links_' + str(i)]['link_id']
        link_length = link_list_hdmap['links_' + str(i)]['link_length']  # 路段的长度:[m]
        link_type = link_list_hdmap['links_' + str(i)]['type']  # 道路类型:[/],(0,0,3),[/],(1,0),区分路段为高速、匝道、城区
        lane_attributelists = link_list_hdmap['links_' + str(i)]['lane_attributelists']  # 车道属性集合
        lane_lines_sets = link_list_hdmap['links_' + str(i)]['lines']  # 车道线集合
        plot_circle((0, 0), r=5, colorstr='r')
        for j in range(0, lane_lines_sets.__len__()):
            lane_line_point_sets = lane_lines_sets['lines_' + str(j)]['line_points']
            line_x = []
            line_y = []
            for k in range(0, lane_line_point_sets.__len__()):
                lane_line_point_sets_cell = lane_line_point_sets['line_points_' + str(k)]
                # x,y=coordinateconvert.wgs84tomercator(lane_line_point_sets_cell['lng'], lane_line_point_sets_cell['lat'])
                x, y = coordinateconvert.millerToXY(lane_line_point_sets_cell['lng'], lane_line_point_sets_cell['lat'])
                line_x.append(x - x0)
                line_y.append(y - y0)
            if lane_lines_sets['lines_' + str(j)]['line_type'] == 0:  # 实线
                plt.plot(line_x, line_y, linestyle='solid', color='b', linewidth=1.0)
                logging.debug('实线')
            elif lane_lines_sets['lines_' + str(j)]['line_type'] == 1:  # 虚线
                plt.plot(line_x, line_y, linestyle='dotted', color='b', linewidth=1)
                logging.debug('虚线')
            elif lane_lines_sets['lines_' + str(j)]['line_type'] == 2:  # 双虚线
                plt.plot(line_x, line_y, color='b',
                         linestyle='dotted', linewidth=1.0)
                line_x2 = [x + 0.1 for x in line_x]
                line_y2 = [x + 0.1 for x in line_y]
                plt.plot(line_x2, line_y2, color='b',
                         linestyle='dotted', linewidth=1.0)
                logging.debug('双虚线')
            elif lane_lines_sets['lines_' + str(j)]['line_type'] == 3:  # 虚实线
                plt.plot(line_x, line_y, color='b',
                         linestyle='dotted', linewidth=1.0)
                line_x2 = [x + 0.1 for x in line_x]
                line_y2 = [x + 0.1 for x in line_y]
                plt.plot(line_x2, line_y2, color='b',
                         linestyle='solid', linewidth=1.0)
                logging.debug('虚实线')
            elif lane_lines_sets['lines_' + str(j)]['line_type'] == 4:  # 实虚线
                plt.plot(line_x, line_y, color='b',
                         linestyle='solid', linewidth=1.0)
                line_x2 = [x + 0.1 for x in line_x]
                line_y2 = [x + 0.1 for x in line_y]
                plt.plot(line_x2, line_y2, color='b',
                         linestyle='dotted', linewidth=1.0)
                logging.debug('实虚线')
            elif lane_lines_sets['lines_' + str(j)]['line_type'] == 5:  # 双实线
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
        lan_line_ground_traffic_light = link_list_hdmap['links_' + str(i)]['traffic_light']  # 交通灯信息
        lan_line_traffic_info = link_list_hdmap['links_' + str(i)]['traffic_info']  # 交通标志牌信息
        lan_line_complex_intersection = link_list_hdmap['links_' + str(i)][
            'complex_intersection']  # 是否为路口link
        lan_line_successive_link_ids = link_list_hdmap['links_' + str(i)]['successive_link_ids']  # 后继link编号
        lan_line_is_routing_path = link_list_hdmap['links_' + str(i)]['is_routing_path']  # 当前link是否在导航路径上
        lan_line_split_merge_list = link_list_hdmap['links_' + str(i)]['split_merge_list']  # 分流汇流状态
        lan_line_is_in_tunnel = link_list_hdmap['links_' + str(i)]['is_in_tunnel']  # link是否为隧道
        lan_line_is_in_toll_booth = link_list_hdmap['links_' + str(i)]['is_in_toll_booth']  # link是否为收费站
        lan_line_is_in_certified_road = link_list_hdmap['links_' + str(i)][
            'is_in_certified_road']  # link是否为检查站
        lan_line_is_in_odd = link_list_hdmap['links_' + str(i)]['is_in_odd']  # link是否在ODD（运行设计域）范围内


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
def Free_Space_Desc(first0,BacauseOfThereIsImage):
    plt.xlim(-40, 40)
    # 设置y轴的刻度范围
    plt.ylim(-40, 40)
    freespace_fc_list = first0[22+BacauseOfThereIsImage]
    freespace_fc_list = ast.literal_eval(freespace_fc_list)
    plot_circle((0, 0), r=2, colorstr='darkred')
    for i in range(0, freespace_fc_list.__len__()):
        freespace_fc_unit = freespace_fc_list['points_' + str(i)]
        motion_state = freespace_fc_unit['motion_state']
        point_type = freespace_fc_unit['type']  # reespace点类型
        position_longitudinal_distance = freespace_fc_unit['position_longitudinal_distance']
        position_lateral_distance = freespace_fc_unit['position_lateral_distance']
        if point_type == 0:  # 忽略：0
            logging.debug('忽略')
            plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1, colorstr='dimgray')
        elif point_type == 1:  # 汽车 ：1
            logging.debug('汽车')
            plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1, colorstr='blue')
        elif point_type == 2:  # 路沿：2
            logging.debug('路边')
            plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1,
                        colorstr='brown')
        elif point_type == 3:  # 行人 ：3
            logging.debug('行人')
            plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1,
                        colorstr='red')
        elif point_type == 4:  # 锥形桶 ：4
            logging.debug('锥形桶')
            plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1,
                        colorstr='coral')
        elif point_type == 5:  # 静态目标：5
            logging.debug('静态目标')
            plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1,
                        colorstr='darkcyan')
        elif point_type == 6:  # 未知：6
            logging.debug('未知')
            plot_circle((position_longitudinal_distance, position_longitudinal_distance), r=1,
                        colorstr='aliceblue')
    return


def plot_img(img_bin):
    # figsize = 11, 9  # 设定图片大小，数字可以调整
    # figure, ax = plt.subplots(figsize=figsize)
    jpg_as_np = np.frombuffer(img_bin, np.uint8)
    img = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
    plt.imshow(img, interpolation='nearest')
    return


def Draw_Dection_POI(object_detection_list):
    for i in range(0, object_detection_list.__len__()):
        # 如果目标目标跟踪ID不为空
        track_id = object_detection_list['objs_' + str(i)]['track_id'];
        if track_id == 0:
            continue;
        # if already_draw_one == 2:
        #     break;
        if track_id != 0:
            # 获取目标类别
            objs_classification = object_detection_list['objs_' + str(i)]['classification']
            # 获取目标纵向和横向距离
            objs_longitudinal = object_detection_list['objs_' + str(i)]['longitudinal_distance']
            objs_lateral = object_detection_list['objs_' + str(i)]['lateral_distance']
            # 获取trackID
            track_id = object_detection_list['objs_' + str(i)]['track_id']
            track_id_txt = str(track_id)
            # print(type(track_id_txt))
            objs_length = object_detection_list['objs_' + str(i)]['length']
            objs_width = object_detection_list['objs_' + str(i)]['width']
            heading_angle = object_detection_list['objs_' + str(i)]['heading_angle']
            # 体地协同转换矩阵
            Cbe = np.transpose(
                np.array([[np.cos(heading_angle), np.sin(heading_angle), 0],
                          [- np.sin(heading_angle), np.cos(heading_angle), 0], [0, 0, 1]]))

            # print( objs_lateral,objs_longitudinal,objs_length,objs_width,track_id)
            longitudinal = objs_longitudinal / 2  # 目标纵向距离驾驶目标纵向宽度
            lateral = objs_lateral / 2  # 目标纵向距离驾驶目标横向宽度
            length = objs_length / 2
            width = objs_width / 2
            txt = str(objs_classification) + ':' + track_id_txt + ',(' + str(objs_longitudinal) + ',' + str(
                objs_lateral) + '),', str(objs_length), str(objs_width), \
                # txt = str(heading_angle)
            logging.debug(
                'objs_classification:track_id_txt,(objs_longitudinal,objs_lateral),objs_length,objs_width,heading_angle')
            logging.debug('目标是', txt)
            # if abs(objs_longitudinal) <= 10:
            # object_classification_class = ['未知目标', '汽车', '卡车', '摩托车', '行人', '自行车', '动物',
            #                                '巴士', '其他', '购物车',
            #                                '柱子', '锥桶', '已被锁上的车位锁', '未被锁上的车位锁 ']
            if objs_classification == 0:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='m')
                logging.debug("未知目标")
            elif objs_classification == 1:
                Draw_Rectangle_With_angle(objs_lateral, objs_longitudinal, objs_width, objs_length,
                                          heading_angle, 'y', '汽车')
                # plt.Rectangle(
                #          (objs_lateral, objs_longitudinal), 25, 25, fill=True, linewidth=1,edgecolor='red',facecolor='red')
                # #plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='y')
                logging.debug("汽车")
            elif objs_classification == 2:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='y')
                logging.debug("卡车")
            elif objs_classification == 3:
                logging.debug("摩托车")
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='r')
            elif objs_classification == 4:
                logging.debug("行人")
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='r')
            elif objs_classification == 5:
                logging.debug("自行车")
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='m')
            elif objs_classification == 6:
                logging.debug("动物")
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='m')
            elif objs_classification == 7:
                logging.debug("巴士")
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='g')
            elif objs_classification == 8:
                logging.debug("其他")
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='g')
            elif objs_classification == 9:
                logging.debug("购物车")
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='c')
            elif objs_classification == 10:
                logging.debug("柱子")
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='c')
            elif objs_classification == 11:
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='b')
                logging.debug("锥桶")
            elif objs_classification == 12:
                logging.debug("已被锁上的车位锁")
                plot_circle((objs_lateral, objs_longitudinal), r=1, colorstr='b')
            else:
                logging.debug("未被锁上的车位锁")
                plot_circle((objs_lateral, objs_longitudinal), r=1)
            cut_status = object_detection_list['objs_' + str(i)]['cut_status'] #Cut-in状态
            status = object_detection_list['objs_' + str(i)]['status']  # 目标运动状态
            longitudinal_relative_velocity = object_detection_list['objs_' + str(i)]['longitudinal_relative_velocity']  # 纵向相对速度
            lateral_relative_velocity = object_detection_list['objs_' + str(i)][
                'lateral_relative_velocity']  # 横向相对速度
            obj_refer_points = object_detection_list['objs_' + str(i)][
                'obj_refer_points']  # 目标测量参考点：下一级为纵向距离、横向距离
            # plt.Rectangle(
            #     (objs_lateral, objs_longitudinal), objs_length, objs_width, fill=True, edgecolor='red',
            #     linewidth=1)
            # plot_circle((objs_lateral, objs_longitudinal), r=1)
            plt.text(lateral + objs_width / 2, longitudinal + objs_length / 2, '%s' %
                     object_classification_class[objs_classification], ha='center', va='bottom', fontsize=7)
            # ax.add_patch(plt.Rectangle(
            #     (objs_lateral, objs_longitudinal), objs_length, objs_width))
            # ax.add_patch(plt.Rectangle((0, 0), 1, 1))
            # ax = fig.add_subplot(111)
            already_draw_one = 1
    return


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


def Draw_Lane_Line(Lane_line_list):
    for i in range(0, Lane_line_list.__len__()):
        Lane_line_list['lines_' + str(i)]
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
                plt.plot(x, y, linestyle='solid', color='b', linewidth=1.0)
                logging.debug('实线')
            elif Lane_line_list['lines_' + str(i)]['type'] == 1:  # 虚线
                plt.plot(x, y, linestyle='dotted', color='b', linewidth=1)
                logging.debug('虚线')
            elif Lane_line_list['lines_' + str(i)]['type'] == 2:  # 双虚线
                plt.plot(x, y, color='b',
                         linestyle='dotted', linewidth=1.0)
                plt.plot(x + 0.1, y + 0.1, color='b',
                         linestyle='dotted', linewidth=1.0)
                logging.debug('双虚线')
            elif Lane_line_list['lines_' + str(i)]['type'] == 3:  # 虚实线
                plt.plot(x, y, color='b',
                         linestyle='dotted', linewidth=1.0)
                plt.plot(x + 0.1, y + 0.1, color='b',
                         linestyle='solid', linewidth=1.0)
                logging.debug('虚实线')
            elif Lane_line_list['lines_' + str(i)]['type'] == 4:  # 实虚线
                plt.plot(x, y, color='b',
                         linestyle='solid', linewidth=1.0)
                plt.plot(x + 0.1, y + 0.1, color='b',
                         linestyle='dotted', linewidth=1.0)
                logging.debug('实虚线')
            elif Lane_line_list['lines_' + str(i)]['type'] == 5:  # 双实线
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


def plot_circle(center=(3, 3), r=2, colorstr='k'):
    try:
        x = np.linspace(center[0] - r, center[0] + r, 5000)
        y1 = np.sqrt(r ** 2 - (x - center[0]) ** 2) + center[1]
        y2 = -np.sqrt(r ** 2 - (x - center[0]) ** 2) + center[1]
        plt.plot(x, y1, c=colorstr)
        plt.plot(x, y2, c=colorstr)
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
    # print_hi('PyCharm')
    # for i in range(-360, 360):
    #     print('t(', i, '):', heading_angle_Cali(i))
    declare_a_global_variable()
    plt.figure(figsize=(30, 20), dpi=600)  # 字符型linestyle使用方法

    # plt.xlim(-40, 40)
    # # 设置y轴的刻度范围
    # plt.ylim(-40, 40)

    #plt.rcParams['font.sans-serif'] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['figure.figsize'] = (8.0, 6.0)
    # plt.rcParams['figure.dpi'] = 300 #分辨率
    # splt.rcParams['savefig.dpi'] = 150  # 图片像素

    prepare_data(1, 1,1)
    plt.legend()

    plt.show()
    #

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
