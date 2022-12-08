# coding=utf-8
import csv
import getopt
import glob
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import time
from scipy.interpolate import make_interp_spline as spline
from sympy.stats.drv_types import scipy

# Simulator options.
options = {}
options['FIG_SIZE'] = [8, 8]
options['FULL_RECALCULATE'] = True
options['OBSTACLES'] = False

# drive the car from a start position to an end position quickest while driving under the speed limit
def sim_run(x_,y_,v_,dangerM_,complex,options, MPC):
    start = time.process_time()
    # Simulator Options
    FIG_SIZE = options['FIG_SIZE'] # [Width, Height]
    OBSTACLES = options['OBSTACLES']

    mpc = MPC()

    num_inputs = 2
    velocity = v_  #np.zeros(mpc.horizon*num_inputs)
    bounds = []

    # Set bounds for inputs bounded optimization.
    for i in range(mpc.horizon):
        bounds += [[-1, 1]]
        bounds += [[-0.8, 0.8]]

    ref_1 = mpc.reference1
    ref_2 = mpc.reference2
    ref = ref_1

    state_i = np.array([[0,0,0,0]])
    u_i = np.array([[0,0]])
    sim_total = 120
    predict_info = [state_i]

    for i in range(1,sim_total+1):
        #似乎是随机产生速度值
        velocity = np.delete(velocity,0)
        velocity = np.delete(velocity,0)
        velocity = np.append(velocity, velocity[-2])
        velocity = np.append(velocity, velocity[-2])
        start_time = time.time()

        # Non-linear optimization.
        # minimize是scipy中optimize模块的一个函数
        # res = opt.minimize(fun, x0, args=(), method=None, jac=None, hess=None,
        #                    hessp=None, bounds=None, constraints=(), tol=None,
        #                    callback=None, options=None)
        # # fun：该参数就是costFunction你要去最小化的损失函数，将costFunction的名字传给fun
        # # x0: 猜测的初始值
        # # args=():优化的附加参数，默认从第二个开始
        # # method：该参数代表采用的方式，默认是BFGS, L-BFGS-B, SLSQP中的一种，可选TNC
        # # options：用来控制最大的迭代次数，以字典的形式来进行设置，例如：options={‘maxiter’:400}
        # # constraints: 约束条件，针对fun中为参数的部分进行约束限制,多个约束如下：
        # '''cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},\
        #    {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},\
        #    {'type': 'ineq', 'fun': lambda x: x[1] - x2min},\
        #    {'type': 'ineq', 'fun': lambda x: -x[1] + x2max})'''
        # # tol: 目标函数误差范围，控制迭代结束
        # # callback: 保留优化过程
        u_solution = minimize(mpc.cost_function, velocity, (state_i[-1], ref,float(dangerM_[i]),float(complex[i])),
                                method='SLSQP',
                                bounds=bounds,
                                tol = 1e-5)
        print('Step ' + str(i) + ' of ' + str(sim_total) + '   Time ' + str(round(time.time() - start_time,5)))
        velocity = u_solution.x
        # pedal 踏板, steering 转向
        y = mpc.plant_model(state_i[-1], mpc.dt, velocity[0], velocity[1])
        if (i > 130 and ref_2 != None):
            ref = ref_2
        predicted_state = np.array([y])
        for j in range(1, mpc.horizon):
            predicted = mpc.plant_model(predicted_state[-1], mpc.dt, velocity[2*j], velocity[2*j+1])
            predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)
        predict_info += [predicted_state]
        state_i = np.append(state_i, np.array([y]), axis=0)
        u_i = np.append(u_i, np.array([(velocity[0], velocity[1])]), axis=0)


    ###################
    # SIMULATOR DISPLAY

    # Total Figure
    fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
    gs = gridspec.GridSpec(8,8)

    # Elevator plot settings.
    ax = fig.add_subplot(gs[:8, :8])

    plt.xlim(-3, 17)
    ax.set_ylim([-3, 17])
    plt.xticks(np.arange(0,11, step=2))
    plt.yticks(np.arange(0,11, step=2))
    plt.title('MPC 2D')

    # Time display.
    time_text = ax.text(6, 0.5, '', fontsize=15)

    # Main plot info.
    car_width = 1.0
    patch_car = mpatches.Rectangle((0, 0), car_width, 2.5, fc='k', fill=False)
    patch_goal = mpatches.Rectangle((0, 0), car_width, 2.5, fc='b',
                                    ls='dashdot', fill=False)

    ax.add_patch(patch_car)
    ax.add_patch(patch_goal)
    predict, = ax.plot([], [], 'r--', linewidth = 1)

    # Car steering and throttle position.
    # 汽车转向和油门位置
    telem = [3,14]
    patch_wheel = mpatches.Circle((telem[0]-3, telem[1]), 2.2)
    ax.add_patch(patch_wheel)
    wheel_1, = ax.plot([], [], 'k', linewidth = 3)
    wheel_2, = ax.plot([], [], 'k', linewidth = 3)
    wheel_3, = ax.plot([], [], 'k', linewidth = 3)
    throttle_outline, = ax.plot([telem[0], telem[0]], [telem[1]-2, telem[1]+2],
                                'b', linewidth = 20, alpha = 0.4)
    throttle, = ax.plot([], [], 'k', linewidth = 20)
    brake_outline, = ax.plot([telem[0]+3, telem[0]+3], [telem[1]-2, telem[1]+2],
                            'b', linewidth = 20, alpha = 0.2)
    brake, = ax.plot([], [], 'k', linewidth = 20)
    throttle_text = ax.text(telem[0], telem[1]-3, 'Forward', fontsize = 15,
                        horizontalalignment='center')
    brake_text = ax.text(telem[0]+3, telem[1]-3, 'Reverse', fontsize = 15,
                        horizontalalignment='center')

    # Obstacles
    if OBSTACLES:
        patch_obs = mpatches.Circle((mpc.x_obs, mpc.y_obs),0.5)
        ax.add_patch(patch_obs)

    # Shift xy, centered on rear of car to rear left corner of car.
    def car_patch_pos(x, y, psi):
        #return [x,y]
        x_new = x - np.sin(psi)*(car_width/2)
        y_new = y + np.cos(psi)*(car_width/2)
        return [x_new, y_new]

    def steering_wheel(wheel_angle):
        wheel_1.set_data([telem[0]-3, telem[0]-3+np.cos(wheel_angle)*2],
                         [telem[1], telem[1]+np.sin(wheel_angle)*2])
        wheel_2.set_data([telem[0]-3, telem[0]-3-np.cos(wheel_angle)*2],
                         [telem[1], telem[1]-np.sin(wheel_angle)*2])
        wheel_3.set_data([telem[0]-3, telem[0]-3+np.sin(wheel_angle)*2],
                         [telem[1], telem[1]-np.cos(wheel_angle)*2])

    def update_plot(num):
        # Car.
        patch_car.set_xy(car_patch_pos(state_i[num,0], state_i[num,1], state_i[num,2]))
        patch_car.angle = np.rad2deg(state_i[num,2])-90
        # Car wheels
        np.rad2deg(state_i[num,2])
        steering_wheel(u_i[num,1]*2)
        # throttle 汽车油门
        throttle.set_data([telem[0],telem[0]],
                        [telem[1]-2, telem[1]-2+max(0,u_i[num,0]/5*4)])
        # brake 汽车刹车
        brake.set_data([telem[0]+3, telem[0]+3],
                        [telem[1]-2, telem[1]-2+max(0,-u_i[num,0]/5*4)])

        # Goal.
        if (num <= 130 or ref_2 == None):
            patch_goal.set_xy(car_patch_pos(ref_1[0],ref_1[1],ref_1[2]))
            patch_goal.angle = np.rad2deg(ref_1[2])-90
        else:
            patch_goal.set_xy(car_patch_pos(ref_2[0],ref_2[1],ref_2[2]))
            patch_goal.angle = np.rad2deg(ref_2[2])-90

        #print(str(state_i[num,3]))
        predict.set_data(predict_info[num][:,0],predict_info[num][:,1])
        # Timer.
        #time_text.set_text(str(100-t[num]))

        return patch_car, time_text


    print("Compute Time: ", round(time.process_time() - start, 3), "seconds.")
    # Animation.
    car_ani = animation.FuncAnimation(fig, update_plot, frames=range(1,len(state_i)), interval=100, repeat=True, blit=False)
    #car_ani.save('mpc-video.mp4')

    plt.show()

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 20
        self.dt = 0.2
        # every element in input array u will be remain for dt seconds
        # 输入数组u中的每个元素将保留dt秒
        # here with horizon 20 and dt 0.2 we will predict 4 seconds ahead(20*0.2)
        # 在地平线20和dt 0.2的情况下，我们将预测前方4秒（20*0.2）
        # we can't predict too much ahead in time because that might be pointless and take too much computational time
        # 我们不能提前预测太多，因为这可能毫无意义，而且需要太多的计算时间
        # we can't predict too less ahead in time because that might end up overshooting from end point as it won't be able to see the end goal in time
        # 我们不能提前预测得太少，因为这可能会超出终点，因为它无法及时看到最终目标
        # Reference or set point the controller will achieve.
        # 控制器将达到的参考值或设定值。
        self.reference1 = [50, 0, 0]
        self.reference2 = None
        self.x_obs = 5
        self.y_obs = 0.1

    def plant_model(self, prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        v_t = prev_state[3] # m/s
        a_t = pedal

        x_t_1 = x_t + v_t*dt  # distance = speed*time
        v_t_1 = v_t + a_t*dt - v_t/25.0  # v = u + at (-v_t/25 is a rough estimate for friction)

        return [x_t_1, 0, 0, v_t_1]

    def cost_function(self, u, *args):
        state = args[0]
        ref = args[1]
        x_end = ref[0]
        cost = 0.0
        # u[0] 刹车
        # u[1] 转向
        for i in range(self.horizon):
            state = self.plant_model(state, self.dt, u[i*2], u[i*2+1])
            # u input vector is designed like = [(pedal for t1), (steering for t1), (pedal for t2), (steering for t2)...... (pedal for t(horizon)), (steering for t(horizon))]
            x_current = state[0]
            v_current = state[3]

            cost += (x_end - x_current)**2  # so we keep getting closer to end point

            if v_current*3.6 > 10:  # speed limit is 10km/h, 3.6 is multiplied to convert m/s to km/h
                cost += 100*v_current

            if x_current > x_end:  # so we don't overshoot further than end point
                cost += 10000
        return cost
if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["ifile=", "wdir=", ])
    for opt, arg in opts:
        if opt == '-h':
            print
            '3.HighwaySpeedControl.py -i <inputfile> -w <working directory>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            targetfile = arg
        elif opt in ("-w", "--wdir"):
            path = arg
            if not path.endswith('/'):
                path = path + '/'
    path = 'D:/DataContest/data/image2/mp/'
    targetfile = 'carinfo.csv'
    print(path + targetfile)
    with open(path + targetfile, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        log = []
        for row in file_reader:
            log.append(row)
    # log = np.array(log)
    # 去掉文件第一行
    log = log[1:]

    # 判断图像文件数量是否等于csv日志文件中记录的数量
    # ls_imgs = glob.glob(path + 'scence*.png')
    # # assert len(ls_imgs) == len(log) * 3, 'number of images does not match'
    # avalid_data_length=min(len(ls_imgs),len(log) )
    # # 使用20%的数据作为测试数据
    # validation_ratio = 0.2
    # shape = (128, 128, 3)
    # batch_size = 32
    # nb_epoch = 200
    # esp_vehicle_speed_stp_motion,  # 车速
    # vehicle_pos_lng_hdmap, # 锚点经度坐标
    # vehicle_pos_lat_hdmap, # 锚点维度坐标
    # esp_lat_accel_stp_motion, # 横向加速度
    # esp_long_accel_stp_motion, # 纵向加速度
    # risk_value, # 风险值
    # statics_complex_value + statics_comple_value2 + dynamic_complex_value  #环境复杂度
    x_=[]
    y_=[]
    dangerM_=[]
    v_=[]
    vs_ = []
    complex=[]
    e_ = []
    v0_ =round(float(log[0][1]),2)
    for i in range(0,123):
        v_.append(round(float(log[i][1]),2))
        x_.append(round(float(log[i][2]),8))
        y_.append(round(float(log[i][3]),8))
        dangerM_unit =abs(round(float(log[i][6])*80,6))
        complex_unit = round(float(log[i][7]),2)
        dangerM_.append(dangerM_unit)
        complex.append(round(float(log[i][7]),2))
        if dangerM_unit > 700 :
            v0_=v0_-v0_*0.05
        else:
            v0_ = v0_ + v0_ * 0.05
        if complex_unit > 600:
            v0_ = v0_ - v0_ * 0.05
        # else:
        #     v0_ = v0_ + v0_ * 0.05
        vs_.append(v0_)
        e_.append(round(float(log[i][1]),2)-v0_)



    # tmp_smooth2 = scipy.signal.savgol_filter(vs_, 53, 3)
    # plt.semilogx(f, tmp_smooth2 * 0.5, label='MPC生成速度', color='green')
    #
    # # x_ = log[:, 0]
    x_ = np.array(x_)
    y_ = np.array(y_)
    dangerM_ = np.array(dangerM_)
    v_ = np.array(v_)
    complex =  np.array(complex)
    # y_ = log[:, 3].astype(float)
    # z_ = log[:, 4].astype(float)
    # print(x_.shape)
    # print(y_.shape)
    # print(v_.shape)
    # print(dangerM_.shape)
    # print(complex.shape)
    time_index = np.arange(0, 123, 1)

    plt.figure(figsize=(6, 6), dpi=80)

    plt.figure(1)

    ax1 = plt.subplot(221)
    ax1.set_title('(a) routine')
    plt.plot(x_, y_, color="r", linestyle="--")

    ax2 = plt.subplot(222)
    ax2.set_title('(b) dangerous and complex')
    plt.plot(time_index, complex, color="g", linestyle="-.", label='complex')
    plt.plot(time_index, dangerM_, color="m", linestyle="-.",label='dangerous')
    plt.legend()
    ax3 = plt.subplot(223)
    ax3.set_title('(c) velocity')
    # x_new = np.linspace(min(vs_), max(vs_), 123)
    # vs_smooth = spline(time_index, vs_)(x_new)
    plt.plot(time_index, v_, color="y", linestyle="-",label='real')
    plt.plot(time_index, vs_, color="g", linestyle="-",label='simulation')
    plt.legend()
    ax3 = plt.subplot(224)
    ax3.set_title('(d) velocity of Speed difference')
    # x_new = np.linspace(min(vs_), max(vs_), 123)
    # vs_smooth = spline(time_index, vs_)(x_new)
    plt.plot(time_index, e_, color="y", linestyle="-")

    # sim_run(x_,y_,v_,dangerM_,complex,options, ModelPredictiveControl)
    plt.show()