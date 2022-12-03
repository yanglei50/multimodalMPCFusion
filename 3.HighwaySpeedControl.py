# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import time

# Simulator options.
options = {}
options['FIG_SIZE'] = [8, 8]
options['FULL_RECALCULATE'] = True
options['OBSTACLES'] = False

# drive the car from a start position to an end position quickest while driving under the speed limit
def sim_run(options, MPC):
    start = time.process_time()
    # Simulator Options
    FIG_SIZE = options['FIG_SIZE'] # [Width, Height]
    OBSTACLES = options['OBSTACLES']

    mpc = MPC()

    num_inputs = 2
    u = np.zeros(mpc.horizon*num_inputs)
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
    sim_total = 250
    predict_info = [state_i]

    for i in range(1,sim_total+1):
        u = np.delete(u,0)
        u = np.delete(u,0)
        u = np.append(u, u[-2])
        u = np.append(u, u[-2])
        start_time = time.time()

        # Non-linear optimization.
        u_solution = minimize(mpc.cost_function, u, (state_i[-1], ref),
                                method='SLSQP',
                                bounds=bounds,
                                tol = 1e-5)
        print('Step ' + str(i) + ' of ' + str(sim_total) + '   Time ' + str(round(time.time() - start_time,5)))
        u = u_solution.x
        y = mpc.plant_model(state_i[-1], mpc.dt, u[0], u[1])
        if (i > 130 and ref_2 != None):
            ref = ref_2
        predicted_state = np.array([y])
        for j in range(1, mpc.horizon):
            predicted = mpc.plant_model(predicted_state[-1], mpc.dt, u[2*j], u[2*j+1])
            predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)
        predict_info += [predicted_state]
        state_i = np.append(state_i, np.array([y]), axis=0)
        u_i = np.append(u_i, np.array([(u[0], u[1])]), axis=0)


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
        throttle.set_data([telem[0],telem[0]],
                        [telem[1]-2, telem[1]-2+max(0,u_i[num,0]/5*4)])
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
    sim_run(options, ModelPredictiveControl)
