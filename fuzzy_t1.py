#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import time
from std_srvs.srv import Empty
import numpy as np
import skfuzzy as sk
from csv import reader

x = 0
y = 0
theta = 0
x2 = 0
y2 = 0
theta2= 0

def poseCallback1(pose_message):
    global x
    global y
    global theta

    x = pose_message.x
    y = pose_message.y
    theta = pose_message.theta

def poseCallback2(pose_message):
    global x2
    global y2
    global theta2

    x2 = pose_message.x
    y2 = pose_message.y
    theta2 = pose_message.theta
    
##################################################################################
def Index(x, v):
    idx = np.where(x == v)
    if idx[0].shape == (0,):
        closest = 0
        for i in x:
            if abs(v-i) < abs(v-closest):
                closest = i
            idx = np.where(x == closest)
    return idx[0][0]
    
def getValues(da, dl):
    idx_l = Index(Y, dl)
    idx_a = Index(X, da)
    val_a = Za[idx_l][idx_a]

def go_to_goal (xgoal, ygoal):
    while(True):
#a partir de aquí sirve para convertir la separación angular
#entre las dos tortugas en un entero dtheta entre -180 y 180 grados
        if theta > math.pi:
            pm_theta = (theta-(2*math.pi))*(180/math.pi)
        else:
            pm_theta = theta*(180/math.pi)
        gtheta = (math.atan2(y2-y, x2-x))*(180/math.pi)
        dtheta = gtheta - pm_theta
        
        if dtheta>180:
            dtheta = int(dtheta - 360)
        if dtheta<-180:
            dtheta = int(dtheta + 360)
        else:
            dtheta = dtheta
#hasta aquí
#todo lo demás del goal to goal hay que reconstruirlo tomando en cuenta que en un código aparte se genera
#toda la superficie de controly se puede importar como una matriz para solo tomar los valores de cierta coordenada
        dlineal = abs(np.sqrt((x-x2)**2+(y-y2)**2))
        w, v = getValues(dtheta, dlineal)
        msg.angular.z,  msg.linear.x = -w, v
        velocity_publisher.publish(msg)
        if dlineal < 0.01:
            break

##################################################################################
#suscripciones
if __name__ == '__main__':
    try:
        rospy.init_node('turtlesim_motion_pose', anonymous = True)

        cmd_vel_topic = '/turtle1/cmd_vel'
        velocity_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size = 10)

        position_topic1 = "/turtle1/pose"
        pose_subscriber1 = rospy.Subscriber(position_topic1, Pose, poseCallback1)

        position_topic2 = "/turtle2/pose"
        pose_subscriber2 = rospy.Subscriber(position_topic2, Pose, poseCallback2)
        
        with open('surfaceZ_lin.csv', 'r') as read_obj: #Dar path completo
            csv_reader = reader(read_obj)
            list_z_lin = list(csv_reader)
        
        with open('surfaceZ_ang.csv', 'r') as read_obj: #Dar path completo
            csv_reader = reader(read_obj)
            list_z_ang = list(csv_reader)
            
        with open('surfaceX.csv', 'r') as read_obj: #Dar path completo
            csv_reader = reader(read_obj)
            list_x = list(csv_reader)
            
        with open('surfaceY.csv', 'r') as read_obj: #Dar path completo
            csv_reader = reader(read_obj)
            list_y = list(csv_reader)
        #Entrada

        time.sleep(1.0)
        while(True):         
            
            go_to_goal(x2,y2)
                   
    except rospy.ROSInterruptException:        
        pass
