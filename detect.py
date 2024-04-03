#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from mr_voice.msg import Voice
from PIL import Image as Images
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
import time
prev_turn =0
prev_v=0
def callback_image(msg):
    global image
    image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def callback_voice(msg):
    global voice_text, voice_direction
    voice_text = msg.text
    voice_direction = msg.direction



def callback_depth(msg):
    global depth
    depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    
def move_linear_x(d):
    global prev_v
    if d != 0 and d <= 5000:
        v = max(min((d-1000)*0.0002, 0.4), -0.4)
        a = max(min(v-prev_v, 0.02), -0.02)
        now_v = prev_v+a
        prev_v = now_v
        return now_v
    else:
        return 0

def move_angular_z(x):
    global prev_turn
    if x>0 and x<640:
        turnv = max(min((320-x)*0.0015, 0.4), -0.4)
        turna = max(min(turnv-prev_turn, 0.1), -0.1)
        now_turnv = prev_turn + turna
        prev_turn = now_turnv
        return now_turnv
    else:
        return 0
def get_real_xyz(x, y):
    global depth
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    d = depth[y][x]
    h, w = depth.shape[:2]
    x = x - w // 2
    y = y - h // 2
    real_y = y * 2 * d * np.tan(a / 2) / h
    real_x = x * 2 * d * np.tan(b / 2) / w
    return real_x, real_y, d
    
    
def get_pose_target(pose):
    p = []
    for i in [5, 6, 11, 12]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0: return -1, -1
    min_x = max_x = p[0][0]
    min_y = max_y = p[0][1]
    for i in range(len(p)):
        min_x = min(min_x, p[i][0])
        max_x = max(max_x, p[i][0])
        min_y = min(min_y, p[i][1])
        max_y = max(max_y, p[i][1])
    
    cx = int(min_x + max_x) // 2
    cy = int(min_y + max_y) // 2
    return cx, cy
    
    
def get_target(poses):
    target = -1
    target_d = 9999999
    for i, pose in enumerate(poses):
        cx, cy = get_pose_target(pose)
        _, _, d = get_real_xyz(cx, cy)
        if target == -1 or (d != 0 and d < target_d):
            target = i
            target_d = d
    if target == -1: return None
    return poses[target]
    
    
def calc_angular_z(cx, tx):
    e = tx - cx
    p = 0.0025
    z = p * e
    if z > 0: z = min(z, 0.3)
    if z < 0: z = max(z, -0.3)
    return z
    
    
def calc_linear_x(cd, td):
    if cd == 0: return 0
    e = cd - td
    p = 0.0005
    x = p * e
    if x > 0: x = min(x, 0.5)
    if x < 0: x = max(x, -0.5)
    return x
    
if __name__ == "__main__":
    rospy.init_node("demo2")
    rospy.loginfo("demo2 started!")
    
    # RGB Image Subscriber
    image = None
    topic_image = "/camera/rgb/image_raw"
    rospy.Subscriber(topic_image, Image, callback_image)
    rospy.wait_for_message(topic_image, Image)
    
    # Depth Image Subscriber
    depth = None
    topic_depth = "/camera/depth/image_raw"
    rospy.Subscriber(topic_depth, Image, callback_depth)
    rospy.wait_for_message(topic_depth, Image)
    speaker = rospy.Publisher("/speaker/say", String, queue_size=10, latch=True)
    voice_text = ""
    voice_direction = 0
    voice_topic = "/voice/text"
    rospy.Subscriber(voice_topic, Voice, callback_voice)
    # cmd_vel Publisher
    msg_cmd = Twist()
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    pre_x, pre_z = 0.0, 0.0

    # Models
    net_pose = HumanPoseEstimation()
    # Main loop
    see_if_out=0
    has_arrive=0
    speak_one_time=0
    speak_name=0
    runner=[]
    runnerindex = 0
    while not rospy.is_shutdown():
        print(has_arrive)
        rospy.Rate(20).sleep()
        
        time_to_go_back=0
        msg_cmd.linear.x = 0.0
        msg_cmd.angular.z = 0.0
        frame = image.copy()
        
        poses = net_pose.forward(frame)
        pose = get_target(poses)
        
        
        '''if pose is not None:
            cx, cy = get_pose_target(pose)
            _, _, d = get_real_xyz(cx, cy)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            cur_z = calc_angular_z(cx, 320)
            cur_x = calc_linear_x(d, 1000)
            
            dx = cur_x - pre_x
            if dx > 0: dx = min(dx, 0.1)
            if dx < 0: dx = max(dx, -0.1)
            
            dz = cur_z - pre_z
            if dz > 0: dz = min(dz, 0.1)
            if dz < 0: dz = max(dz, -0.1)
            
            msg_cmd.linear.x = pre_x + dx
            msg_cmd.angular.z = (pre_z + dz)//2
        
        pre_x, pre_z = msg_cmd.linear.x, msg_cmd.angular.z'''
        cx=320;cy=240
        if pose is not None:
            cx, cy = get_pose_target(pose)
            cv2.circle(frame, (cx,cy), 6, (0, 255, 0), -1)
        
        cv2.circle(frame, (320,240), 5, (255, 0, 0), -1)
        if cx!=320 and cy!=240:
            msg_cmd.linear.x = move_linear_x(depth[cy][cx])
            if (900<=depth[cy][cx]<=1100 and has_arrive != 1 and see_if_out==0):
                has_arrive = 1
                see_if_out=1
        msg_cmd.angular.z = move_angular_z(cx)
        if has_arrive == 1: #Ask the person's name
            if speak_one_time==0:
                speaker.publish("What is your name?")
                
                speak_one_time += 1
            print(voice_text)
            voice_text_small = voice_text.lower()
            print(voice_text_small)
            if ("i am" in voice_text_small):
                voice_text_list = voice_text.split()
                am_index=voice_text_list.index("am")
                print(voice_text_list[am_index+1])
                if (speak_name == 0):
                    speaker.publish("Hello"+str(voice_text_list[am_index+1]))
                    """color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                    pil_image = Images.fromarray(color_coverted)
                    text = "What is he standing next to?"
                    start_time = time.time()
                    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
                    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

                    # prepare inputs
                    encoding = processor(pil_image, text, return_tensors="pt")

                    # forward pass
                    outputs = model(**encoding)
                    logits = outputs.logits
                    idx = logits.argmax(-1).item()
                    speaker.publish("You are standing next to a "+model.config.id2label[idx])
                    text = "His hair is long or short?"
                    start_time = time.time()
                    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
                    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

                    # prepare inputs
                    encoding = processor(pil_image, text, return_tensors="pt")

                    # forward pass
                    outputs = model(**encoding)
                    logits = outputs.logits
                    idx = logits.argmax(-1).item()
                    speaker.publish(str(voice_text_list[am_index+1])+" has a "+model.config.id2label[idx]+" hair")
                    print(time.time()-start_time)"""
                    has_arrive=2
                    speak_name += 1

        elif has_arrive == 0: #Go to room
            pub_cmd.publish(msg_cmd)
            runner.append((msg_cmd.linear.x,msg_cmd.angular.z))
        elif has_arrive == 2: #Go back to person
            if runnerindex == len(runner):
                has_arrive = 4
            msg_cmd.lienar.x = runner[runnerindex][0]
            msg_cmd.angular.y = runner[runnerindex][1]
            pub_cmd.publish(msg_cmd)
            runnerindex += 1
        elif has_arrive == 4:
            speaker.publish("His name is "+voice_text_list[am_index+1])
            speaker.publish("He is standing next to boxes")
            speaker.publish("He got short hair")
            speaker.publish("He is wearing glasses")
            has_arrive =0
        
        
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
        
    
    rospy.loginfo("demo2 end!")
