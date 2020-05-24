#!/usr/bin/env python3

import rospy
from fastai.vision import *
from std_msgs.msg import String

def print_class(x):
    x = int(x)
    if x == 0:
        return 'bottle'
    elif x == 1:
        return 'chair'
    elif x == 2:
        return 'sofa'
    elif x == 3:
        return 'table'

def prediction():
    defaults.device = torch.device('cpu')
    path = Path('/home/hitech/practice_ws/src/multiClassification/src/data')
    img = open_image(path/'test'/'123.jpeg')
    learn = load_learner(path,'export.pkl')
    pred_class,pred_idx,oututs = learn.predict(img)
    return print_class(pred_class)

def talker():
    predicted = prediction()
    print('Category is ',predicted)
    pub = rospy.Publisher('predic', String, queue_size = 10)
    rospy.init_node('multiClass', anonymous = True)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pred_str = predicted + '%s' % rospy.get_time()
        rospy.loginfo(pred_str)
        pub.publish(pred_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
