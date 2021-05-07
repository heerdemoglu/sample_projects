#!/usr/bin/env python
#######################################################
# Code developed by Hasan Emre Erdemoglu, 200377106
# Code developed as a part of
# Advanced Robotic Systems Course (QMUL-ECS7004P)
#######################################################

import rospy
import random
from std_msgs.msg import Float64

def square_size_generator():
    # Initialize the node with its name:
    rospy.init_node('generator', anonymous = True)

    # Construct the publisher:
    # Since the message is basic (no structs involved), I built it directly
    # Within this node.
    square_size_pub = rospy.Publisher('sq_size', Float64, queue_size = 10)

    # Set the rate to 1/20 Hz, publish every 20 seconds:
    rate = rospy.Rate(0.05)

    # Run indefinitely until shutdown:
    while not rospy.is_shutdown():
        # Create a random value and publish it every cycle:
        val = random.uniform(0.05, 0.20)
        rospy.loginfo(val) # Logs info to the terminal.
        square_size_pub.publish(val)
        rate.sleep()

if __name__ == '__main__':
	try:
		square_size_generator()
	except rospy.ROSInterruptException:
		pass
