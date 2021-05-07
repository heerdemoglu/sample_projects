#!/usr/bin/env python

import rospy

from AR_week4_test.srv import compute_cubic_traj
from AR_week4_test.msg import *

# Publisher to publish trajectory coefficients:
pub = rospy.Publisher('traj_coeffs', cubic_traj_coeffs, queue_size=0)

# Open service call:
srv = rospy.ServiceProxy('compute_srv', compute_cubic_traj)

# When new messages arrive, callback is invoked as the first argument
# Waits for the service, returns the query and publishes new message
# Now plan the trajectory: (Client side)
def callback(data):

    try:
        response = srv(data)

        # Construct trajectory message:
        message = cubic_traj_coeffs()
        message.a0 = response.a0
        message.a1 = response.a1
        message.a2 = response.a2
        message.a3 = response.a3
        message.t0 = data.t0 # t0 also in response
        message.tf = data.tf # tf also in response

        # Publish this data: - Used for Node 4: plot_cubic_traj
        pub.publish(message)

    except rospy.ServiceException as e:
        print("Service call failed: " + str(e))

def plan_trajectory():

    # First construct the subscriber
    rospy.init_node('traj_planner', anonymous=True)

    # Subscribes to points_generator.py, sends message as data to "callback".
    rospy.Subscriber('points_msg', cubic_traj_params, callback)

    # Wait for compute_cubic_coeffs service:
    rospy.wait_for_service('compute_srv')

    rospy.spin() # continue forever

if __name__ == '__main__':
    plan_trajectory()
