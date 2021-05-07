#!/usr/bin/env python

import rospy
import numpy as np
from AR_week4_test.srv import compute_cubic_traj

# Callback to run this:
def handle_compute_srv(req):

    # Matrix of time: - Follow from W4-S16
    tm = np.matrix([[1, req.msg.t0, req.msg.t0**2, req.msg.t0**3],
             [0, 1, 2*req.msg.t0, 3*req.msg.t0**2],
             [1, req.msg.tf, req.msg.tf**2, req.msg.tf**3],
             [0, 1, 2*req.msg.tf, 3*req.msg.tf**2]])

    # Matrix of coefficients: - column vector
    cf = [req.msg.p0, req.msg.v0, req.msg.pf, req.msg.vf]

    # Inverse of tm:
    inv_tm = np.linalg.inv(tm)

    # Compute cubic trajectory: - convert to list to send as msg
    return inv_tm.dot(cf).tolist()[0]

def compute_coeffs():
    # Initialize the node:
    rospy.init_node('compute_cubic_coeffs')

    # Start the service: - compute_cubic_traj has positions, velocities & time
    srv = rospy.Service('compute_srv', compute_cubic_traj, handle_compute_srv)

    # Keep alive:
    rospy.spin()

if __name__ == "__main__":
    compute_coeffs()
