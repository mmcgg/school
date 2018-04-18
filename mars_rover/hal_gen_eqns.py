#
# Copyright (c) 2013, Georgia Tech Research Corporation
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# \\ author Marc Killpack (Healthcare Robotics Lab, Georgia Tech.)
# \\ adviser Charlie Kemp (Healthcare Robotics Lab, Georgia Tech.)
# based on some sample code from Cristovao Sousa and his sympybotics library


import sympy
import sympybotics
import math

#defining length of the end effector for forward kinematics
l_ee = 0.203
pi = sympy.pi
q = sympybotics.robotdef.q


# defining constants for the offset dh params from the body to the first joint
x_offset = 0
y_offset = 0
z_offset = 0
 

### First, create a robot object:
### list of tuples with Denavit-Hartenberg parameters (alpha, a, d, theta)
#Standard DH PARAMS
#                alpha    a                        d          theta
dh_params = [(pi/2,   0.10795,                     0.0889,    q+pi/2),
             (0,      0.4572,                      0.0,       q),
             (pi/2,   0.06985,                     0.0,       q+pi/2),
             (-pi/2,  0.0,                         0.4064,    q),
             (pi/2,   0.0,                         0.0,       q),
             (0,      0.0,                         0.254,     q)]

rbtdef = sympybotics.RobotDef('Hal', dh_params , 'standard')


# defining which friction model to use
rbtdef.frictionmodel = {'viscous'} #'simple' # options are 'simple' and None, defaults to None

# printing the definition of which variables are used as dynamic parameters (i.e. mass, center of mass, etc.)
print(rbtdef.dynparms())
rbt = sympybotics.RobotDynCode(rbtdef)

arm = 'hal_arm'

offsets = str(x_offset)+', '+str(y_offset)+', '+str(z_offset)
f_kin = open('./'+arm+'_kinematics.py', 'w+')
print >> f_kin, "from math import sin, cos"
#print >> f_kin, "from offset_util import offset_and_reshape"
print >> f_kin, "import numpy as np"
print >> f_kin, "pi = np.pi\n\n\n"
fk_dict = "FK = {"
for i in range(len(rbt.geo.T)):
    joint_fk_code = sympy.cse(rbt.geo.T[i])
    fk_string = sympybotics.robotcodegen.robot_code_to_func('python', joint_fk_code, 'pose', 'joint_fk' + str(i).zfill(2), rbtdef)
    fk_list_string = fk_string.split('\n')
#    fk_list_string.insert(-1, '    pose = offset_and_reshape(pose,'+str(x_offset)+','
#                           +str(y_offset)+','
#                           +str(z_offset)+')')
    fk_final = "\n".join(fk_list_string)
    print >> f_kin, fk_final + '\n\n\n'
    fk_dict = fk_dict+str(i)+":joint_fk"+str(i).zfill(2)+", "

print >> f_kin, fk_dict+"}\n\n\n"

jac_dict = "J = {"
for i in xrange(len(rbt.kin.J)):
    joint_jac_code = sympy.cse(rbt.kin.J[i])
    jac_string = sympybotics.robotcodegen.robot_code_to_func('python', joint_jac_code, 'jacobian', 'jacobian' + str(i).zfill(2), rbtdef)

    jac_list_string = jac_string.split('\n')
    jac_list_string.insert(-1, '    jacobian = np.array(jacobian).reshape(6,'+str(rbt.dof)+')')
    jac_final = "\n".join(jac_list_string)
    print >> f_kin, jac_final + '\n\n\n'
    jac_dict = jac_dict + str(i)+":jacobian"+str(i).zfill(2)+", "

print >> f_kin, jac_dict+"}"

f_kin.close()

