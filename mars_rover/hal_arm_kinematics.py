from math import sin, cos
import numpy as np
pi = np.pi



def joint_fk00(q):
#
    pose = [0]*16
#
    x0 = sin(q[0])
    x1 = cos(q[0])
#
    pose[0] = -x0
    pose[1] = 0
    pose[2] = x1
    pose[3] = -0.10795*x0
    pose[4] = x1
    pose[5] = 0
    pose[6] = x0
    pose[7] = 0.10795*x1
    pose[8] = 0
    pose[9] = 1
    pose[10] = 0
    pose[11] = 0.0889000000000000
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
#
    return pose



def joint_fk01(q):
#
    pose = [0]*16
#
    x0 = sin(q[0])
    x1 = cos(q[1])
    x2 = x0*x1
    x3 = sin(q[1])
    x4 = cos(q[0])
    x5 = x1*x4
#
    pose[0] = -x2
    pose[1] = x0*x3
    pose[2] = x4
    pose[3] = -0.10795*x0 - 0.4572*x2
    pose[4] = x5
    pose[5] = -x3*x4
    pose[6] = x0
    pose[7] = 0.10795*x4 + 0.4572*x5
    pose[8] = x3
    pose[9] = x1
    pose[10] = 0
    pose[11] = 0.4572*x3 + 0.0889
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
#
    return pose



def joint_fk02(q):
#
    pose = [0]*16
#
    x0 = sin(q[0])
    x1 = sin(q[1])
    x2 = cos(q[2])
    x3 = x1*x2
    x4 = x0*x3
    x5 = sin(q[2])
    x6 = cos(q[1])
    x7 = x5*x6
    x8 = x0*x7
    x9 = cos(q[0])
    x10 = x1*x5
    x11 = x2*x6
    x12 = 0.4572*x6
    x13 = x3*x9
    x14 = x7*x9
#
    pose[0] = x4 + x8
    pose[1] = x9
    pose[2] = x0*x10 - x0*x11
    pose[3] = -x0*x12 - 0.10795*x0 + 0.06985*x4 + 0.06985*x8
    pose[4] = -x13 - x14
    pose[5] = x0
    pose[6] = -x10*x9 + x11*x9
    pose[7] = x12*x9 - 0.06985*x13 - 0.06985*x14 + 0.10795*x9
    pose[8] = -x10 + x11
    pose[9] = 0
    pose[10] = x3 + x7
    pose[11] = 0.4572*x1 - 0.06985*x10 + 0.06985*x11 + 0.0889
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
#
    return pose



def joint_fk03(q):
#
    pose = [0]*16
#
    x0 = sin(q[3])
    x1 = cos(q[0])
    x2 = cos(q[3])
    x3 = sin(q[0])
    x4 = sin(q[1])
    x5 = cos(q[2])
    x6 = x4*x5
    x7 = x3*x6
    x8 = sin(q[2])
    x9 = cos(q[1])
    x10 = x8*x9
    x11 = x10*x3
    x12 = x11 + x7
    x13 = x5*x9
    x14 = x13*x3
    x15 = x4*x8
    x16 = x15*x3
    x17 = 0.4572*x9
    x18 = x1*x6
    x19 = x1*x10
    x20 = -x18 - x19
    x21 = x1*x15
    x22 = x1*x13
    x23 = x13 - x15
#
    pose[0] = x0*x1 + x12*x2
    pose[1] = x14 - x16
    pose[2] = -x0*x12 + x1*x2
    pose[3] = 0.06985*x11 - 0.4064*x14 + 0.4064*x16 - x17*x3 - 0.10795*x3 + 0.06985*x7
    pose[4] = x0*x3 + x2*x20
    pose[5] = x21 - x22
    pose[6] = -x0*x20 + x2*x3
    pose[7] = x1*x17 + 0.10795*x1 - 0.06985*x18 - 0.06985*x19 - 0.4064*x21 + 0.4064*x22
    pose[8] = x2*x23
    pose[9] = -x10 - x6
    pose[10] = -x0*x23
    pose[11] = 0.4064*x10 + 0.06985*x13 - 0.06985*x15 + 0.4572*x4 + 0.4064*x6 + 0.0889
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
#
    return pose



def joint_fk04(q):
#
    pose = [0]*16
#
    x0 = sin(q[4])
    x1 = sin(q[0])
    x2 = cos(q[1])
    x3 = cos(q[2])
    x4 = x2*x3
    x5 = x1*x4
    x6 = sin(q[1])
    x7 = sin(q[2])
    x8 = x6*x7
    x9 = x1*x8
    x10 = x5 - x9
    x11 = cos(q[4])
    x12 = sin(q[3])
    x13 = cos(q[0])
    x14 = cos(q[3])
    x15 = x3*x6
    x16 = x1*x15
    x17 = x2*x7
    x18 = x1*x17
    x19 = x16 + x18
    x20 = x12*x13 + x14*x19
    x21 = 0.4572*x2
    x22 = x13*x8
    x23 = x13*x4
    x24 = x22 - x23
    x25 = x13*x15
    x26 = x13*x17
    x27 = -x25 - x26
    x28 = x1*x12 + x14*x27
    x29 = -x15 - x17
    x30 = x4 - x8
    x31 = x14*x30
#
    pose[0] = x0*x10 + x11*x20
    pose[1] = -x12*x19 + x13*x14
    pose[2] = x0*x20 - x10*x11
    pose[3] = -x1*x21 - 0.10795*x1 + 0.06985*x16 + 0.06985*x18 - 0.4064*x5 + 0.4064*x9
    pose[4] = x0*x24 + x11*x28
    pose[5] = x1*x14 - x12*x27
    pose[6] = x0*x28 - x11*x24
    pose[7] = x13*x21 + 0.10795*x13 - 0.4064*x22 + 0.4064*x23 - 0.06985*x25 - 0.06985*x26
    pose[8] = x0*x29 + x11*x31
    pose[9] = -x12*x30
    pose[10] = x0*x31 - x11*x29
    pose[11] = 0.4064*x15 + 0.4064*x17 + 0.06985*x4 + 0.4572*x6 - 0.06985*x8 + 0.0889
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
#
    return pose



def joint_fk05(q):
#
    pose = [0]*16
#
    x0 = sin(q[5])
    x1 = cos(q[0])
    x2 = cos(q[3])
    x3 = sin(q[3])
    x4 = sin(q[0])
    x5 = sin(q[1])
    x6 = cos(q[2])
    x7 = x5*x6
    x8 = x4*x7
    x9 = sin(q[2])
    x10 = cos(q[1])
    x11 = x10*x9
    x12 = x11*x4
    x13 = x12 + x8
    x14 = x1*x2 - x13*x3
    x15 = cos(q[5])
    x16 = sin(q[4])
    x17 = x10*x6
    x18 = x17*x4
    x19 = x5*x9
    x20 = x19*x4
    x21 = x18 - x20
    x22 = cos(q[4])
    x23 = x1*x3 + x13*x2
    x24 = x16*x21 + x22*x23
    x25 = x21*x22
    x26 = x16*x23
    x27 = 0.4572*x10
    x28 = x1*x7
    x29 = x1*x11
    x30 = -x28 - x29
    x31 = x2*x4 - x3*x30
    x32 = x1*x19
    x33 = x1*x17
    x34 = x32 - x33
    x35 = x2*x30 + x3*x4
    x36 = x16*x34 + x22*x35
    x37 = x22*x34
    x38 = x16*x35
    x39 = x17 - x19
    x40 = x3*x39
    x41 = -x11 - x7
    x42 = x2*x39
    x43 = x16*x41 + x22*x42
    x44 = x22*x41
    x45 = x16*x42
#
    pose[0] = x0*x14 + x15*x24
    pose[1] = -x0*x24 + x14*x15
    pose[2] = -x25 + x26
    pose[3] = 0.06985*x12 - 0.4064*x18 + 0.4064*x20 - 0.254*x25 + 0.254*x26 - x27*x4 - 0.10795*x4 + 0.06985*x8
    pose[4] = x0*x31 + x15*x36
    pose[5] = -x0*x36 + x15*x31
    pose[6] = -x37 + x38
    pose[7] = x1*x27 + 0.10795*x1 - 0.06985*x28 - 0.06985*x29 - 0.4064*x32 + 0.4064*x33 - 0.254*x37 + 0.254*x38
    pose[8] = -x0*x40 + x15*x43
    pose[9] = -x0*x43 - x15*x40
    pose[10] = -x44 + x45
    pose[11] = 0.4064*x11 + 0.06985*x17 - 0.06985*x19 - 0.254*x44 + 0.254*x45 + 0.4572*x5 + 0.4064*x7 + 0.0889
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
#
    return pose



FK = {0:joint_fk00, 1:joint_fk01, 2:joint_fk02, 3:joint_fk03, 4:joint_fk04, 5:joint_fk05, }



def jacobian00(q):
#
    jacobian = [0]*36
#

    jacobian[0] = -0.10795*cos(q[0])
    jacobian[1] = 0
    jacobian[2] = 0
    jacobian[3] = 0
    jacobian[4] = 0
    jacobian[5] = 0
    jacobian[6] = -0.10795*sin(q[0])
    jacobian[7] = 0
    jacobian[8] = 0
    jacobian[9] = 0
    jacobian[10] = 0
    jacobian[11] = 0
    jacobian[12] = 0
    jacobian[13] = 0
    jacobian[14] = 0
    jacobian[15] = 0
    jacobian[16] = 0
    jacobian[17] = 0
    jacobian[18] = 0
    jacobian[19] = 0
    jacobian[20] = 0
    jacobian[21] = 0
    jacobian[22] = 0
    jacobian[23] = 0
    jacobian[24] = 0
    jacobian[25] = 0
    jacobian[26] = 0
    jacobian[27] = 0
    jacobian[28] = 0
    jacobian[29] = 0
    jacobian[30] = 1
    jacobian[31] = 0
    jacobian[32] = 0
    jacobian[33] = 0
    jacobian[34] = 0
    jacobian[35] = 0
#
    jacobian = np.array(jacobian).reshape(6,6)
    return jacobian



def jacobian01(q):
#
    jacobian = [0]*36
#
    x0 = cos(q[0])
    x1 = cos(q[1])
    x2 = 0.4572*x0
    x3 = sin(q[1])
    x4 = sin(q[0])
    x5 = 0.4572*x4
    x6 = 0.4572*x1
#
    jacobian[0] = -0.10795*x0 - x1*x2
    jacobian[1] = x3*x5
    jacobian[2] = 0
    jacobian[3] = 0
    jacobian[4] = 0
    jacobian[5] = 0
    jacobian[6] = -x1*x5 - 0.10795*x4
    jacobian[7] = -x2*x3
    jacobian[8] = 0
    jacobian[9] = 0
    jacobian[10] = 0
    jacobian[11] = 0
    jacobian[12] = 0
    jacobian[13] = x0**2*x6 + x4**2*x6
    jacobian[14] = 0
    jacobian[15] = 0
    jacobian[16] = 0
    jacobian[17] = 0
    jacobian[18] = 0
    jacobian[19] = x0
    jacobian[20] = 0
    jacobian[21] = 0
    jacobian[22] = 0
    jacobian[23] = 0
    jacobian[24] = 0
    jacobian[25] = x4
    jacobian[26] = 0
    jacobian[27] = 0
    jacobian[28] = 0
    jacobian[29] = 0
    jacobian[30] = 1
    jacobian[31] = 0
    jacobian[32] = 0
    jacobian[33] = 0
    jacobian[34] = 0
    jacobian[35] = 0
#
    jacobian = np.array(jacobian).reshape(6,6)
    return jacobian



def jacobian02(q):
#
    jacobian = [0]*36
#
    x0 = cos(q[0])
    x1 = cos(q[1])
    x2 = 0.4572*x1
    x3 = x0*x2
    x4 = sin(q[1])
    x5 = cos(q[2])
    x6 = 0.06985*x4*x5
    x7 = x0*x6
    x8 = sin(q[2])
    x9 = 0.06985*x1*x8
    x10 = x0*x9
    x11 = sin(q[0])
    x12 = 0.06985*x1*x5 - 0.06985*x4*x8
    x13 = x12 + 0.4572*x4
    x14 = x11*x6 + x11*x9
    x15 = -x11*x2 + x14
    x16 = -x10 - x7
#
    jacobian[0] = -0.10795*x0 + x10 - x3 + x7
    jacobian[1] = x11*x13
    jacobian[2] = x11*x12
    jacobian[3] = 0
    jacobian[4] = 0
    jacobian[5] = 0
    jacobian[6] = -0.10795*x11 + x15
    jacobian[7] = -x0*x13
    jacobian[8] = -x0*x12
    jacobian[9] = 0
    jacobian[10] = 0
    jacobian[11] = 0
    jacobian[12] = 0
    jacobian[13] = x0*(x16 + x3) - x11*x15
    jacobian[14] = x0*x16 - x11*x14
    jacobian[15] = 0
    jacobian[16] = 0
    jacobian[17] = 0
    jacobian[18] = 0
    jacobian[19] = x0
    jacobian[20] = x0
    jacobian[21] = 0
    jacobian[22] = 0
    jacobian[23] = 0
    jacobian[24] = 0
    jacobian[25] = x11
    jacobian[26] = x11
    jacobian[27] = 0
    jacobian[28] = 0
    jacobian[29] = 0
    jacobian[30] = 1
    jacobian[31] = 0
    jacobian[32] = 0
    jacobian[33] = 0
    jacobian[34] = 0
    jacobian[35] = 0
#
    jacobian = np.array(jacobian).reshape(6,6)
    return jacobian



def jacobian03(q):
#
    jacobian = [0]*36
#
    x0 = cos(q[0])
    x1 = cos(q[1])
    x2 = 0.4572*x1
    x3 = x0*x2
    x4 = sin(q[1])
    x5 = sin(q[2])
    x6 = x4*x5
    x7 = x0*x6
    x8 = 0.4064*x7
    x9 = 0.06985*x0
    x10 = cos(q[2])
    x11 = x10*x4
    x12 = x11*x9
    x13 = x1*x5
    x14 = x13*x9
    x15 = x1*x10
    x16 = x0*x15
    x17 = 0.4064*x16
    x18 = sin(q[0])
    x19 = 0.4064*x11 + 0.4064*x13
    x20 = 0.06985*x15 + x19 - 0.06985*x6
    x21 = x20 + 0.4572*x4
    x22 = x16 - x7
    x23 = x11 + x13
    x24 = x17 - x8
    x25 = x18*x6
    x26 = x15*x18
    x27 = 0.4064*x25 - 0.4064*x26
    x28 = 0.06985*x18
    x29 = x11*x28 + x13*x28 + x27
    x30 = -x18*x2 + x29
    x31 = x25 - x26
    x32 = -x12 - x14 + x24
#
    jacobian[0] = -0.10795*x0 + x12 + x14 - x17 - x3 + x8
    jacobian[1] = x18*x21
    jacobian[2] = x18*x20
    jacobian[3] = x19*x22 - x23*x24
    jacobian[4] = 0
    jacobian[5] = 0
    jacobian[6] = -0.10795*x18 + x30
    jacobian[7] = -x0*x21
    jacobian[8] = -x0*x20
    jacobian[9] = -x19*x31 + x23*x27
    jacobian[10] = 0
    jacobian[11] = 0
    jacobian[12] = 0
    jacobian[13] = x0*(x3 + x32) - x18*x30
    jacobian[14] = x0*x32 - x18*x29
    jacobian[15] = -x22*x27 + x24*x31
    jacobian[16] = 0
    jacobian[17] = 0
    jacobian[18] = 0
    jacobian[19] = x0
    jacobian[20] = x0
    jacobian[21] = x31
    jacobian[22] = 0
    jacobian[23] = 0
    jacobian[24] = 0
    jacobian[25] = x18
    jacobian[26] = x18
    jacobian[27] = x22
    jacobian[28] = 0
    jacobian[29] = 0
    jacobian[30] = 1
    jacobian[31] = 0
    jacobian[32] = 0
    jacobian[33] = x23
    jacobian[34] = 0
    jacobian[35] = 0
#
    jacobian = np.array(jacobian).reshape(6,6)
    return jacobian



def jacobian04(q):
#
    jacobian = [0]*36
#
    x0 = cos(q[0])
    x1 = cos(q[1])
    x2 = 0.4572*x1
    x3 = x0*x2
    x4 = sin(q[1])
    x5 = sin(q[2])
    x6 = x4*x5
    x7 = x0*x6
    x8 = 0.4064*x7
    x9 = cos(q[2])
    x10 = x4*x9
    x11 = x0*x10
    x12 = 0.06985*x11
    x13 = x1*x5
    x14 = x0*x13
    x15 = 0.06985*x14
    x16 = x1*x9
    x17 = x0*x16
    x18 = 0.4064*x17
    x19 = sin(q[0])
    x20 = 0.4064*x10 + 0.4064*x13
    x21 = 0.06985*x16 + x20 - 0.06985*x6
    x22 = x21 + 0.4572*x4
    x23 = x17 - x7
    x24 = x10 + x13
    x25 = x18 - x8
    x26 = x19*x6
    x27 = x16*x19
    x28 = 0.4064*x26 - 0.4064*x27
    x29 = x10*x19
    x30 = x13*x19
    x31 = x28 + 0.06985*x29 + 0.06985*x30
    x32 = -x19*x2 + x31
    x33 = x26 - x27
    x34 = -x12 - x15 + x25
    x35 = cos(q[3])
    x36 = sin(q[3])
#
    jacobian[0] = -0.10795*x0 + x12 + x15 - x18 - x3 + x8
    jacobian[1] = x19*x22
    jacobian[2] = x19*x21
    jacobian[3] = x20*x23 - x24*x25
    jacobian[4] = 0
    jacobian[5] = 0
    jacobian[6] = -0.10795*x19 + x32
    jacobian[7] = -x0*x22
    jacobian[8] = -x0*x21
    jacobian[9] = -x20*x33 + x24*x28
    jacobian[10] = 0
    jacobian[11] = 0
    jacobian[12] = 0
    jacobian[13] = x0*(x3 + x34) - x19*x32
    jacobian[14] = x0*x34 - x19*x31
    jacobian[15] = -x23*x28 + x25*x33
    jacobian[16] = 0
    jacobian[17] = 0
    jacobian[18] = 0
    jacobian[19] = x0
    jacobian[20] = x0
    jacobian[21] = x33
    jacobian[22] = x0*x35 - x36*(x29 + x30)
    jacobian[23] = 0
    jacobian[24] = 0
    jacobian[25] = x19
    jacobian[26] = x19
    jacobian[27] = x23
    jacobian[28] = x19*x35 - x36*(-x11 - x14)
    jacobian[29] = 0
    jacobian[30] = 1
    jacobian[31] = 0
    jacobian[32] = 0
    jacobian[33] = x24
    jacobian[34] = -x36*(x16 - x6)
    jacobian[35] = 0
#
    jacobian = np.array(jacobian).reshape(6,6)
    return jacobian



def jacobian05(q):
#
    jacobian = [0]*36
#
    x0 = cos(q[0])
    x1 = cos(q[1])
    x2 = 0.4572*x1
    x3 = x0*x2
    x4 = sin(q[1])
    x5 = sin(q[2])
    x6 = x4*x5
    x7 = x0*x6
    x8 = 0.4064*x7
    x9 = cos(q[2])
    x10 = x4*x9
    x11 = x0*x10
    x12 = 0.06985*x11
    x13 = x1*x5
    x14 = x0*x13
    x15 = 0.06985*x14
    x16 = x1*x9
    x17 = x0*x16
    x18 = 0.4064*x17
    x19 = cos(q[4])
    x20 = x19*(-x17 + x7)
    x21 = 0.254*x20
    x22 = sin(q[4])
    x23 = sin(q[0])
    x24 = sin(q[3])
    x25 = cos(q[3])
    x26 = -x11 - x14
    x27 = x22*(x23*x24 + x25*x26)
    x28 = 0.254*x27
    x29 = x19*(-x10 - x13)
    x30 = x16 - x6
    x31 = x22*x25*x30
    x32 = -0.254*x29 + 0.254*x31
    x33 = 0.4064*x10 + 0.4064*x13 + x32
    x34 = 0.06985*x16 + x33 - 0.06985*x6
    x35 = x34 + 0.4572*x4
    x36 = x17 - x7
    x37 = x10 + x13
    x38 = -x21 + x28
    x39 = x18 + x38 - x8
    x40 = x23*x25 - x24*x26
    x41 = x24*x30
    x42 = -x20 + x27
    x43 = -x29 + x31
    x44 = x10*x23
    x45 = x13*x23
    x46 = x23*x6
    x47 = x16*x23
    x48 = x19*(-x46 + x47)
    x49 = x44 + x45
    x50 = x22*(x0*x24 + x25*x49)
    x51 = -0.254*x48 + 0.254*x50
    x52 = 0.4064*x46 - 0.4064*x47 + x51
    x53 = 0.06985*x44 + 0.06985*x45 + x52
    x54 = -x2*x23 + x53
    x55 = x46 - x47
    x56 = x0*x25 - x24*x49
    x57 = -x48 + x50
    x58 = -x12 - x15 + x39
#
    jacobian[0] = -0.10795*x0 + x12 + x15 - x18 + x21 - x28 - x3 + x8
    jacobian[1] = x23*x35
    jacobian[2] = x23*x34
    jacobian[3] = x33*x36 - x37*x39
    jacobian[4] = x32*x40 + x38*x41
    jacobian[5] = x32*x42 - x38*x43
    jacobian[6] = -0.10795*x23 + x54
    jacobian[7] = -x0*x35
    jacobian[8] = -x0*x34
    jacobian[9] = -x33*x55 + x37*x52
    jacobian[10] = -x32*x56 - x41*x51
    jacobian[11] = -x32*x57 + x43*x51
    jacobian[12] = 0
    jacobian[13] = x0*(x3 + x58) - x23*x54
    jacobian[14] = x0*x58 - x23*x53
    jacobian[15] = -x36*x52 + x39*x55
    jacobian[16] = x38*x56 - x40*x51
    jacobian[17] = x38*x57 - x42*x51
    jacobian[18] = 0
    jacobian[19] = x0
    jacobian[20] = x0
    jacobian[21] = x55
    jacobian[22] = x56
    jacobian[23] = x57
    jacobian[24] = 0
    jacobian[25] = x23
    jacobian[26] = x23
    jacobian[27] = x36
    jacobian[28] = x40
    jacobian[29] = x42
    jacobian[30] = 1
    jacobian[31] = 0
    jacobian[32] = 0
    jacobian[33] = x37
    jacobian[34] = -x41
    jacobian[35] = x43
#
    jacobian = np.array(jacobian).reshape(6,6)
    return jacobian



J = {0:jacobian00, 1:jacobian01, 2:jacobian02, 3:jacobian03, 4:jacobian04, 5:jacobian05, }
