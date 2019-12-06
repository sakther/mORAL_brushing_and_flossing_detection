from utils.merge_two_datastream import *
from utils.hand_orientation import *
from utils.orientation_mapping import *

def get_magnitude(ax, ay, az):
    return math.sqrt(ax * ax + ay * ay + az * az)


def merge_accel_gyro_and_compute_basic_features(accel, gyro):
    '''

    :param accel:
    :param gyro:
    :return:list of (ts, ax, ay, az, gx, gy, gz, Amag, Gmag, roll, pitch, yaw)
    '''
    if len(accel) == 0 or len(gyro) == 0:
        return []
    AG = merge_accel_gyro(accel, gyro)
    Amag = [get_magnitude(v[1], v[2], v[3]) for v in AG]
    Gmag = [get_magnitude(v[4], v[5], v[6]) for v in AG]
    AGM = list(np.column_stack((AG, Amag, Gmag)))

    AGMO = apply_complementary_filter(AGM)
    Omag = [get_magnitude(v[9], v[10], v[11]) for v in AGMO]
    AGMO = list(np.column_stack((AGMO, Omag)))
    return AGMO
