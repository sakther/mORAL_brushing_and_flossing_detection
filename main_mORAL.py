from features.compute_featues_for_candidate_segment import *
from utils.filter_candidates import *
from input.import_stream_processor_inputs import *
from utils.annotations_gt_utils import *
from utils.file_utils import save_as_csv
from utils.generate_candidates import generate_brushing_candidates, generate_flossing_candidates
from model.brushing_flossing_classifier import *
from datetime import datetime

data_dir = '\\\\MD2K_LAB\\md2k_lab_share\\Data\\ROBAS\\from_phone_storage\\'
output_feature_dir = data_dir + 'features_with_labels\\'


# def plot_candidates(cands, AGMO):
#     t = [v[0] for v in AGMO]
#     ay = [v[2] for v in AGMO]
#     ax = [v[3] for v in AGMO]
#     plt.plot(t, ay, label='Ay')
#     plt.plot(t, ax, label='Ax')
#     plt.legend()
#     plt.savefig('plot/all.png', bbox_inches='tight')
#     plt.close()
#
#     for i, cand in enumerate(cands):
#         st = t[cand[0]]
#         et = t[cand[1]]
#         tt = t[cand[0]:cand[1]]
#         ayt = ay[cand[0]:cand[1]]
#         axt = ax[cand[0]:cand[1]]
#         plt.plot(tt, ayt, label='Ay')
#         plt.plot(tt, axt, label='Ax')
#         plt.title(str(et-st) + '  ' + datetime.utcfromtimestamp(st).strftime('%Y-%m-%d %H:%M:%S'))
#         plt.legend()
#         plt.savefig('plot/' + str(i) + '.png', bbox_inches='tight')
#         plt.close()
#


def process_brushing(pid, sid, AGMO_r, AGMO_l):
    featurenames = create_brushing_featurenames()

    cand_l = generate_brushing_candidates(AGMO_l)
    cand_l = filter_for_brushing(cand_l, AGMO_l)

    cand_r = generate_brushing_candidates(AGMO_r)
    cand_r = filter_for_brushing(cand_r, AGMO_r)

    # plot_candidates(cand_r, AGMO_r)

    y_brushing = []
    if len(cand_l) > 0:
        X_left = generate_all_window_and_compute_brushing_features(pid, sid, AGMO_l, cand_l)
        X_df_left = pd.DataFrame(np.array(X_left), columns=featurenames)
        y_left = classify_brushing(X_df_left)
        y_brushing.extend([[v[2], v[3], v[3] - v[2], 1] for i, v in enumerate(X_left) if y_left[i] == 1])
    if len(cand_r) > 0:
        X_right = generate_all_window_and_compute_brushing_features(pid, sid, AGMO_r, cand_r)
        X_df_right = pd.DataFrame(np.array(X_right), columns=featurenames)
        y_right = classify_brushing(X_df_right)
        y_brushing.extend([[v[2], v[3], v[3] - v[2], 2] for i, v in enumerate(X_right) if y_right[i] == 1])
    if len(y_brushing) > 0:
        y_brushing.sort(key=lambda x: x[0])
        y_df_brushing = pd.DataFrame(y_brushing, columns=['start_timestamp', 'end_timestamp', 'duration', 'label'])
        print('Brushing events', y_df_brushing)
    else:
        print('No brushing events detected')

    return y_brushing


def process_flossing(pid, sid, AGMO_r, AGMO_l):
    featurenames = create_flossing_featurenames()

    cand_l = generate_flossing_candidates(AGMO_l)
    cand_l = filter_for_flossing(cand_l, AGMO_l)

    cand_r = generate_flossing_candidates(AGMO_r)
    cand_r = filter_for_flossing(cand_r, AGMO_r)

    cands = combine_left_right(cand_l, cand_r)

    y_flossing = []
    if len(cands) > 0:
        X = generate_all_window_and_compute_flossing_features(pid, sid, AGMO_l, AGMO_r, cands)
        X_df = pd.DataFrame(np.array(X), columns=featurenames)

        y = classify_flossing(X_df)
        y_flossing = [[v[2], v[3], v[3] - v[2], 1] for i, v in enumerate(X) if y[i] == 1]
        y_df_flossing = pd.DataFrame(y_flossing, columns=['start_timestamp', 'end_timestamp', 'duration', 'label'])
        print('Flossing events', y_df_flossing)
    else:
        print('No flossing events detected')

    return y_flossing


def process_mORAL(accel_left, gyro_left, accel_right, gyro_right, pid, sid):
    '''
    This is the main function for computing brushing and flossing events from accelerometer and gyroscope
    :param accel_left: list of [ts, ax, ay, az]
    :param gyro_left: list of [ts, gx, gy, gz]
    :param accel_right: list of [ts, ax, ay, az]
    :param gyro_right: list of [ts, gx, gy, gz]
    :return:
    '''
    AGMO_l = merge_accel_gyro_and_compute_basic_features(accel_left, gyro_left)
    AGMO_r = merge_accel_gyro_and_compute_basic_features(accel_right, gyro_right)

    if len(AGMO_l) == 0 or len(AGMO_r) == 0:
        print('---------DATA MISSING----', pid, sid, '#left', len(AGMO_l), '#right', len(AGMO_r))
        return [], []

    detected_brushing = process_brushing(pid, sid, AGMO_r, AGMO_l)
    AGMO_l = aline_datastream(AGMO_l, [v[0] for v in AGMO_r])
    detected_flossing = process_flossing(pid, sid, AGMO_r, AGMO_l)

    return detected_brushing, detected_flossing


def do_proccess_for_all_events_per_pid(pids, data_dir):
    for pid in pids:
        print('start', pid)

        basedir = data_dir + pid + '/'
        # print('-----', basedir)
        sids = [d for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d)) and d.startswith('s')]
        sids.sort()
        for sid in sids:
            print('---START ----', pid, sid)
            cur_dir = data_dir + pid + '/' + sid + '/'

            accel_left, gyro_left = get_accel_gyro(cur_dir, LEFT_WRIST)
            accel_right, gyro_right = get_accel_gyro(cur_dir, RIGHT_WRIST)

            detected_brushing, detected_flossing = process_mORAL(accel_left, gyro_left, accel_right, gyro_right, pid,
                                                                 sid)


if __name__ == "__main__":
    # pids = [d for d in os.listdir(phone_data_dir) if os.path.isdir(os.path.join(phone_data_dir, d)) and d.startswith('0')]
    pids = ['001', '003', '004']
    pids.sort()
    print(pids)
    do_proccess_for_all_events_per_pid(pids, data_dir)
