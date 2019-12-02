from features.compute_featues_for_candidate_segment import *
from utils.filter_candidates import *
from input.import_stream_processor_inputs import *
from utils.annotations_gt_utils import *
from utils.file_utils import save_as_csv
from utils.generate_candidates import generate_brushing_candidates, generate_flossing_candidates
from model.brushing_flossing_classifier import *

data_dir = '\\\\MD2K_LAB\\md2k_lab_share\\Data\\ROBAS\\'
output_feature_dir = data_dir + 'features_with_labels\\'


def process_brushing(pid, sid, AGMO_r, AGMO_l):
    cand_l = generate_brushing_candidates(AGMO_l)
    cand_l = filter_for_brushing(cand_l, AGMO_l)

    cand_r = generate_brushing_candidates(AGMO_r)
    cand_r = filter_for_brushing(cand_r, AGMO_r)

    X_left = generate_all_window_and_compute_brushing_features(pid, sid, AGMO_l, cand_l)
    X_right = generate_all_window_and_compute_brushing_features(pid, sid, AGMO_r, cand_r)

    y_left = classify_brushing(X_left)
    y_right = classify_brushing(X_right)

    y_brushing = [[v[2], v[3], 1] for i, v in enumerate(X_left) if y_left[i] == 1]
    y_brushing.extend([[v[2], v[3], 2] for i, v in enumerate(X_right) if y_right[i] == 1])
    y_brushing.sort(key=lambda x: x[0])

    featurenames = create_brushing_featurenames()
    output_filename = pid + '_' + sid + '_brushing_left_feature_and_label.csv'
    save_as_csv(X_right, y_right, featurenames, output_feature_dir, output_filename=output_filename)
    output_filename = pid + '_' + sid + '_brushing_right_feature_and_label.csv'
    save_as_csv(X_left, y_left, featurenames, output_feature_dir, output_filename=output_filename)
    return y_brushing


def process_flossing(pid, sid, AGMO_r, AGMO_l):
    cand_l = generate_flossing_candidates(AGMO_l)
    cand_l = filter_for_flossing(cand_l, AGMO_l)

    cand_r = generate_flossing_candidates(AGMO_r)
    cand_r = filter_for_flossing(cand_r, AGMO_r)

    cands = combine_left_right(cand_l, cand_r)

    X = generate_all_window_and_compute_flossing_features(pid, sid, AGMO_l, AGMO_r, cands)

    y = classify_flossing(X)

    featurenames = create_flossing_featurenames()
    output_filename = pid + '_' + sid + '_flossing_feature_and_label.csv'
    save_as_csv(X, y, featurenames, output_feature_dir, output_filename=output_filename)
    return y


def do_proccess_for_all_events_per_pid(pids, data_dir):
    for pid in pids:
        print('start', pid)

        basedir = data_dir + pid + '/'
        # print('-----', basedir)
        sids = [d for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d)) and d.startswith('s')]
        sids.sort()

        for sid in sids:
            cur_dir = data_dir + pid + '/' + sid + '/'

            AGMO_r = get_accel_gyro_mag_orientation(cur_dir, RIGHT_WRIST)
            AGMO_l = get_accel_gyro_mag_orientation(cur_dir, LEFT_WRIST)

            if len(AGMO_l) == 0 or len(AGMO_r) == 0:
                print('---------DATA MISSING----', pid, sid, '#left', len(AGMO_l), '#right', len(AGMO_r))
                continue

            AGMO_l = aline_datastream(AGMO_l, [v[0] for v in AGMO_r])
            detected_brushing = process_brushing(pid, sid, AGMO_r, AGMO_l)
            detected_flossing = process_flossing(pid, sid, AGMO_r, AGMO_l)


if __name__ == "__main__":
    # pids = [d for d in os.listdir(phone_data_dir) if os.path.isdir(os.path.join(phone_data_dir, d)) and d.startswith('0')]
    pids = ['001', '003', '004']
    pids.sort()
    print(pids)
    do_proccess_for_all_events_per_pid(pids, data_dir)
