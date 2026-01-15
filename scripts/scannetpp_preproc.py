import os
import math
import json
import argparse
from tqdm import tqdm


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='data/scannetpp_v2/data')
    parser.add_argument('--split_dir', default='data/scannetpp_v2/splits')

    parser.add_argument('--splits', default=[], nargs='*')

    parser.add_argument('--ids', default=[], nargs='*')
    # parser.add_argument('--ids', default=['08bbbdcc3d'], nargs='*')
    # parser.add_argument('--ids', default=['7b6477cb95', 'c50d2d1d42', 'cc5237fd77', 'acd95847c5', 'fb5a96b1a2', 'a24f64f7fb', '1ada7a0617', '5eb31827b7', '3e8bba0176', '3f15a9266d', '21d970d8de', '5748ce6f01', 'c4c04e6d6c', '7831862f02', 'bde1e479ad', '38d58a7a31', '5ee7c22ba0', 'f9f95681fd', '3864514494', '40aec5fffa', '13c3e046d7', 'e398684d27', 'a8bf42d646', '45b0dac5e3', '31a2c91c43', 'e7af285f7d', '286b55a2bf', '7bc286c1b6', 'f3685d06a9', 'b0a08200c9', '825d228aec', 'a980334473', 'f2dc06b1d2', '5942004064', '25f3b7a318', 'bcd2436daf', 'f3d64c30f8', '0d2ee665be', '3db0a1c8f3', 'ac48a9b736', 'c5439f4607', '578511c8a9', 'd755b3d9d8', '99fa5c25e1', '09c1414f1b', '5f99900f09', '9071e139d9', '6115eddb86', '27dd4da69e', 'c49a8c6cff'], nargs='*')

    parser.add_argument('--is_test_hidden', default=False, action='store_true')
    # parser.add_argument('--ids', default=['ca0e09014e', 'beb802368c', 'ebff4de90b', 'd228e2d9dd', '9e019d8be1', '11b696efba', '471cc4ba84', 'f20e7b5640', 'dfe9cbd72a', 'ccdc33dc2a', '124974734e', 'c0cbb1fea1', '047fb766c4', '7b37cccb03', '8283161f1b', 'c3e279be54', '5a14f9da39', 'cd7973d92b', '5298ec174f', 'e0e83b4ca3', '64ea6b73c2', 'f00bd5fa8a', '02a980c994', 'be91f7884d', '1c876c250f', '15155a88fb', '633f9a9f06', 'd6419f6478', 'f0b0a42ba3', 'a46b21d949', '74ff105c0d', '77596f5d2a', 'ecb5d01065', 'c9bf4c8b62', 'b074ca565a', '49c758655e', 'd4d2019f5d', '319787e6ec', '84b48f2614', 'bee11d6a41', '9a9e32c768', '9b365a9b68', '54e7ffaea3', '7d72f01865', '252652d5ba', '651dc6b4f1', '03f7a0e617', 'fe94fc30cf', 'd1b9dff904', '4bc04e0cde'], nargs='*')
    args = parser.parse_args()

    if len(args.splits) > 0:
        args.ids = []
        for split in args.splits:
            with open(os.path.join(args.split_dir, f"{split}.txt")) as f:
                args.ids.extend(f.read().strip().split())
        print(args.ids)

    for scene_id in tqdm(args.ids):
        in_scene_dir = os.path.join(args.indir, scene_id, 'dslr')
        out_scene_dir = os.path.join(in_scene_dir, 'svraster_inputs')

        os.system(f'mkdir -p {out_scene_dir}')

        with open(os.path.join(in_scene_dir, 'nerfstudio', 'transforms_undistorted.json')) as f:
            meta = json.load(f)

        cx_p = meta['cx'] / meta['w']
        cy_p = meta['cy'] / meta['h']
        camera_angle_x = focal2fov(meta['fl_x'], meta['w'])
        camera_angle_y = focal2fov(meta['fl_y'], meta['h'])

        new_metas_lst = []
        for key in ['frames', 'test_frames']:
            new_metas_lst.append(dict(
                camera_angle_x=0,
                colmap={
                    'path': '../colmap',
                    'transform': [
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, -1],
                    ],
                },
                frames=[]))
            for frame in meta[key]:
                new_metas_lst[-1]['frames'].append({
                    'camera_angle_x': camera_angle_x,
                    'camera_angle_y': camera_angle_y,
                    'cx_p': cx_p,
                    'cy_p': cy_p,
                    'file_path': f"../undistorted_images/{frame['file_path']}",
                    'depth_path': f"../undistorted_depths/{frame['file_path'].replace('.JPG', '.png')}",
                    'transform_matrix': frame['transform_matrix'],
                    'is_bad': frame['is_bad'],
                    'heldout': args.is_test_hidden and (key == 'test_frames'),
                    'w': meta['w'],
                    'h': meta['h'],
                })

        new_train_meta, new_test_meta = new_metas_lst

        with open(os.path.join(out_scene_dir, 'transforms_train.json'), 'w') as f:
            json.dump(new_train_meta, f, indent=2)
        with open(os.path.join(out_scene_dir, 'transforms_test.json'), 'w') as f:
            json.dump(new_test_meta, f, indent=2)
