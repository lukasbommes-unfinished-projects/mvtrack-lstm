scales = [1.0]#[1.0, 0.75, 0.5, 0.25]  # encode video at 100 %, 75 % and 50 % of original size

frame_rates = {
    "MOT17": {
            "train": [30,30,14,30,30,30,25],
            "test": [30,30,14,30,30,30,25]
        },
    "MOT16": {
        "train": [30,30,14,30,30,30,25],
        "test": [30,30,14,30,30,30,25]
        },
    "MOT15": {
        "train": [25,25,7,14,14,14,30,30,10,10,30],
        "test": [25,7,14,14,14,5/2,30,30,10,10,30]
        }
}

dir_names = {
"MOT17": {
        "train": [
            'MOT17-02-FRCNN',
            'MOT17-04-FRCNN',
            'MOT17-05-FRCNN',
            'MOT17-09-FRCNN',
            'MOT17-10-FRCNN',
            'MOT17-11-FRCNN',
            'MOT17-13-FRCNN',
        ],
        "test": [
            'MOT17-01-FRCNN',
            'MOT17-03-FRCNN',
            'MOT17-06-FRCNN',
            'MOT17-07-FRCNN',
            'MOT17-08-FRCNN',
            'MOT17-12-FRCNN',
            'MOT17-14-FRCNN',
        ]
    },
"MOT16": {
        "train": [
            'MOT16-02',
            'MOT16-04',
            'MOT16-05',
            'MOT16-09',
            'MOT16-10',
            'MOT16-11',
            'MOT16-13',
        ],
        "test": [
            'MOT16-01',
            'MOT16-03',
            'MOT16-06',
            'MOT16-07',
            'MOT16-08',
            'MOT16-12',
            'MOT16-14',
        ]
    },
"MOT15": {
        "train": [
            'TUD-Stadtmitte',
            'TUD-Campus',
            'PETS09-S2L1',
            'ETH-Bahnhof',
            'ETH-Sunnyday',
            'ETH-Pedcross2',
            'ADL-Rundle-6',
            'ADL-Rundle-8',
            'KITTI-13',
            'KITTI-17',
            'Venice-2'
        ],
        "test": [
            'TUD-Crossing',
            'PETS09-S2L2',
            'ETH-Jelmoli',
            'ETH-Linthescher',
            'ETH-Crossing',
            'AVG-TownCentre',
            'ADL-Rundle-1',
            'ADL-Rundle-3',
            'KITTI-16',
            'KITTI-19',
            'Venice-1'
        ]
    }
}
