data_path = './data/waymo_sf/'
db_data_path = './data/waymo_sf/'
point_cloud_range = [-75.2, -75.2, -2., 75.2, 75.2, 4.]

train_pipeline = [
    dict(
        type='LoadSFFromFile',
        use_dim=[0,1,2],
        with_fg_mask=False),
    dict(
        type='DataBaseSampler',
        info_path=db_data_path+'waymo_dbinfos_train.pkl',
        data_root=db_data_path,
        rate=1.0,
        prepare=dict(
            filter_by_difficulty=[-1],
            filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
        sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
        classes=['Car', 'Pedestrian', 'Cyclist'],
        filter_by_min_points=dict(Car=100, Pedestrian=10, Cyclist=20),
        point_cloud_range=point_cloud_range,
        height_axis=2,
        pillar_size=1.6,
        pillar_points_thresh=10,
        db_load_dim=5,
        db_use_dim=[0,1,2],
        sim_velocity={
            'Car':60/3.6/5,
            'Pedestrian':5/3.6/5,
            'Cyclist':30/3.6/5}),
    dict(
        type='RandomFlipSF',
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        shift_height=False),
    
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='RandomPointSample', num_points=8192),
    dict(
        type='Collection',
        meta_keys=[],
        list_keys=[],
        stack_keys=['sf_data'])
]
val_pipeline = [
    dict(
        type='LoadSFFromFile',
        use_dim=[0,1,2],
        with_fg_mask=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='RandomPointSample', num_points=8192),
    dict(
        type='Collection',
        meta_keys=[],
        list_keys=[],
        stack_keys=['sf_data'])
]

data = dict(
    train = dict(
        dataset = dict(
            type='WaymoSFDataset',
            data_path=data_path,
            scene_list=list(range(80)),
            pipline=train_pipeline
        ),
        dataloader = dict(
            pin_memory=True,
            shuffle=True,
            drop_last=False,
            timeout=0
        )
    ),
    val = dict(
        dataset = dict(
            type='WaymoSFDataset',
            data_path=data_path,
            scene_list=list(range(80,100)),
            pipline=val_pipeline
        ),
        dataloader = dict(
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            timeout=0
        )
    )
)