class Config:
    # Threshold = 0.92
    exp = ''
    phase = 'train'
    data_dir = 'data'
    model_name = "seresnext26d_32x4d"

    label_smoothing_rate = 0.1
    fold_num = 5
    batch_size = 64
    num_workers = 4
    seed = 555
    deterministic = True if phase == 'test' else False
    tta = True
    # SOTA
    # NUMWORKERS 0 - 8, else 4

    #! LAST PHASE 1 TEST

    #! LAST PHASE 2 TEST

    #! LAST PHASE 3 TEST

    #! PHASE 1 RECAP
