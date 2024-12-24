import os


def init_args(args):
    if args.debug:
        args.ae_epochs = 10
    
    args.gt_path = args.test_path
    
    args.pose_path = {
            'train' : os.path.join(args.data_dir, 'pose', 'training/tracked_person/'),
            'test' : os.path.join(args.data_dir, 'pose', 'testing/tracked_person/')}

    args.ckpt_dir = create_experiment_dirs(args)

    return args


def create_experiment_dirs(args):
    dataset = args.dataset_choice
    checkpoints_dir = os.path.join(args.exp_dir, dataset, args.dir_name)
    dirs = [checkpoints_dir]
    if args.create_experiment_dir: 
        try:
            for dir_ in dirs:
                os.makedirs(dir_, exist_ok=True)
            print("Experiment directories created in {}".format(checkpoints_dir))
        except Exception as err:
            print("Experiment directories creation Failed, error {}".format(err))
            exit(-1)
    return checkpoints_dir
    