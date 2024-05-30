import argparse
import yaml
import engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./train.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=False,
        default="/media/arvc/HDD4TB1/Antonio/Minkowski/dataset_recortado_normales_vecinos/kitti_Voxel_neighbour/sequences/train",
    )
    parser.add_argument(
      '--dataset2', '-d2',
      type=str,
      required=False,
      default="/media/arvc/HDD4TB1/Antonio/Minkowski/dataset_recortado_normales_vecinos/Rellis_3D_Voxel_neighbour/train",
    )

    parser.add_argument(
      '--dataset3', '-d3',
      type=str,
      required=False,
      default="/media/arvc/HDD4TB1/Antonio/Minkowski/dataset_recortado_normales_vecinos/usl_voxel_neighbour/sequences",
    )
    

    parser.add_argument(
        '--arch_cfg', '-ac',
        type=str,
        required=False,
        default="config/TE.yaml"
    )

    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default='config/sem-kitti.yaml',
        )
    
    parser.add_argument(
        '--data_cfg2', '-dc2',
        type=str,
        required=False,
        default='config/sem-rellis3D.yaml',
        )

    parser.add_argument(
        '--data_cfg3', '-dc3',
        type=str,
        required=False,
        default='config/sem-usl.yaml',
        )
    
    FLAGS, unparsed = parser.parse_known_args()
   
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("dataset", FLAGS.dataset2)
    print("dataset", FLAGS.dataset3)
    print("arch_cfg", FLAGS.arch_cfg)
    print("Configuration file for SemanticKITTI", FLAGS.data_cfg)
    print("Configuration file for Rellis-3D", FLAGS.data_cfg2)
    print("Configuration file for Semantic-USL", FLAGS.data_cfg3)

    # open arch config file
    try:
        print("Opening arch config file %s" % FLAGS.arch_cfg)
        ARCH_kitti = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg)
        DATA_kitti = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # open data2 config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg2)
        DATA_rellis = yaml.safe_load(open(FLAGS.data_cfg2, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()
    # open data3 config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg3)
        DATA_usl = yaml.safe_load(open(FLAGS.data_cfg3, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    trainer = engine.Trainer(ARCH_kitti, DATA_kitti,DATA_rellis, DATA_usl,  FLAGS.dataset, FLAGS.dataset2, FLAGS.dataset3)
    trainer.train()
