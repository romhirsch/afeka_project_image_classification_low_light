from enum import Enum


class PathDatasets(Enum):
    # EXDARK = r"E:\dataset\ExDark"k
    # COCO_TEST = r"E:\dataset\MyCoco\test"
    # COCO_TRAIN = r"E:\dataset\MyCoco\train"
    # AUG_TEST = r"E:\dataset\augmentation_images"
    Imagenet_aug1 = r"E:\Imagenet_aug_1"
    Imagenet_aug2 = r"E:\Imagenet_aug_2"
    Imagenet_aug3 = r"E:\Imagenet_aug_3"
    Imagenet_aug4 = r"E:\Imagenet_aug_4"
    coco_aug1 = r"E:\coco_aug_1"
    coco_aug2 = r"E:\coco_aug_2"
    coco_aug3 = r"E:\coco_aug_3"
    coco_aug4 = r"E:\coco_aug_4"
    lol_low = r"E:\dataset\lol\low"
    lol_high = r"E:\dataset\lol\high"
    real = r"E:\real_images"
    Imagenet_aug_very_low = r"E:\Imagenet_aug_very_low"
    Imagenet_aug = r"E:\Imagenet_aug"
    Imagenet_test = r"E:\imagenet_test"
    Imagenet_train = r"E:\imagenet_train"
    Imagenet_orig = r"E:\imagenet_orig"
    COCO_TEST_SMALL = r"E:\dataset\MyCoco\test_small"
    COCO_TRAIN_SMALL = r"E:\dataset\MyCoco\train_small"
    EXDARK_TRAIN = r"E:\dataset\ExDark_train"
    EXDARK_TEST = r"E:\dataset\ExDark_test"
    EXDARK_VAL = r"E:\dataset\ExDark_val"
    COCO_AUG_TEST = r"E:\dataset\augmentation_images_small"
    EXDARK = r"E:\dataset\ExDark"
    COCO_TEST = r"E:\coco_test"
    COCO_TRAIN = r"E:\coco_train"
    AUG_TEST = r"E:\dataset\augmentation_images"
    COCO2017_TRAIN = r"E:\COCO2017_Train"
    COCO2017_TEST = r"E:\COCO2017_Test"
    COCO2017_4_TRAIN = r"E:\COCO2017-14_Train"
    COCO2017_4_TEST = r"E:\COCO2017-14_Test"
    COCO_ALL = r"E:\COCO2017-14"
class Models(Enum):
    #dark: normal
    resnet50 = 0
    resnet50_0_10 = 1
    resnet50_1_9 = 2
    resnet50_2_8 = 3
    resnet50_3_7 = 4
    resnet50_4_6 = 5
    resnet50_5_5 = 6
    resnet50_6_4 = 7
    resnet50_7_3 = 8
    resnet50_8_2 = 9
    resnet50_9_1 = 10
    resnet50_10_0 = 11
    EfficientNetB0 = 12
    EfficientNetB0_0_10 = 13
    EfficientNetB0_1_9 = 14
    EfficientNetB0_2_8 = 15
    EfficientNetB0_3_7 = 16
    EfficientNetB0_4_6 = 17
    EfficientNetB0_5_5 = 18
    EfficientNetB0_6_4 = 19
    EfficientNetB0_7_3 = 20
    EfficientNetB0_8_2 = 21
    EfficientNetB0_9_1 = 22
    EfficientNetB0_10_0 = 23


    # [Flatten(), Dense(512, activation='relu')]
    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #      directories[0],
    #      validation_split=0.25,
    #      subset="training",
    #      seed=123,
    #      image_size=(image_size, image_size),
    #      batch_size=1)
    # def get_key(my_dict, val):
    #     for key, value in my_dict.items():
    #         if val == value:
    #             return key
    #
    #     return "key doesn't exist"
    #
    # images, labels = tuple(zip(*train_ds))
    # fig, m_axs = plt.subplots(3, 5, figsize=(16, 16))
    # for (c_x, c_y, c_ax) in zip(images, labels, m_axs.flatten()):
    #     c_ax.imshow(np.array(c_x[0,:, :, :]).astype(np.uint64), cmap='bone', vmin=-1.5, vmax=1.5)
    #     c_ax.set_title(f', {get_key(dp.train_generator.class_indices,c_y)}')
    #     c_ax.axis('off')

    # ds_test = DPhandler.load_dataset(PathDatasets.Imagenet_test.value, preprocess_input, classes)
    # ds_aug1 = DPhandler.load_dataset(PathDatasets.Imagenet_aug1.value, preprocess_input, classes)
    # ds_aug2 = DPhandler.load_dataset(PathDatasets.Imagenet_aug2.value, preprocess_input, classes)
    # ds_aug3 = DPhandler.load_dataset(PathDatasets.Imagenet_aug3.value, preprocess_input, classes)
    # ds_aug4 = DPhandler.load_dataset(PathDatasets.Imagenet_aug4.value, preprocess_input, classes)
    # ds_tests = [ds_test, ds_aug1, ds_aug2, ds_aug3, ds_aug4]
    # ds_names = ['Test', 'Aug1', 'Aug2', 'Aug3', 'Aug4']

    # import shutil
    # target_path = r"E:\Exdark_split\Single"
    # for image_path in Single_df['Images']:
    #     shutil.copyfile(image_path, os.path.join(target_path, os.path.basename(image_path)))
