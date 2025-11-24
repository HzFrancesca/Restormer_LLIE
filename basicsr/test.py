import logging
import time
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt["path"]["log"], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name="basicsr", log_level=logging.INFO, log_file=log_file
    )
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt["num_gpu"],
            dist=opt["dist"],
            sampler=None,
            seed=opt["manual_seed"],
        )
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt)

    # Initialize timing variables
    total_inference_time = 0.0
    total_images = 0

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt["name"]
        logger.info(f"Testing {test_set_name}...")
        rgb2bgr = opt["val"].get("rgb2bgr", True)
        # wheather use uint8 image to compute metrics
        use_image = opt["val"].get("use_image", True)

        # Record start time for this test set
        start_time = time.perf_counter()

        model.validation(
            test_loader,
            current_iter=opt["name"],
            tb_logger=None,
            save_img=opt["val"]["save_img"],
            rgb2bgr=rgb2bgr,
            use_image=use_image,
        )

        # Record end time and calculate elapsed time
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        total_inference_time += elapsed_time

        # Count images in this test set
        num_images = len(test_loader.dataset)
        total_images += num_images

        # Display timing for this test set
        avg_time_per_image = elapsed_time / num_images if num_images > 0 else 0
        logger.info(
            f"{test_set_name} - Total time: {elapsed_time:.4f}s, "
            f"Images: {num_images}, "
            f"Avg time per image: {avg_time_per_image:.4f}s"
        )

    # Display overall timing statistics
    if total_images > 0:
        overall_avg_time = total_inference_time / total_images
        logger.info("=" * 80)
        logger.info("Overall Statistics:")
        logger.info(f"Total inference time: {total_inference_time:.4f}s")
        logger.info(f"Total images processed: {total_images}")
        logger.info(
            f"Average time per image: {overall_avg_time:.4f}s ({overall_avg_time * 1000:.2f}ms)"
        )
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
