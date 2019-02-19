# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys


# LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s "  # 配置输出日志格式
# DATE_FORMAT = '%Y-%m-%d  %H:%M:%S %a '  # 配置输出时间的格式，注意月份和天数不要搞乱了
# logging.basicConfig(level=logging.DEBUG,
#                     format=LOG_FORMAT,
#                     datefmt=DATE_FORMAT,
#                     filename=args.checkpoint_dir+"/train_log.txt"  # 有了filename参数就不会直接输出显示到控制台，而是直接写入文件
#                     )
# logging.info(str(datetime.now()))

def setup_logger(name, save_dir,prefix):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir,prefix+"log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
