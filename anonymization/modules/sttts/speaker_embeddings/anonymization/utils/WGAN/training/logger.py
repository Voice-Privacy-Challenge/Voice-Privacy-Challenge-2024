import os
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


def setup_logger():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    log_dateformat =  '%d-%m-%y_%H:%M'

    name_of_this_file = os.path.splitext(os.path.basename(__file__))[0]
    fileHandler = logging.FileHandler('{}/{}.log'.format(log_dir, 'log_timestamp'))
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                               log_dateformat))

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(logging.Formatter('%(name)s - %(message)s', log_dateformat))

    logging.basicConfig(datefmt=log_dateformat, level=logging.DEBUG, handlers=[fileHandler, consoleHandler])

    logger = logging.getLogger(__name__)
    logger.info('Execute \'%s\''%(__file__))
    return logger

def setup_tensorboard(logger):
    # logging for Tensorboard
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # create timestamp for gan
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%m-%Y-%H-%M-%S")

    tensorboard_log_dir = Path(log_dir, f'log_{timestampStr}')
    tensorboard_log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    return writer, timestampStr