import logging
import os
from typing import Tuple


def configure_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)


def _ensure_file_with_header(path: str, header: str) -> None:
    if not os.path.exists(path):
        with open(path, 'w') as outfile:
            outfile.write(header + '\n')


def init_metric_logs(results_dir: str) -> Tuple[str, str, str, str]:
    os.makedirs(results_dir, exist_ok=True)

    train_log_path = os.path.join(results_dir, 'moist_exchange_loss.dat')
    val_log_path = os.path.join(results_dir, 'moist_exchange_loss_test.dat')
    train_step_log_path = os.path.join(results_dir, 'moist_exchange_train_steps.dat')
    val_step_log_path = os.path.join(results_dir, 'moist_exchange_val_steps.dat')

    _ensure_file_with_header(
        train_log_path,
        'epoch\ttrain_loss_v\ttrain_loss_l\ttrain_loss_i\ttrain_loss_e\ttrain_loss_total\ttrain_loss_mA\ttrain_nA_gt0\ttrain_time_s\tlr\tgrad_norm'
    )
    _ensure_file_with_header(
        val_log_path,
        'epoch\tval_loss_v\tval_loss_l\tval_loss_i\tval_loss_e\tval_loss_total\tval_loss_mA\tval_nA_gt0\tval_time_s\tepoch_time_s'
    )
    _ensure_file_with_header(
        train_step_log_path,
        'epoch\tstep\ttrain_loss_v\ttrain_loss_l\ttrain_loss_i\ttrain_loss_e\ttrain_loss_mA\ttrain_loss_total\tlr\tstep_time_s'
    )
    _ensure_file_with_header(
        val_step_log_path,
        'epoch\tstep\tval_loss_v\tval_loss_l\tval_loss_i\tval_loss_e\tval_loss_mA\tval_loss_total\tstep_time_s'
    )

    return train_log_path, val_log_path, train_step_log_path, val_step_log_path


def append_line(path: str, line: str) -> None:
    with open(path, 'a') as outfile:
        outfile.write(line + '\n')
