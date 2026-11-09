import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from training_logging import configure_logger, init_metric_logs, append_line
from data_cache import prebuild_default_caches, load_array_cached, load_filtered_array


'''
simple neural network from R^5 -> R^3
inputs:
    q_v : mass fraction of water vapor phase (scaled by the density)
    q_l : mass fraction of water liquid phase (scaled by the density)
    q_i : mass fraction of water ice phase (scaled by the density)
    eta : entropy (thermal equilibrium, same for all phases)
    rho : density (thermal equilibrium, same for all phases)
outputs:
    vapor to liquid exchange, normalised by the sum of the vapor and liquid mass (B)
    vapor to ice exchange, normalised by the sum of the vapor and ice mass       (C)
    liquid to ice exchange, normalised by the sum of the liquid and ice mass     (D)
'''
class MoistExchangesNN(nn.Module):
    def __init__(self):
        super(MoistExchangesNN, self).__init__()
        self.fc1 = nn.Linear(5,10)
        self.fc2 = nn.Linear(10,60)
        self.fc3 = nn.Linear(60,60)
        self.fc4 = nn.Linear(60,60)
        self.fc5 = nn.Linear(60,60)
        self.fc6 = nn.Linear(60,12)
        self.fc7 = nn.Linear(12,3)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()
        self.relu4 = nn.LeakyReLU()
        self.relu5 = nn.LeakyReLU()
        self.relu6 = nn.LeakyReLU()

        # set some initial values
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
        nn.init.kaiming_normal_(self.fc5.weight)
        nn.init.zeros_(self.fc5.bias)
        nn.init.kaiming_normal_(self.fc6.weight)
        nn.init.zeros_(self.fc6.bias)
        nn.init.kaiming_normal_(self.fc7.weight)
        nn.init.zeros_(self.fc7.bias)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        return x

restart = False

# Training/runtime configuration
batch_size = 16384 #8192 #4096 #2048 #512
start_lr = 1.4e-3 #1.8e-3 #1.2e-3 #8e-4 #1.0e-3
min_lr = 1.0e-8
num_workers = 8
num_epochs = 800
log_every_n_steps = 50
validate_every_n_epochs = 1

logger = configure_logger(__name__)

results_dir = 'results'
cache_dir = os.path.join(results_dir, 'cache_npy')
os.makedirs(cache_dir, exist_ok=True)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(16)

torch.manual_seed(101)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Using device: {device}')

# load the training data from file
path = '/g/data/dp9/dl9118/lfric_ral_training/scripts/updated_11/'

logger.info(f'Data loading starts from: {path}')
data_load_t0 = time.perf_counter()

load_workers = min(16, max(1, (os.cpu_count() or 1)))
prebuild_default_caches(path, cache_dir, load_workers, logger=logger)

logger.info('Loading unfiltered transport arrays used to build mask')
t_mask = time.perf_counter()
Dq_v = load_array_cached(os.path.join(path, 'Dq_v_loss.txt'), cache_dir, logger=logger)
Dq_l = load_array_cached(os.path.join(path, 'Dq_l_loss.txt'), cache_dir, logger=logger)
Dq_i = load_array_cached(os.path.join(path, 'Dq_i_loss.txt'), cache_dir, logger=logger)
logger.info(f'Loaded mask arrays in {time.perf_counter() - t_mask:.2f} s')

use_v = np.abs(Dq_v) > 1.0e-7
use_l = np.abs(Dq_l) > 1.0e-7
use_i = np.abs(Dq_i) > 1.0e-7
use_vl = np.logical_or(use_v, use_l)
use = np.logical_or(use_vl, use_i)
logger.info(f'Mask selected {int(np.count_nonzero(use))} / {use.shape[0]} samples')

Dq_v = Dq_v[use]
Dq_l = Dq_l[use]
Dq_i = Dq_i[use]
logger.info('Loading filtered thermodynamic/state arrays')
mu_v = load_filtered_array(path, 'mu_v_loss.txt', use, cache_dir, logger=logger)
mu_l = load_filtered_array(path, 'mu_l_loss.txt', use, cache_dir, logger=logger)
mu_i = load_filtered_array(path, 'mu_i_loss.txt', use, cache_dir, logger=logger)
temp = load_filtered_array(path, 'temp_loss.txt', use, cache_dir, logger=logger)
rho  = load_filtered_array(path, 'rho_loss.txt', use, cache_dir, logger=logger)
q_v = load_filtered_array(path, 'qv_loss.txt', use, cache_dir, logger=logger)
q_l = load_filtered_array(path, 'ql_loss.txt', use, cache_dir, logger=logger)
q_i = load_filtered_array(path, 'qi_loss.txt', use, cache_dir, logger=logger)
eta = load_filtered_array(path, 'eta_loss.txt', use, cache_dir, logger=logger)

logger.info(f'Data loading complete in {time.perf_counter() - data_load_t0:.2f} s')

Dq_v_t = torch.from_numpy(Dq_v)#.float()
Dq_l_t = torch.from_numpy(Dq_l)#.float()
Dq_i_t = torch.from_numpy(Dq_i)#.float()
mu_v_t = torch.from_numpy(mu_v)#.float()
mu_l_t = torch.from_numpy(mu_l)#.float()
mu_i_t = torch.from_numpy(mu_i)#.float()
temp_t = torch.from_numpy(temp)#.float()
rho_t = torch.from_numpy(rho)#.float()
q_v_t = torch.from_numpy(q_v)#.float()
q_l_t = torch.from_numpy(q_l)#.float()
q_i_t = torch.from_numpy(q_i)#.float()
eta_t = torch.from_numpy(eta)#.float()

trial_cpu = torch.stack([Dq_v_t[0::2],Dq_l_t[0::2],Dq_i_t[0::2],eta_t[0::2],mu_v_t[0::2],mu_l_t[0::2],mu_i_t[0::2],temp_t[0::2],rho_t[0::2],q_v_t[0::2],q_l_t[0::2],q_i_t[0::2]],dim=1)
test_cpu = torch.stack([Dq_v_t[1::4],Dq_l_t[1::4],Dq_i_t[1::4],eta_t[1::4],mu_v_t[1::4],mu_l_t[1::4],mu_i_t[1::4],temp_t[1::4],rho_t[1::4],q_v_t[1::4],q_l_t[1::4],q_i_t[1::4]],dim=1)

loader_workers = max(0, num_workers)
loader_kwargs = {
    'pin_memory': True,
    'num_workers': loader_workers,
}
if loader_workers > 0:
    loader_kwargs['persistent_workers'] = True
    loader_kwargs['prefetch_factor'] = 4

train_loader = DataLoader(
    TensorDataset(trial_cpu),
    batch_size=batch_size,
    shuffle=True,
    **loader_kwargs,
)
test_loader = DataLoader(
    TensorDataset(test_cpu),
    batch_size=batch_size,
    shuffle=False,
    **loader_kwargs,
)
logger.info(
    f'DataLoader ready: train_samples={trial_cpu.shape[0]}, val_samples={test_cpu.shape[0]}, '
    f'batch_size={batch_size}, workers={loader_workers}, '
    f'prefetch_factor={loader_kwargs.get("prefetch_factor", "n/a")}'
)

use_pred = False
if restart:
    checkpoint = torch.load('/g/data/dp9/dl9118/lfric_ral_training/runs/nAdv_nEta_wBatch/prev_71/model_min_loss_4.pt', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model = MoistExchangesNN()
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint
    model.eval()
    use_pred = True
else:
    model = MoistExchangesNN()

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=start_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8, min_lr=min_lr
)
penalty_magnitude = torch.tensor(1.0e+0,device=device)

small_number = torch.tensor(1.0e-15,device=device)
transfer_scale = torch.tensor(1.0e-12,device=device)

mse_loss = nn.MSELoss()

best_val_epoch_loss = np.inf
best_epoch = -1

train_log_path, val_log_path, train_step_log_path, val_step_log_path = init_metric_logs(results_dir)

logger.info(
    f'Config: batch_size={batch_size}, start_lr={start_lr:.2e}, '
    f'min_lr={min_lr:.2e}, '
    f'num_workers={loader_workers}, num_epochs={num_epochs}, '
    f'log_every_n_steps={log_every_n_steps}, validate_every_n_epochs={validate_every_n_epochs}'
)

#pred = small_number*torch.randn([batch_size,3],dtype=torch.float32,device=device,requires_grad=True)
pred = small_number*torch.randn([batch_size,3],device=device,requires_grad=True)
counter = 0

for epoch in range(num_epochs):
    epoch_t0 = time.perf_counter()
    #if epoch == 10:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 1.0e-4

    model.train()

    train_nA_gt0 = 0
    epoch_loss_t = 0.0
    epoch_loss_v = 0.0
    epoch_loss_l = 0.0
    epoch_loss_i = 0.0
    epoch_loss_e = 0.0
    epoch_loss_A = 0.0
    train_batches = 0
    epoch_grad_norm = 0.0
    train_t0 = time.perf_counter()
    for step, (x_cpu,) in enumerate(train_loader, start=1):
        step_t0 = time.perf_counter()
        # try to do async CPU to GPU copy, and only do this for the current batch, not the entire TensorDataset(trial_cpu)
        x = x_cpu.to(device, non_blocking=True)

        optimizer.zero_grad() # clear the calculated gradients for previous batch
        if use_pred:
            pred = model(x[:,[9,10,11,3,8]])

        pred.requires_grad_ = True
        use_pred = True

        l_v = x[:, 8] * (x[:, 9] + x[:,10]) * (x[:, 4] - x[:, 5]) * pred[:, 0] + x[:, 8] * (x[:, 9] + x[:,11]) * (x[:, 4] - x[:, 6]) * pred[:, 1]
        l_l = x[:, 8] * (x[:, 9] + x[:,10]) * (x[:, 5] - x[:, 4]) * pred[:, 0] + x[:, 8] * (x[:,10] + x[:,11]) * (x[:, 5] - x[:, 6]) * pred[:, 2]
        l_i = x[:, 8] * (x[:, 9] + x[:,11]) * (x[:, 6] - x[:, 4]) * pred[:, 1] + x[:, 8] * (x[:,10] + x[:,11]) * (x[:, 6] - x[:, 5]) * pred[:, 2]
 
        loss_v = mse_loss(x[:, 0], l_v)
        loss_l = mse_loss(x[:, 1], l_l)
        loss_i = mse_loss(x[:, 2], l_i)

        # lagrange multiplier to penalise the loss function where the entropy generation is -ve
        mA = ((x[:, 4] - x[:, 5])*(x[:, 4] - x[:, 5]) * pred[:, 0] + \
              (x[:, 4] - x[:, 6])*(x[:, 4] - x[:, 6]) * pred[:, 1] + \
              (x[:, 5] - x[:, 6])*(x[:, 5] - x[:, 6]) * pred[:, 2]) / (x[:, 7] * x[:, 7])

        fail_2nd_law = mA > 0.0
        mA_gt0 = mA[fail_2nd_law]
        train_nA_gt0 += mA_gt0.shape[0]

        l_e = -mA * x[:, 8] * x[:, 7]
        loss_e = mse_loss(x[:, 3], l_e)

        scale = penalty_magnitude
        if mA_gt0.shape[0] > 0:
            scale = scale/mA_gt0.shape[0]
        else:
            scale = 0.0
        loss_mA = scale*torch.dot(mA_gt0, mA_gt0)

        # Requested behavior: do not include loss_mA in the optimization objective.
        loss = loss_v + loss_l + loss_i

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        epoch_grad_norm += grad_norm.item()
        optimizer.step()

        epoch_loss_v += loss_v.item()
        epoch_loss_l += loss_l.item()
        epoch_loss_i += loss_i.item()
        epoch_loss_e += loss_e.item()
        epoch_loss_t += loss.item()
        epoch_loss_A += loss_mA.item()
        train_batches += 1

        if step % log_every_n_steps == 0:
            append_line(
                train_step_log_path,
                f'{epoch}\t{step}\t{loss_v.item():.8e}\t{loss_l.item():.8e}\t{loss_i.item():.8e}\t'
                f'{loss_e.item():.8e}\t{loss_mA.item():.8e}\t{loss.item():.8e}\t'
                f'{optimizer.param_groups[0]["lr"]:.2e}\t{time.perf_counter() - step_t0:.4f}'
            )

    # Use mean epoch losses for diagnostics and checkpointing.
    train_epoch_loss_v = epoch_loss_v / max(train_batches, 1)
    train_epoch_loss_l = epoch_loss_l / max(train_batches, 1)
    train_epoch_loss_i = epoch_loss_i / max(train_batches, 1)
    train_epoch_loss_e = epoch_loss_e / max(train_batches, 1)
    train_epoch_loss_t = epoch_loss_t / max(train_batches, 1)
    train_epoch_loss_A = epoch_loss_A / max(train_batches, 1)
    train_time_s = time.perf_counter() - train_t0
    mean_grad_norm = epoch_grad_norm / max(train_batches, 1)
    current_lr = optimizer.param_groups[0]['lr']

    append_line(
        train_log_path,
        f'{epoch}\t{train_epoch_loss_v:.8e}\t{train_epoch_loss_l:.8e}\t{train_epoch_loss_i:.8e}\t{train_epoch_loss_e:.8e}\t{train_epoch_loss_t:.8e}\t{train_epoch_loss_A:.8e}\t{train_nA_gt0}\t{train_time_s:.4f}\t{current_lr:.2e}\t{mean_grad_norm:.4e}'
    )

    epoch_loss_t = 0.0
    epoch_loss_v = 0.0
    epoch_loss_l = 0.0
    epoch_loss_i = 0.0
    epoch_loss_e = 0.0
    epoch_loss_A = 0.0
    val_nA_gt0 = 0
    val_batches = 0
    run_validation = ((epoch + 1) % validate_every_n_epochs == 0)
    if run_validation:
        val_t0 = time.perf_counter()
        with torch.no_grad():
            for step, (x_cpu,) in enumerate(test_loader, start=1):
                step_t0 = time.perf_counter()
                x = x_cpu.to(device, non_blocking=True)

                pred = model(x[:,[9,10,11,3,8]])

                l_v = x[:, 8] * (x[:, 9] + x[:,10]) * (x[:, 4] - x[:, 5]) * pred[:, 0] + x[:, 8] * (x[:, 9] + x[:,11]) * (x[:, 4] - x[:, 6]) * pred[:, 1]
                l_l = x[:, 8] * (x[:, 9] + x[:,10]) * (x[:, 5] - x[:, 4]) * pred[:, 0] + x[:, 8] * (x[:,10] + x[:,11]) * (x[:, 5] - x[:, 6]) * pred[:, 2]
                l_i = x[:, 8] * (x[:, 9] + x[:,11]) * (x[:, 6] - x[:, 4]) * pred[:, 1] + x[:, 8] * (x[:,10] + x[:,11]) * (x[:, 6] - x[:, 5]) * pred[:, 2]
 
                loss_v = mse_loss(x[:, 0], l_v)
                loss_l = mse_loss(x[:, 1], l_l)
                loss_i = mse_loss(x[:, 2], l_i)

                # lagrange multiplier to penalise the loss function where the entropy generation is -ve
                mA = ((x[:, 4] - x[:, 5]) * (x[:, 4] - x[:, 5]) * pred[:, 0] + \
                      (x[:, 4] - x[:, 6]) * (x[:, 4] - x[:, 6]) * pred[:, 1] + \
                      (x[:, 5] - x[:, 6]) * (x[:, 5] - x[:, 6]) * pred[:, 2]) / (x[:, 7] * x[:, 7])

                fail_2nd_law = mA > 0.0
                mA_gt0 = mA[fail_2nd_law]
                val_nA_gt0 += mA_gt0.shape[0]

                l_e = -mA * x[:, 8] * x[:, 7]
                loss_e = mse_loss(x[:, 3], l_e)

                #scale = penalty_magnitude
                scale = 1.0
                if mA_gt0.shape[0] > 0:
                    scale = scale/mA_gt0.shape[0]
                else:
                    scale = 0.0
                loss_mA = scale*torch.dot(mA_gt0, mA_gt0)

                loss = loss_v + loss_l + loss_i

                epoch_loss_v += loss_v.item()
                epoch_loss_l += loss_l.item()
                epoch_loss_i += loss_i.item()
                epoch_loss_t += loss.item()
                epoch_loss_e += loss_e.item()
                epoch_loss_A += loss_mA.item()
                val_batches += 1

                if step % log_every_n_steps == 0:
                    append_line(
                        val_step_log_path,
                        f'{epoch}\t{step}\t{loss_v.item():.8e}\t{loss_l.item():.8e}\t{loss_i.item():.8e}\t'
                        f'{loss_e.item():.8e}\t{loss_mA.item():.8e}\t{loss.item():.8e}\t'
                        f'{time.perf_counter() - step_t0:.4f}'
                    )

            val_epoch_loss_v = epoch_loss_v / max(val_batches, 1)
            val_epoch_loss_l = epoch_loss_l / max(val_batches, 1)
            val_epoch_loss_i = epoch_loss_i / max(val_batches, 1)
            val_epoch_loss_e = epoch_loss_e / max(val_batches, 1)
            val_epoch_loss_t = epoch_loss_t / max(val_batches, 1)
            val_epoch_loss_A = epoch_loss_A / max(val_batches, 1)
            val_time_s = time.perf_counter() - val_t0
            epoch_time_s = time.perf_counter() - epoch_t0

            # Step LR scheduler based on validation epoch loss.
            scheduler.step(val_epoch_loss_t)

            # Save best checkpoint based on the minimum validation epoch loss.
            if val_epoch_loss_t < best_val_epoch_loss:
                best_val_epoch_loss = val_epoch_loss_t
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_epoch_loss_total': train_epoch_loss_t,
                    'val_epoch_loss_total': val_epoch_loss_t,
                }, os.path.join(results_dir, f'model_min_loss_{counter}_{epoch}.pt'))
                counter = counter + 1

            logger.info(
                f'Epoch {epoch:04d} | '
                f'lr={current_lr:.2e} grad_norm={mean_grad_norm:.2e} | '
                f'time(train={train_time_s:.2f}s, val={val_time_s:.2f}s, total={epoch_time_s:.2f}s) | '
                f'train_total={train_epoch_loss_t:.6e} '
                f'(v={train_epoch_loss_v:.3e}, l={train_epoch_loss_l:.3e}, i={train_epoch_loss_i:.3e}, e={train_epoch_loss_e:.3e}, mA={train_epoch_loss_A:.3e}, nA={train_nA_gt0}) | '
                f'val_total={val_epoch_loss_t:.6e} '
                f'(v={val_epoch_loss_v:.3e}, l={val_epoch_loss_l:.3e}, i={val_epoch_loss_i:.3e}, e={val_epoch_loss_e:.3e}, mA={val_epoch_loss_A:.3e}, nA={val_nA_gt0}) | '
                f'best_val={best_val_epoch_loss:.6e} @ epoch {best_epoch:04d}'
            )

            append_line(
                val_log_path,
                f'{epoch}\t{val_epoch_loss_v:.8e}\t{val_epoch_loss_l:.8e}\t{val_epoch_loss_i:.8e}\t{val_epoch_loss_e:.8e}\t{val_epoch_loss_t:.8e}\t{val_epoch_loss_A:.8e}\t{val_nA_gt0}\t{val_time_s:.4f}\t{epoch_time_s:.4f}'
            )
    else:
        logger.info(
            f'Epoch {epoch:04d} | '
            f'lr={current_lr:.2e} grad_norm={mean_grad_norm:.2e} | '
            f'time(train={train_time_s:.2f}s, val=skipped, total={time.perf_counter() - epoch_t0:.2f}s) | '
            f'train_total={train_epoch_loss_t:.6e} '
            f'(v={train_epoch_loss_v:.3e}, l={train_epoch_loss_l:.3e}, i={train_epoch_loss_i:.3e}, e={train_epoch_loss_e:.3e}, mA={train_epoch_loss_A:.3e}, nA={train_nA_gt0}) | '
            f'validation skipped (validate_every_n_epochs={validate_every_n_epochs}) | '
            f'best_val={best_val_epoch_loss:.6e} @ epoch {best_epoch:04d}'
        )

torch.save(model, os.path.join(results_dir, 'model_complete.pt'))
