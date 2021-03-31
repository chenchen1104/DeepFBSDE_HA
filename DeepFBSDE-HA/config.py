class Config(object):
    batch_size = 64
    valid_size = 1280
    num_iterations = 6000
    logging_frequency = 10
    verbose = True
    constrained = True
    y_init_range = [0, 1]
    max_grad_norm = 10000
    lstm = False

class AircraftConfig(Config):
    DELTA_CLIP = 50.0
    weight_decay = 1e-5
    alpha_threshold_max = 10
    alpha_threshold_min = -1
    u_threhold_max = 20
    u_threhold_min = -20
    num_iterations = 1000
    umax = 20
    dim = 3
    model_save_path = "./model.pth"

    total_time = 1.5
    delta_t = 0.02
    lr_value = 2e-3
    num_hiddens = [16, 16]
    lstm_num_layers = 1
    lstm_hidden_size = 16
    y_init_range = [1, 2]
    z_init_range = [-0.1, 0.1]


def get_config(name):
    try:
        return globals()[name + 'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")
