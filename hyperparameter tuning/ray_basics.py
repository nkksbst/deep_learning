from functools import partial

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Data Loaders 
# pass a global data directory that will be shared among trials
def load_data(data_dir = "./data"):
    pass

# Train Function
# must define this function all the time as this will be the function to be run by all trials
def train(config, checkpoint_dir = None, data_dir = None):
    """
    config: contain all necessary hyperparameters with which ray will sample from
    checkpoint_dir: used for restoring checkpoints
    data_dir: path for data used by all trials
    """
    pass

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    # the search space for the hyperparameters are defined in config dictionary
    config = {
        "hyperparam1": tune.loguniform(1e-3, 1e-1),
        "hyperparam2": tune.loguniform(1e-3, 1e-1),
        "hyperparam3": tune.choice([16, 32, 64, 128]),
        "hyperparam4": tune.loguniform(1e-4, 1e-1),
        "hyperparam5": args
    }
    # the scheduler schedules trials and uses the metric to decide early terminations
    # by maximizing/minimizing (depending on the mode) the defined metric 
    scheduler = ASHAScheduler(
        metric="episode_reward",
        mode="max",
        time_attr = "epoch", # time unit used for quantifying time
        max_t=max_num_epochs, # max time units per trial
        grace_period=1, # only stop trials that have this minimum 'age'
        reduction_factor=2)

    # the reporter prints the current progress (in a table)
    # for trials with the following parameters and metrics
    reporter = CLIReporter(
        parameter_columns   =   ["param1", "param2", "param3", "param4"],
        metric_columns      =   ["episode_reward", "task_loss"]
    )

    data_dir = './data'

    result = tune.run(
    partial(train, data_dir=data_dir),
    resources_per_trial={"gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    name = 'ray_basics',
    progress_reporter=reporter)

    print('result: ', result)

    best_trial = result.get_best_trial("episode_reward", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["val_accuracy"]))
    print("Best trial final episode_reward accuracy: {}".format(
        best_trial.last_result["episode_reward"]))

    #args.model_state_path = best_trial.checkpoint.value
    #best_trained_model = setup_vit(args)
    #best_trained_model.to(args.device)

    #best_checkpoint_dir = best_trial.checkpoint.value
    #model_state, optimizer_state = torch.load(os.path.join(
    #best_checkpoint_dir, "checkpoint"))
    #best_trained_model.load_state_dict(model_state)

    #test_acc = test_accuracy(best_trained_model, args.device)
    #print("Best trial test set accuracy: {}".format(test_acc))

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=20, max_num_epochs=10, gpus_per_trial=1)