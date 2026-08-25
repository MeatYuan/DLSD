from Algorithm import get_algorithm
from args import parse_args
from datasets.Data import Create_Data
from TargetModels.main import get_target_models
from utils.network import Sample_Client
from utils.utils import init_logging, set_seed


def main(args):
    init_logging(args)
    set_seed(args)

    # Client data and the per-client target models. Both are built before the
    # algorithm is chosen, so every algorithm sees an identical starting point.
    Client_Data = Create_Data(args)
    models, model_list = get_target_models(args)

    # Seed the shared client sampler, so the participation sequence is reproducible
    # and identical across algorithms.
    Sample_Client(num_clients=args.num_clients, init_seed=args.random_seed)

    run = get_algorithm(args.algorithm)
    print(f"Running {args.algorithm} Algorithm")
    return run(Client_Data=Client_Data, models=models, model_list=model_list, args=args)


if __name__ == "__main__":
    main(parse_args())
