from dataloader import SquadDataset
from parser_utils import get_args
import logging
from copy import deepcopy
import configargparse
from constants import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)


if __name__ == "__main__":

    ##################################################
    # Collect main args
    ##################################################

    args, parser, device = get_args()

    parser = SquadDataset.add_argparse_args(parser)

    # Workaround to parse args from config file twice
    _actions = deepcopy(parser._actions)
    _option_string_actions = deepcopy(parser._option_string_actions)
    parser = configargparse.ArgParser(default_config_files=["parameters.ini"])
    parser._actions = _actions
    parser._option_string_actions = _option_string_actions
    # End workaround

    args, unknown = parser.parse_known_args()
 
    logger.info(f"parsed args: {args}")
    if len(unknown) > 0:
        logger.info(f"Unknown args: {unknown}")

    ##################################################
    # Setup data loader
    ##################################################
    
    dataset = SquadDataset(
        model_name = args.model_name,
        batch_size = args.batch_size,
        dev_ratio = args.dev_split,
        test_ratio = args.train_split,
        seed = args.seed,
        llmv3_max_length = args.llmv3_max_length,
        roberta_max_length = args.roberta_max_length,
        num_workers=args.num_workers
        )
    
    train_dataloader = dataset.dataloader('train')
    eval_dataloader = dataset.dataloader('dev')
    test_dataloader = dataset.dataloader('test')

    batch = next(iter(train_dataloader)) # just for this PR visualization
    logger.info(batch) # just for this PR visualization