import configs
from trainer import SemEvalTrainer


# retrieving the args
args = configs.parser.parse_args()

trainer = SemEvalTrainer(args)

trainer.train()



