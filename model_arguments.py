from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # kd setting
    alpha_kd: float = field(
        default=1.0,
        metadata={
            "help": "The weight of kd loss"
        },
    )
    # mode: str = field(
    #     default="kd",
    #     metadata={"help": "The type of kd loss"},
    # )
    temperature: float = field(
        default=1,
        metadata={
            "help": "The temperature."
        },
    )

    # teacher model setting
    teacher_model: str = field(
        default=None,
        metadata={
            "help": "Path of teacher model."
        },
    )
    train_teacher: bool = field(
        default=False,
        metadata={
            "help": "Train teacher or not."
        },
    )
    t_alpha_kd: float = field(
        default=0.4,
        metadata={
            "help": "The weight of kd loss if train_teacher is True."
        },
    )
    t_learning_rate: float = field(
        default=3e-5,
        metadata={
            "help": "The learning rate of teacher."
        },
    )

    # lgtm setting
    use_lgtm: bool = field(
        default=False,
        metadata={
            "help": "Use LGTM or not."
        },
    )
    init_classifier_to_zero: bool = field(
        default=False,
        metadata={
            "help": "Initialize the classifier of the teacher and student to zero."
        },
    )
    num_layers: int = field(
        default=6,
        metadata={
            "help": "The layer number of the student model."
        },
    )