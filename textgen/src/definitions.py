"""
Constants to be used throughout the project these include:
 - string constants (such as dates), common phrases, dates
 - numeric constants such as factors and single inputs
 - all file paths (data, figures etc)
 """
from datetime import datetime
from pathlib import Path

date_today: str = str(datetime.now().date())


class Dirs:
    """namespace for directory paths"""
    ROOT = Path(__file__).parent.parent
    DATA = ROOT / 'data'  # store for intermediate analysis
    OUTPUT = ROOT / 'output'
    CHARTS = OUTPUT / 'charts'
    EXTERNAL = DATA / 'external'
    INTERIM = DATA / 'interim'
    PROCESSED = DATA / 'processed'
    RAW = DATA / 'raw'

    MODELS = ROOT / 'models'

    FIGURES = ROOT / 'reports' / 'figures'

    TEXT: Path = ROOT.home() / r'Google Drive (a.lewzey@gmail.com)/data/text'


class External:
    """namespace for external data file paths"""
    LYRICS55000 = Dirs.TEXT / '55000-song-lyrics' / 'songdata.csv'


class Interim:
    """namespace for interim data file paths"""
    pass


class Processed:
    """namespace for processed data file paths"""
    pass


class Raw:
    """namespace for raw data file paths"""
    pass


class Models:
    """namespace for serialised model paths"""
    TEXT_CLASS = Dirs.MODELS / f'text_class_{date_today}'

    CKPT_DIR_TEXT_SEQ = Dirs.MODELS / 'ckpts_text_seq'
    CHECKPOINT_TEXT_SEQ = CKPT_DIR_TEXT_SEQ / 'ckpt_{epoch:02d}'

    CKPT_DIR_TABULAR = Dirs.MODELS / 'ckpts_tabular'
    CKPT_TABULAR = CKPT_DIR_TABULAR / 'ckpt_{epoch:02d}'


class Charts:
    """namespace for chart file paths"""
    pass


# dates


# string constants


# numeric constants
MIL: int = 1_000_000
