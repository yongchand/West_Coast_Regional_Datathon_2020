# Code originated from ML CLass by Yi-Chieh Wu

# python modules
from abc import ABC

# sklearn modules
from sklearn import preprocessing
from sklearn import impute
######################################################################
# classes
######################################################################

class Preprocessor(ABC):
    """Base class for preprocessor with hyper-parameter optimization.
    Attributes
    --------------------
        transformer_  -- transformer object
            This is assumed to implement the scikit-learn transformer interface.
        param_grid_ -- dict or list of dictionaries
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.
    """
    def __init__(self):
        self.transformer_ = None
        self.param_grid_ = None


class Imputer(Preprocessor):
    def __init__(self):
        self.param_grid_ = None
        self.transformer_ = impute.SimpleImputer(strategy='constant', fill_value=0)

class Scaler(Preprocessor):
    def __init__(self):
        self.param_grid_ = None
        self.transformer_ = preprocessing.StandardScaler()

######################################################################
# globals
######################################################################

PREPROCESSORS = [c.__name__ for c in Preprocessor.__subclasses__()]