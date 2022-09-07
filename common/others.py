"""Other stuff"""

class FormattedKeyError(KeyError):
    """
    KeyError with formatted error message.
    Python's `KeyError` has special casing around formatting
    (see https://bugs.python.org/issue2651). Use this class when the error
    message has newlines and other special format characters.
    """

    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message
