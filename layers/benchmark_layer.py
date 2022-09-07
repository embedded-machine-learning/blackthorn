"""Common layer class for benchmarking"""

from layers.layer_definitions import ConvolutionTuple


class BenchmarkLayer():
    """
    Common layer class for benchmarking.
    Every layer is passed to the specified backend using this class!
    """

    def __init__(self, layer_definition) -> None:

        self.type = layer_definition.__name__

        self._params = [p.value for p in layer_definition]
        self._sweep_param = None

        # Dict holding the layer configuration
        self._config = dict.fromkeys(self._params)


    # Methods
    def get_layer(self):
        """Return layer as namedtuple according to the layer type"""

        if not all(list(self._config.values())):
            raise ValueError('Layer configuration contains at least one "None" value: {}'.format(self._config))

        if self.type == 'Convolution':
            return ConvolutionTuple(**self._config)

        raise NotImplementedError('Layer of type {} is currently not implemented'.format(self.type))

    def set_estimation_param(self, param: str) -> None:
        """Set parameter which will be varied for estimation"""

        if not param in self._params:
            raise KeyError('Parameter {} not part of this layer type ({})'.format(param, self.type))
        self._sweep_param = param

    def update_estimation_value(self, value: int) -> None:
        """Update value of the varied parameter"""

        self._params[self._sweep_param] = value

    def update_config(self, config: dict) -> None:
        """Update layer configuration"""

        if not all(e in self._params for e in list(config.keys())):
            raise KeyError(('At least one parameter given ({}) is not part of this layer type ({}). '
                            'Possible parameters are: {}').format(list(config.keys()), self.type, self._params))

        self._config.update(config)
