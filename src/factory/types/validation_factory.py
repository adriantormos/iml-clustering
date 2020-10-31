from src.validation.validation_method import ValidationMethod
from src.validation.types.davies_bouldin import DaviesBouldinScore


class ValidationFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_validation(config, output_path, verbose) -> ValidationMethod:
        name = config['name']
        if name == 'daviesbouldin':
            method = DaviesBouldinScore(config, output_path, verbose)
        else:
            raise Exception('The algorithm with name ' + name + ' does not exist')
        if issubclass(type(method), ValidationMethod):
            return method
