from abc import ABC, abstractmethod

class ILinguisticVariableStore(ABC):
    @abstractmethod
    def add_variable(self, name, variable):
        pass

    @abstractmethod
    def get_variable(self, name):
        pass

    @abstractmethod
    def has_variable(self, name):
        pass

    @abstractmethod
    def remove_variable(self, name):
        pass

    @abstractmethod
    def get_all_variables(self):
        pass

class LocalLinguisticVariableStore(ILinguisticVariableStore):
    def __init__(self):
        self.variables = {}

    def add_variable(self, name, variable):
        self.variables[name] = variable

    def get_variable(self, name):
        return self.variables.get(name)

    def has_variable(self, name):
        return name in self.variables

    def remove_variable(self, name):
        if name in self.variables:
            del self.variables[name]

    def get_all_variables(self):
        return self.variables
