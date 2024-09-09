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

    def update_variable_terms(self, name, new_fuzzy_sets):
        """
        Update the terms (fuzzy sets) for a given linguistic variable.
        
        Args:
            name (str): Name of the linguistic variable to update.
            new_fuzzy_sets (list): List of new fuzzy sets to replace the old ones.
        """
        if self.has_variable(name):
            linguistic_variable = self.get_variable(name)
            linguistic_variable._FS_list = new_fuzzy_sets  # Update the fuzzy sets
            print(f"Updated terms for variable '{name}'")
        else:
            print(f"Variable '{name}' not found in the store.")
