class BaseEnv:
    """
    Base class for all environments.
    This class should be inherited by all specific environment classes.
    """

    def __init__(self):
        """
        Initialize the environment.
        This method can be overridden in subclasses to set up specific environment parameters.
        """
        pass
    
    def reset(self):
        """
        Reset the environment to its initial state.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("reset method must be implemented in subclasses.")
    
    def feedback(self, x, action):
        """
        Provide feedback for a given action on a specific input.
        
        Args:
            x: The input data (e.g., prompt).
            action: The action taken (e.g., model selection).
        
        Returns:
            response: The response from the environment.
            reward: The reward received for the action.
            cost: The cost incurred for the action.
        """
        raise NotImplementedError("feedback method must be implemented in subclasses.")
    
    def support_length(self):
        """
        Return the maximum number of rounds the environment can support.

        If the environment does not have a limit, return None.
        """
        return None
    
    def get_sample(self):
        """
        Get a sample from the environment dataset.
        """
        raise NotImplementedError("get_sample method must be implemented in subclasses.")