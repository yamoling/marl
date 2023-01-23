from typing import Tuple
import torch


class ICM_NN(torch.nn.Module):
    """Neural network used by the intrinsic curiosity module (ICM)"""

    def __init__(self, input_shape, n_actions, feature_extractor) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor

        self.inverse_model = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0] * 2, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, n_actions)
        )

        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0] + n_actions, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, input_shape[0])
        )
        self.inverse_loss_f = torch.nn.CrossEntropyLoss()
        self.forward_loss_f = torch.nn.MSELoss()

    def forward(self, obs: torch.FloatTensor, obs_: torch.FloatTensor, actions: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Forward pass"""
        # Inverse model
        inverse_input = torch.concat([obs, obs_], dim=-1)
        predicted_action_logits = self.inverse_model(inverse_input)

        forward_input = torch.concat([obs, actions], dim=-1)
        predicted_new_state = self.forward_model(forward_input)
        return predicted_action_logits, predicted_new_state
