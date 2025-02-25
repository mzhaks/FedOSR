import torch
import torch.nn.functional as F
from torch.autograd import Variable

def l2_norm(x, keepdim=True):
    return torch.norm(x, p=2, dim=tuple(range(1, x.dim())), keepdim=keepdim)



class Attack:
    """
    Implements adversarial attack methods: 
    - DUS (Direct Uncertainty Sampling)
    - i_DUS (Iterative Direct Uncertainty Sampling)

    Args:
        known_class (int, optional): Class label considered known. Defaults to None.
        eps (float, optional): Perturbation step size. Defaults to 5.0 * 16 / 255.0.
        num_steps (int, optional): Number of iterations for iterative attacks. Defaults to 10.
        x_val_min (float, optional): Minimum allowable value for input. Defaults to -2.5.
        x_val_max (float, optional): Maximum allowable value for input. Defaults to 2.5.
    """

    def __init__(self, known_class=None, eps=5.0 * 16 / 255.0, num_steps=10, x_val_min=-2.5, x_val_max=2.5):
        self.eps = eps
        self.num_steps = num_steps
        self.criterion = F.cross_entropy  # Loss function
        self.x_val_min = x_val_min
        self.x_val_max = x_val_max
        self.known_class = known_class

    def DUS(self, model, inputs, targets, eps=0.03):
        """
        Implements the Direct Uncertainty Sampling (DUS) attack.
        Generates an adversarial example using a single-step attack.

        Args:
            model (torch.nn.Module): The target model.
            inputs (torch.Tensor): Input samples.
            targets (torch.Tensor): Corresponding labels.
            eps (float, optional): Perturbation magnitude. Defaults to 0.03.

        Returns:
            torch.Tensor: Adversarial inputs.
            torch.Tensor: Model outputs for adversarial inputs.
        """
        x_adv = Variable(inputs.clone().detach(), requires_grad=True)
        model.eval()  # Set model to evaluation mode
        outputs = model(x_adv)

        # Compute the adversarial loss (negative cross-entropy)
        loss = -self.criterion(outputs, targets)

        model.zero_grad()  # Ensure gradients are zero before backpropagation
        if x_adv.grad is not None:
            x_adv.grad.data.zero_()

        # Backpropagate the loss
        loss.backward()

        # Compute adversarial perturbation
        x_adv = x_adv - eps * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, self.x_val_min, self.x_val_max)  # Clip values

        # Get model output for adversarial input
        adv_outputs = model(x_adv)
        model.train()  # Switch model back to training mode

        return x_adv, adv_outputs

    def i_DUS(self, model, inputs, targets):
        """
        Implements Iterative Direct Uncertainty Sampling (i_DUS).
        Generates adversarial examples using an iterative approach.

        Args:
            model (torch.nn.Module): The target model.
            inputs (torch.Tensor): Input samples.
            targets (torch.Tensor): Corresponding labels.

        Returns:
            torch.Tensor: Adversarial inputs.
            torch.Tensor: Corresponding adversarial targets.
        """
        x_adv = inputs.clone().detach() # (is_boundary.sum(), 256, 4, 4)
        model.eval()  # Set model to evaluation mode

        for _ in range(self.num_steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)

            # Get model output
            outputs = model.discrete_forward(x_adv)["outputs"]  # shape (is_boundary.sum(), 11)

            # Compute adversarial loss
            loss = -self.criterion(outputs, targets)

            model.zero_grad()  # Reset gradients
            if x_adv.grad is not None:
                x_adv.grad.data.zero_()

            # Backpropagate the loss
            loss.backward()

            # Update adversarial example
            x_adv = x_adv - self.eps * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, self.x_val_min, self.x_val_max)  # Clip values

        model.train()  # Switch model back to training mode
        return x_adv.detach(), targets.detach()  #(batch, 256,4,4),  (batch,)